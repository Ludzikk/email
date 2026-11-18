"""Ekstrakcja bogatych informacji z Enron (CSV/maildir) i Nazario MBOX.

Nie pobiera linkow, nie otwiera zalacznikow, nie zapisuje ich zawartosci.
Wymaga: pandas. Uruchom: py prepare_email_datasets.py
"""
from __future__ import annotations

import html
import ipaddress
import json
import mailbox
import os
import random
import re
from collections import Counter
from email import policy
from email.parser import BytesParser, Parser
from email.utils import getaddresses, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

ROOT = Path(__file__).resolve().parent
ENRON_DIR = ROOT / "data" / "raw" / "enron"
ENRON_INPUT = ENRON_DIR / "emails.csv"
NAZARIO_DIR = ROOT / "data" / "raw" / "nazario"
OUTPUT_DIR = ROOT / "data" / "processed"
ENRON_SAMPLE_SIZE = 50_000  # None = wszystkie wiadomosci Enron (moze byc bardzo duzo)
CHUNK_SIZE = 5_000
SEED = 42
NAZARIO_YEARS = (2018, 2019, 2020, 2021, 2024, 2025)

# 'date' jest celowo pominiete, zgodnie z wybranym schematem projektu.
BASE = ["sender", "receiver", "reply_to", "subject", "message", "label", "source"]
HEADERS = {
    "cc": "Cc", "bcc": "Bcc", "return_path": "Return-Path",
    "message_id": "Message-ID", "in_reply_to": "In-Reply-To",
    "references": "References", "sender_header": "Sender",
    "content_type": "Content-Type", "content_transfer_encoding": "Content-Transfer-Encoding",
    "mime_version": "MIME-Version", "user_agent": "User-Agent", "x_mailer": "X-Mailer",
    "organization": "Organization", "priority": "Priority", "x_priority": "X-Priority",
    "importance": "Importance", "authentication_results": "Authentication-Results",
    "received_spf": "Received-SPF", "dkim_signature": "DKIM-Signature",
    "arc_authentication_results": "ARC-Authentication-Results",
    "delivered_to": "Delivered-To", "list_unsubscribe": "List-Unsubscribe",
    "list_id": "List-Id", "x_originating_ip": "X-Originating-IP",
    "x_enron_from": "X-From", "x_enron_to": "X-To", "x_enron_folder": "X-Folder",
    "x_enron_origin": "X-Origin", "x_enron_filename": "X-FileName",
}
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'\]\[(){}]+")

# Niektore surowe maile Enron zawieraja osierocone surogaty Unicode.
# PyArrow/CSV wymagaja poprawnego Unicode (UTF-8).
SURROGATE_RE = re.compile("[\ud800-\udfff]")


def safe_unicode(value):
    """Zamienia niepoprawne surogaty na znak zastepczy U+FFFD."""
    return SURROGATE_RE.sub("\ufffd", value) if isinstance(value, str) else value


def compact(value):
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def get_header(msg, key):
    try:
        return compact(str(msg.get(key) or ""))
    except Exception:
        return ""


def domain_from_header(value):
    addrs = getaddresses([value or ""])
    for _, addr in addrs:
        if "@" in addr:
            return addr.rsplit("@", 1)[-1].lower().strip().strip(">")
    return ""


def count_addresses(value):
    return len([addr for _, addr in getaddresses([value or ""]) if "@" in addr])


def part_text(part):
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except (UnicodeError, LookupError, ValueError, TypeError, KeyError):
        pass
    try:
        raw = part.get_payload(decode=True)
        if isinstance(raw, bytes):
            return raw.decode(part.get_content_charset() or "utf-8", errors="replace")
    except Exception:
        pass
    return ""


class HTMLFacts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.visible = []
        self._ignore = 0
        self.forms = 0
        self.images = 0
        self.scripts = 0
        self.iframes = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self._ignore += 1
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "form":
            self.forms += 1
        if tag == "img":
            self.images += 1
        if tag == "script":
            self.scripts += 1
        if tag == "iframe":
            self.iframes += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._ignore:
            self._ignore -= 1

    def handle_data(self, data):
        if not self._ignore:
            self.visible.append(data)


def clean_url(url):
    return html.unescape(url).strip().rstrip(".,;:!?)\"'")


def host_from_url(url):
    try:
        value = url if "://" in url else "https://" + url
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


def to_record(msg, label, source):
    record = {
        "sender": get_header(msg, "From"), "receiver": get_header(msg, "To"),
        "reply_to": get_header(msg, "Reply-To"), "subject": get_header(msg, "Subject"),
        "label": label, "source": source,
    }
    record.update({key: get_header(msg, name) for key, name in HEADERS.items()})

    # Wszystkie oryginalne naglowki (takze niestandardowe), bez ograniczania do listy HEADERS.
    # JSON zachowuje powtarzajace sie klucze takie jak Received.
    record["all_headers_json"] = json.dumps(
        [[str(k), str(v)] for k, v in msg.raw_items()], ensure_ascii=False
    )
    record["header_names_json"] = json.dumps(
        sorted({str(k).lower() for k in msg.keys()}), ensure_ascii=False
    )
    record["header_count"] = len(list(msg.keys()))
    record["received_count"] = len(msg.get_all("Received", []))
    record["has_spf_header"] = int(bool(msg.get("Received-SPF")))
    record["has_dkim_signature"] = int(bool(msg.get("DKIM-Signature")))
    record["has_auth_results"] = int(bool(msg.get("Authentication-Results")))
    record["is_multipart"] = int(msg.is_multipart())
    record["sender_domain"] = domain_from_header(record["sender"])
    record["receiver_domain"] = domain_from_header(record["receiver"])
    record["reply_to_domain"] = domain_from_header(record["reply_to"])
    record["return_path_domain"] = domain_from_header(record["return_path"])
    record["cc_count"] = count_addresses(record["cc"])
    record["bcc_count"] = count_addresses(record["bcc"])
    record["to_count"] = count_addresses(record["receiver"])
    record["sender_replyto_domain_mismatch"] = int(
        bool(record["sender_domain"] and record["reply_to_domain"])
        and record["sender_domain"] != record["reply_to_domain"]
    )
    record["sender_returnpath_domain_mismatch"] = int(
        bool(record["sender_domain"] and record["return_path_domain"])
        and record["sender_domain"] != record["return_path_domain"]
    )
    record["charset"] = msg.get_content_charset() or ""

    plain_parts, html_parts, filenames, ext, part_types, part_charsets = [], [], [], [], [], []
    attach_count = 0
    attach_bytes = 0
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        part_types.append(ctype)
        part_charsets.append(part.get_content_charset() or "")
        filename = part.get_filename()
        disposition = part.get_content_disposition()
        is_attachment = disposition == "attachment" or bool(filename)
        if is_attachment:
            attach_count += 1
            if filename:
                filenames.append(str(filename))
                ext.append(Path(str(filename)).suffix.lower())
            try:
                payload = part.get_payload(decode=True)
                attach_bytes += len(payload) if isinstance(payload, bytes) else 0
            except Exception:
                pass
            continue
        if ctype == "text/plain":
            plain_parts.append(part_text(part))
        elif ctype == "text/html":
            html_parts.append(part_text(part))

    html_source = "\n".join(html_parts)
    facts = HTMLFacts()
    try:
        facts.feed(html_source)
    except Exception:
        pass
    html_visible = compact(" ".join(facts.visible))
    plain = compact("\n".join(plain_parts))
    body = plain or html_visible
    if not body and not msg.is_multipart() and not attach_count:
        body = compact(part_text(msg))
    record["message"] = body
    record["body_plain"] = plain
    record["body_html"] = html_source  # oryginalny HTML, jesli jest
    record["body_html_visible"] = html_visible
    record["has_plain_part"] = int(bool(plain_parts))
    record["has_html_part"] = int(bool(html_parts))
    record["html_link_count"] = len(facts.links)
    record["html_form_count"] = facts.forms
    record["html_image_count"] = facts.images
    record["html_script_count"] = facts.scripts
    record["html_iframe_count"] = facts.iframes
    record["attachment_count"] = attach_count
    record["has_attachment"] = int(attach_count > 0)
    record["attachment_total_bytes"] = attach_bytes
    record["attachment_filenames_json"] = json.dumps(filenames, ensure_ascii=False)
    record["attachment_extensions_json"] = json.dumps(ext, ensure_ascii=False)
    record["mime_types_json"] = json.dumps(part_types, ensure_ascii=False)
    record["mime_charsets_json"] = json.dumps(part_charsets, ensure_ascii=False)
    record["mime_parts_count"] = len(part_types)

    urls = [clean_url(x) for x in URL_RE.findall(plain + " " + html_visible + " " + html_source)]
    urls.extend(clean_url(u) for u in facts.links if u.lower().startswith(("http://", "https://", "www.")))
    # Unikalne URL zachowujac kolejnosc; adresy widoczne w HTML nie sa liczone podwojnie.
    urls = list(dict.fromkeys(url for url in urls if url))
    hosts = list(dict.fromkeys(host for host in (host_from_url(u) for u in urls) if host))
    record["urls_json"] = json.dumps(urls, ensure_ascii=False)
    record["url_domains_json"] = json.dumps(hosts, ensure_ascii=False)
    record["url_count"] = len(urls)
    record["unique_url_domain_count"] = len(hosts)
    record["has_url"] = int(bool(urls))
    record["url_ip_count"] = sum(
        1 for h in hosts if is_ip_address(h)
    )
    record["subject_length"] = len(record["subject"])
    record["message_length"] = len(body)
    record["message_word_count"] = len(body.split())
    record["exclamation_count"] = body.count("!")
    record["question_count"] = body.count("?")
    record["uppercase_ratio"] = round(
        sum(ch.isupper() for ch in body) / max(1, sum(ch.isalpha() for ch in body)), 6
    )
    # Oczysc wszystkie tekstowe pola, w tym JSON, przed tworzeniem DataFrame.
    return {key: safe_unicode(value) for key, value in record.items()}


def is_ip_address(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def enron_maildir_root():
    for folder in (ENRON_DIR / "maildir", ENRON_DIR):
        if folder.is_dir() and any(p.is_dir() for p in folder.iterdir()):
            return folder
    raise FileNotFoundError("Nie znaleziono Enron emails.csv ani maildir")


def iter_mail_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.startswith(".") and name.lower() not in {"thumbs.db", "desktop.ini"}:
                yield Path(dirpath) / name


def sample_enron():
    rng = random.Random(SEED)
    reservoir, seen = [], 0
    if ENRON_INPUT.is_file():
        print("[ENRON] Wczytywanie emails.csv")
        for chunk in pd.read_csv(ENRON_INPUT, usecols=["message"], chunksize=CHUNK_SIZE,
                                 encoding="utf-8", encoding_errors="replace", on_bad_lines="warn"):
            for value in chunk["message"].dropna():
                seen += 1
                if ENRON_SAMPLE_SIZE is None or len(reservoir) < ENRON_SAMPLE_SIZE:
                    reservoir.append(str(value))
                else:
                    j = rng.randrange(seen)
                    if j < ENRON_SAMPLE_SIZE:
                        reservoir[j] = str(value)
        fmt = "csv"
    else:
        root = enron_maildir_root()
        print(f"[ENRON] Wczytywanie maildir: {root}")
        for path in iter_mail_files(root):
            seen += 1
            if ENRON_SAMPLE_SIZE is None or len(reservoir) < ENRON_SAMPLE_SIZE:
                reservoir.append(path)
            else:
                j = rng.randrange(seen)
                if j < ENRON_SAMPLE_SIZE:
                    reservoir[j] = path
        fmt = "maildir"
    rng.shuffle(reservoir)
    print(f"[ENRON] Przejrzano {seen}; wybrano {len(reservoir)}")
    records, failed = [], 0
    for item in reservoir:
        try:
            if fmt == "csv":
                msg = Parser(policy=policy.default).parsestr(item)
            else:
                with item.open("rb") as f:
                    msg = BytesParser(policy=policy.default).parse(f)
            records.append(to_record(msg, 0, "enron"))
        except Exception:
            failed += 1
    print(f"[ENRON] Pominięte przez błędy parsowania: {failed}")
    return pd.DataFrame.from_records(records)


def nazario_path(year):
    for name in (f"phishing-{year}", f"phishing-{year}.mbox"):
        path = NAZARIO_DIR / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Brak Nazario {year} w {NAZARIO_DIR}")


def process_nazario():
    records = []
    for year in NAZARIO_YEARS:
        path = nazario_path(year)
        archive = mailbox.mbox(str(path), create=False)
        count, failed = 0, 0
        try:
            for item in archive:
                try:
                    msg = BytesParser(policy=policy.default).parsebytes(item.as_bytes())
                    records.append(to_record(msg, 1, f"nazario_{year}"))
                    count += 1
                except Exception:
                    failed += 1
        finally:
            archive.close()
        print(f"[NAZARIO] {year}: {count} wiadomosci, bledy={failed}")
    return pd.DataFrame.from_records(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enron = sample_enron()
    nazario = process_nazario()
    # Identyczny zestaw kolumn nawet jesli jakies pole jest nieobecne w calym zrodle.
    columns = list(dict.fromkeys(BASE + list(HEADERS) + list(enron.columns) + list(nazario.columns)))
    for name, frame in (("enron_processed.csv", enron), ("nazario_processed.csv", nazario)):
        frame = frame.reindex(columns=columns, fill_value="")
        path = OUTPUT_DIR / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[ZAPIS] {path}: {len(frame)} wierszy, {len(frame.columns)} kolumn")
    print("Gotowe: wszystkie lata Nazario w pliku processed; brak podzialu na recente.")
    print("UWAGA: nie uzywaj source, Message-ID, X-Enron ani all_headers_json jako cech modelu.")


if __name__ == "__main__":
    main()
