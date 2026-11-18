
"""Laczy Enron, Nazario (wszystkie lata) i legalne SpamAssassin.

Wejscie:
  data/processed/enron_processed.csv
  data/processed/nazario_processed.csv
  data/raw/extra/SpamAssasin.csv

Wyjscie: data/final_80_20/{combined,train,test}.csv
Podzial 80/20, bez walidacji i backupow. XLSX to tylko kopie pogladowe.
Uruchomienie: py build_datasets.py
"""
from pathlib import Path
from email.utils import getaddresses
from urllib.parse import urlsplit
import ipaddress
import hashlib
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
SPAMASSASSIN_FILE = ROOT / "data" / "raw" / "extra" / "SpamAssasin.csv"
OUTPUT = ROOT / "data" / "splits"
SEED = 42
TEST_SIZE = 0.20
HAM_TO_PHISHING_RATIO = 1.5  # ok. 60% ham / 40% phishing; None = wszystkie ham
BASIC = ["sender", "receiver", "reply_to", "subject", "message", "label", "source"]
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'\]\[(){}]+")
SURROGATES = re.compile("[\ud800-\udfff]")


def clean_string(value):
    return SURROGATES.sub("\ufffd", str(value)).strip()


def normalized(value):
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def fingerprint(value):
    return hashlib.sha256(normalized(value).encode("utf-8")).hexdigest()


def read_processed(filename, expected_label):
    path = PROCESSED / filename
    if not path.is_file():
        raise FileNotFoundError(f"Brak pliku: {path}. Uruchom prepare_email_datasets.py")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, keep_default_na=False)
    missing = {"sender", "receiver", "subject", "message", "label", "source"} - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: brak kolumn {sorted(missing)}")
    if not pd.to_numeric(df["label"], errors="coerce").eq(expected_label).all():
        raise ValueError(f"{filename}: nieoczekiwane etykiety")
    df = df.drop(columns=["date"], errors="ignore").copy()
    for col in BASIC:
        if col not in df.columns:
            df[col] = ""
    for col in df.columns:
        if col != "label":
            df[col] = df[col].fillna("").map(clean_string)
    df["label"] = expected_label
    return df


def first_domain(addresses):
    for _, addr in getaddresses([str(addresses)]):
        if "@" in addr:
            return addr.rsplit("@", 1)[-1].lower().strip().strip(">")
    return ""


def address_count(addresses):
    return sum("@" in addr for _, addr in getaddresses([str(addresses)]))


def from_spamassassin():
    if not SPAMASSASSIN_FILE.is_file():
        raise FileNotFoundError(
            f"Brak {SPAMASSASSIN_FILE}. Umiesc tam SpamAssasin.csv (z jednym 's' w Assasin)."
        )
    df = pd.read_csv(
        SPAMASSASSIN_FILE, encoding="utf-8-sig", low_memory=False,
        keep_default_na=False,
    )
    required = {"sender", "receiver", "subject", "body", "label"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"SpamAssassin: brak kolumn {sorted(missing)}")
    labels = pd.to_numeric(df["label"], errors="coerce")
    print(f"[SPAMASSASSIN] Rekordy: {len(df)}; ham: {(labels == 0).sum()}; spam: {(labels == 1).sum()}")
    df = df.loc[labels.eq(0), ["sender", "receiver", "subject", "body"]].copy()
    df = df.rename(columns={"body": "message"})
    df["reply_to"] = ""  # nie ma takiego pola w tym CSV
    df["label"] = 0
    df["source"] = "spamassassin_ham"
    for col in ("sender", "receiver", "subject", "message"):
        df[col] = df[col].fillna("").map(clean_string)

    # Tylko cechy, ktore mozna rzeczywiscie odtworzyc z tego CSV.
    df["sender_domain"] = df["sender"].map(first_domain)
    df["receiver_domain"] = df["receiver"].map(first_domain)
    df["to_count"] = df["receiver"].map(address_count)
    df["subject_length"] = df["subject"].str.len()
    df["message_length"] = df["message"].str.len()
    df["message_word_count"] = df["message"].str.split().str.len()
    df["exclamation_count"] = df["message"].str.count("!")
    df["question_count"] = df["message"].str.count(r"\?")
    df["uppercase_ratio"] = df["message"].map(
        lambda s: round(sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s)), 6)
    )

    urls = df["message"].map(lambda s: list(dict.fromkeys(URL_RE.findall(s))))
    df["urls_json"] = urls.map(lambda arr: json.dumps(arr, ensure_ascii=False))
    df["url_count"] = urls.map(len)
    df["has_url"] = df["url_count"].gt(0).astype(int)

    # Domeny URL wyliczamy z rzeczywistych adresow w tresci.
    # Nie traktujemy oryginalnej kolumny "urls" (0/1) jako listy URL.
    def domains_of(url_list):
        domains = []
        for url in url_list:
            try:
                parsed = urlsplit(url if "://" in url else "https://" + url)
                host = (parsed.hostname or "").lower()
            except ValueError:
                host = ""
            if host and host not in domains:
                domains.append(host)
        return domains

    domains = urls.map(domains_of)
    df["url_domains_json"] = domains.map(lambda x: json.dumps(x, ensure_ascii=False))
    df["unique_url_domain_count"] = domains.map(len)

    def count_ip_domains(domains_list):
        n = 0
        for host in domains_list:
            try:
                ipaddress.ip_address(host)
                n += 1
            except ValueError:
                pass
        return n

    df["url_ip_count"] = domains.map(count_ip_domains)
    print(f"[SPAMASSASSIN] Do zbioru wybieram {len(df)} legalnych maili; spam pomijam.")
    return df


def select_ham(ham, phishing_count):
    if HAM_TO_PHISHING_RATIO is None:
        return ham
    target = round(phishing_count * HAM_TO_PHISHING_RATIO)
    if len(ham) < target:
        raise ValueError(f"Za malo legalnych maili do 60/40: potrzeba {target}, jest {len(ham)}")

    # Zachowaj mniej wiecej polowe Enron i polowe SpamAssassin
    # (jesli oba zrodla maja wystarczajaco duzo rekordow).
    enron = ham.loc[ham["source"].eq("enron")]
    spam = ham.loc[ham["source"].eq("spamassassin_ham")]
    spam_n = min(len(spam), (target + 1) // 2)
    enron_n = min(len(enron), target - spam_n)
    spam_n = min(len(spam), target - enron_n)
    if enron_n + spam_n < target:
        raise ValueError("Brak wystarczajacej liczby ham z Enron i SpamAssassin")
    return pd.concat([
        enron.sample(n=enron_n, random_state=SEED),
        spam.sample(n=spam_n, random_state=SEED),
    ], ignore_index=True)


def main():
    enron = read_processed("enron_processed.csv", 0)
    nazario = read_processed("nazario_processed.csv", 1)
    spam_ham = from_spamassassin()
    print(f"[WEJSCIE] Enron={len(enron)}, Nazario={len(nazario)}, SpamAssassin ham={len(spam_ham)}")

    # Zachowaj wszystkie kolumny Enron/Nazario; brakujace pola SpamAssassin beda puste.
    data = pd.concat([enron, nazario, spam_ham], ignore_index=True, sort=False)
    data = data.drop(columns=["date"], errors="ignore")
    data = data.fillna("")
    empty = data["message"].map(lambda s: not normalized(s))
    technical = data["subject"].astype(str).str.contains(
        "DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA", case=False, regex=False, na=False
    )
    print(f"[CZYSZCZENIE] Puste={int(empty.sum())}, techniczne={int(technical.sum())}")
    data = data.loc[~(empty | technical)].copy()

    data["_fingerprint"] = data["message"].map(fingerprint)
    conflicts = data.groupby("_fingerprint")["label"].nunique()
    bad = data["_fingerprint"].isin(conflicts[conflicts > 1].index)
    print(f"[CZYSZCZENIE] Sprzeczne etykiety (usuniete rekordy)={int(bad.sum())}")
    data = data.loc[~bad].copy()
    old_count = len(data)
    data = data.drop_duplicates("_fingerprint", keep="first").drop(columns="_fingerprint")
    print(f"[CZYSZCZENIE] Duplikaty={old_count - len(data)}")

    ham = data.loc[data["label"].eq(0)]
    phish = data.loc[data["label"].eq(1)]
    if len(ham) < 2 or len(phish) < 2:
        raise ValueError("Za malo danych ktorejs klasy")
    selected_ham = select_ham(ham, len(phish))
    combined = pd.concat([selected_ham, phish], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train, test = train_test_split(
        combined, test_size=TEST_SIZE, stratify=combined["label"], random_state=SEED
    )
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Raport dostepnosci metadanych. Brak naglowka to pusty tekst,
    # natomiast obliczone 0 (np. brak URL) to prawidlowa wartosc.
    check_cols = [
        "sender", "receiver", "reply_to", "sender_domain", "receiver_domain",
        "to_count", "subject_length", "message_length", "message_word_count",
        "url_count", "url_domains_json", "unique_url_domain_count",
        "received_count", "has_dkim_signature", "authentication_results",
        "attachment_count", "has_html_part",
    ]
    report_rows = []
    for source, group in combined.groupby("source", sort=True):
        for col in check_cols:
            if col not in combined.columns:
                continue
            values = group[col].astype("string").fillna("").str.strip()
            available = int(values.ne("").sum())
            report_rows.append({
                "source": source,
                "column": col,
                "rows": len(group),
                "available": available,
                "missing": len(group) - available,
                "available_percent": round(100 * available / len(group), 2),
            })
    coverage = pd.DataFrame(report_rows)
    coverage_path = OUTPUT / "metadata_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    print(f"[METADANE] Raport dostepnosci: {coverage_path}")
    spam_report = coverage.loc[
        coverage["source"].eq("spamassassin_ham") &
        coverage["column"].isin(["sender", "receiver", "received_count", "has_dkim_signature"])
    ]
    if not spam_report.empty:
        print(spam_report[["column", "available", "missing"]].to_string(index=False))

    for name, frame in (("combined", combined), ("train", train), ("test", test)):
        path = OUTPUT / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        counts = frame["label"].value_counts().to_dict()
        print(f"{name}: {len(frame)}; legalne={counts.get(0, 0)}, phishing={counts.get(1, 0)} -> {path}")
        print("  Zrodla:", frame["source"].value_counts().to_dict())

    try:
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    except ImportError:
        print("Pominieto XLSX: py -m pip install openpyxl")
    else:
        xlsx_dir = ROOT / "data" / "xlsx"
        xlsx_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in (("combined", combined), ("train", train), ("test", test)):
            preview = frame.copy()
            for col in preview.select_dtypes(include=["object"]).columns:
                preview[col] = preview[col].map(
                    lambda x: ILLEGAL_CHARACTERS_RE.sub("", x)[:32767]
                    if isinstance(x, str) else x
                )
            path = xlsx_dir / f"{name}.xlsx"
            preview.to_excel(path, index=False, sheet_name="Wiadomosci", engine="openpyxl")
            print(f"Excel: {path}")

    print("Gotowe: Enron + Nazario + legalne SpamAssassin, tylko split 80/20.")
    print("Uwaga: pola MIME i naglowki niedostepne w SpamAssassin pozostaja puste.")
    print("Do modelu metadanych stosuj wspolne cechy z metadata_features.py,")
    print("a nie puste naglowki techniczne i nie kolumne source.")


if __name__ == "__main__":
    main()
