
"""
12 eksperymentow:
Word / Char TF-IDF + URL A/B/C + metadane + cechy tekstowe.

Wejscie:
    data/splits/train.csv
    data/splits/test.csv

Uruchomienie:
    py train_url_experiments.py

Wymagania:
    pandas numpy scipy scikit-learn tldextract
"""

from pathlib import Path
from email.utils import getaddresses
from html import unescape
from ipaddress import ip_address
from urllib.parse import urlsplit, parse_qsl
import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    import tldextract
except ImportError as exc:
    raise SystemExit(
        "Brakuje tldextract. Zainstaluj: py -m pip install tldextract"
    ) from exc


# ============================================================
# KONFIGURACJA
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "splits"
RESULT_DIR = ROOT / "results" / "final_80_20"

SEED = 42

TEXT_WEIGHT = 0.1
METADATA_WEIGHT = 0.1
URL_WEIGHT = 0.1

PREFIX = "word_char_url_weight005"


# ============================================================
# GRUPY CECH
# ============================================================

LENGTH = [
    "subject_length",
    "message_length",
    "message_word_count",
]

STYLE = [
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
]

ALL_TEXT = LENGTH + STYLE


# Metadane nadawcy

SENDER = [
    "sender_address_length",
    "sender_local_length",
    "sender_domain_length",
    "sender_domain_digit_count",
    "sender_domain_hyphen_count",
    "sender_address_has_digit",
]


# Metadane odbiorcy

RECEIVER = [
    "receiver_address_length",
    "receiver_local_length",
    "receiver_domain_length",
    "receiver_count",
    "sender_receiver_domain_match",
    "sender_receiver_domain_length_diff",
]

META_C = SENDER + RECEIVER


# URL A - struktura i liczba linkow

URL_A = [
    "url_count",
    "url_max_length",
    "url_avg_length",
    "url_https_ratio",
    "url_max_path_length",
    "url_max_query_param_count",
]


# URL B - struktura domen

URL_B = [
    "url_unique_domain_count",
    "url_max_domain_length",
    "url_max_subdomain_count",
    "url_max_domain_digit_count",
    "url_max_domain_hyphen_count",
    "url_has_ip_domain",
]

URL_C = URL_A + URL_B


# ============================================================
# WARIANTY EKSPERYMENTOW
# ============================================================

# Kazdy wariant zawiera:
# (cechy tekstowe, metadane adresowe, cechy URL)

VARIANTS = {
    "bazowy": (
        [], [], []
    ),

    "url_A_struktura": (
        [], [], URL_A
    ),

    "url_B_domeny": (
        [], [], URL_B
    ),

    "url_C_wszystkie": (
        [], [], URL_C
    ),

    "meta_C_url_C": (
        [], META_C, URL_C
    ),

    "all_tekst_meta_C_url_C": (
        ALL_TEXT, META_C, URL_C
    ),
}


# ============================================================
# EKSTRAKCJA DOMEN I URL
# ============================================================

# Korzystamy z lokalnej listy sufiksow biblioteki.
# Nie pobieramy listy podczas uruchomienia.

EXTRACT_DOMAIN = tldextract.TLDExtract(
    suffix_list_urls=()
)

# Rozpoznajemy jawne http://, https:// i www.
# Nie zgadujemy adresow zapisanych bez tych prefiksow.

URL_RE = re.compile(
    r'''(?i)(?<![\w@])(?:https?://|www\.)[^\s<>"'`]+'''
)


# ============================================================
# WCZYTYWANIE DANYCH
# ============================================================

def load(name):

    path = DATA_DIR / f"{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Brakuje pliku: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "subject",
        "message",
        "sender",
        "receiver",
        "source",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: brak kolumn {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{path.name} jest pusty"
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    if (
        not df["label"].isin([0, 1]).all()
        or df["label"].nunique() != 2
    ):
        raise ValueError(
            f"{path.name}: etykiety musza zawierac 0 i 1"
        )

    return df.reset_index(drop=True)


# ============================================================
# TEKST DO TF-IDF
# ============================================================

def make_text(df):

    return (
        df["subject"].astype(str)
        + " "
        + df["message"].astype(str)
    ).tolist()


# ============================================================
# 6 CECH TEKSTOWYCH
# ============================================================

def text_features(df):

    s = df["subject"].astype(str)
    m = df["message"].astype(str)

    def capitals(t):

        letters = sum(
            c.isalpha() for c in t
        )

        if letters == 0:
            return 0.0

        uppercase = sum(
            c.isupper() for c in t
        )

        return uppercase / letters

    return pd.DataFrame({
        "subject_length": s.str.len(),

        "message_length": m.str.len(),

        "message_word_count": (
            m.str.split().str.len()
        ),

        "exclamation_count": (
            m.str.count("!")
        ),

        "question_count": (
            m.str.count(r"\?")
        ),

        "uppercase_ratio": (
            m.map(capitals)
        ),
    })[ALL_TEXT].astype(float)


# ============================================================
# ADRESY NADAWCY I ODBIORCY
# ============================================================

def addresses(raw):

    result = []

    for _, addr in getaddresses([str(raw)]):

        addr = addr.strip().lower()

        if "@" not in addr:
            continue

        local, domain = addr.rsplit("@", 1)

        if local and domain:
            result.append(addr)

    return result


# ============================================================
# 12 METADANYCH ADRESOWYCH
# ============================================================

def metadata_features(df):

    rows = []

    for sender, receiver in zip(
        df["sender"],
        df["receiver"],
    ):

        a = addresses(sender)
        b = addresses(receiver)

        sa = a[0] if a else ""
        ra = b[0] if b else ""

        if sa:
            sl, sd = sa.rsplit("@", 1)
        else:
            sl, sd = "", ""

        if ra:
            rl, rd = ra.rsplit("@", 1)
        else:
            rl, rd = "", ""

        rows.append({
            "sender_address_length": len(sa),

            "sender_local_length": len(sl),

            "sender_domain_length": len(sd),

            "sender_domain_digit_count": sum(
                c.isdigit() for c in sd
            ),

            "sender_domain_hyphen_count": (
                sd.count("-")
            ),

            "sender_address_has_digit": int(
                any(c.isdigit() for c in sa)
            ),

            "receiver_address_length": len(ra),

            "receiver_local_length": len(rl),

            "receiver_domain_length": len(rd),

            "receiver_count": len(b),

            "sender_receiver_domain_match": int(
                bool(sd and rd) and sd == rd
            ),

            "sender_receiver_domain_length_diff": abs(
                len(sd) - len(rd)
            ),
        })

    return pd.DataFrame(
        rows,
        columns=META_C,
    ).astype(float)


# ============================================================
# ODCZYTYWANIE URL Z TEMATU I TRESCI
# ============================================================

def parse_urls(subject, message):

    combined = unescape(
        str(subject) + " " + str(message)
    )

    records = []

    for match in URL_RE.finditer(combined):

        url = match.group(0).rstrip(
            ".,;:!?)]}"
        )

        if not url:
            continue

        # Adresy www. analizujemy jako HTTP technicznie,
        # ale ich faktyczny protokol pozostaje nieznany.

        scheme_known = not url.lower().startswith(
            "www."
        )

        if scheme_known:
            candidate = url
        else:
            candidate = "http://" + url

        try:
            parsed = urlsplit(candidate)

            host = (
                (parsed.hostname or "")
                .lower()
                .rstrip(".")
            )

            if not host:
                continue

            is_ip = False

            try:
                ip_address(host)
                is_ip = True
            except ValueError:
                pass

            if is_ip:
                subdomains = 0
            else:
                parts = EXTRACT_DOMAIN(host)

                if parts.subdomain:
                    subdomains = len(
                        parts.subdomain.split(".")
                    )
                else:
                    subdomains = 0

            records.append({
                "url": url,
                "host": host,

                "scheme": (
                    parsed.scheme.lower()
                    if scheme_known
                    else "unknown"
                ),

                "path_len": len(
                    parsed.path
                ),

                "query_params": len(
                    parse_qsl(
                        parsed.query,
                        keep_blank_values=True,
                    )
                ),

                "subdomains": subdomains,
                "is_ip": is_ip,
            })

        except ValueError:
            continue

    return records


# ============================================================
# 12 CECH URL
# ============================================================

def url_features(df):

    rows = []

    for subject, message in zip(
        df["subject"],
        df["message"],
    ):

        urls = parse_urls(subject, message)

        count = len(urls)

        lengths = [
            len(x["url"])
            for x in urls
        ]

        hosts = [
            x["host"]
            for x in urls
        ]

        known = [
            x for x in urls
            if x["scheme"] in ("http", "https")
        ]

        if known:
            https_ratio = (
                sum(
                    x["scheme"] == "https"
                    for x in known
                )
                / len(known)
            )
        else:
            https_ratio = 0.0

        rows.append({
            "url_count": count,

            "url_max_length": max(
                lengths,
                default=0,
            ),

            "url_avg_length": (
                sum(lengths) / count
                if count
                else 0.0
            ),

            "url_https_ratio": https_ratio,

            "url_max_path_length": max(
                (
                    x["path_len"]
                    for x in urls
                ),
                default=0,
            ),

            "url_max_query_param_count": max(
                (
                    x["query_params"]
                    for x in urls
                ),
                default=0,
            ),

            "url_unique_domain_count": len(
                set(hosts)
            ),

            "url_max_domain_length": max(
                (
                    len(h)
                    for h in hosts
                ),
                default=0,
            ),

            "url_max_subdomain_count": max(
                (
                    x["subdomains"]
                    for x in urls
                ),
                default=0,
            ),

            "url_max_domain_digit_count": max(
                (
                    sum(
                        c.isdigit()
                        for c in h
                    )
                    for h in hosts
                ),
                default=0,
            ),

            "url_max_domain_hyphen_count": max(
                (
                    h.count("-")
                    for h in hosts
                ),
                default=0,
            ),

            "url_has_ip_domain": int(
                any(
                    x["is_ip"]
                    for x in urls
                )
            ),
        })

    return pd.DataFrame(
        rows,
        columns=URL_C,
    ).astype(float)


# ============================================================
# AUDYT ISTNIEJACYCH KOLUMN URL
# ============================================================

def stored_json_url_count(raw):

    if raw is None or str(raw).strip() == "":
        return np.nan

    try:
        obj = json.loads(str(raw))

        if isinstance(obj, list):
            return float(len(obj))

        return np.nan

    except (ValueError, TypeError):
        return np.nan


def build_audit(split, df, url_df):

    audit = pd.DataFrame({
        "split": split,
        "source": df["source"].astype(str),
        "label": df["label"].astype(int),

        "detected_url_count": (
            url_df["url_count"].astype(int)
        ),
    })

    if "url_count" in df.columns:

        audit["stored_url_count"] = (
            pd.to_numeric(
                df["url_count"],
                errors="coerce",
            )
        )

    if "urls_json" in df.columns:

        audit["stored_urls_json_count"] = (
            df["urls_json"].map(
                stored_json_url_count
            )
        )

    return audit


# ============================================================
# RAPORT OBECNOSCI LINKOW
# ============================================================

def make_coverage(audit):

    out = audit.copy()

    out["has_url"] = (
        out["detected_url_count"] > 0
    ).astype(int)

    rows = out.groupby(
        ["split", "source", "label"],
        as_index=False,
    ).agg(
        messages=("has_url", "size"),

        messages_with_url=("has_url", "sum"),

        detected_urls_total=(
            "detected_url_count",
            "sum",
        ),

        mean_urls_per_message=(
            "detected_url_count",
            "mean",
        ),
    )

    rows["messages_without_url"] = (
        rows["messages"]
        - rows["messages_with_url"]
    )

    rows["messages_with_url_pct"] = (
        100
        * rows["messages_with_url"]
        / rows["messages"]
    ).round(2)

    rows["mean_urls_per_message"] = (
        rows["mean_urls_per_message"]
        .round(3)
    )

    return rows


def make_audit_summary(audit):

    cols = ["split", "source", "label"]
    records = []

    for keys, group in audit.groupby(
        cols,
        sort=True,
    ):

        row = dict(zip(cols, keys))

        row["messages"] = len(group)

        row["detected_urls_total"] = int(
            group["detected_url_count"].sum()
        )

        for col in (
            "stored_url_count",
            "stored_urls_json_count",
        ):

            if col not in group.columns:
                continue

            valid = group[col].notna()

            row[f"{col}_available_rows"] = int(
                valid.sum()
            )

            row[f"{col}_presence_mismatches"] = int(
                (
                    (group.loc[valid, col] > 0)
                    != (
                        group.loc[
                            valid,
                            "detected_url_count"
                        ] > 0
                    )
                ).sum()
            )

            row[f"{col}_exact_count_mismatches"] = int(
                (
                    group.loc[valid, col]
                    != group.loc[
                        valid,
                        "detected_url_count"
                    ]
                ).sum()
            )

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# SKALOWANIE CECH
# ============================================================

def scale(train, test):

    scaler = StandardScaler()

    x = scaler.fit_transform(train)
    y = scaler.transform(test)

    return (
        pd.DataFrame(
            x,
            columns=train.columns,
            index=train.index,
        ),

        pd.DataFrame(
            y,
            columns=test.columns,
            index=test.index,
        ),
    )


# ============================================================
# WORD / CHAR TF-IDF
# ============================================================

def make_vectorizer(kind):

    if kind == "Word":

        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )

    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
    )


# ============================================================
# LACZENIE CECH
# ============================================================

def combine(
    x_text_train,
    x_text_test,
    tx_train,
    tx_test,
    mt_train,
    mt_test,
    ur_train,
    ur_test,
    chosen,
):

    selected_text, selected_meta, selected_url = chosen

    a = [x_text_train]
    b = [x_text_test]

    groups = (
        (
            selected_text,
            TEXT_WEIGHT,
            tx_train,
            tx_test,
        ),

        (
            selected_meta,
            METADATA_WEIGHT,
            mt_train,
            mt_test,
        ),

        (
            selected_url,
            URL_WEIGHT,
            ur_train,
            ur_test,
        ),
    )

    for names, weight, train, test in groups:

        if not names:
            continue

        a.append(
            csr_matrix(
                train[names].to_numpy() * weight
            )
        )

        b.append(
            csr_matrix(
                test[names].to_numpy() * weight
            )
        )

    if len(a) == 1:
        return a[0], b[0]

    return (
        hstack(a, format="csr"),
        hstack(b, format="csr"),
    )


# ============================================================
# PROGRAM GLOWNY
# ============================================================

def main():

    print("=" * 78)
    print("WORD / CHAR TF-IDF + 12 CECH URL")
    print("12 EKSPERYMENTOW")
    print("=" * 78)

    train = load("train")
    test = load("test")

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    print(
        f"Train: {len(train)} | Test: {len(test)}"
    )

    print(
        f"Wagi: tekst={TEXT_WEIGHT}, "
        f"metadane={METADATA_WEIGHT}, "
        f"URL={URL_WEIGHT}"
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # EKSTRAKCJA CECH
    # --------------------------------------------------------

    print(
        "\nEkstrakcja 6 cech tekstowych, "
        "12 adresowych i 12 URL..."
    )

    tt = text_features(train)
    te = text_features(test)

    mt = metadata_features(train)
    me = metadata_features(test)

    ut = url_features(train)
    ue = url_features(test)

    # --------------------------------------------------------
    # SKALOWANIE: FIT TYLKO NA TRAIN
    # --------------------------------------------------------

    tts, tes = scale(tt, te)
    mts, mes = scale(mt, me)
    uts, ues = scale(ut, ue)

    # --------------------------------------------------------
    # KONTROLA EKSTRAKCJI URL
    # --------------------------------------------------------

    audit = pd.concat(
        [
            build_audit("train", train, ut),
            build_audit("test", test, ue),
        ],
        ignore_index=True,
    )

    coverage = make_coverage(audit)
    mismatch = make_audit_summary(audit)

    coverage.to_csv(
        RESULT_DIR / f"{PREFIX}_url_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mismatch.to_csv(
        RESULT_DIR / f"{PREFIX}_url_count_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nUDZIAL WIADOMOSCI Z URL W KORPUSACH:"
    )

    print(
        coverage.to_string(index=False)
    )

    print(
        "\nKontrola zgodnosci z kolumnami oryginalnymi:"
    )

    print(
        mismatch.to_string(index=False)
    )

    if (
        "stored_url_count_presence_mismatches"
        in mismatch.columns
    ):

        differences = mismatch[
            "stored_url_count_presence_mismatches"
        ].sum()

        if differences > 0:

            print(
                "\nUWAGA: Nowa ekstrakcja URL "
                "i istniejaca kolumna url_count "
                "nie zawsze sa zgodne."
            )

            print(
                "Sprawdz raport audytu przed "
                "interpretacja wynikow."
            )

    # --------------------------------------------------------
    # ZAPIS CECH URL DLA TESTU
    # --------------------------------------------------------

    feature_test = ue.copy()

    feature_test.insert(
        0,
        "label",
        y_test,
    )

    feature_test.insert(
        0,
        "source",
        test["source"].astype(str),
    )

    feature_test.insert(
        0,
        "test_row",
        np.arange(len(test)),
    )

    feature_test.to_csv(
        RESULT_DIR
        / f"{PREFIX}_url_features_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # TRENOWANIE 12 MODELI
    # --------------------------------------------------------

    summary = []

    predictions = pd.DataFrame({
        "test_row": np.arange(len(test)),

        "source": (
            test["source"].astype(str).to_numpy()
        ),

        "actual": y_test,
    })

    text_train = make_text(train)
    text_test = make_text(test)

    for kind in ("Word", "Char"):

        print("\n" + "=" * 78)
        print(f"{kind} TF-IDF")
        print("=" * 78)

        vectorizer = make_vectorizer(kind)

        x_train = vectorizer.fit_transform(
            text_train
        )

        x_test = vectorizer.transform(
            text_test
        )

        print(
            "Liczba cech TF-IDF:",
            x_train.shape[1],
        )

        for variant, groups in VARIANTS.items():

            a, b = combine(
                x_train,
                x_test,
                tts,
                tes,
                mts,
                mes,
                uts,
                ues,
                groups,
            )

            model = LogisticRegression(
                max_iter=2000,
                random_state=SEED,
            )

            model.fit(
                a,
                y_train,
            )

            pred = model.predict(b)

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                pred,
                labels=[0, 1],
            ).ravel()

            row = {
                "model": kind,
                "variant": variant,

                "text_weight": (
                    TEXT_WEIGHT
                    if groups[0] else 0
                ),

                "metadata_weight": (
                    METADATA_WEIGHT
                    if groups[1] else 0
                ),

                "url_weight": (
                    URL_WEIGHT
                    if groups[2] else 0
                ),

                "n_text_features": len(
                    groups[0]
                ),

                "n_metadata_features": len(
                    groups[1]
                ),

                "n_url_features": len(
                    groups[2]
                ),

                "n_train": len(train),
                "n_test": len(test),

                "accuracy": accuracy_score(
                    y_test,
                    pred,
                ),

                "precision": precision_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),

                "recall": recall_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),

                "f1": f1_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),

                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }

            summary.append(row)

            predictions[
                f"{kind}_{variant}"
            ] = pred

            print(
                f"{kind:4s} "
                f"{variant:28s} "
                f"Accuracy={100 * row['accuracy']:.2f}% "
                f"Precision={100 * row['precision']:.2f}% "
                f"Recall={100 * row['recall']:.2f}% "
                f"F1={100 * row['f1']:.2f}% "
                f"FP={fp} FN={fn}"
            )

    # --------------------------------------------------------
    # ZAPIS METRYK I PREDYKCJI
    # --------------------------------------------------------

    metrics = pd.DataFrame(summary)

    metrics.to_csv(
        RESULT_DIR / f"{PREFIX}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        RESULT_DIR / f"{PREFIX}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # ANALIZA BLEDOW WEDLUG ZRODLA
    # --------------------------------------------------------

    errors = []

    for kind in ("Word", "Char"):

        for variant in VARIANTS:

            predcol = f"{kind}_{variant}"

            tmp = predictions[
                ["source", "actual", predcol]
            ].copy()

            tmp["FP"] = (
                (tmp["actual"] == 0)
                & (tmp[predcol] == 1)
            ).astype(int)

            tmp["FN"] = (
                (tmp["actual"] == 1)
                & (tmp[predcol] == 0)
            ).astype(int)

            grouped = tmp.groupby(
                "source",
                as_index=False,
            ).agg(
                messages=("actual", "size"),
                FP=("FP", "sum"),
                FN=("FN", "sum"),
            )

            grouped.insert(
                0,
                "variant",
                variant,
            )

            grouped.insert(
                0,
                "model",
                kind,
            )

            errors.append(grouped)

    pd.concat(
        errors,
        ignore_index=True,
    ).to_csv(
        RESULT_DIR
        / f"{PREFIX}_errors_by_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # PODSUMOWANIE
    # --------------------------------------------------------

    display = metrics[
        [
            "model",
            "variant",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "FP",
            "FN",
        ]
    ].copy()

    for col in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):

        display[col] = (
            100 * display[col]
        ).round(2)

    print("\n" + "=" * 78)
    print("PODSUMOWANIE 12 EKSPERYMENTOW")
    print("=" * 78)

    print(
        display.to_string(index=False)
    )

    print(
        "\nWyniki zapisane w:",
        RESULT_DIR,
    )

    print(
        "Prefiks plikow:",
        PREFIX,
    )

    print(
        "Dataset i podzial train/test bez zmian."
    )


if __name__ == "__main__":
    main()
