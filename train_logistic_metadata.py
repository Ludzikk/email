
"""
EKSPERYMENTY WORD / CHAR TF-IDF + CECHY TEKSTOWE I METADANE

Dataset:
    Enron + Nazario + SpamAssassin (tylko ham)

Wejscie:
    data/splits/train.csv
    data/splits/test.csv

16 eksperymentow:
    WORD i CHAR, kazdy z wariantami:
    1. bazowy
    2. dlugosc
    3. styl
    4. wszystkie cechy tekstowe
    5. metadane C
    6. dlugosc + metadane C
    7. styl + metadane C
    8. wszystkie cechy tekstowe + metadane C

Wagi:
    TEXT_WEIGHT = 0.05
    METADATA_WEIGHT = 0.05

Skalowanie StandardScaler tylko na zbiorze treningowym.
Bez osobnego zbioru walidacyjnego.

Uruchomienie:
    py train_logistic_metadata.py
"""

from pathlib import Path
from email.utils import getaddresses

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


# ============================================================
# KONFIGURACJA
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "final_80_20"

SEED = 42

TEXT_WEIGHT = 0.1
METADATA_WEIGHT = 0.1

# Nazwa serii, aby nie nadpisac starszych wynikow.
EXPERIMENT_NAME = "word_char_text_metadata_weight005"


# ============================================================
# CECHY TEKSTOWE
# ============================================================

LENGTH_FEATURES = [
    "subject_length",
    "message_length",
    "message_word_count",
]

STYLE_FEATURES = [
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
]

ALL_TEXT_FEATURES = (
    LENGTH_FEATURES + STYLE_FEATURES
)


# ============================================================
# METADANE ADRESOWE - 12 CECH
# ============================================================

SENDER_FEATURES = [
    "sender_address_length",
    "sender_local_length",
    "sender_domain_length",
    "sender_domain_digit_count",
    "sender_domain_hyphen_count",
    "sender_address_has_digit",
]

RECEIVER_FEATURES = [
    "receiver_address_length",
    "receiver_local_length",
    "receiver_domain_length",
    "receiver_count",
    "sender_receiver_domain_match",
    "sender_receiver_domain_length_diff",
]

METADATA_FEATURES = (
    SENDER_FEATURES + RECEIVER_FEATURES
)


# ============================================================
# WARIANTY EKSPERYMENTOW
# ============================================================

# Kazdy wariant zawiera:
# - liste dodatkowych cech tekstowych
# - liste metadanych adresowych


VARIANTS = {
    "bazowy": {
        "text": [],
        "metadata": [],
    },

    "styl": {
        "text": STYLE_FEATURES,
        "metadata": [],
    },

    "dlugosc": {
        "text": LENGTH_FEATURES,
        "metadata": [],
    },

    "all_tekstowe": {
        "text": ALL_TEXT_FEATURES,
        "metadata": [],
    },

    "meta_A_nadawca": {
        "text": [],
        "metadata": SENDER_FEATURES,
    },

    "meta_B_odbiorca": {
        "text": [],
        "metadata": RECEIVER_FEATURES,
    },

    "meta_C_wszystkie": {
        "text": [],
        "metadata": METADATA_FEATURES,
    },

    "meta_C_dlugosc": {
        "text": LENGTH_FEATURES,
        "metadata": METADATA_FEATURES,
    },

    "meta_C_styl": {
        "text": STYLE_FEATURES,
        "metadata": METADATA_FEATURES,
    },

    "meta_C_all_tekstowe": {
        "text": ALL_TEXT_FEATURES,
        "metadata": METADATA_FEATURES,
    },
}



# ============================================================
# WCZYTYWANIE DANYCH
# ============================================================

def load_split(name):

    path = DATA_DIR / f"{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Brak pliku: {path}"
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
        "label",
        "source",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: brak kolumn {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"Plik {path.name} jest pusty."
        )

    labels = pd.to_numeric(
        df["label"],
        errors="raise",
    )

    if (
        not labels.isin([0, 1]).all()
        or labels.nunique() != 2
    ):
        raise ValueError(
            f"{path.name}: bledne etykiety lub brak klasy"
        )

    df["label"] = labels.astype(int)

    return df


# ============================================================
# PRZYGOTOWANIE TEKSTU TF-IDF
# ============================================================

def prepare_text(df):

    subject = (
        df["subject"]
        .fillna("")
        .astype(str)
    )

    message = (
        df["message"]
        .fillna("")
        .astype(str)
    )

    return (
        subject + " " + message
    ).tolist()


# ============================================================
# CECHY STATYSTYCZNE I STYLISTYCZNE
# ============================================================

def uppercase_ratio(text):

    letters = sum(
        character.isalpha()
        for character in text
    )

    if letters == 0:
        return 0.0

    uppercase = sum(
        character.isupper()
        for character in text
    )

    return uppercase / letters


def make_text_features(df):

    subject = (
        df["subject"]
        .fillna("")
        .astype(str)
    )

    message = (
        df["message"]
        .fillna("")
        .astype(str)
    )

    features = pd.DataFrame({
        "subject_length": (
            subject.str.len()
        ),

        "message_length": (
            message.str.len()
        ),

        "message_word_count": (
            message.str.split().str.len()
        ),

        "exclamation_count": (
            message.str.count("!")
        ),

        "question_count": (
            message.str.count(r"\?")
        ),

        "uppercase_ratio": (
            message.map(uppercase_ratio)
        ),
    }, index=df.index)

    return features[
        ALL_TEXT_FEATURES
    ].astype("float64")


# ============================================================
# WYDOBYWANIE ADRESOW E-MAIL
# ============================================================

def extract_addresses(value):

    if value is None:
        return []

    value = str(value).strip()

    if not value:
        return []

    result = []

    for _, address in getaddresses([value]):

        address = address.strip().lower()

        if "@" not in address:
            continue

        local, domain = address.rsplit("@", 1)

        if not local or not domain:
            continue

        result.append(address)

    return result


# ============================================================
# CECHY METADANYCH ADRESOWYCH
# ============================================================

def make_metadata_features(df):

    rows = []

    sender_available = []
    receiver_available = []

    for sender, receiver in zip(
        df["sender"],
        df["receiver"],
    ):

        senders = extract_addresses(sender)
        receivers = extract_addresses(receiver)

        sender_available.append(
            bool(senders)
        )

        receiver_available.append(
            bool(receivers)
        )

        sender_address = (
            senders[0] if senders else ""
        )

        receiver_address = (
            receivers[0] if receivers else ""
        )

        if sender_address:
            sender_local, sender_domain = (
                sender_address.rsplit("@", 1)
            )
        else:
            sender_local = ""
            sender_domain = ""

        if receiver_address:
            receiver_local, receiver_domain = (
                receiver_address.rsplit("@", 1)
            )
        else:
            receiver_local = ""
            receiver_domain = ""

        same_domain = int(
            bool(sender_domain and receiver_domain)
            and sender_domain == receiver_domain
        )

        row = {
            "sender_address_length": (
                len(sender_address)
            ),

            "sender_local_length": (
                len(sender_local)
            ),

            "sender_domain_length": (
                len(sender_domain)
            ),

            "sender_domain_digit_count": sum(
                c.isdigit()
                for c in sender_domain
            ),

            "sender_domain_hyphen_count": (
                sender_domain.count("-")
            ),

            "sender_address_has_digit": int(
                any(
                    c.isdigit()
                    for c in sender_address
                )
            ),

            "receiver_address_length": (
                len(receiver_address)
            ),

            "receiver_local_length": (
                len(receiver_local)
            ),

            "receiver_domain_length": (
                len(receiver_domain)
            ),

            "receiver_count": (
                len(receivers)
            ),

            "sender_receiver_domain_match": (
                same_domain
            ),

            "sender_receiver_domain_length_diff": abs(
                len(sender_domain)
                - len(receiver_domain)
            ),
        }

        rows.append(row)

    metadata = pd.DataFrame(
        rows,
        columns=METADATA_FEATURES,
        index=df.index,
    ).astype("float64")

    availability = pd.DataFrame({
        "source": df["source"].astype(str),
        "sender_available": sender_available,
        "receiver_available": receiver_available,
    }, index=df.index)

    return metadata, availability


# ============================================================
# RAPORT DOSTEPNOSCI ADRESOW
# ============================================================

def make_coverage_report(
    train_info,
    test_info,
):

    all_reports = []

    for split_name, info in (
        ("train", train_info),
        ("test", test_info),
    ):

        report = info.groupby(
            "source",
            as_index=False,
        ).agg(
            messages=("source", "size"),

            sender_available=(
                "sender_available",
                "sum",
            ),

            receiver_available=(
                "receiver_available",
                "sum",
            ),
        )

        report.insert(
            0,
            "split",
            split_name,
        )

        for side in ("sender", "receiver"):

            report[f"{side}_missing"] = (
                report["messages"]
                - report[f"{side}_available"]
            )

            report[f"{side}_available_pct"] = (
                100
                * report[f"{side}_available"]
                / report["messages"]
            ).round(2)

        all_reports.append(report)

    return pd.concat(
        all_reports,
        ignore_index=True,
    )


# ============================================================
# WEKTORYZATORY WORD I CHAR
# ============================================================

def create_vectorizer(model_type):

    if model_type == "Word":

        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )

    if model_type == "Char":

        return TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=2,
            max_features=40000,
            sublinear_tf=True,
        )

    raise ValueError(
        f"Nieznany model: {model_type}"
    )


# ============================================================
# SKALOWANIE DODATKOWYCH CECH
# ============================================================

def scale_features(
    train_features,
    test_features,
):

    # Skalowanie wszystkich kolumn danej grupy
    # niezaleznie. Fit tylko na treningu.

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_features
    )

    test_scaled = scaler.transform(
        test_features
    )

    train_scaled = pd.DataFrame(
        train_scaled,
        columns=train_features.columns,
        index=train_features.index,
    )

    test_scaled = pd.DataFrame(
        test_scaled,
        columns=test_features.columns,
        index=test_features.index,
    )

    return train_scaled, test_scaled


# ============================================================
# BUDOWANIE MACIERZY DLA WARIANTU
# ============================================================

def combine_features(
    X_train_text,
    X_test_text,
    text_train,
    text_test,
    metadata_train,
    metadata_test,
    selected_text,
    selected_metadata,
):

    train_blocks = [
        X_train_text
    ]

    test_blocks = [
        X_test_text
    ]

    # Dodatkowe cechy tekstowe.

    if selected_text:

        numeric_train = (
            text_train[selected_text]
            .to_numpy()
            * TEXT_WEIGHT
        )

        numeric_test = (
            text_test[selected_text]
            .to_numpy()
            * TEXT_WEIGHT
        )

        train_blocks.append(
            csr_matrix(numeric_train)
        )

        test_blocks.append(
            csr_matrix(numeric_test)
        )

    # Metadane adresowe.

    if selected_metadata:

        numeric_train = (
            metadata_train[selected_metadata]
            .to_numpy()
            * METADATA_WEIGHT
        )

        numeric_test = (
            metadata_test[selected_metadata]
            .to_numpy()
            * METADATA_WEIGHT
        )

        train_blocks.append(
            csr_matrix(numeric_train)
        )

        test_blocks.append(
            csr_matrix(numeric_test)
        )

    if len(train_blocks) == 1:

        return (
            X_train_text,
            X_test_text,
        )

    X_train = hstack(
        train_blocks,
        format="csr",
    )

    X_test = hstack(
        test_blocks,
        format="csr",
    )

    return X_train, X_test


# ============================================================
# OBLICZANIE METRYK
# ============================================================

def evaluate(
    actual,
    predicted,
    model_type,
    variant,
    selected_text,
    selected_metadata,
):

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "model": model_type,

        "variant": variant,

        "text_weight": (
            TEXT_WEIGHT if selected_text else 0
        ),

        "metadata_weight": (
            METADATA_WEIGHT
            if selected_metadata else 0
        ),

        "n_text_features": len(
            selected_text
        ),

        "n_metadata_features": len(
            selected_metadata
        ),

        "text_features": ", ".join(
            selected_text
        ),

        "metadata_features": ", ".join(
            selected_metadata
        ),

        "n_train": len(actual) * 0 + 0,
        # n_train ustawiamy pozniej

        "n_test": len(actual),

        "accuracy": accuracy_score(
            actual,
            predicted,
        ),

        "precision": precision_score(
            actual,
            predicted,
            zero_division=0,
        ),

        "recall": recall_score(
            actual,
            predicted,
            zero_division=0,
        ),

        "f1": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# GLOWNY PROGRAM
# ============================================================

def main():

    print("=" * 78)
    print("WORD / CHAR TF-IDF + TEKST + METADANE")
    print("16 EKSPERYMENTOW, WAGI 0.05")
    print("=" * 78)

    train = load_split("train")
    test = load_split("test")

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    print(
        f"\nTrening: {len(train)} wiadomosci"
    )

    print(
        f"Test:    {len(test)} wiadomosci"
    )

    print(
        f"Trening: ham={(y_train == 0).sum()}, "
        f"phishing={(y_train == 1).sum()}"
    )

    print(
        f"Test: ham={(y_test == 0).sum()}, "
        f"phishing={(y_test == 1).sum()}"
    )

    print(
        f"\nTEXT_WEIGHT: {TEXT_WEIGHT}"
    )

    print(
        f"METADATA_WEIGHT: {METADATA_WEIGHT}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PRZYGOTOWANIE TEKSTU
    # ========================================================

    train_text = prepare_text(train)
    test_text = prepare_text(test)

    # ========================================================
    # EKSTRAKCJA CECH LICZBOWYCH
    # ========================================================

    text_features_train = make_text_features(
        train
    )

    text_features_test = make_text_features(
        test
    )

    metadata_train, availability_train = (
        make_metadata_features(train)
    )

    metadata_test, availability_test = (
        make_metadata_features(test)
    )

    # ========================================================
    # OSOBNE SKALOWANIE OBU GRUP
    # ========================================================

    text_train_scaled, text_test_scaled = (
        scale_features(
            text_features_train,
            text_features_test,
        )
    )

    metadata_train_scaled, metadata_test_scaled = (
        scale_features(
            metadata_train,
            metadata_test,
        )
    )

    # ========================================================
    # RAPORT BRAKUJACYCH ADRESOW
    # ========================================================

    coverage = make_coverage_report(
        availability_train,
        availability_test,
    )

    coverage_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_address_coverage.csv"
    )

    coverage.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDOSTEPNOSC ADRESOW:")
    print(coverage.to_string(index=False))

    # ========================================================
    # EKSPERYMENTY
    # ========================================================

    all_results = []

    predictions = pd.DataFrame({
        "test_row": np.arange(len(test)),
        "source": (
            test["source"].astype(str).to_numpy()
        ),
        "actual": y_test,
    })

    for model_type in (
        "Word",
        "Char",
    ):

        print("\n" + "=" * 78)
        print(f"MODEL: {model_type} TF-IDF")
        print("=" * 78)

        vectorizer = create_vectorizer(
            model_type
        )

        print("Tworzenie TF-IDF...")

        X_train_text = vectorizer.fit_transform(
            train_text
        )

        X_test_text = vectorizer.transform(
            test_text
        )

        print(
            f"Liczba cech TF-IDF: "
            f"{X_train_text.shape[1]}"
        )

        for variant, groups in VARIANTS.items():

            selected_text = groups["text"]
            selected_metadata = groups["metadata"]

            print("\n" + "-" * 78)
            print(
                f"EKSPERYMENT: "
                f"{model_type} + {variant}"
            )
            print("-" * 78)

            X_train, X_test = combine_features(
                X_train_text=X_train_text,
                X_test_text=X_test_text,

                text_train=text_train_scaled,
                text_test=text_test_scaled,

                metadata_train=metadata_train_scaled,
                metadata_test=metadata_test_scaled,

                selected_text=selected_text,
                selected_metadata=selected_metadata,
            )

            model = LogisticRegression(
                max_iter=2000,
                random_state=SEED,
            )

            model.fit(
                X_train,
                y_train,
            )

            predicted = model.predict(
                X_test
            )

            result = evaluate(
                actual=y_test,
                predicted=predicted,
                model_type=model_type,
                variant=variant,
                selected_text=selected_text,
                selected_metadata=selected_metadata,
            )

            result["n_train"] = len(y_train)

            all_results.append(result)

            prediction_column = (
                f"{model_type}_{variant}"
            )

            predictions[prediction_column] = (
                predicted
            )

            print(
                f"Accuracy : "
                f"{result['accuracy'] * 100:.2f}%"
            )

            print(
                f"Precision: "
                f"{result['precision'] * 100:.2f}%"
            )

            print(
                f"Recall   : "
                f"{result['recall'] * 100:.2f}%"
            )

            print(
                f"F1-score : "
                f"{result['f1'] * 100:.2f}%"
            )

            print(
                f"TN={result['TN']} "
                f"FP={result['FP']} "
                f"FN={result['FN']} "
                f"TP={result['TP']}"
            )

    # ========================================================
    # ZAPIS WYNIKOW
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # BLEDY WEDLUG ZRODLA
    # ========================================================

    by_source_rows = []

    for model_type in ("Word", "Char"):

        for variant in VARIANTS:

            prediction_column = (
                f"{model_type}_{variant}"
            )

            temp = predictions[
                ["source", "actual", prediction_column]
            ].copy()

            temp["FP"] = (
                (temp["actual"] == 0)
                & (temp[prediction_column] == 1)
            ).astype(int)

            temp["FN"] = (
                (temp["actual"] == 1)
                & (temp[prediction_column] == 0)
            ).astype(int)

            report = temp.groupby(
                "source",
                as_index=False,
            ).agg(
                messages=("actual", "size"),
                FP=("FP", "sum"),
                FN=("FN", "sum"),
            )

            report.insert(
                0,
                "variant",
                variant,
            )

            report.insert(
                0,
                "model",
                model_type,
            )

            by_source_rows.append(report)

    by_source_df = pd.concat(
        by_source_rows,
        ignore_index=True,
    )

    by_source_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_errors_by_source.csv"
    )

    by_source_df.to_csv(
        by_source_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    summary = results_df[
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

        summary[col] = (
            summary[col] * 100
        ).round(2)

    print("\n" + "=" * 78)
    print("PODSUMOWANIE 16 EKSPERYMENTOW")
    print("=" * 78)

    print(
        summary.to_string(index=False)
    )

    print("\nZAPISANE PLIKI:")
    print(results_path)
    print(predictions_path)
    print(by_source_path)
    print(coverage_path)

    print("\nGotowe.")
    print(
        "Nie zmieniono datasetu ani podzialu train/test."
    )


if __name__ == "__main__":
    main()
