
"""
EKSPERYMENTY: TF-IDF + STATYSTYCZNE CECHY TEKSTU

Zrodla:
    Enron + Nazario + SpamAssassin (ham)

Dane:
    data/splits/train.csv
    data/splits/test.csv

Podzial:
    80% train / 20% test
    Bez walidacji

Eksperymenty:
    Word TF-IDF:
        bazowy
        A - dlugosc
        B - styl
        C - wszystkie

    Char TF-IDF:
        bazowy
        A - dlugosc
        B - styl
        C - wszystkie

Wyniki:
    results/final_80_20/text_statistics_experiments.csv

Uruchomienie:
    py train_text_statistics_experiments.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# KONFIGURACJA
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Stala, ustalona przed eksperymentami waga cech liczbowych.
# Nie zmieniamy jej na podstawie wyniku na zbiorze testowym.
NUMERIC_WEIGHT = 0.05


# ============================================================
# GRUPY CECH
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

ALL_FEATURES = LENGTH_FEATURES + STYLE_FEATURES

VARIANTS = {
    "bazowy": [],
    "A_dlugosc": LENGTH_FEATURES,
    "B_styl": STYLE_FEATURES,
    "C_wszystkie": ALL_FEATURES,
}


# ============================================================
# WCZYTYWANIE DANYCH
# ============================================================

def load_data(name):

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
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"W {path.name} brakuje: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"Plik {path.name} jest pusty."
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            f"Nieprawidlowe etykiety: {path.name}"
        )

    if df["label"].nunique() != 2:
        raise ValueError(
            f"Brakuje jednej klasy: {path.name}"
        )

    return df


# ============================================================
# PRZYGOTOWANIE TEKSTU
# ============================================================

def prepare_text(df):

    # Taka sama kolejnosc jak w bazowych modelach:
    # subject + spacja + message

    subject = df["subject"].fillna("").astype(str)
    message = df["message"].fillna("").astype(str)

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


def make_numeric_features(df):

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

    # Cechy przeliczamy z tekstu we wszystkich trzech
    # zrodlach, wedlug identycznych regul.
    #
    # Odpowiadaja one szesciu kolumnom naszego datasetu.
    # Dzieki temu nie pomylimy brakujacej wartosci
    # z prawdziwym zerem.

    features = pd.DataFrame({
        "subject_length": subject.str.len(),

        "message_length": message.str.len(),

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
    })

    features.index = df.index

    return features[ALL_FEATURES].astype("float64")


# ============================================================
# MODELE TF-IDF
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
# LACZENIE TF-IDF Z CECHAMI LICZBOWYMI
# ============================================================

def combine_features(
    X_train_text,
    X_test_text,
    numeric_train,
    numeric_test,
    selected_columns,
):

    # Wariant bazowy - wyłącznie TF-IDF.

    if not selected_columns:
        return X_train_text, X_test_text

    # Skalowanie jest dopasowywane wylacznie na train.

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        numeric_train[selected_columns]
    )

    test_scaled = scaler.transform(
        numeric_test[selected_columns]
    )

    # Stala waga cech liczbowych.

    train_scaled = train_scaled * NUMERIC_WEIGHT
    test_scaled = test_scaled * NUMERIC_WEIGHT

    # Laczymy macierz TF-IDF z dodatkowymi kolumnami.

    X_train = hstack(
        [
            X_train_text,
            csr_matrix(train_scaled),
        ],
        format="csr",
    )

    X_test = hstack(
        [
            X_test_text,
            csr_matrix(test_scaled),
        ],
        format="csr",
    )

    return X_train, X_test


# ============================================================
# POJEDYNCZY EKSPERYMENT
# ============================================================

def run_experiment(
    model_type,
    variant,
    selected_columns,
    X_train_text,
    X_test_text,
    numeric_train,
    numeric_test,
    y_train,
    y_test,
):

    X_train, X_test = combine_features(
        X_train_text,
        X_test_text,
        numeric_train,
        numeric_test,
        selected_columns,
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    results = {
        "model": model_type,
        "variant": variant,
        "n_extra_features": len(selected_columns),
        "extra_features": ", ".join(selected_columns),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return results


# ============================================================
# GLOWNY PROGRAM
# ============================================================

def main():

    print("=" * 72)
    print("WORD / CHAR TF-IDF + CECHY TEKSTOWE")
    print("=" * 72)

    train = load_data("train")
    test = load_data("test")

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    train_text = prepare_text(train)
    test_text = prepare_text(test)

    numeric_train = make_numeric_features(train)
    numeric_test = make_numeric_features(test)

    print(f"\nTrain: {len(train)} wiadomosci")
    print(f"Test:  {len(test)} wiadomosci")

    print(
        f"Train: ham={np.sum(y_train == 0)}, "
        f"phishing={np.sum(y_train == 1)}"
    )

    print(
        f"Test:  ham={np.sum(y_test == 0)}, "
        f"phishing={np.sum(y_test == 1)}"
    )

    print(
        f"Waga dodatkowych cech: "
        f"{NUMERIC_WEIGHT}"
    )

    all_results = []

    for model_type in ("Word", "Char"):

        print("\n" + "=" * 72)
        print(f"MODEL: {model_type} TF-IDF")
        print("=" * 72)

        vectorizer = create_vectorizer(
            model_type
        )

        print("Tworzenie macierzy TF-IDF...")

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

        model_results = []

        for variant, columns in VARIANTS.items():

            print(
                f"\nTrenowanie wariantu: "
                f"{variant}"
            )

            result = run_experiment(
                model_type=model_type,
                variant=variant,
                selected_columns=columns,
                X_train_text=X_train_text,
                X_test_text=X_test_text,
                numeric_train=numeric_train,
                numeric_test=numeric_test,
                y_train=y_train,
                y_test=y_test,
            )

            all_results.append(result)
            model_results.append(result)

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

        # Porownanie wariantow z bazowym modelem
        # tego samego typu.

        baseline = model_results[0]["f1"]

        print("\nPOROWNANIE F1 Z BAZOWYM:")

        for result in model_results:

            difference = (
                result["f1"] - baseline
            ) * 100

            print(
                f"{result['variant']:15s} "
                f"F1={result['f1'] * 100:6.2f}% "
                f"zmiana={difference:+.2f} pp"
            )

    # ========================================================
    # ZAPIS WYNIKOW
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(all_results)

    results_path = (
        RESULTS_DIR
        / "text_statistics_experiments.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("PODSUMOWANIE WSZYSTKICH EKSPERYMENTOW")
    print("=" * 72)

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

    for column in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):
        summary[column] = (
            summary[column] * 100
        ).round(2)

    print(summary.to_string(index=False))

    print(f"\nWyniki zapisane do: {results_path}")
    print("Gotowe: 8 eksperymentow, ten sam train/test.")


if __name__ == "__main__":
    main()
