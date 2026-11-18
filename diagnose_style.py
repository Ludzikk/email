
"""
DIAGNOSTYKA WPLYWU CECH STYLISTYCZNYCH

Sprawdza:
1. Skutecznosc samych cech stylistycznych.
2. Normy wektorow TF-IDF i cech stylistycznych.
3. Wklad obu grup cech do decyzji modelu.
4. Czy Word + styl i Char + styl myla sie na tych samych mailach.

Dane:
    data/splits/train.csv
    data/splits/test.csv

Wyniki:
    results/final_80_20/style_diagnostics_*.csv

Nie zmienia datasetu ani zapisanych modeli.
Uruchomienie: py diagnose_style.py
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
STYLE_WEIGHT = 0.1

STYLE_COLUMNS = [
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
]


# ============================================================
# WCZYTYWANIE
# ============================================================

def load_split(split):

    path = DATA_DIR / f"{split}.csv"

    if not path.exists():
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
            f"Pusty plik: {path}"
        )

    y = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError(
            f"Nieprawidlowe etykiety: {path.name}"
        )

    if y.nunique() != 2:
        raise ValueError(
            f"Brakuje jednej klasy: {path.name}"
        )

    return df, y.to_numpy()


# ============================================================
# TEKST I CECHY STYLISTYCZNE
# ============================================================

def prepare_text(df):

    return (
        df["subject"].astype(str)
        + " "
        + df["message"].astype(str)
    ).tolist()


def uppercase_ratio(text):

    letters = sum(
        c.isalpha()
        for c in text
    )

    if letters == 0:
        return 0.0

    uppercase = sum(
        c.isupper()
        for c in text
    )

    return uppercase / letters


def make_style_features(df):

    message = df["message"].astype(str)

    features = pd.DataFrame({
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
        STYLE_COLUMNS
    ].astype("float64")


# ============================================================
# WEKTORYZATORY
# ============================================================

def create_vectorizer(kind):

    if kind == "Word":

        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )

    if kind == "Char":

        return TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=2,
            max_features=40000,
            sublinear_tf=True,
        )

    raise ValueError(
        f"Nieznany model: {kind}"
    )


# ============================================================
# METRYKI
# ============================================================

def evaluate(y, predicted, name):

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "wariant": name,

        "accuracy": accuracy_score(
            y, predicted
        ),

        "precision": precision_score(
            y, predicted,
            zero_division=0,
        ),

        "recall": recall_score(
            y, predicted,
            zero_division=0,
        ),

        "f1": f1_score(
            y, predicted,
            zero_division=0,
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# NORMY WEKTOROW
# ============================================================

def norm_rows(matrix):

    if hasattr(matrix, "multiply"):

        squared = matrix.multiply(matrix)

        return np.sqrt(
            np.asarray(
                squared.sum(axis=1)
            ).ravel()
        )

    return np.linalg.norm(
        matrix,
        axis=1,
    )


def describe(values):

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


# ============================================================
# DIAGNOSTYKA
# ============================================================

def main():

    train, y_train = load_split("train")
    test, y_test = load_split("test")

    train_text = prepare_text(train)
    test_text = prepare_text(test)

    print("=" * 70)
    print("DIAGNOSTYKA CECH STYLISTYCZNYCH")
    print("=" * 70)

    print(f"Train: {len(train)}")
    print(f"Test:  {len(test)}")
    print(f"Waga stylu: {STYLE_WEIGHT}")

    # Skalowanie TYLKO na train.

    scaler = StandardScaler()

    numeric_train = scaler.fit_transform(
        make_style_features(train)
    )

    numeric_test = scaler.transform(
        make_style_features(test)
    )

    # Identyczna waga jak w poprzednim eksperymencie.

    train_weighted = (
        numeric_train * STYLE_WEIGHT
    )

    test_weighted = (
        numeric_test * STYLE_WEIGHT
    )

    # ========================================================
    # 1. SAM STYL
    # ========================================================

    # W modelu samodzielnym uzywamy standardowego
    # skalowania, bez wagi 0.1, poniewaz nie laczymy
    # tutaj dwoch roznych grup cech.

    style_only = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
    )

    style_only.fit(
        numeric_train,
        y_train,
    )

    style_pred = style_only.predict(
        numeric_test
    )

    summary = [
        evaluate(
            y_test,
            style_pred,
            "Sam_styl",
        )
    ]

    predictions = pd.DataFrame({
        "test_row": np.arange(len(test)),

        "label": y_test,

        "source": (
            test["source"].astype(str).to_numpy()
            if "source" in test.columns
            else np.repeat("nieznane", len(test))
        ),

        "Sam_styl": style_pred,
    })

    norms = []
    contributions = []

    # ========================================================
    # 2. WORD I CHAR
    # ========================================================

    for kind in ("Word", "Char"):

        print("\n" + "=" * 70)
        print(f"MODEL: {kind}")
        print("=" * 70)

        vectorizer = create_vectorizer(kind)

        X_train_tfidf = vectorizer.fit_transform(
            train_text
        )

        X_test_tfidf = vectorizer.transform(
            test_text
        )

        n_text = X_train_tfidf.shape[1]

        # ----------------------------------------------------
        # MODEL BAZOWY
        # ----------------------------------------------------

        baseline = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        )

        baseline.fit(
            X_train_tfidf,
            y_train,
        )

        base_pred = baseline.predict(
            X_test_tfidf
        )

        summary.append(
            evaluate(
                y_test,
                base_pred,
                f"{kind}_bazowy",
            )
        )

        # ----------------------------------------------------
        # MODEL TF-IDF + STYL
        # ----------------------------------------------------

        X_train_both = hstack(
            [
                X_train_tfidf,
                csr_matrix(train_weighted),
            ],
            format="csr",
        )

        X_test_both = hstack(
            [
                X_test_tfidf,
                csr_matrix(test_weighted),
            ],
            format="csr",
        )

        combined = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        )

        combined.fit(
            X_train_both,
            y_train,
        )

        combined_pred = combined.predict(
            X_test_both
        )

        summary.append(
            evaluate(
                y_test,
                combined_pred,
                f"{kind}_styl",
            )
        )

        predictions[f"{kind}_bazowy"] = (
            base_pred
        )

        predictions[f"{kind}_styl"] = (
            combined_pred
        )

        # ----------------------------------------------------
        # 3. WPLYW GRUP CECH NA DECYZJE
        # ----------------------------------------------------

        # Regresja logistyczna:
        #
        # logit = TF-IDF * wspolczynniki_tekstu
        #       + styl * wspolczynniki_stylu
        #       + intercept
        #
        # Pozwala to zmierzyc osobny wklad
        # tekstu i stylu do decyzji.

        coefficients = combined.coef_[0]

        text_coefficients = (
            coefficients[:n_text]
        )

        style_coefficients = (
            coefficients[n_text:]
        )

        text_contrib = np.asarray(
            X_test_tfidf @ text_coefficients
        ).ravel()

        style_contrib = (
            test_weighted @ style_coefficients
        )

        abs_text = np.abs(
            text_contrib
        )

        abs_style = np.abs(
            style_contrib
        )

        # Udzial bezwzglednego wkladu stylu.
        # Nie jest to procent waznosci modelu,
        # tylko pomocnicza miara diagnostyczna.

        style_share = (
            abs_style
            / np.maximum(
                abs_text + abs_style,
                1e-12,
            )
        )

        style_dominates = (
            abs_style > abs_text
        )

        predictions[
            f"{kind}_text_contrib"
        ] = text_contrib

        predictions[
            f"{kind}_style_contrib"
        ] = style_contrib

        predictions[
            f"{kind}_style_abs_share"
        ] = style_share

        contributions.append({
            "model": kind,

            "avg_abs_text_logit": float(
                abs_text.mean()
            ),

            "avg_abs_style_logit": float(
                abs_style.mean()
            ),

            "median_style_abs_share": float(
                np.median(style_share)
            ),

            "style_abs_gt_text_percent": float(
                100 * style_dominates.mean()
            ),

            "changed_decisions_vs_baseline": int(
                (base_pred != combined_pred).sum()
            ),

            "combined_errors": int(
                (combined_pred != y_test).sum()
            ),

            "features_tfidf": int(
                n_text
            ),
        })

        # ----------------------------------------------------
        # 4. NORMY WEKTOROW
        # ----------------------------------------------------

        for split_name, X_tf, scaled in (
            ("train", X_train_tfidf, train_weighted),
            ("test", X_test_tfidf, test_weighted),
        ):

            for group_name, matrix in (
                ("tfidf", X_tf),
                ("style_scaled_x0_1", scaled),
            ):

                values = norm_rows(matrix)

                norms.append({
                    "model": kind,
                    "split": split_name,
                    "group": group_name,
                    **describe(values),
                })

    # ========================================================
    # 5. POROWNANIE BLEDOW WORD I CHAR
    # ========================================================

    word_errors = set(
        np.where(
            predictions["Word_styl"].to_numpy()
            != y_test
        )[0]
    )

    char_errors = set(
        np.where(
            predictions["Char_styl"].to_numpy()
            != y_test
        )[0]
    )

    common_errors = (
        word_errors & char_errors
    )

    only_word = (
        word_errors - char_errors
    )

    only_char = (
        char_errors - word_errors
    )

    print("\nPOROWNANIE BLEDOW")
    print(f"Word + styl: {len(word_errors)}")
    print(f"Char + styl: {len(char_errors)}")
    print(f"Wspolne bledy: {len(common_errors)}")
    print(f"Tylko Word: {len(only_word)}")
    print(f"Tylko Char: {len(only_char)}")

    print(
        "Indeksy bledow Word:",
        sorted(word_errors),
    )

    print(
        "Indeksy bledow Char:",
        sorted(char_errors),
    )

    # ========================================================
    # 6. ZAPIS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(
        summary
    )

    norms_df = pd.DataFrame(
        norms
    )

    contrib_df = pd.DataFrame(
        contributions
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "style_diagnostics_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    norms_df.to_csv(
        RESULTS_DIR
        / "style_diagnostics_norms.csv",
        index=False,
        encoding="utf-8-sig",
    )

    contrib_df.to_csv(
        RESULTS_DIR
        / "style_diagnostics_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        RESULTS_DIR
        / "style_diagnostics_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # WYSWIETLENIE WYNIKOW
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("METRYKI")
    print("=" * 70)

    display_metrics = (
        summary_df.copy()
    )

    for column in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):
        display_metrics[column] = (
            display_metrics[column] * 100
        ).round(2)

    print(
        display_metrics.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("NORMY BLOKOW CECH")
    print("=" * 70)

    print(
        norms_df.round(4).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("WPLYW CECH NA DECYZJE MODELU")
    print("=" * 70)

    print(
        contrib_df.round(4).to_string(index=False)
    )

    print("\nZapisano wyniki w:")
    print(RESULTS_DIR)

    print(
        "\nUwaga: jest to diagnostyka na ustalonym "
        "zbiorze testowym, a nie strojenie wag."
    )


if __name__ == "__main__":
    main()
