
"""Char TF-IDF + Logistic Regression.

Dane:
    data/final_80_20/train.csv
    data/final_80_20/test.csv

Podzial: 80% trening / 20% test, bez walidacji.

Uruchomienie:
    py train_logistic_regression_char.py
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_FILE = MODELS_DIR / "logistic_tfidf_char_raw_80_20.joblib"
METRICS_FILE = RESULTS_DIR / "logistic_regression_char_metrics.csv"
CONFUSION_FILE = RESULTS_DIR / "logistic_regression_char_confusion_matrix.csv"


def load_split(name):
    path = DATA_DIR / f"{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Brak pliku: {path}\n"
            "Najpierw uruchom: py build_dataset.py"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    required = {"subject", "message", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"W pliku {path.name} brakuje kolumn: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"Zbior {name} jest pusty.")

    labels = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    if not labels.isin([0, 1]).all():
        raise ValueError("Etykiety musza miec wartosc 0 lub 1.")

    if labels.nunique() != 2:
        raise ValueError(
            f"Zbior {name} musi zawierac obie klasy."
        )

    # Laczymy temat i tresc wiadomosci.
    texts = (
        df["subject"].fillna("").astype(str).str.strip()
        + " "
        + df["message"].fillna("").astype(str).str.strip()
    ).str.strip()

    return texts, labels


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                min_df=2,
                max_features=40_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def evaluate(model, texts, labels):
    predicted = model.predict(texts)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predicted,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": "char_tfidf",
        "dataset": "test",
        "count": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(
            labels, predicted, zero_division=0
        ),
        "recall": recall_score(
            labels, predicted, zero_division=0
        ),
        "f1": f1_score(
            labels, predicted, zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\n" + "=" * 60)
    print("WYNIKI TESTOWE - CHAR TF-IDF")
    print("=" * 60)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    print(f"\nTN={tn} FP={fp} FN={fn} TP={tp}")

    print("\nRaport klasyfikacji:")
    print(
        classification_report(
            labels,
            predicted,
            labels=[0, 1],
            target_names=["legitimate (0)", "phishing (1)"],
            zero_division=0,
        )
    )

    return metrics, [[int(tn), int(fp)], [int(fn), int(tp)]]


def main():
    train_text, y_train = load_split("train")
    test_text, y_test = load_split("test")

    print("=" * 60)
    print("CHAR TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    print(
        f"Train: {len(y_train)} "
        f"(legalne={(y_train == 0).sum()}, "
        f"phishing={(y_train == 1).sum()})"
    )

    print(
        f"Test:  {len(y_test)} "
        f"(legalne={(y_test == 0).sum()}, "
        f"phishing={(y_test == 1).sum()})"
    )

    model = build_model()

    print("\nTrenowanie modelu...")
    model.fit(train_text, y_train)

    print("Trening zakonczony.")

    metrics, cm = evaluate(model, test_text, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Slownik zachowuje zgodnosc z poprzednim formatem char_raw.
    joblib.dump(
        {
            "pipeline": model,
            "normalize_unicode": False,
            "variant": "char_raw",
        },
        MODEL_FILE,
    )

    pd.DataFrame([metrics]).to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(
        CONFUSION_FILE,
        encoding="utf-8-sig",
    )

    print("\nZapisano:")
    print(MODEL_FILE)
    print(METRICS_FILE)
    print(CONFUSION_FILE)


if __name__ == "__main__":
    main()
