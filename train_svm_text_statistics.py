"""
SVM (LinearSVC): Word i Char TF-IDF + statystyczne cechy tekstu.

Osiem wariantow: Word/Char x bazowy, A_dlugosc, B_styl, C_wszystkie.
Wejscie: data/splits/train.csv i data/splits/test.csv
Uruchomienie: py train_svm_text_statistics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "final_80_20"
SEED = 42
NUMERIC_WEIGHT = 0.05
SVM_C = 1.0

LENGTH_FEATURES = [
    "subject_length", "message_length", "message_word_count"
]
STYLE_FEATURES = [
    "exclamation_count", "question_count", "uppercase_ratio"
]
ALL_FEATURES = LENGTH_FEATURES + STYLE_FEATURES
VARIANTS = {
    "bazowy": [],
    "A_dlugosc": LENGTH_FEATURES,
    "B_styl": STYLE_FEATURES,
    "C_wszystkie": ALL_FEATURES,
}


def load_data(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Brak pliku: {path}")

    df = pd.read_csv(
        path, encoding="utf-8-sig", keep_default_na=False, low_memory=False
    )
    required = {"subject", "message", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"W {path.name} brakuje kolumn: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Plik {path.name} jest pusty")
    labels = pd.to_numeric(df["label"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError(f"Bledne etykiety albo brak jednej klasy w {path.name}")
    df["label"] = labels.astype(int)
    return df.reset_index(drop=True)


def prepare_text(df):
    subject = df["subject"].astype(str)
    message = df["message"].astype(str)
    return (subject + " " + message).tolist()


def uppercase_ratio(text):
    letters = sum(ch.isalpha() for ch in text)
    return sum(ch.isupper() for ch in text) / letters if letters else 0.0


def make_numeric_features(df):
    subject = df["subject"].astype(str)
    message = df["message"].astype(str)
    # Kazde zrodlo danych jest przeliczane tymi samymi regulami.
    return pd.DataFrame({
        "subject_length": subject.str.len(),
        "message_length": message.str.len(),
        "message_word_count": message.str.split().str.len(),
        "exclamation_count": message.str.count("!"),
        "question_count": message.str.count(r"\?"),
        "uppercase_ratio": message.map(uppercase_ratio),
    })[ALL_FEATURES].astype("float64")


def create_vectorizer(model_type):
    if model_type == "Word":
        return TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2,
            max_features=30000, sublinear_tf=True
        )
    if model_type == "Char":
        return TfidfVectorizer(
            analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
            min_df=2, max_features=40000, sublinear_tf=True
        )
    raise ValueError(f"Nieznany model: {model_type}")


def combine_features(x_train_text, x_test_text,
                     numeric_train, numeric_test, selected_columns):
    if not selected_columns:
        return x_train_text, x_test_text

    # Fit tylko na train; osobno dla kazdego wariantu cech.
    scaler = StandardScaler()
    extra_train = scaler.fit_transform(numeric_train[selected_columns]) * NUMERIC_WEIGHT
    extra_test = scaler.transform(numeric_test[selected_columns]) * NUMERIC_WEIGHT
    x_train = hstack([x_train_text, csr_matrix(extra_train)], format="csr")
    x_test = hstack([x_test_text, csr_matrix(extra_test)], format="csr")
    return x_train, x_test


def main():
    train = load_data("train")
    test = load_data("test")
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    train_text = prepare_text(train)
    test_text = prepare_text(test)
    numeric_train = make_numeric_features(train)
    numeric_test = make_numeric_features(test)

    print("=" * 78)
    print("SVM LinearSVC: WORD / CHAR TF-IDF + CECHY TEKSTOWE")
    print("=" * 78)
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"SVM C={SVM_C} | waga cech liczbowych={NUMERIC_WEIGHT}")
    print("Reprezentacja tekstu: subject + message")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = pd.DataFrame({
        "test_row": np.arange(len(test)),
        "source": (test["source"].astype(str).to_numpy()
                   if "source" in test.columns else [""] * len(test)),
        "actual": y_test,
    })

    for model_type in ("Word", "Char"):
        vectorizer = create_vectorizer(model_type)
        x_train_text = vectorizer.fit_transform(train_text)
        x_test_text = vectorizer.transform(test_text)
        print(f"\n{model_type} TF-IDF: {x_train_text.shape[1]} cech")

        for variant, selected_columns in VARIANTS.items():
            x_train, x_test = combine_features(
                x_train_text, x_test_text,
                numeric_train, numeric_test, selected_columns
            )
            clf = LinearSVC(C=SVM_C, max_iter=10000, random_state=SEED)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            row = {
                "model": model_type,
                "variant": variant,
                "classifier": "LinearSVC",
                "C": SVM_C,
                "numeric_weight": NUMERIC_WEIGHT if selected_columns else 0.0,
                "n_extra_features": len(selected_columns),
                "n_train": len(train),
                "n_test": len(test),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            }
            results.append(row)
            predictions[f"{model_type}_{variant}"] = pred
            print(f"  {variant:15s} F1={row['f1']*100:.2f}% FP={fp} FN={fn}")

    metrics = pd.DataFrame(results)
    metrics_path = RESULTS_DIR / "svm_text_statistics_experiments.csv"
    preds_path = RESULTS_DIR / "svm_text_statistics_predictions.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(preds_path, index=False, encoding="utf-8-sig")

    summary = metrics[
        ["model", "variant", "accuracy", "precision", "recall", "f1", "FP", "FN"]
    ].copy()
    for col in ("accuracy", "precision", "recall", "f1"):
        summary[col] = (summary[col] * 100).round(2)

    print("\n" + "=" * 78)
    print("PODSUMOWANIE 8 EKSPERYMENTOW SVM")
    print("=" * 78)
    print(summary.to_string(index=False))
    print(f"\nWyniki: {metrics_path}")
    print(f"Predykcje: {preds_path}")


if __name__ == "__main__":
    main()
