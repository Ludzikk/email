"""
Bazowe SVM: Word TF-IDF i Char TF-IDF na subject + message.
Te same parametry TF-IDF i podzial co w eksperymentach Logistic Regression.
Uruchomienie: py train_svm_baseline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "final_80_20"
SEED = 42


def load_data(split):
    path = DATA_DIR / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    df = pd.read_csv(
        path, encoding="utf-8-sig", keep_default_na=False, low_memory=False
    )
    required = {"subject", "message", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn w {path.name}: {sorted(missing)}")

    labels = pd.to_numeric(df["label"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"Kolumna label w {path.name} musi zawierac tylko 0 i 1.")
    df["label"] = labels.astype(int)
    return df.reset_index(drop=True)


def get_text(df):
    subject = df["subject"].astype(str)
    body = df["message"].astype(str)
    return (subject + " " + body).tolist()

def make_vectorizer(kind):
    if kind == "Word":
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
            sublinear_tf=True,
        )
    if kind == "Char":
        return TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            max_features=40000,
            min_df=2,
            sublinear_tf=True,
        )
    raise ValueError(f"Nieznany wariant: {kind}")


def main():
    train = load_data("train")
    test = load_data("test")
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    train_text = get_text(train)
    test_text = get_text(test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("SVM (LinearSVC) - WORD I CHAR TF-IDF - SUBJECT + BODY")
    print("=" * 70)
    print(f"Train: {len(train)} | Test: {len(test)}")
    print("Cechy dodatkowe: brak | C=1.0 | random_state=42")

    results = []
    predictions = pd.DataFrame({
        "test_row": np.arange(len(test)),
        "source": test["source"].astype(str).to_numpy(),
        "actual": y_test,
    })

    for kind in ("Word", "Char"):
        vectorizer = make_vectorizer(kind)
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)

        svm = LinearSVC(C=1.0, max_iter=10000, random_state=SEED)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

        result = {
            "model": kind,
            "classifier": "LinearSVC",
            "variant": "bazowy_subject_body",
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "n_features": X_train.shape[1],
            "n_train": len(train), "n_test": len(test),
            "C": 1.0,
        }
        results.append(result)
        predictions[f"{kind}_SVM_bazowy"] = pred

        print(f"\n{kind} TF-IDF | cech: {X_train.shape[1]}")
        print(f"Accuracy : {result['accuracy'] * 100:.2f}%")
        print(f"Precision: {result['precision'] * 100:.2f}%")
        print(f"Recall   : {result['recall'] * 100:.2f}%")
        print(f"F1-score : {result['f1'] * 100:.2f}%")
        print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    metrics_df = pd.DataFrame(results)
    metrics_path = RESULTS_DIR / "svm_baseline_word_char_metrics.csv"
    preds_path = RESULTS_DIR / "svm_baseline_word_char_predictions.csv"
    errors_path = RESULTS_DIR / "svm_baseline_word_char_errors_by_source.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(preds_path, index=False, encoding="utf-8-sig")

    error_rows = []
    for kind in ("Word", "Char"):
        col = f"{kind}_SVM_bazowy"
        tmp = predictions[["source", "actual", col]].copy()
        tmp["FP"] = ((tmp["actual"] == 0) & (tmp[col] == 1)).astype(int)
        tmp["FN"] = ((tmp["actual"] == 1) & (tmp[col] == 0)).astype(int)
        by_source = tmp.groupby("source", as_index=False).agg(
            messages=("actual", "size"), FP=("FP", "sum"), FN=("FN", "sum")
        )
        by_source.insert(0, "model", kind)
        error_rows.append(by_source)
    pd.concat(error_rows, ignore_index=True).to_csv(
        errors_path, index=False, encoding="utf-8-sig"
    )

    summary = metrics_df[
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
    for col in ("accuracy", "precision", "recall", "f1"):
        summary[col] = (summary[col] * 100).round(2)
    print("\n" + "=" * 70)
    print("PODSUMOWANIE SVM (F1 W %)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nZapisano:")
    print(metrics_path)
    print(preds_path)
    print(errors_path)


if __name__ == "__main__":
    main()
