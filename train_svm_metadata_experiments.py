"""SVM LinearSVC: 20 eksperymentow Word/Char TF-IDF + tekst + metadane.
Uruchom: py train_svm_metadata_experiments.py
"""
from pathlib import Path
from email.utils import getaddresses

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "final_80_20"

SEED = 42
SVM_C = 1.0
TEXT_WEIGHT = 0.05
METADATA_WEIGHT = 0.05
PREFIX = "svm_word_char_text_metadata_20variants"

LENGTH = ["subject_length", "message_length", "message_word_count"]
STYLE = ["exclamation_count", "question_count", "uppercase_ratio"]
ALL_TEXT = LENGTH + STYLE
SENDER = [
    "sender_address_length", "sender_local_length", "sender_domain_length",
    "sender_domain_digit_count", "sender_domain_hyphen_count",
    "sender_address_has_digit",
]
RECEIVER = [
    "receiver_address_length", "receiver_local_length", "receiver_domain_length",
    "receiver_count", "sender_receiver_domain_match",
    "sender_receiver_domain_length_diff",
]
META_C = SENDER + RECEIVER

# Takie same 10 wariantow jak w przeslanym skrypcie LR.
VARIANTS = {
    "bazowy": ([], []),
    "styl": (STYLE, []),
    "dlugosc": (LENGTH, []),
    "all_tekstowe": (ALL_TEXT, []),
    "meta_A_nadawca": ([], SENDER),
    "meta_B_odbiorca": ([], RECEIVER),
    "meta_C_wszystkie": ([], META_C),
    "meta_C_dlugosc": (LENGTH, META_C),
    "meta_C_styl": (STYLE, META_C),
    "meta_C_all_tekstowe": (ALL_TEXT, META_C),
}


def load_split(split):
    path = DATA_DIR / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    required = {"subject", "message", "sender", "receiver", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: brakuje kolumn {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path.name} jest pusty")
    labels = pd.to_numeric(df["label"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError(f"{path.name}: label musi zawierac obie klasy 0 i 1")
    df["label"] = labels.astype(int)
    return df.reset_index(drop=True)


def text_for_tfidf(df):
    return (df["subject"].astype(str) + " " + df["message"].astype(str)).tolist()


def uppercase_ratio(s):
    letters = sum(ch.isalpha() for ch in s)
    return sum(ch.isupper() for ch in s) / letters if letters else 0.0


def text_features(df):
    subject = df["subject"].astype(str)
    message = df["message"].astype(str)
    return pd.DataFrame({
        "subject_length": subject.str.len(),
        "message_length": message.str.len(),
        "message_word_count": message.str.split().str.len(),
        "exclamation_count": message.str.count("!"),
        "question_count": message.str.count(r"\?"),
        "uppercase_ratio": message.map(uppercase_ratio),
    })[ALL_TEXT].astype("float64")


def addresses(raw):
    found = []
    for _, addr in getaddresses([str(raw)]):
        addr = addr.strip().lower()
        if "@" in addr:
            local, domain = addr.rsplit("@", 1)
            if local and domain:
                found.append(addr)
    return found


def metadata_features(df):
    rows = []
    coverage = []
    for sender, receiver in zip(df["sender"], df["receiver"]):
        senders, receivers = addresses(sender), addresses(receiver)
        sa = senders[0] if senders else ""
        ra = receivers[0] if receivers else ""
        sl, sd = sa.rsplit("@", 1) if sa else ("", "")
        rl, rd = ra.rsplit("@", 1) if ra else ("", "")
        rows.append({
            "sender_address_length": len(sa),
            "sender_local_length": len(sl),
            "sender_domain_length": len(sd),
            "sender_domain_digit_count": sum(c.isdigit() for c in sd),
            "sender_domain_hyphen_count": sd.count("-"),
            "sender_address_has_digit": int(any(c.isdigit() for c in sa)),
            "receiver_address_length": len(ra),
            "receiver_local_length": len(rl),
            "receiver_domain_length": len(rd),
            "receiver_count": len(receivers),
            "sender_receiver_domain_match": int(bool(sd and rd) and sd == rd),
            "sender_receiver_domain_length_diff": abs(len(sd) - len(rd)),
        })
        coverage.append((bool(senders), bool(receivers)))
    values = pd.DataFrame(rows, columns=META_C).astype("float64")
    available = pd.DataFrame(coverage, columns=["sender_available", "receiver_available"])
    available["source"] = df["source"].astype(str)
    return values, available


def standardize(train, test):
    scaler = StandardScaler()
    return (
        pd.DataFrame(scaler.fit_transform(train), columns=train.columns),
        pd.DataFrame(scaler.transform(test), columns=test.columns),
    )


def vectorizer_for(kind):
    if kind == "Word":
        return TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2,
            max_features=30000, sublinear_tf=True,
        )
    return TfidfVectorizer(
        analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
        min_df=2, max_features=40000, sublinear_tf=True,
    )


def combine(tf_train, tf_test, txt_train, txt_test,
            meta_train, meta_test, selected_text, selected_meta):
    train_parts, test_parts = [tf_train], [tf_test]
    if selected_text:
        train_parts.append(csr_matrix(txt_train[selected_text].to_numpy() * TEXT_WEIGHT))
        test_parts.append(csr_matrix(txt_test[selected_text].to_numpy() * TEXT_WEIGHT))
    if selected_meta:
        train_parts.append(csr_matrix(meta_train[selected_meta].to_numpy() * METADATA_WEIGHT))
        test_parts.append(csr_matrix(meta_test[selected_meta].to_numpy() * METADATA_WEIGHT))
    if len(train_parts) == 1:
        return tf_train, tf_test
    return hstack(train_parts, format="csr"), hstack(test_parts, format="csr")


def main():
    train, test = load_split("train"), load_split("test")
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SVM LinearSVC - 20 EKSPERYMENTOW: WORD/CHAR + TEKST + METADANE")
    print("=" * 78)
    print(f"Train={len(train)}, Test={len(test)}")
    print(f"SVM_C={SVM_C}, TEXT_WEIGHT={TEXT_WEIGHT}, METADATA_WEIGHT={METADATA_WEIGHT}")

    text_train = text_for_tfidf(train)
    text_test = text_for_tfidf(test)
    numeric_train, numeric_test = standardize(text_features(train), text_features(test))
    raw_meta_train, available_train = metadata_features(train)
    raw_meta_test, available_test = metadata_features(test)
    meta_train, meta_test = standardize(raw_meta_train, raw_meta_test)

    # Raport dostepnosci adresow (tylko diagnostyka, nigdy wejscie do modelu).
    coverage_frames = []
    for split, available in (("train", available_train), ("test", available_test)):
        frame = available.groupby("source", as_index=False).agg(
            messages=("source", "size"),
            sender_available=("sender_available", "sum"),
            receiver_available=("receiver_available", "sum"),
        )
        frame.insert(0, "split", split)
        coverage_frames.append(frame)
    pd.concat(coverage_frames, ignore_index=True).to_csv(
        RESULTS_DIR / f"{PREFIX}_address_coverage.csv", index=False, encoding="utf-8-sig"
    )

    results = []
    predictions = pd.DataFrame({
        "test_row": np.arange(len(test)),
        "source": test["source"].astype(str).to_numpy(),
        "actual": y_test,
    })

    for kind in ("Word", "Char"):
        tfidf = vectorizer_for(kind)
        X_tf_train = tfidf.fit_transform(text_train)
        X_tf_test = tfidf.transform(text_test)
        print(f"\n{kind} TF-IDF: {X_tf_train.shape[1]} cech")

        for variant, (text_cols, meta_cols) in VARIANTS.items():
            X_train, X_test = combine(
                X_tf_train, X_tf_test, numeric_train, numeric_test,
                meta_train, meta_test, text_cols, meta_cols,
            )
            clf = LinearSVC(C=SVM_C, max_iter=10000, random_state=SEED)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            result = {
                "model": kind, "variant": variant, "classifier": "LinearSVC",
                "svm_C": SVM_C,
                "text_weight": TEXT_WEIGHT if text_cols else 0.0,
                "metadata_weight": METADATA_WEIGHT if meta_cols else 0.0,
                "n_text_features": len(text_cols), "n_metadata_features": len(meta_cols),
                "n_train": len(train), "n_test": len(test),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            }
            results.append(result)
            predictions[f"{kind}_{variant}"] = pred
            print(f"{variant:23s} F1={result['f1']*100:6.2f}% FP={fp} FN={fn}")

    metrics = pd.DataFrame(results)
    metrics_path = RESULTS_DIR / f"{PREFIX}_metrics.csv"
    predictions_path = RESULTS_DIR / f"{PREFIX}_predictions.csv"
    errors_path = RESULTS_DIR / f"{PREFIX}_errors_by_source.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    errors = []
    for kind in ("Word", "Char"):
        for variant in VARIANTS:
            pcol = f"{kind}_{variant}"
            temp = predictions[["source", "actual", pcol]].copy()
            temp["FP"] = ((temp["actual"] == 0) & (temp[pcol] == 1)).astype(int)
            temp["FN"] = ((temp["actual"] == 1) & (temp[pcol] == 0)).astype(int)
            grouped = temp.groupby("source", as_index=False).agg(
                messages=("actual", "size"), FP=("FP", "sum"), FN=("FN", "sum")
            )
            grouped.insert(0, "variant", variant)
            grouped.insert(0, "model", kind)
            errors.append(grouped)
    pd.concat(errors, ignore_index=True).to_csv(
        errors_path, index=False, encoding="utf-8-sig"
    )

    summary = metrics[["model", "variant", "accuracy", "precision", "recall", "f1", "FP", "FN"]].copy()
    for col in ("accuracy", "precision", "recall", "f1"):
        summary[col] = (summary[col] * 100).round(2)
    print("\n" + "=" * 78)
    print("PODSUMOWANIE 20 EKSPERYMENTOW SVM")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("\nPliki:")
    print(metrics_path)
    print(predictions_path)
    print(errors_path)
    print(RESULTS_DIR / f"{PREFIX}_address_coverage.csv")


if __name__ == "__main__":
    main()
