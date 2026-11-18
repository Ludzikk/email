
from pathlib import Path
import json

import pandas as pd

from train_url_experiments import parse_urls


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "results" / "final_80_20"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def decode_urls(value):
    try:
        result = json.loads(str(value))
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


records = []

for split in ("train", "test"):
    df = pd.read_csv(
        DATA_DIR / f"{split}.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )

    for row_number, row in df.iterrows():
        subject = str(row["subject"])
        message = str(row["message"])

        new_urls = [
            item["url"]
            for item in parse_urls(subject, message)
        ]

        old_urls = decode_urls(
            row.get("urls_json", "")
        )

        old_count = pd.to_numeric(
            row.get("url_count", 0),
            errors="coerce",
        )

        if pd.isna(old_count):
            old_count = 0

        records.append({
            "split": split,
            "row": row_number,
            "source": str(row["source"]),
            "label": row["label"],
            "old_count": int(old_count),
            "new_count": len(new_urls),
            "old_urls": str(old_urls)[:500],
            "new_urls": str(new_urls)[:500],
            "subject": subject[:150],
            "message_preview": message[:300],
        })


result = pd.DataFrame(records)

result["mismatch"] = (
    result["old_count"] != result["new_count"]
)

result["old_yes_new_no"] = (
    (result["old_count"] > 0)
    & (result["new_count"] == 0)
)

print("\n" + "=" * 75)
print("NIEZGODNOSCI URL WEDLUG ZRODLA")
print("=" * 75)

summary = result.groupby(
    "source",
    as_index=False,
).agg(
    messages=("source", "size"),
    mismatches=("mismatch", "sum"),
    old_yes_new_no=("old_yes_new_no", "sum"),
)

print(summary.to_string(index=False))


# Najpierw pokazujemy przypadki Nazario,
# w ktorych stara metoda wykryla URL,
# a nowa nie wykryla zadnego.

nazario = result[
    result["source"].str.startswith("nazario")
    & result["old_yes_new_no"]
].head(10)

print("\n" + "=" * 75)
print("10 PRZYKLADOW NAZARIO")
print("Stary ekstraktor: URL jest")
print("Nowy ekstraktor:  URL nie zostal wykryty")
print("=" * 75)

for _, item in nazario.iterrows():
    print("\n" + "-" * 75)

    print("Split:", item["split"])
    print("Wiersz:", item["row"])
    print("Zrodlo:", item["source"])

    print("Stara liczba URL:", item["old_count"])
    print("Nowa liczba URL:", item["new_count"])

    print("STARE URL:", item["old_urls"])
    print("NOWE URL:", item["new_urls"])

    print("TEMAT:", item["subject"])
    print("FRAGMENT TRESCI:", item["message_preview"])


output_path = (
    OUT_DIR / "url_mismatch_diagnostics.csv"
)

result[result["mismatch"]].to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print("\nZapisano:", output_path)
print("\nGotowe.")
