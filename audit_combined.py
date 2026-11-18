import pandas as pd

df = pd.read_csv("data/splits/combined.csv", low_memory=False)

total = len(df)

print(f"Liczba wszystkich wiadomosci: {total}")

print("\nROZKLAD KLAS:")

for label in (0, 1):
    count = (df["label"] == label).sum()
    percent = count / total * 100

    print(f"Klasa {label}: {count} ({percent:.2f}%)")

print("\nBrakujace etykiety:", df["label"].isna().sum())

print("\nROZKLAD KLAS WEDLUG ZRODLA:")

print(pd.crosstab(df["source"], df["label"]))
