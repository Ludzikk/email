"""Read-only check of CSV structure and labels in phishing project splits.
Run from project root: python audit_splits.py
"""
from pathlib import Path
from collections import Counter
import csv
import sys

try:
    import pandas as pd
except ImportError:
    pd = None

ROOT = Path(__file__).resolve().parent
FILES = [
    ROOT / 'data/splits/train.csv',
    ROOT / 'data/splits/validation.csv',
    ROOT / 'data/splits/test.csv',
    ROOT / 'data/splits/nazario_recent_phishing.csv',
]
EXPECTED_LABELS = {'0', '1'}
MAX_EXAMPLES = 5

try:
    csv.field_size_limit(2_147_483_647)
except OverflowError:
    csv.field_size_limit(100_000_000)


def compact(value, limit=90):
    s = str(value).replace('\\r', ' ').replace('\\n', ' ').replace('\r', ' ').replace('\n', ' ')
    return repr(s[:limit] + ('...' if len(s) > limit else ''))


def inspect(path):
    print('\n' + '=' * 72)
    print(f'PLIK: {path.relative_to(ROOT)}')
    if not path.exists():
        print('BRAK PLIKU — pomijam')
        return False

    issues = []
    counts = Counter()
    total = 0
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as fh:
            reader = csv.reader(fh, strict=True)
            header = next(reader, None)
            if not header:
                print('BLAD: pusty plik lub brak naglowka')
                return False
            print('Kolumny:', ', '.join(header))
            if len(header) != len(set(header)):
                issues.append(('naglowek', 'powtarzajace sie nazwy kolumn'))
            if 'label' not in header:
                print('BLAD: brak kolumny label')
                return False
            label_idx = header.index('label')
            for row in reader:
                total += 1
                if len(row) != len(header):
                    if len(issues) < MAX_EXAMPLES:
                        issues.append((total, f'liczba pol: {len(row)} zamiast {len(header)}'))
                    continue
                label = row[label_idx].strip()
                counts[label if label in EXPECTED_LABELS else 'NIEPOPRAWNA'] += 1
                if label not in EXPECTED_LABELS and len(issues) < MAX_EXAMPLES:
                    issues.append((total, 'niepoprawne label=' + compact(row[label_idx])))
    except (csv.Error, UnicodeError, OSError) as e:
        print('BLAD odczytu CSV:', type(e).__name__, str(e))
        return False

    print(f'Rekordy CSV (nie linie tekstu): {total}')
    print(f'Label 0: {counts["0"]}, label 1: {counts["1"]}, niepoprawne: {counts["NIEPOPRAWNA"]}')

    pandas_ok = True
    if pd is not None:
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig', on_bad_lines='error')
            p_counts = df['label'].str.strip().value_counts(dropna=False).to_dict()
            pandas_ok = len(df) == total and all(k in EXPECTED_LABELS for k in p_counts)
            print('Pandas: rekordow =', len(df), '; label =', p_counts)
            if not pandas_ok:
                print('UWAGA: odczyt Pandas nie zgadza sie z oczekiwaniami')
        except Exception as e:
            pandas_ok = False
            print('BLAD odczytu Pandas:', type(e).__name__, str(e))

    if issues:
        print('Wykryte problemy (maksymalnie pierwsze 5):')
        for at, desc in issues[:MAX_EXAMPLES]:
            print(f'  rekord {at}: {desc}')

    is_ok = not issues and counts['NIEPOPRAWNA'] == 0 and pandas_ok
    print('WYNIK:', 'OK — etykiety i struktura CSV poprawne' if is_ok else 'UWAGA — sprawdz problemy powyzej')
    return is_ok


def main():
    results = [inspect(path) for path in FILES]
    print('\n' + '=' * 72)
    if all(results):
        print('PODSUMOWANIE: wszystkie dostepne pliki przeszly kontrole.')
    else:
        print('PODSUMOWANIE: co najmniej jeden plik wymaga uwagi (lub nie istnieje).')
        sys.exit(1)


if __name__ == '__main__':
    main()
