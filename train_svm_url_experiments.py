"""12 wariantow Word/Char TF-IDF z LinearSVC i URL pobranymi z urls_json.

Wykorzystuje identyczne ekstraktory i definicje cech jak
train_url_experiments_json.py (regresja logistyczna).

Uruchom: py train_svm_url_experiments.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import train_url_experiments_json as features

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results' / 'final_80_20'
SEED = 42
SVM_C = 1.0
TEXT_WEIGHT = 0.05
METADATA_WEIGHT = 0.05
URL_WEIGHT = 0.05
PREFIX = 'svm_word_char_url_json_text005_meta010_url005'


def main():
    # Wspolne przetwarzanie danych dla SVM i regresji logistycznej.
    features.TEXT_WEIGHT = TEXT_WEIGHT
    features.METADATA_WEIGHT = METADATA_WEIGHT
    features.URL_WEIGHT = URL_WEIGHT

    train = features.load('train')
    test = features.load('test')
    y_train = train['label'].to_numpy()
    y_test = test['label'].to_numpy()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('SVM LinearSVC + URLS_JSON | Word i Char TF-IDF | 12 eksperymentow')
    print('=' * 78)
    print(f'Train: {len(train)} | Test: {len(test)} | SVM_C: {SVM_C}')
    print(f'TEXT_WEIGHT={TEXT_WEIGHT}, METADATA_WEIGHT={METADATA_WEIGHT}, URL_WEIGHT={URL_WEIGHT}')

    # Takie same cechy jak w skrypcie regresji logistycznej.
    text_train = features.make_text(train)
    text_test = features.make_text(test)
    tx_train, tx_test = features.scale(
        features.text_features(train), features.text_features(test)
    )
    mt_train, mt_test = features.scale(
        features.metadata_features(train), features.metadata_features(test)
    )
    url_train_raw, audit_train = features.url_features(train)
    url_test_raw, audit_test = features.url_features(test)
    ur_train, ur_test = features.scale(url_train_raw, url_test_raw)

    for split_name, df, audit_df in (
        ('train', train, audit_train), ('test', test, audit_test)
    ):
        audit_df.insert(0, 'label', df['label'].to_numpy())
        audit_df.insert(0, 'source', df['source'].to_numpy())
        audit_df.insert(0, 'split', split_name)

    extraction = pd.concat([audit_train, audit_test], ignore_index=True)
    extraction.to_csv(
        RESULTS_DIR / f'{PREFIX}_extraction_diagnostics.csv',
        index=False, encoding='utf-8-sig'
    )
    print('\nJAKOSC ODCZYTU URLS_JSON:')
    print(extraction.groupby(['split', 'source', 'json_status'], as_index=False).agg(
        messages=('json_status', 'size'),
        json_items=('json_items', 'sum'),
        valid_urls=('valid_urls', 'sum'),
        ignored_items=('ignored_items', 'sum'),
    ).to_string(index=False))

    audit = pd.concat([
        features.build_audit('train', train, url_train_raw),
        features.build_audit('test', test, url_test_raw),
    ], ignore_index=True)
    features.make_coverage(audit).to_csv(
        RESULTS_DIR / f'{PREFIX}_url_coverage.csv',
        index=False, encoding='utf-8-sig'
    )
    features.make_audit_summary(audit).to_csv(
        RESULTS_DIR / f'{PREFIX}_url_count_audit.csv',
        index=False, encoding='utf-8-sig'
    )

    results = []
    predictions = pd.DataFrame({
        'test_row': np.arange(len(test)),
        'source': test['source'].astype(str).to_numpy(),
        'actual': y_test,
    })

    for kind in ('Word', 'Char'):
        vectorizer = features.make_vectorizer(kind)
        x_train_text = vectorizer.fit_transform(text_train)
        x_test_text = vectorizer.transform(text_test)
        print(f'\n{kind} TF-IDF: {x_train_text.shape[1]} cech')

        for variant, groups in features.VARIANTS.items():
            x_train, x_test = features.combine(
                x_train_text, x_test_text,
                tx_train, tx_test, mt_train, mt_test, ur_train, ur_test, groups
            )
            clf = LinearSVC(C=SVM_C, max_iter=10000, random_state=SEED)
            clf.fit(x_train, y_train)
            predicted = clf.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(
                y_test, predicted, labels=[0, 1]
            ).ravel()
            results.append({
                'model': kind, 'variant': variant,
                'classifier': 'LinearSVC', 'svm_C': SVM_C,
                'text_weight': TEXT_WEIGHT if groups[0] else 0.0,
                'metadata_weight': METADATA_WEIGHT if groups[1] else 0.0,
                'url_weight': URL_WEIGHT if groups[2] else 0.0,
                'n_text_features': len(groups[0]),
                'n_metadata_features': len(groups[1]),
                'n_url_features': len(groups[2]),
                'n_train': len(train), 'n_test': len(test),
                'accuracy': accuracy_score(y_test, predicted),
                'precision': precision_score(y_test, predicted, zero_division=0),
                'recall': recall_score(y_test, predicted, zero_division=0),
                'f1': f1_score(y_test, predicted, zero_division=0),
                'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            })
            predictions[f'{kind}_{variant}'] = predicted
            print(f'{variant:28} F1={f1_score(y_test, predicted) * 100:.2f}% FP={fp} FN={fn}')

    metrics = pd.DataFrame(results)
    metrics.to_csv(
        RESULTS_DIR / f'{PREFIX}_metrics.csv',
        index=False, encoding='utf-8-sig'
    )
    predictions.to_csv(
        RESULTS_DIR / f'{PREFIX}_predictions.csv',
        index=False, encoding='utf-8-sig'
    )

    errors = []
    for kind in ('Word', 'Char'):
        for variant in features.VARIANTS:
            prediction_col = f'{kind}_{variant}'
            temp = predictions[['source', 'actual', prediction_col]].copy()
            temp['FP'] = ((temp['actual'] == 0) & (temp[prediction_col] == 1)).astype(int)
            temp['FN'] = ((temp['actual'] == 1) & (temp[prediction_col] == 0)).astype(int)
            grouped = temp.groupby('source', as_index=False).agg(
                messages=('actual', 'size'), FP=('FP', 'sum'), FN=('FN', 'sum')
            )
            grouped.insert(0, 'variant', variant)
            grouped.insert(0, 'model', kind)
            errors.append(grouped)
    pd.concat(errors, ignore_index=True).to_csv(
        RESULTS_DIR / f'{PREFIX}_errors_by_source.csv',
        index=False, encoding='utf-8-sig'
    )

    table = metrics[
        ['model', 'variant', 'accuracy', 'precision', 'recall', 'f1', 'FP', 'FN']
    ].copy()
    for col in ('accuracy', 'precision', 'recall', 'f1'):
        table[col] = (table[col] * 100).round(2)
    print('\n' + '=' * 78)
    print('PODSUMOWANIE 12 EKSPERYMENTOW SVM')
    print('=' * 78)
    print(table.to_string(index=False))
    print(f'\nPliki zapisano w: {RESULTS_DIR}')
    print(f'Prefiks plikow: {PREFIX}')


if __name__ == '__main__':
    main()
