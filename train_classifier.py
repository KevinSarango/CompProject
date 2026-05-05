"""
train_classifier.py

Trains a Decision Tree classifier on the VNAT Feature Dataframe
(Release 1) using all 129 available features directly — no manual
column mapping needed since we now know the exact schema.

Feature groups in the dataset:
  - Inter-arrival times     : out_iat_*, in_iat_*, flow_iat_*, active_*, idle_*
  - Log byte/packet counts  : log_bytes_per_sec, log_total_*
  - Wavelet relative energy : in_rel_eng_0..12, out_rel_eng_0..12
  - Shannon entropy         : in_shannon_entropy_0..12, out_shannon_entropy_0..12
  - Wavelet detail coeffs   : in/out_log_mean_abs_detail_coeffs_0..12
                              in/out_log_std_dev_detail_coeffs_0..12
  - Label column            : labels

Requirements:
    pip install pandas tables scikit-learn matplotlib seaborn joblib

Usage:
    python3 train_classifier.py --data VNAT_Feature_Dataframe_release_1.h5
"""

import argparse
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------------
# All 129 VNAT feature columns (everything except 'labels')
# -------------------------------------------------------------------
FEATURE_COLS = [
    # Inter-arrival times
    'out_iat_min', 'out_iat_max', 'out_iat_mean', 'out_iat_std_dev',
    'in_iat_min',  'in_iat_max',  'in_iat_mean',  'in_iat_std_dev',
    'flow_iat_min','flow_iat_max','flow_iat_mean','flow_iat_std_dev',
    'active_min',  'active_max',  'active_mean',  'active_std_dev',
    'idle_min',    'idle_max',    'idle_mean',    'idle_std_dev',

    # Log-scale byte and packet counts
    'log_bytes_per_sec',
    'log_total_outgoing_packets',
    'log_total_incoming_packets',
    'log_total_outgoing_bytes',
    'log_total_incoming_bytes',

    # Wavelet relative energy (incoming and outgoing, 13 levels each)
    *[f'in_rel_eng_{i}'  for i in range(13)],
    *[f'out_rel_eng_{i}' for i in range(13)],

    # Shannon entropy per wavelet level
    *[f'in_shannon_entropy_{i}'  for i in range(13)],
    *[f'out_shannon_entropy_{i}' for i in range(13)],

    # Log mean absolute detail coefficients per wavelet level
    *[f'in_log_mean_abs_detail_coeffs_{i}'  for i in range(13)],
    *[f'out_log_mean_abs_detail_coeffs_{i}' for i in range(13)],

    # Log std dev of detail coefficients per wavelet level
    *[f'in_log_std_dev_detail_coeffs_{i}'  for i in range(13)],
    *[f'out_log_std_dev_detail_coeffs_{i}' for i in range(13)],
]

LABEL_COL   = 'labels'
CLASS_NAMES = ['C2', 'CHAT', 'FILE_TRANSFER', 'STREAMING', 'VOIP']


# -------------------------------------------------------------------
# Inference speed benchmark
# -------------------------------------------------------------------

def benchmark_inference(model, X_test, n_runs=200):
    times = []
    for _ in range(n_runs):
        sample = X_test[np.random.randint(len(X_test))].reshape(1, -1)
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)

    batch_times = []
    for batch_size in [10, 50, 100, 500]:
        if batch_size > len(X_test):
            continue
        t0 = time.perf_counter()
        model.predict(X_test[:batch_size])
        batch_times.append({
            'batch_size': batch_size,
            'ms': (time.perf_counter() - t0) * 1000
        })

    return {
        'single_mean_ms': np.mean(times),
        'single_p99_ms':  np.percentile(times, 99),
        'single_min_ms':  np.min(times),
        'batch_results':  batch_times,
    }


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, label_names,
                          path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Traffic Classification — Confusion Matrix (VNAT)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved -> {path}")


def plot_top_feature_importance(model, feature_names, top_n=20,
                                path='feature_importance.png'):
    """
    Shows only the top N most important features — more readable
    than plotting all 129 at once.
    """
    importances = model.named_steps['clf'].feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = [feature_names[i] for i in indices]
    top_vals    = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_vals[::-1], color='steelblue')
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances — Decision Tree (VNAT)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved -> {path}")


def plot_class_distribution(y, label_names, path='class_distribution.png'):
    unique, counts = np.unique(y, return_counts=True)
    names = [label_names[i] for i in unique]
    plt.figure(figsize=(7, 4))
    plt.bar(names, counts, color='steelblue')
    plt.title('Class Distribution — VNAT Dataset')
    plt.ylabel('Sample Count')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved -> {path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', default='VNAT_Feature_Dataframe_release_1.h5',
        help='Path to VNAT_Feature_Dataframe_release_1.h5'
    )
    args = parser.parse_args()

    # 1. Load
    print(f"[DATA] Loading {args.data} ...")
    df = pd.read_hdf(args.data)
    print(f"[DATA] Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # 2. Verify all expected feature columns exist
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"[WARN] Missing expected columns: {missing_cols}")
    print(f"[DATA] Using {len(FEATURE_COLS)} feature columns + '{LABEL_COL}' label\n")

    # 3. Extract features and labels
    X = df[FEATURE_COLS].values.astype(np.float32)

    raw_labels = df[LABEL_COL].astype(str).str.strip().str.upper()
    print(f"[LABELS] Unique values: {sorted(raw_labels.unique())}")

    unknown = set(raw_labels.unique()) - set(CLASS_NAMES)
    if unknown:
        print(f"[WARN] Dropping {(~raw_labels.isin(CLASS_NAMES)).sum()} "
              f"rows with unknown labels: {unknown}")
        mask = raw_labels.isin(CLASS_NAMES)
        X    = X[mask]
        raw_labels = raw_labels[mask]

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y  = le.transform(raw_labels)

    print(f"[DATA] Final dataset: {X.shape[0]} samples, "
          f"{X.shape[1]} features, {len(le.classes_)} classes\n")

    # Replace NaN/inf with 0 (some wavelet features may be NaN for short flows)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    plot_class_distribution(y, list(le.classes_))

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Decision Tree pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=2,
            random_state=42
        ))
    ])

    print("[TRAIN] Training Decision Tree on VNAT features...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[TRAIN] Done in {time.time() - t0:.2f}s\n")

    # 6. Evaluate
    y_pred        = model.predict(X_test)
    present_idx   = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_names = [le.classes_[i] for i in present_idx]

    print("=" * 55)
    print("CLASSIFICATION REPORT (VNAT)")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=present_names))

    cv_scores = cross_val_score(model, X, y, cv=5,
                                scoring='f1_macro', n_jobs=-1)
    print(f"5-Fold CV F1 (macro): {cv_scores.mean():.4f} "
          f"+/- {cv_scores.std():.4f}\n")

    # 7. Inference benchmark
    print("=" * 55)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 55)
    bench = benchmark_inference(model, X_test)
    print(f"Single-sample: mean={bench['single_mean_ms']:.3f}ms  "
          f"p99={bench['single_p99_ms']:.3f}ms  "
          f"min={bench['single_min_ms']:.3f}ms")
    for b in bench['batch_results']:
        print(f"  Batch {b['batch_size']:4d}: {b['ms']:.3f}ms total  "
              f"({b['ms']/b['batch_size']:.4f}ms per flow)")

    # 8. Plots
    plot_confusion_matrix(y_test, y_pred, present_names)
    plot_top_feature_importance(model, FEATURE_COLS, top_n=20)

    # 9. Save model + label encoder + feature list together
    joblib.dump({
        'model':         model,
        'label_encoder': le,
        'feature_cols':  FEATURE_COLS,
    }, 'traffic_classifier.pkl')
    print("\n[SAVE] Model saved -> traffic_classifier.pkl")
    print("[INFO] Start Ryu with: ryu-manager ryu_classifier_controller.py")