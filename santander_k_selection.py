#!/usr/bin/env python
"""
Santander Customer Segmentation – K Selection Helper

This script helps you choose a reasonable number of clusters (K) for KMeans by:
- Loading a cleaned segmentation CSV (from santander_segmentation_prep.py).
- Building a feature matrix from product ownership + simple numeric features.
- Optionally subsampling for speed.
- Fitting KMeans for K = min_k..max_k.
- Computing:
    * Inertia (within-cluster SSE)
    * Average silhouette score
    * Calinski–Harabasz index
    * Davies–Bouldin index
- Saving a CSV with all metrics and (optionally) PNG plots.

Example:
    python santander_k_selection.py \
        --data-path segmentation_20160528.csv \
        --min-k 2 --max-k 10 \
        --sample-frac 0.2 \
        --out-prefix k_selection_20160528
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Choose number of clusters (K) for Santander customer segmentation using KMeans + metrics."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to segmentation_YYYYMMDD.csv (output from santander_segmentation_prep.py).",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Minimum K to try (inclusive). Default: 2.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Maximum K to try (inclusive). Default: 10.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.2,
        help="Fraction of rows to sample for metric computation (0 < f <= 1). Default: 0.2 (20%% of data).",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="If set, do NOT subsample; use all rows (can be slow on large data).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and KMeans. Default: 42.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Prefix for output files. Default: based on input filename.",
    )
    return parser.parse_args()


# Product columns (same list as in the other scripts)
PRODUCT_COLS = [
    "ind_ahor_fin_ult1",   # Saving Account
    "ind_aval_fin_ult1",   # Guarantees
    "ind_cco_fin_ult1",    # Current Accounts
    "ind_cder_fin_ult1",   # Derivada Account
    "ind_cno_fin_ult1",    # Payroll Account
    "ind_ctju_fin_ult1",   # Junior Account
    "ind_ctma_fin_ult1",   # Más particular Account
    "ind_ctop_fin_ult1",   # Particular Account
    "ind_ctpp_fin_ult1",   # Particular Plus Account
    "ind_deco_fin_ult1",   # Short-term Deposits
    "ind_deme_fin_ult1",   # Medium-term Deposits
    "ind_dela_fin_ult1",   # Long-term Deposits
    "ind_ecue_fin_ult1",   # e-Account
    "ind_fond_fin_ult1",   # Funds
    "ind_hip_fin_ult1",    # Mortgage
    "ind_plan_fin_ult1",   # Pensions
    "ind_pres_fin_ult1",   # Loans
    "ind_reca_fin_ult1",   # Taxes
    "ind_tjcr_fin_ult1",   # Credit Card
    "ind_valo_fin_ult1",   # Securities
    "ind_viv_fin_ult1",    # Home Account
    "ind_nomina_ult1",     # Payroll
    "ind_nom_pens_ult1",   # Pensions (nom)
    "ind_recibo_ult1",     # Direct Debit
]


def load_segmentation_data(path: Path) -> pd.DataFrame:
    print(f"Loading segmentation data from: {path}")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build feature matrix X for clustering / metrics.

    - Use all available product columns (0/1).
    - Use numeric features if present: num_products, age, renta.

    Returns:
        X          : DataFrame of features
        prod_cols  : list of product column names used
        num_cols   : list of numeric (non-binary) column names used
    """
    # Products: keep only those that exist
    prod_cols = [c for c in PRODUCT_COLS if c in df.columns]
    missing = [c for c in PRODUCT_COLS if c not in df.columns]
    print(f"Product columns found: {len(prod_cols)}")
    if missing:
        print(f"Warning: {len(missing)} product columns are missing and will be ignored.")

    # Numeric features (optional)
    candidate_num = ["num_products", "age", "renta"]
    num_cols = [c for c in candidate_num if c in df.columns]
    print(f"Numeric features used: {num_cols}")

    feature_cols = prod_cols + num_cols
    if not feature_cols:
        raise ValueError("No feature columns found. Check your input data.")

    X = df[feature_cols].copy()

    # Ensure numeric dtype
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fill missing numeric values with column median
    X = X.fillna(X.median(numeric_only=True))

    return X, prod_cols, num_cols


def maybe_subsample(X: pd.DataFrame, frac: float, random_state: int, no_sample: bool) -> pd.DataFrame:
    if no_sample or frac >= 1.0:
        print("Using full dataset for K selection (no subsampling).")
        return X

    if frac <= 0.0:
        raise ValueError("sample-frac must be in (0, 1]")

    print(f"Subsampling {frac * 100:.1f}% of rows for K selection.")
    X_sample = X.sample(frac=frac, random_state=random_state)
    print("Sampled shape:", X_sample.shape)
    return X_sample


def compute_k_metrics(
    X: pd.DataFrame,
    min_k: int,
    max_k: int,
    random_state: int,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )
    from sklearn.preprocessing import StandardScaler

    # Standardize non-binary numeric features (heuristic: columns not strictly 0/1)
    X_scaled = X.copy()
    binary_like = []
    numeric_like = []

    for col in X.columns:
        unique_vals = X[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            binary_like.append(col)
        else:
            numeric_like.append(col)

    if numeric_like:
        print(f"Standardizing numeric-like columns: {numeric_like}")
        scaler = StandardScaler()
        X_scaled[numeric_like] = scaler.fit_transform(X_scaled[numeric_like])
    else:
        print("No numeric-like columns found to standardize.")

    ks = list(range(min_k, max_k + 1))
    results = []

    for k in ks:
        print(f"\nFitting KMeans for K={k} ...")
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto",
        )
        labels = kmeans.fit_predict(X_scaled)

        inertia = kmeans.inertia_

        # For some edge cases (e.g. too few points), metrics can fail; guard with try/except
        try:
            sil = silhouette_score(X_scaled, labels)
        except Exception as e:
            print(f"  Warning: silhouette_score failed for K={k}: {e}")
            sil = np.nan

        try:
            ch = calinski_harabasz_score(X_scaled, labels)
        except Exception as e:
            print(f"  Warning: calinski_harabasz_score failed for K={k}: {e}")
            ch = np.nan

        try:
            db = davies_bouldin_score(X_scaled, labels)
        except Exception as e:
            print(f"  Warning: davies_bouldin_score failed for K={k}: {e}")
            db = np.nan

        print(f"  Inertia (SSE): {inertia:.2f}")
        print(f"  Silhouette   : {sil:.4f}")
        print(f"  Calinski-Harabasz: {ch:.2f}")
        print(f"  Davies-Bouldin   : {db:.4f}")

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
            }
        )

    metrics_df = pd.DataFrame(results)
    return metrics_df


def save_metrics_and_plots(metrics_df: pd.DataFrame, out_prefix: Path):
    # Save CSV
    csv_path = out_prefix.with_name(out_prefix.name + "_k_metrics.csv")
    print(f"\nSaving K metrics to: {csv_path}")
    metrics_df.to_csv(csv_path, index=False)

    # Try to create plots (optional)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed, skipping plots. "
            "Install with: pip install matplotlib"
        )
        return

    # Inertia plot
    plt.figure()
    plt.plot(metrics_df["k"], metrics_df["inertia"], marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow method – inertia vs K")
    inertia_png = out_prefix.with_name(out_prefix.name + "_inertia.png")
    plt.savefig(inertia_png, bbox_inches="tight")
    plt.close()
    print(f"Saved inertia plot to: {inertia_png}")

    # Silhouette plot
    plt.figure()
    plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Average silhouette score")
    plt.title("Silhouette vs K")
    sil_png = out_prefix.with_name(out_prefix.name + "_silhouette.png")
    plt.savefig(sil_png, bbox_inches="tight")
    plt.close()
    print(f"Saved silhouette plot to: {sil_png}")

    # Optional: Calinski–Harabasz and Davies–Bouldin plot
    plt.figure()
    plt.plot(metrics_df["k"], metrics_df["calinski_harabasz"], marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Calinski-Harabasz index")
    plt.title("Calinski-Harabasz vs K")
    ch_png = out_prefix.with_name(out_prefix.name + "_calinski_harabasz.png")
    plt.savefig(ch_png, bbox_inches="tight")
    plt.close()
    print(f"Saved Calinski-Harabasz plot to: {ch_png}")

    plt.figure()
    plt.plot(metrics_df["k"], metrics_df["davies_bouldin"], marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Davies-Bouldin index")
    plt.title("Davies-Bouldin vs K")
    db_png = out_prefix.with_name(out_prefix.name + "_davies_bouldin.png")
    plt.savefig(db_png, bbox_inches="tight")
    plt.close()
    print(f"Saved Davies-Bouldin plot to: {db_png}")


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    df = load_segmentation_data(data_path)

    X, prod_cols, num_cols = build_feature_matrix(df)

    X_sample = maybe_subsample(
        X,
        frac=args.sample_frac,
        random_state=args.random_state,
        no_sample=args.no_sample,
    )

    metrics_df = compute_k_metrics(
        X_sample,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
    )

    if args.out_prefix is None:
        # Default prefix from data file stem
        out_prefix = data_path.with_name("k_selection_" + data_path.stem)
    else:
        out_prefix = Path(args.out_prefix)

    save_metrics_and_plots(metrics_df, out_prefix)

    print("\nDone. Inspect the CSV and PNGs to choose a good K.")
    print("Tip: look for an 'elbow' in inertia and a reasonably high silhouette.")


if __name__ == "__main__":
    main()
