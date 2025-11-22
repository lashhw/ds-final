#!/usr/bin/env python
"""
Santander Customer Segmentation – Cluster Analysis Script

This script:
- Loads a segmentation CSV with a 'cluster' column.
- Computes:
    * Cluster sizes (count and percentage)
    * Global product ownership rates
    * Cluster-level product ownership rates
    * Per-cluster "most distinctive" products vs global average
    * Demographic stats (age, renta, num_products)
    * Category distributions (segmento, sexo, pais_residencia if available)
- Builds a concise summary table per cluster and saves it.

Usage:
    python santander_cluster_analysis.py \
        --data-path segmentation_20160528_clusters.csv \
        --out-prefix cluster_analysis_20160528

Outputs:
    <out-prefix>_cluster_sizes.csv
    <out-prefix>_cluster_summary.csv
    <out-prefix>_cluster_product_means.csv
    <out-prefix>_cluster_product_diff_vs_global.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Santander customer clusters from a segmentation CSV."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to segmentation_YYYYMMDD_clusters.csv (must contain a 'cluster' column).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="cluster_analysis",
        help="Prefix for output CSV files (default: cluster_analysis).",
    )
    parser.add_argument(
        "--cluster-col",
        type=str,
        default="cluster",
        help="Name of the cluster column (default: 'cluster').",
    )
    return parser.parse_args()


# Product columns (same list as in the prep script)
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


def load_data(path: Path, cluster_col: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)

    if cluster_col not in df.columns:
        raise ValueError(
            f"Cluster column '{cluster_col}' not found in file. "
            "Make sure you used the prep script that adds KMeans clusters."
        )

    # Ensure cluster is treated as a category for groupby readability
    df[cluster_col] = df[cluster_col].astype("int64")
    print("Data shape:", df.shape)
    print("Clusters found:", sorted(df[cluster_col].unique()))
    return df


def get_available_product_cols(df: pd.DataFrame):
    # Only keep product columns that actually exist in the dataframe
    available = [c for c in PRODUCT_COLS if c in df.columns]
    missing = [c for c in PRODUCT_COLS if c not in df.columns]
    print(f"\nProduct columns found: {len(available)}")
    if missing:
        print(f"Warning: {len(missing)} product columns not found and will be ignored:")
        print(", ".join(missing))
    return available


def analyze_cluster_sizes(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    print("\n=== Cluster sizes ===")
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    cluster_sizes_pct = (cluster_sizes / len(df) * 100).round(2)

    sizes_df = pd.DataFrame(
        {"cluster": cluster_sizes.index, "count": cluster_sizes.values, "percent": cluster_sizes_pct.values}
    ).reset_index(drop=True)

    print(sizes_df)
    return sizes_df


def analyze_products(df: pd.DataFrame, cluster_col: str, product_cols: list):
    print("\n=== Global product ownership rate ===")
    global_mean = df[product_cols].mean().sort_values(ascending=False)
    print(global_mean.head(10))

    cluster_mean = df.groupby(cluster_col)[product_cols].mean()
    print("\n=== Cluster-level product ownership (first few products) ===")
    print(cluster_mean[global_mean.index[:8]])

    # Difference vs global mean
    diff_from_global = cluster_mean.subtract(global_mean, axis=1)

    print("\n=== Most distinctive products per cluster (above global avg) ===")
    for c in sorted(df[cluster_col].unique()):
        print(f"\nCluster {c}:")
        top_diff = diff_from_global.loc[c].sort_values(ascending=False).head(5)
        print(top_diff)

    return global_mean, cluster_mean, diff_from_global


def analyze_demographics(df: pd.DataFrame, cluster_col: str):
    # Numeric demo columns we expect to exist (if not, they will be skipped safely)
    numeric_cols = [col for col in ["age", "renta", "num_products"] if col in df.columns]
    if numeric_cols:
        print("\n=== Demographic stats (numeric) per cluster ===")
        demo_stats = df.groupby(cluster_col)[numeric_cols].agg(["mean", "median", "min", "max"])
        print(demo_stats)
    else:
        demo_stats = pd.DataFrame()
        print("\nNo numeric demographic columns (age/renta/num_products) found for analysis.")

    # Categorical columns (optional)
    cat_cols = [c for c in ["segmento", "sexo", "pais_residencia"] if c in df.columns]
    cat_tables = {}

    for col in cat_cols:
        print(f"\n=== {col} distribution per cluster (row-normalized) ===")
        ct = pd.crosstab(df[cluster_col], df[col], normalize="index").round(2)
        print(ct)
        cat_tables[col] = ct

    return demo_stats, cat_tables


def build_summary_table(
    df: pd.DataFrame,
    cluster_col: str,
    product_cols: list,
    cluster_sizes_df: pd.DataFrame,
    cluster_mean: pd.DataFrame,
) -> pd.DataFrame:
    """Build a concise summary table per cluster."""

    base = df.groupby(cluster_col).agg(
        n_customers=("ncodpers", "count") if "ncodpers" in df.columns else (cluster_col, "count"),
        avg_age=("age", "mean") if "age" in df.columns else (cluster_col, "size"),
        avg_renta=("renta", "mean") if "renta" in df.columns else (cluster_col, "size"),
        avg_num_products=("num_products", "mean") if "num_products" in df.columns else (cluster_col, "size"),
    )

    # Fix column names in case fallback aggregations were used
    base = base.rename(
        columns={
            cluster_col: "n_customers",
        }
    )

    # Add cluster percentage from cluster_sizes_df
    sizes_map = cluster_sizes_df.set_index("cluster")["percent"]
    base["percent_of_total"] = sizes_map

    # Add top-3 products per cluster
    top_products_per_cluster = {}
    for c in base.index:
        prod_rates = cluster_mean.loc[c, product_cols]
        top3 = prod_rates.sort_values(ascending=False).head(3).index.tolist()
        top_products_per_cluster[c] = ", ".join(top3)

    base["top_products"] = base.index.map(top_products_per_cluster.get)

    # Optional: add an empty "label" column for manual naming in Excel
    base["segment_label"] = ""  # you can fill this manually later

    # Round numeric columns for neat printing
    round_cols = [col for col in ["avg_age", "avg_renta", "avg_num_products", "percent_of_total"] if col in base.columns]
    base[round_cols] = base[round_cols].round(2)

    base = base.reset_index().rename(columns={cluster_col: "cluster"})
    print("\n=== Cluster summary table ===")
    print(base)

    return base


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    out_prefix = args.out_prefix
    cluster_col = args.cluster_col

    df = load_data(data_path, cluster_col=cluster_col)

    product_cols = get_available_product_cols(df)

    # 1. Cluster sizes
    cluster_sizes_df = analyze_cluster_sizes(df, cluster_col=cluster_col)

    # 2. Product analysis
    global_mean, cluster_mean, diff_from_global = analyze_products(
        df, cluster_col=cluster_col, product_cols=product_cols
    )

    # 3. Demographics & categories
    demo_stats, cat_tables = analyze_demographics(df, cluster_col=cluster_col)

    # 4. Summary table for report
    summary_df = build_summary_table(
        df,
        cluster_col=cluster_col,
        product_cols=product_cols,
        cluster_sizes_df=cluster_sizes_df,
        cluster_mean=cluster_mean,
    )

    # === Save outputs ===
    out_prefix = Path(out_prefix)

    sizes_path = out_prefix.with_name(out_prefix.name + "_cluster_sizes.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_cluster_summary.csv")
    prod_means_path = out_prefix.with_name(out_prefix.name + "_cluster_product_means.csv")
    prod_diff_path = out_prefix.with_name(out_prefix.name + "_cluster_product_diff_vs_global.csv")

    print(f"\nSaving cluster sizes to: {sizes_path}")
    cluster_sizes_df.to_csv(sizes_path, index=False)

    print(f"Saving cluster summary to: {summary_path}")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saving cluster product means to: {prod_means_path}")
    cluster_mean.to_csv(prod_means_path)

    print(f"Saving product diff vs global to: {prod_diff_path}")
    diff_from_global.to_csv(prod_diff_path)

    print("\nDone. You can now open the CSVs (especially *_cluster_summary.csv) for your report.")


if __name__ == "__main__":
    main()
