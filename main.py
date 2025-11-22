#!/usr/bin/env python
"""
Santander Customer Segmentation – Data Preparation Script

Usage (basic):
    python santander_segmentation_prep.py \
        --data-path train_ver2.csv \
        --month 2016-05-28

Options:
    --chunksize N       Read CSV in chunks of N rows (default: 200000).
    --no-chunks         Disable chunked reading; load all at once.
    --n-clusters K      Number of KMeans clusters for quick test (default: 4).

Outputs:
    segmentation_<YYYYMMDD>.csv
    segmentation_<YYYYMMDD>.parquet
    segmentation_<YYYYMMDD>_clusters.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Santander Kaggle dataset for customer segmentation."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to train_ver2.csv",
    )
    parser.add_argument(
        "--month",
        type=str,
        default="2016-05-28",
        help="Reference month (fecha_dato) to filter on, e.g. 2016-05-28",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for reading CSV. Ignored if --no-chunks is set.",
    )
    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="If set, load the CSV in one go instead of chunked.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of KMeans clusters for quick test.",
    )
    return parser.parse_args()


# Columns you care about
ID_COLS = ["ncodpers", "fecha_dato"]

DEMOGRAPHIC_COLS = [
    "age",
    "sexo",
    "renta",
    "segmento",
    "pais_residencia",
    "indrel",
    "ind_nuevo",
    "antiguedad",
    "ind_actividad_cliente",
]

# Product columns from the Santander competition
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

USECOLS = ID_COLS + DEMOGRAPHIC_COLS + PRODUCT_COLS


def load_filtered_month(path: Path, month: str, use_chunks: bool, chunksize: int) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    print(f"Filtering to month: {month}")
    print(f"Columns used: {len(USECOLS)}")

    if not use_chunks:
        # Simple load – OK if you have enough RAM
        df = pd.read_csv(path, usecols=USECOLS, dtype=str)
        df = df[df["fecha_dato"] == month].copy()
        print(f"Shape after filtering: {df.shape}")
        return df

    # Chunked loading – safer on low memory
    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(path, usecols=USECOLS, dtype=str, chunksize=chunksize)
    ):
        subset = chunk[chunk["fecha_dato"] == month]
        if len(subset) > 0:
            chunks.append(subset)
        print(f"Processed chunk {i}, kept {len(subset)} rows")

    if not chunks:
        raise ValueError(f"No rows found for month {month}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Final shape for {month}: {df.shape}")
    return df


def clean_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Drop fecha_dato, we only have one month
    df = df.drop(columns=["fecha_dato"])

    # Deduplicate customers (keep first if repeated in this month)
    before = df.shape[0]
    df = df.drop_duplicates(subset=["ncodpers"])
    after = df.shape[0]
    print(f"Deduplicated customers: {before} -> {after}")

    # --- Numeric cleaning ---

    # Age
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    # Cap insane ages
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = pd.NA
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median).astype("float32")

    # Antigüedad (seniority in months)
    df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
    ant_median = df["antiguedad"].median()
    df["antiguedad"] = df["antiguedad"].fillna(ant_median).astype("float32")

    # Renta (income)
    df["renta"] = pd.to_numeric(df["renta"], errors="coerce")
    renta_median = df["renta"].median()
    df["renta"] = df["renta"].fillna(renta_median).astype("float32")

    # --- Categorical cleaning ---
    for col in ["sexo", "segmento", "pais_residencia", "indrel", "ind_nuevo", "ind_actividad_cliente"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype("category")

    # --- Product columns: ensure integer 0/1 ---
    for col in PRODUCT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = df[col].clip(0, 1).astype("int8")

    # Number of products owned
    df["num_products"] = df[PRODUCT_COLS].sum(axis=1).astype("int16")

    # Keep a compact segmentation dataset
    seg_cols = ["ncodpers"] + DEMOGRAPHIC_COLS + PRODUCT_COLS + ["num_products"]
    df_seg = df[seg_cols].copy()

    print("Finished cleaning & feature engineering.")
    print("Segmentation dataset shape:", df_seg.shape)
    return df_seg


def quick_eda(df_seg: pd.DataFrame):
    print("\n=== Quick EDA ===")
    print("\nNumber of products per customer:")
    print(df_seg["num_products"].describe())
    print("\nTop counts for num_products:")
    print(df_seg["num_products"].value_counts().head(10))

    print("\nAge distribution:")
    print(df_seg["age"].describe())

    print("\nIncome (renta) distribution:")
    print(df_seg["renta"].describe())

    print("\nMost common products (ownership rate):")
    product_popularity = df_seg[PRODUCT_COLS].mean().sort_values(ascending=False)
    print(product_popularity.head(10))


def basic_kmeans_test(df_seg: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print(
            "\n[Warning] scikit-learn is not installed. "
            "Skipping KMeans test. Install with: pip install scikit-learn"
        )
        return df_seg

    print(f"\n=== Running basic KMeans with K={n_clusters} ===")

    feature_cols = PRODUCT_COLS + ["num_products", "age", "renta"]
    X = df_seg[feature_cols].copy()

    # Standardize only numeric columns; product columns are already 0/1
    numeric_cols = ["num_products", "age", "renta"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df_seg = df_seg.copy()
    df_seg["cluster"] = kmeans.fit_predict(X)

    print("\nCluster sizes:")
    print(df_seg["cluster"].value_counts().sort_index())

    # Basic cluster-level product summary
    print("\nCluster-level average number of products:")
    print(df_seg.groupby("cluster")["num_products"].mean())

    print("\nCluster-level ownership rate for the 5 most common products:")
    # pick the global top 5 products, then show cluster means
    top5_products = df_seg[PRODUCT_COLS].mean().sort_values(ascending=False).head(5).index
    cluster_product_means = df_seg.groupby("cluster")[top5_products].mean()
    print(cluster_product_means)

    return df_seg


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    month = args.month
    use_chunks = not args.no_chunks
    chunksize = args.chunksize
    n_clusters = args.n_clusters

    df_raw = load_filtered_month(
        path=data_path,
        month=month,
        use_chunks=use_chunks,
        chunksize=chunksize,
    )

    df_seg = clean_and_engineer(df_raw)

    # Save clean segmentation dataset
    date_tag = month.replace("-", "")
    csv_out = f"segmentation_{date_tag}.csv"
    parquet_out = f"segmentation_{date_tag}.parquet"

    print(f"\nSaving cleaned segmentation dataset to: {csv_out}")
    df_seg.to_csv(csv_out, index=False)

    try:
        import pyarrow  # noqa: F401
        print(f"Saving Parquet to: {parquet_out}")
        df_seg.to_parquet(parquet_out, index=False)
    except ImportError:
        print(
            "\n[Warning] pyarrow is not installed, skipping Parquet export. "
            "Install with: pip install pyarrow"
        )

    # Quick EDA
    quick_eda(df_seg)

    # Basic KMeans test
    df_seg_clusters = basic_kmeans_test(df_seg, n_clusters=n_clusters)

    # Save clusters if KMeans ran
    if "cluster" in df_seg_clusters.columns:
        clusters_out = f"segmentation_{date_tag}_clusters.csv"
        print(f"\nSaving segmentation with clusters to: {clusters_out}")
        df_seg_clusters.to_csv(clusters_out, index=False)

    print("\nAll done.")


if __name__ == "__main__":
    main()
