"""
Santander customer segmentation and product strategy pipeline.

This script filters the large Kaggle Santander Product Recommendation dataset
(train.csv) to one or two months, cleans demographic/product features, clusters
customers, profiles clusters, and generates prompt snippets for LLM personas.

How to run (Python 3.12 assumed; optional virtualenv commands):
    uv venv venv --python 3.12
    . venv/bin/activate
    uv pip install numpy pandas scikit-learn matplotlib seaborn

Then execute:
    python main.py

Expected files in the working directory:
- train.csv (full Kaggle training set)
- test.csv (optional; not required for the main flow)

Outputs are written to ./output/ including cluster assignments, summaries,
plots, and prompt text for ChatGPT or other LLMs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Configuration defaults
# ----------------------------
REFERENCE_MONTH = "2016-05-28"
NEXT_MONTH = "2016-06-28"  # optional; used for adoption analysis if present
TRAIN_PATH = Path("train.csv")
OUTPUT_DIR = Path("output")
CHUNKSIZE = 500_000  # chunk rows when reading the massive CSV
CLUSTER_RANGE = range(3, 9)  # K values to scan
MAX_SILHOUETTE_SAMPLE = 50_000  # sample size for silhouette to stay efficient
RANDOM_STATE = 42
USE_MINI_BATCH = True  # MiniBatchKMeans scales better when many rows exist

# Product columns (24)
PRODUCT_COLUMNS = [
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1",
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1",
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1",
    "ind_nomina_ult1",
    "ind_nom_pens_ult1",
    "ind_recibo_ult1",
]

# Demographic/status columns (24) from the dataset description
PROFILE_COLUMNS = [
    "fecha_dato",
    "ncodpers",
    "ind_empleado",
    "pais_residencia",
    "sexo",
    "age",
    "fecha_alta",
    "ind_nuevo",
    "antiguedad",
    "indrel",
    "ult_fec_cli_1t",
    "indrel_1mes",
    "tiprel_1mes",
    "indresi",
    "indext",
    "conyuemp",
    "canal_entrada",
    "indfall",
    "tipodom",
    "cod_prov",
    "nomprov",
    "ind_actividad_cliente",
    "renta",
    "segmento",
]

ALL_COLUMNS = PROFILE_COLUMNS + PRODUCT_COLUMNS

NUMERIC_FEATURES = [
    "age",
    "antiguedad",
    "renta_filled",
    "log_renta",
    "num_products",
    "deposit_count",
    "loan_credit_count",
    "investment_count",
    "activity_index",
]

CATEGORICAL_FEATURES = ["sexo", "segmento"]


@dataclass
class ClusterMetrics:
    inertia: float
    silhouette: float


# ----------------------------
# Data loading
# ----------------------------

def _dtype_map() -> Dict[str, str]:
    """Return dtype mapping to keep memory reasonable when reading CSV.

    Strings are loaded as pandas string dtype; product columns are float32 to
    allow coercion and later casting to small ints.
    """

    dtypes: Dict[str, str] = {
        "fecha_dato": "string",
        "ncodpers": "int64",
        "ind_empleado": "string",
        "pais_residencia": "string",
        "sexo": "string",
        "age": "string",
        "fecha_alta": "string",
        "ind_nuevo": "float32",
        "antiguedad": "string",
        "indrel": "float32",
        "ult_fec_cli_1t": "string",
        "indrel_1mes": "string",
        "tiprel_1mes": "string",
        "indresi": "string",
        "indext": "string",
        "conyuemp": "float32",
        "canal_entrada": "string",
        "indfall": "string",
        "tipodom": "float32",
        "cod_prov": "float32",
        "nomprov": "string",
        "ind_actividad_cliente": "float32",
        "renta": "string",
        "segmento": "string",
    }
    dtypes.update({col: "float32" for col in PRODUCT_COLUMNS})
    return dtypes


def load_filtered_months(
    csv_path: Path,
    target_months: Iterable[str],
    usecols: Optional[List[str]] = None,
    chunksize: int = CHUNKSIZE,
) -> Dict[str, pd.DataFrame]:
    """Read the huge CSV in chunks and return DataFrames for requested months.

    Only the needed months are kept in memory to stay memory efficient.
    """

    months = list(target_months)
    month_frames: Dict[str, List[pd.DataFrame]] = {m: [] for m in months}

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=_dtype_map(),
        chunksize=chunksize,
        low_memory=False,
    ):
        filtered = chunk[chunk["fecha_dato"].isin(months)]
        if filtered.empty:
            continue
        for month in months:
            part = filtered[filtered["fecha_dato"] == month]
            if not part.empty:
                month_frames[month].append(part)

    return {m: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for m, frames in month_frames.items()}


# ----------------------------
# Cleaning and feature engineering
# ----------------------------

def _clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure product columns are numeric 0/1 and int8."""

    df_products = df.copy()
    for col in PRODUCT_COLUMNS:
        df_products[col] = pd.to_numeric(df_products[col], errors="coerce").fillna(0)
        df_products[col] = df_products[col].clip(lower=0, upper=1).astype("int8")
    return df_products


def _impute_renta(df: pd.DataFrame) -> pd.Series:
    """Impute renta with province median then global median."""

    renta_num = pd.to_numeric(df["renta"], errors="coerce")
    global_median = renta_num.median()
    prov_median = renta_num.groupby(df["cod_prov"]).transform("median")
    filled = renta_num.fillna(prov_median)
    filled = filled.fillna(global_median)
    return filled


def clean_customer_month(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean a month-level customer snapshot.

    - Drops duplicate customer rows within the month.
    - Converts numeric fields and imputes age and renta.
    - Normalises product flags.
    - Adds aggregate product counts and log income.
    """

    if raw_df.empty:
        return raw_df

    df = raw_df.drop_duplicates(subset=["fecha_dato", "ncodpers"], keep="last").copy()

    # Numeric conversions
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].clip(lower=18, upper=100)
    df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["ind_nuevo"] = pd.to_numeric(df["ind_nuevo"], errors="coerce")
    df["ind_actividad_cliente"] = pd.to_numeric(df["ind_actividad_cliente"], errors="coerce")

    df["age"] = df["age"].fillna(df["age"].median())
    df["antiguedad"] = df["antiguedad"].fillna(df["antiguedad"].median())
    df["ind_nuevo"] = df["ind_nuevo"].fillna(0)
    df["ind_actividad_cliente"] = df["ind_actividad_cliente"].fillna(0)

    df["renta_filled"] = _impute_renta(df)
    df["log_renta"] = np.log1p(df["renta_filled"]).replace(-np.inf, np.nan)
    df["log_renta"] = df["log_renta"].fillna(df["log_renta"].median())

    # Clean categorical fields
    for col in [
        "ind_empleado",
        "pais_residencia",
        "sexo",
        "segmento",
        "indrel_1mes",
        "tiprel_1mes",
        "canal_entrada",
    ]:
        df[col] = df[col].fillna("Unknown").astype("category")

    df_products = _clean_products(df[PRODUCT_COLUMNS])
    df[PRODUCT_COLUMNS] = df_products

    # Aggregate product counts
    deposits = ["ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1"]
    loan_credit = ["ind_pres_fin_ult1", "ind_tjcr_fin_ult1", "ind_hip_fin_ult1"]
    investments = ["ind_fond_fin_ult1", "ind_valo_fin_ult1", "ind_plan_fin_ult1"]

    df["num_products"] = df_products.sum(axis=1)
    df["deposit_count"] = df_products[deposits].sum(axis=1)
    df["loan_credit_count"] = df_products[loan_credit].sum(axis=1)
    df["investment_count"] = df_products[investments].sum(axis=1)
    df["activity_index"] = df["ind_actividad_cliente"]

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create the matrix used for clustering and return it with feature names."""

    if df.empty:
        return pd.DataFrame(), []

    categorical = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES, dtype=np.int8)
    feature_df = pd.concat([
        df[PRODUCT_COLUMNS],
        df[NUMERIC_FEATURES],
        categorical,
    ], axis=1)

    # Standardize numeric columns; leave binaries as 0/1.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df[NUMERIC_FEATURES])
    feature_df.loc[:, NUMERIC_FEATURES] = scaled
    feature_names = list(feature_df.columns)
    return feature_df, feature_names


# ----------------------------
# Clustering
# ----------------------------

def _sample_for_eval(feature_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(feature_df) <= sample_size:
        return feature_df
    return feature_df.sample(n=sample_size, random_state=RANDOM_STATE)


def find_best_k(feature_df: pd.DataFrame, cluster_range: Iterable[int]) -> Tuple[int, Dict[int, ClusterMetrics]]:
    """Evaluate K candidates using silhouette on a sample and pick the best."""

    metrics: Dict[int, ClusterMetrics] = {}
    eval_df = _sample_for_eval(feature_df, MAX_SILHOUETTE_SAMPLE)
    X = eval_df.to_numpy()

    for k in cluster_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        inertia = model.inertia_
        sil = float("nan")
        if k > 1 and len(eval_df) >= k:
            sil = silhouette_score(X, labels)
        metrics[k] = ClusterMetrics(inertia=inertia, silhouette=sil)

    # Choose best silhouette; fall back to smallest inertia if silhouettes are nan
    valid_sils = {k: v.silhouette for k, v in metrics.items() if not np.isnan(v.silhouette)}
    if valid_sils:
        best_k = max(valid_sils, key=valid_sils.get)
    else:
        best_k = min(metrics, key=lambda k: metrics[k].inertia)
    return best_k, metrics


def fit_clusters(feature_df: pd.DataFrame, n_clusters: int) -> Tuple[np.ndarray, object]:
    """Fit clustering model and return labels and model."""

    X = feature_df.to_numpy()
    if USE_MINI_BATCH:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            batch_size=10_000,
            n_init=10,
        )
    else:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    return labels, model


# ----------------------------
# Analysis and visualization
# ----------------------------

def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Return aggregated statistics per cluster."""

    profiled = df.copy()
    profiled["cluster"] = labels

    stats = {
        "cluster": [],
        "size": [],
        "avg_age": [],
        "median_renta": [],
        "avg_products": [],
        "top_segment": [],
    }

    for cluster_id, group in profiled.groupby("cluster"):
        stats["cluster"].append(cluster_id)
        stats["size"].append(len(group))
        stats["avg_age"].append(group["age"].mean())
        stats["median_renta"].append(group["renta_filled"].median())
        stats["avg_products"].append(group["num_products"].mean())
        stats["top_segment"].append(group["segmento"].mode().iloc[0] if not group["segmento"].empty else "Unknown")

    return pd.DataFrame(stats)


def cluster_product_means(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    profiled = df.copy()
    profiled["cluster"] = labels
    return profiled.groupby("cluster")[PRODUCT_COLUMNS].mean().reset_index()


def compute_adoption(
    df_t: pd.DataFrame,
    df_t1: pd.DataFrame,
    labels_t: np.ndarray,
) -> Optional[pd.DataFrame]:
    """Compute product adoption rates per cluster using next-month data."""

    if df_t.empty or df_t1.empty:
        return None

    curr = df_t[["ncodpers"] + PRODUCT_COLUMNS].copy()
    nxt = df_t1[["ncodpers"] + PRODUCT_COLUMNS].copy()
    merged = curr.merge(nxt, on="ncodpers", suffixes=("_t", "_t1"))
    cluster_map = dict(zip(df_t["ncodpers"], labels_t))
    merged["cluster"] = merged["ncodpers"].map(cluster_map)
    merged = merged.dropna(subset=["cluster"])
    merged["cluster"] = merged["cluster"].astype(int)

    adoption_rates = {}
    grouped = merged.groupby("cluster")
    for col in PRODUCT_COLUMNS:
        curr_col = f"{col}_t"
        next_col = f"{col}_t1"
        adoption_rates[col] = grouped.apply(lambda g: (g[next_col] > g[curr_col]).mean())

    return pd.DataFrame(adoption_rates).reset_index()


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_product_rates(product_means: pd.DataFrame) -> Path:
    """Plot ownership rates for top products by overall prevalence."""

    overall = product_means.drop(columns=["cluster"]).mean().sort_values(ascending=False)
    top_products = list(overall.head(10).index)

    melted = product_means.melt(id_vars="cluster", value_vars=top_products, var_name="product", value_name="rate")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="product", y="rate", hue="cluster")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ownership rate")
    plt.title("Top product ownership rates by cluster")
    plt.tight_layout()
    path = OUTPUT_DIR / "cluster_product_rates.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_age_distribution(df: pd.DataFrame, labels: np.ndarray) -> Path:
    data = df[["age"]].copy()
    data["cluster"] = labels

    plt.figure(figsize=(10, 6))
    for cluster_id, group in data.groupby("cluster"):
        sns.kdeplot(group["age"], label=f"Cluster {cluster_id}", fill=True, alpha=0.25)
    plt.xlabel("Age")
    plt.title("Age distribution by cluster")
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "cluster_age_distribution.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_pca(feature_df: pd.DataFrame, labels: np.ndarray) -> Path:
    """2D PCA scatter (sampled) colored by cluster."""

    sample = _sample_for_eval(feature_df, 50_000)
    sample_labels = labels[sample.index]
    X = sample.to_numpy()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=sample_labels, cmap="tab10", s=5, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Customer clusters (PCA view)")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    path = OUTPUT_DIR / "cluster_pca.png"
    plt.savefig(path)
    plt.close()
    return path


# ----------------------------
# Generative AI prompt helper
# ----------------------------

def build_cluster_prompt_file(
    df: pd.DataFrame,
    labels: np.ndarray,
    product_rates: pd.DataFrame,
    adoption_rates: Optional[pd.DataFrame] = None,
    filepath: Path = OUTPUT_DIR / "cluster_prompts_for_gpt.txt",
) -> Path:
    """Write cluster summaries and prompt templates for LLM persona generation."""

    profiled = df.copy()
    profiled["cluster"] = labels

    lines: List[str] = []
    for cluster_id, group in profiled.groupby("cluster"):
        prod_mean = product_rates[product_rates["cluster"] == cluster_id].drop(columns=["cluster"])
        top_products = prod_mean.T.squeeze().sort_values(ascending=False).head(5)

        adoption_summary: List[str] = []
        if adoption_rates is not None:
            row = adoption_rates[adoption_rates["cluster"] == cluster_id]
            if not row.empty:
                top_adopt = row.drop(columns=["cluster"]).T.squeeze().sort_values(ascending=False).head(3)
                adoption_summary = [
                    f"Top adoption next month: {', '.join(f'{p} ({rate:.2%})' for p, rate in top_adopt.items())}"
                ]

        lines.append(f"===== CLUSTER {cluster_id} =====")
        lines.append(f"Size: {len(group):,} customers")
        lines.append(f"Avg age: {group['age'].mean():.1f} | Median income: {group['renta_filled'].median():.0f}")
        lines.append(f"Avg products: {group['num_products'].mean():.2f}")
        lines.append(f"Top segment: {group['segmento'].mode().iloc[0] if not group['segmento'].empty else 'Unknown'}")
        lines.append("Top owned products: " + ", ".join(f"{p} ({rate:.1%})" for p, rate in top_products.items()))
        if adoption_summary:
            lines.extend(adoption_summary)
        lines.append("")
        lines.append("PROMPT FOR CHATGPT:")
        lines.append(
            "You are a marketing strategist for a retail bank. Based on the cluster description, "
            "write a short persona (2-3 sentences) and propose 2-3 priority banking products to cross-sell, "
            "with a brief rationale. Use concise business language."
        )
        lines.append("Cluster description (bullet points):")
        lines.append(f"- Cluster ID: {cluster_id}")
        lines.append(f"- Size: {len(group):,} customers")
        lines.append(f"- Avg age: {group['age'].mean():.1f}; median renta: {group['renta_filled'].median():.0f}")
        lines.append(f"- Avg number of products: {group['num_products'].mean():.2f}")
        lines.append(f"- Most common segment: {group['segmento'].mode().iloc[0] if not group['segmento'].empty else 'Unknown'}")
        lines.append("- Top owned products: " + ", ".join(f"{p} ({rate:.1%})" for p, rate in top_products.items()))
        if adoption_summary:
            lines.extend("- " + line for line in adoption_summary)
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


# Optional, lightweight skeleton for calling an LLM API directly.
def call_llm_stub(prompt: str, api_key: Optional[str] = None) -> str:
    """Minimal example of invoking an OpenAI-compatible chat completion API.

    Not used by default; safe to ignore. Provide api_key and install the
    `openai` package to activate, or adapt to another client.
    """

    if api_key is None:
        return "LLM call skipped (no API key provided)."
    try:
        import openai  # type: ignore
    except ImportError:
        return "Install the `openai` package to enable direct LLM calls."

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]


# ----------------------------
# Main pipeline
# ----------------------------

def run_pipeline() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError("train.csv not found in the working directory")

    _ensure_output_dir()

    target_months = [REFERENCE_MONTH, NEXT_MONTH]
    print("Loading data for target months...")
    month_dfs = load_filtered_months(TRAIN_PATH, target_months, usecols=ALL_COLUMNS)

    df_ref = month_dfs.get(REFERENCE_MONTH, pd.DataFrame())
    df_next = month_dfs.get(NEXT_MONTH, pd.DataFrame())

    if df_ref.empty:
        raise RuntimeError(f"No data found for reference month {REFERENCE_MONTH}")

    print(f"Cleaning reference month {REFERENCE_MONTH}...")
    df_ref_clean = clean_customer_month(df_ref)

    df_next_clean = pd.DataFrame()
    if not df_next.empty:
        print(f"Cleaning next month {NEXT_MONTH} for adoption analysis...")
        df_next_clean = clean_customer_month(df_next)

    print("Building feature matrix...")
    feature_df, feature_names = build_feature_matrix(df_ref_clean)
    if feature_df.empty:
        raise RuntimeError("Feature matrix is empty")

    print("Selecting cluster count...")
    best_k, metrics = find_best_k(feature_df, CLUSTER_RANGE)
    print(f"Selected K={best_k} based on silhouette/inertia")

    print("Fitting clustering model...")
    labels, model = fit_clusters(feature_df, best_k)

    assignments = df_ref_clean[["ncodpers", "fecha_dato"]].copy()
    assignments["cluster"] = labels
    assignments_path = OUTPUT_DIR / "customer_clusters.csv"
    assignments.to_csv(assignments_path, index=False)

    print("Computing cluster profiles...")
    profile_df = cluster_profiles(df_ref_clean, labels)
    profile_path = OUTPUT_DIR / "cluster_profiles.csv"
    profile_df.to_csv(profile_path, index=False)

    product_means = cluster_product_means(df_ref_clean, labels)

    adoption_rates = None
    if not df_next_clean.empty:
        print("Computing adoption rates using next month...")
        adoption_rates = compute_adoption(df_ref_clean, df_next_clean, labels)
        if adoption_rates is not None:
            adoption_rates.to_csv(OUTPUT_DIR / "cluster_adoption_rates.csv", index=False)

    print("Generating plots...")
    plot_product_rates(product_means)
    plot_age_distribution(df_ref_clean, labels)
    plot_pca(feature_df, labels)

    print("Writing LLM prompt file...")
    build_cluster_prompt_file(df_ref_clean, labels, product_means, adoption_rates)

    # Optional: save metrics summary
    metrics_path = OUTPUT_DIR / "cluster_metrics.csv"
    pd.DataFrame([
        {"k": k, "inertia": m.inertia, "silhouette": m.silhouette}
        for k, m in metrics.items()
    ]).to_csv(metrics_path, index=False)

    print("Pipeline complete. Outputs saved to ./output")


if __name__ == "__main__":
    run_pipeline()
