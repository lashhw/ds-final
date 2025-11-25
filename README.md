# Santander Customer Segmentation

Pipeline for customer segmentation and lightweight marketing prompt generation using the Kaggle Santander Product Recommendation dataset. The code filters the large `train.csv` by month, cleans demographic/product features, clusters customers, profiles segments, plots summaries, and writes ready-to-paste LLM prompts for each cluster.

## Setup

Use Python 3.12. Example virtualenv with uv:
```bash
uv venv venv --python 3.12
. venv/bin/activate
uv pip install numpy pandas scikit-learn matplotlib seaborn
```
Place `train.csv` (and optionally `test.csv`) in the repository root alongside `main.py`.

## Run

```bash
python main.py
```

The script reads `train.csv` in chunks, keeps only the target months (defaults: 2016-05-28 for clustering and 2016-06-28 for optional adoption analysis), builds features, selects K via silhouette/inertia, fits K-Means/MiniBatchKMeans, and writes outputs to `output/`.

## Outputs (in `output/`)

- `customer_clusters.csv` — `ncodpers`, `fecha_dato`, and cluster label.
- `cluster_profiles.csv` — per-cluster size/age/income/product counts.
- `cluster_metrics.csv` — silhouette/inertia per tested K.
- `cluster_product_rates.png` — bar plot of top product ownership by cluster.
- `cluster_age_distribution.png` — age KDEs by cluster.
- `cluster_pca.png` — 2D PCA scatter colored by cluster.
- `cluster_prompts_for_gpt.txt` — cluster summaries + prompt snippets for LLM persona/offer generation.
- `cluster_adoption_rates.csv` — (if next month present) product adoption rates by cluster.
- `cluster_cross_sell_opportunities.csv` — top cross-sell candidates per cluster with ownership gaps/adoption lift.
- `cluster_churn_rates.csv` — simple churn/inactivity proxy per cluster using next month activity flags.
- `strategy_prompts_for_gpt.txt` — strategy-focused LLM prompts that blend size, churn, and cross-sell opportunity data.
- `customer_samples_for_gpt.txt` — optional small customer-level snippets per cluster for example-based reasoning.

## Configuration

Tune constants near the top of `main.py`:
- `REFERENCE_MONTH`, `NEXT_MONTH` — which months to load.
- `CLUSTER_RANGE` — Ks to evaluate.
- `USE_MINI_BATCH` — toggle MiniBatchKMeans.
- `CHUNKSIZE`, `OUTPUT_DIR`, `PRODUCT_COLUMNS` — advanced tweaks.

## Optional LLM call

`call_llm_stub` shows how to invoke an OpenAI-compatible API. It is unused by default; pass an API key and install `openai` if you want to experiment. The pipeline works without any LLM dependency.

## Notes

- `train.csv` is ~2.1GB; chunked loading and dtype hints keep memory manageable.
- `test.csv` is not required; the main flow focuses on segmentation, not the Kaggle MAP@7 task.
