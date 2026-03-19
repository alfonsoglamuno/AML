# AML Financial Features Baseline

Anti-Money Laundering (AML) detection on the **IBM HI-Medium** synthetic transaction dataset. This repository implements and evaluates a tabular XGBoost baseline (Model A) built exclusively from account-level financial features, and prepares the ground for a comparison with Graph Neural Networks (GNNs).

---

## Study case

**Dataset:** IBM AML HI-Medium — a synthetic dataset of ~31 million financial transactions labelled as legitimate or money laundering (~0.1% positive rate). The generator and datasets are described in:

> Altman, E., Blanusa, J., von Niederhaüsern, L., Egressy, B., Anghel, A., & Atasu, K. (2023).
> *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.*
> NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.16424.

**Task:** Binary classification at the transaction level. The operational objective is to rank transactions by fraud risk so that a compliance analyst reviewing the top-K alerts captures the highest possible fraction of real laundering cases (Recall@K / PR-AUC).

**Challenge:** Extreme class imbalance (~0.1% fraud), temporal distribution shift, and the structural limitation that tabular models cannot propagate fraud signals across the transaction network.

---

## Model A — Financial Features

Model A is a pure tabular approach. No graph structure is used. All features are derived from individual account behaviour aggregated over the training period.

### Feature groups (43 total)

| Group | Count | Description |
|-------|-------|-------------|
| Transaction-level | 13 | Amounts (raw + log), cross-border flag, same-bank flag, self-loop, hour, weekday, weekend, night |
| Sender aggregates | 17 | Volume, frequency, structuring flags, fan-out ratio, destination concentration (HHI), CV |
| Receiver aggregates | 8 | Volume, frequency, fan-in ratio, CV |
| Categorical | 5 | From Bank, To Bank, Payment Format, Receiving/Payment Currency |

Key engineered features:
- **`pct_structured_sent`** — fraction of transactions just below reporting thresholds (structuring / smurfing typology)
- **`concentration_dest`** — Herfindahl-Hirschman Index of destination accounts (layering detection)
- **`fanout_ratio`** — unique counterparties / total transactions (money mule detection)
- **`pct_cross_border_sent`** — fraction of cross-border outgoing transactions

### Model details

| Setting | Value |
|---------|-------|
| Algorithm | XGBoost (histogram-based, categorical support) |
| Hyperparameter selection | Optuna — 30 trials, PR-AUC on validation |
| Class rebalancing | `scale_pos_weight` = sqrt(neg/pos) ≈ 34 |
| Primary evaluation metric | PR-AUC (validation set) |
| Calibration | Isotonic regression (fit on val, applied to test) |
| Temporal split | 60 / 20 / 20 (strict chronological, no leakage) |

### Results (test set)

| Metric | Value | Notes |
|--------|-------|-------|
| PR-AUC | **0.1504** | Primary metric — ROC-AUC is misleading at 0.1% imbalance |
| ROC-AUC | **0.9716** | Reported for completeness |
| Precision @ 100 alerts | ~82% | High-confidence top alerts |
| Recall @ 1,000 alerts | ~7.8% | |
| Recall @ 5,000 alerts | ~18.9% | |
| Payment Format ablation | ~-40% PR-AUC | Single most impactful feature |

### Key findings

**1. Payment Format dominates — but not through a single mechanism.**
ACH (Automated Clearing House) has a fraud rate of ~6.28% vs the dataset average of ~0.088% — approximately 71x higher. This is not just a statistical outlier; ACH transactions also cluster with cross-border activity and structuring patterns, creating a compound risk signal.

**2. Self-loop and same-bank transactions are the strongest individual signals.**
Contrary to expectation, `self_loop` and `same_bank` rank above `Payment Format` in XGBoost gain-based importance. Circular transactions (sender = receiver) are the clearest synthetic marker of round-tripping laundering patterns.

**3. The model partially memorises account identities.**
PR-AUC drops 2× for accounts unseen in training (0.160 seen vs ~0.075 unseen). The large train-val gap in the learning curve (0.28–0.40 PR-AUC) confirms this. Adding more training data helps modestly but does not close the gap — the structural problem remains.

**4. Precision-anchored risk tiers make operational sense.**
Risk thresholds are calibrated so each tier guarantees a minimum precision level:

| Tier | Min. precision | Interpretation |
|------|---------------|----------------|
| Critical | ≥ 15% | ~1 in 7 alerts is genuine fraud — immediate investigation |
| High | ≥ 5% | ~1 in 20 — priority review queue |
| Medium-High | ≥ 2% | ~1 in 50 — standard investigation queue |
| Medium | ≥ 0.5% | ~6× baseline — secondary screening |
| Low | ≥ 0.2% | ~2× baseline — automated flag |
| Very Low | ≥ 0.1% | Just above random — watchlist only |

---

## Repository structure

```
.
├── aml_model_a.ipynb          # Main notebook — Model A end-to-end
├── README.md
└── data/                      # Not tracked — place dataset here
    └── raw/
        ├── HI-Medium_Trans.csv        # Raw IBM dataset
        ├── HI-Medium_days_parquet/    # Day-partitioned cache (auto-generated)
        └── outputs/                   # Model artifacts (auto-generated)
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install xgboost scikit-learn pandas numpy matplotlib seaborn shap optuna

# 2. Place the raw CSV at data/raw/HI-Medium_Trans.csv
#    OR set an environment variable pointing to your data/raw directory:
export AML_DATA_ROOT=/path/to/your/data/raw

# 3. Run the notebook
jupyter notebook aml_model_a.ipynb
```

> The first run partitions the CSV into daily Parquet files (~5 min). Subsequent runs reuse the cache (`REBUILD_PARQUET = False`).
> The first training run takes ~40 min. Subsequent runs reuse the saved model (`FORCE_RETRAIN = False`).

---

## Notebook walkthrough

| Section | Description |
|---------|-------------|
| 0–1 | Setup and path resolution (no hardcoded paths — uses `AML_DATA_ROOT` env var or auto-search) |
| 2 | CSV to daily Parquet partitioning |
| 3 | Temporal split (60/20/20) with anomalous-day filtering |
| 4 | EDA: class balance, daily volume, amount distributions, fraud rate by payment format, hour, flags |
| 4a | Payment Format deep-dive: volume vs fraud count, fraud rate multipliers, cross-border correlation |
| 5 | Feature engineering: sender/receiver account aggregates |
| 5a | Feature distributions: fraud vs legitimate (structuring, HHI, fan-out, cross-border) |
| 5b | Correlation matrix: Pearson correlations with fraud label and inter-feature heatmap |
| 5c | Feature matrix construction |
| 6 | Evaluation utilities (PR-AUC, Recall@K, workload table) |
| 7 | Hyperparameter selection: Optuna search space, final values, rationale |
| 8 | XGBoost training with sqrt-SPW class rebalancing (cached) |
| 9 | Full evaluation: metrics, calibration, 6-panel plot, workload curve, memorization diagnostic, learning curve |
| 10 | Payment Format ablation (~-40% PR-AUC) |
| 11 | SHAP explanations (global importance, beeswarm, waterfall, laundering vs legitimate) |
| 12 | Precision-anchored risk tiers (Critical / High / Medium-High / Medium / Low / Very Low) |
| 13 | Artifact export (model JSON, calibrator, scored CSV, summary JSON) |
| 14 | GNN motivation: graph structure, multi-hop laundering patterns, evidence from memorization |
| 15 | Summary and key findings |

---

## Next step: GNN comparison

### Why GNNs?

Model A treats every transaction as an independent event. Account aggregates capture individual behaviour over the training window, but they are blind to **relationships between accounts**. A GNN treats the transaction history as a directed graph and propagates fraud signal along edges — an account 3 hops away from a known launderer receives an elevated score even if its own history looks clean.

### What questions can the GNN comparison answer?

| Research question | How Model A fails | What GNN addresses |
|-------------------|-------------------|-------------------|
| Does graph neighbourhood predict fraud? | No multi-hop signal | k-hop message passing propagates fraud labels |
| Can we score accounts unseen in training? | 2× PR-AUC drop for unseen accounts | Scores via connected known fraudsters |
| Do layering chains (A→B→C→D) leave detectable traces? | Only 1-hop aggregates (fan-out) | Graph paths are traversed explicitly |
| Does community membership matter? | No community signal | GNN learns community embeddings implicitly |
| Is temporal order within chains informative? | Aggregates only | Sequential GNN variants (EvolveGCN, TGNN) |

### Where is the improvement expected to be largest?

Based on the memorization diagnostic in Model A:
- **Unseen accounts**: PR-AUC 0.160 (seen) → 0.075 (unseen). GNN should close most of this gap.
- **Layering patterns**: Multi-hop chains connecting placement → layering → integration nodes.
- **Shell company detection**: Nodes with clean individual history but centrally connected to fraud clusters.

### Planned comparison

- Same 60/20/20 temporal split and PR-AUC / Recall@K evaluation protocol
- GNN architecture candidates: **GraphSAGE** or **heterogeneous graph transformer (HGT)** on the sender-receiver bipartite graph
- Edge features: Amount Paid, Payment Format, cross-border flag, timestamp
- IBM benchmark reference: GIN+EU F1 = 54.9 vs MLP F1 = 28.2 (95% improvement) on HI-Medium

### Expected outcome and open questions

The GNN should improve recall on unseen accounts (the main weakness of Model A) and detect layering chains that are invisible to the tabular baseline. Several questions can only be answered by running the comparison:

1. **Does the improvement appear in the top-K alerts or only in aggregate recall?** If GNN gains are concentrated in the tail (high K), the Critical/High tiers may not improve much in practice.
2. **Does GNN also memorise node identities?** If GNN embeddings are dominated by node IDs rather than structural patterns, the gap for unseen accounts may remain.
3. **What is the effective hop distance of fraud signal?** If most laundering chains are 2–3 hops, a shallow GNN (2 layers) should suffice; deeper models may propagate noise.
4. **Are edge features (amount, payment format) necessary, or does graph topology alone explain the gain?** This can be tested with a topology-only ablation (edges without features).
5. **Does calibration still work well on GNN scores?** GNN outputs are often less calibrated than XGBoost; a second-stage calibrator may be needed.

### Related work

- Altman et al. (2023) — *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023. [arXiv:2306.16424](https://arxiv.org/abs/2306.16424)
- Johannessen & Jullum (2023) — *Finding Money Launderers Using Heterogeneous Graph Neural Networks.* [arXiv:2307.13499](https://arxiv.org/abs/2307.13499)
- IBM Multi-GNN implementation: [github.com/IBM/Multi-GNN](https://github.com/IBM/Multi-GNN)

---

## Data

The IBM AML dataset is publicly available. Download it and place it at `data/raw/HI-Medium_Trans.csv`.

> Altman, E., et al. (2023). *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023. arXiv:2306.16424.
