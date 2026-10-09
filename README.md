# SC-CaTH Synthetic Professional Network Dataset

A fully synthetic, reproducible dataset for research prototyping and temporal evaluation of SC-CaTH-style professional people-to-people recommendation. All identifiers, profiles, companies, institutions, regions, skills, title text, and events are generated locally. **No real people, PII, real company names, scraped LinkedIn data, or Faker are used.**

## Quick start

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python generate_dataset.py
python validate_dataset.py
```

`python generate_dataset.py` regenerates all CSV outputs, the raw-duplicate example, schema, data dictionary, samples, and summary statistics with `seed=42`. `python validate_dataset.py` checks the generated files and writes `reports/validation_report.json`. No network connection is needed.

## Project layout

```text
SC-CaTH_synthetic_dataset/
├── generate_dataset.py
├── validate_dataset.py
├── requirements.txt
├── README.md
├── schema.json
├── data_dictionary.md
├── data/                         # complete cleaned CSV datasets
├── sample_data/                  # first 20 rows from every data CSV + raw sample
├── raw/
│   └── user_user_relations_raw.csv  # 1.7% duplicate-rate example, before cleaning
└── reports/
    ├── summary_stats.json
    └── validation_report.json
```

## Dataset summary

| Table / component | Clean records or value |
|---|---:|
| Professional users | 1,404 |
| Companies | 318 |
| Industries / functions / occupations | 24 / 18 / 96 |
| Synthetic skills | 240 |
| Career events | 2,871 |
| Unique undirected user-user relations | 5,612 |
| Interest titles | 9,891 |
| User-interest likes | 3,336 |
| Hyperedges | 7,020 (exactly 5 per user) |
| Supply-chain-specific hyperedges | 2,808 (40%) |
| Temporal events | 12,461 |
| Train / validation / test positive pairs | 3,500 / 1,300 / 812 |
| Train non-positive samples | 14,000 (4 per positive pair) |
| Validation non-positive samples | 128,700 (99 per positive pair) |
| Test non-positive samples | 80,388 (99 per positive pair) |
| Inductive cold-start users | 220 |

Observation window: **2023-01-01 to 2025-03-01** (26 months). Temporal cutoffs are `t1 = 2024-05-01 00:00:00` and `t2 = 2024-11-01 00:00:00`.

The generated graph has average degree **7.9943** and density **0.005698** (about 0.57%). The user-interest matrix density is **0.0002402** (about 0.024%). The `raw/user_user_relations_raw.csv` file contains 5,709 rows, including 97 copied duplicate rows, for a duplicate-row rate of **1.699%**. The cleaned `data/user_user_relations.csv` contains exactly 5,612 unique pairs.

## Temporal split and cold-start protocol

- **Train:** relations with timestamp `<= t1` (3,500 positive pairs).
- **Validation:** relations with timestamp `> t1` and `<= t2` (1,300 positive pairs).
- **Test:** relations with timestamp `> t2` (812 positive pairs), including an event exactly at the inclusive endpoint `2025-03-01 00:00:00`.
- User-user relations are undirected. Each pair appears once, with `user_id_1 < user_id_2`; `is_reciprocal` is always 1.
- There are 1,184 warm users with at least five pre-`t1` relations. The remaining 220 users have zero user-user relations before `t1` and are reserved for inductive testing. They are absent from train/validation positive pairs and candidate pools. Their pre-`t1` interest-like count is also zero.
- Train non-positive candidates are sampled uniformly without replacement from warm users who are not connected to the anchor anywhere in the training window. Validation/test non-positive candidates are sampled uniformly without replacement from users not connected to the anchor strictly before that candidate's decision timestamp. A pair that is unobserved is called **non-positive**; this does not assert the underlying professional relationship is absent.
- `train_pairs.csv`, `val_pairs.csv`, and `test_pairs.csv` contain one row per positive pair and a pipe-separated `negative_user_ids` list. Flat non-positive rows are supplied separately. `val_negative_samples.csv` is included in addition to the requested train/test flat files.

## Leakage-safe feature construction

- `users.csv` contains baseline profile attributes, not a future/current snapshot. Career evolution belongs in `career_events.csv` and must be reconstructed using only events at or before the prediction timestamp.
- `user_skills.csv`, `user_interest_interactions.csv`, and `hyperedges.csv` carry timestamps. Filter each of them to `timestamp <= decision_time` before making temporal/semantic features or hyperedge sets.
- `pair_context_features.csv` provides event-time examples of `degree_1_pre_event`, `degree_2_pre_event`, and `mutual_connections_pre_event`. These are computed using relation edges with timestamps **strictly earlier** than the target edge timestamp; edges at the same timestamp do not see each other.
- No full-window degree or popularity feature is materialized on users. Do not calculate degree, mutual connections, popularity, or hyperedge aggregates from the full graph for earlier decisions.
- `users.csv.is_cold` and `users.csv.pre_t1_interaction_count` are split metadata. `users.csv.first_interaction_time` is audit metadata computed from the full generated window. Do not feed these columns to a model as unrestricted features. Use them to verify the split only.
- `pair_context_features.csv.signed_counterfactual_sensitivity` is a **synthetic signed proxy** defined for controlled experiments; it is not a causal estimate. Its counterpart `supply_chain_compatibility` is also a synthetic profile-based score.

## Rating-error and ranking experiments

`pair_rating_targets.csv` provides synthetic 1–5 affinity targets for observed positive relation pairs, split temporally into train/validation/test. It supports reproducible RMSE/MAE experiments on these synthetic targets. These values are simulated labels, not actual human ratings. Non-observed pairs do not receive an inferred low rating; they remain non-positive candidates for ranking. For ranking metrics such as Recall@K, NDCG@K, MRR, Hit Rate, and Precision@K, use each positive pair with its sampled non-positive candidates.

## Missing values and structural nulls

Ordinary feature missingness is kept below 6.4%. A few categorical fields use the token `unknown`. The following nulls are **structural**, not ordinary missingness: `career_events.end_time` for the last active position; `career_events.previous_seniority_level` and `career_events.previous_occupation_id` for the first career record; and `hyperedges.sc_role`, `sc_stage`, `sc_path` when `supply_chain_specific=0`. The validator excludes only these documented structural nulls from the per-feature missing-rate limit.

`duration_months` is a synthetic tenure label between 6 and 60 months. For earlier career positions it equals the gap to the next synthetic career record; for the final active position it may reflect tenure accrued before the 26-month observation window. All exported timestamp fields remain inside the stated observation window.

## File guide

| File | Intended use |
|---|---|
| `users.csv`, `companies.csv`, `industries.csv`, `functions.csv`, `occupations.csv`, `skills.csv`, `universities.csv`, `locations.csv` | Multi-view profile and lookup tables |
| `career_events.csv`, `user_skills.csv`, `user_interest_interactions.csv` | Temporal career, skill, and semantic views |
| `user_user_relations.csv`, `graph_edges.csv` | Undirected positive social/professional graph and GNN/LightGCN baselines |
| `interest_titles.csv`, `user_interest_edges.csv` | Auxiliary semantic/bipartite view |
| `hyperedges.csv` | Higher-order heterogeneous hypergraph context |
| `pair_context_features.csv` | Leakage-safe pair context, supply-chain compatibility, signed counterfactual proxy |
| `pair_rating_targets.csv` | Synthetic affinity target for RMSE/MAE |
| `temporal_events.csv` | Unified event stream, exactly 12,461 rows |
| `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv` | Positive ranking queries with embedded non-positive candidate IDs |
| `train_negative_samples.csv`, `val_negative_samples.csv`, `test_negative_samples.csv` | Flat ranking candidates with `label=0` |
| `cold_start_users.csv` | Explicit inductive test cohort |

`schema.json` describes column types, nullability, primary keys, foreign keys, and semantics for every CSV. `data_dictionary.md` is the human-readable companion. Samples in `sample_data/` contain the first 20 rows of each cleaned CSV; the raw relation sample is named `raw_user_user_relations_raw.csv`.

## Scope notes

This release prioritizes deterministic structural validity and transparent split logic for experimentation. The relation backbone is constructed to guarantee at least five pre-`t1` edges for every warm user; additional edges are selected with synthetic profile-affinity weights and a mutual-neighbour bonus based only on already-added edges. Therefore, these data are appropriate for pipeline tests, ablations, metric implementations, and reproducibility checks, but they are **not empirical evidence about LinkedIn or any real professional network**.
