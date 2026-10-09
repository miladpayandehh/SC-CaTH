# SC-CaTH Synthetic Dataset — Data Dictionary

All IDs, labels, profile snippets, companies, institutions, regions, titles, and interactions are synthetic. No real profiles or scraped data are used.

## Temporal and leakage rules

- Observation window: `2023-01-01T00:00:00` through `2025-03-01T00:00:00` (26 months).
- `t1 = 2024-05-01T00:00:00`; `t2 = 2024-11-01T00:00:00`.
- Train: `timestamp <= t1`; validation: `t1 < timestamp <= t2`; test: `timestamp > t2`.
- User-user edges are undirected and stored once (`user_id_1 < user_id_2`).
- Degree/mutual-neighbour features in `pair_context_features.csv` use only edges strictly earlier than the target edge timestamp.
- `users.csv` profile attributes are the baseline snapshot. `is_cold`, `pre_t1_interaction_count`, and `first_interaction_time` are split/audit metadata and must not be used as unrestricted predictive features; `first_interaction_time` is full-window aware.
- `career_events.csv`, `user_skills.csv`, `hyperedges.csv`, and `user_interest_interactions.csv` are timestamped. Filter each source to records at or before the decision time before feature construction.
- A non-positive candidate means an unobserved pair at the relevant cutoff, not proof that a professional relationship does not exist.
- `pair_rating_targets.csv` contains synthetic 1–5 affinity labels for observed positive relation pairs only; do not convert missing interactions into rating 1.
- Nulls in `career_events.end_time`, `previous_seniority_level`, `previous_occupation_id` and `hyperedges.sc_*` are structural, as explained below.

## File schemas

### `users.csv`

Baseline professional profile as of the beginning of the observation window; not a final/current snapshot.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `user_id` | `string` | no | Unique synthetic user identifier U0001... |
| `experience_years` | `float` | yes | Synthetic professional experience at baseline; 2% deliberately null. |
| `industry_id` | `string` | no | FK to industries.industry_id. |
| `function_id` | `string` | no | FK to functions.function_id. |
| `occupation_id` | `string` | no | FK to occupations.occupation_id. |
| `seniority_level` | `integer` | no | Synthetic ordinal level 1..7. |
| `education_level` | `string` | no | Synthetic category; unknown represents unrecorded information. |
| `field_of_study` | `string` | no | Synthetic category; unknown represents unrecorded information. |
| `university_id` | `string` | no | FK to universities; V_UNKNOWN is a valid synthetic lookup row. |
| `company_id` | `string` | no | FK to companies; company at baseline. |
| `company_size` | `string` | no | Baseline company-size category or unknown token. |
| `geographic_region_id` | `string` | no | FK to locations; synthetic region. |
| `supply_chain_role` | `string` | no | One of seven abstract supply-chain roles. |
| `supply_chain_stage` | `string` | no | Upstream, Midstream, or Downstream. |
| `supply_chain_path` | `string` | no | Synthetic path category. |
| `profile_text` | `string` | yes | Synthetic baseline text built from tokens/IDs; 2% null. |
| `occupation_text` | `string` | yes | Synthetic occupation text; 2% null. |
| `skills_text` | `string` | yes | Baseline skill IDs separated by \|; 2% null. |
| `is_cold` | `boolean` | no | True when user has fewer than 5 user-user relation events at or before t1; split metadata, not a model feature. |
| `first_interaction_time` | `timestamp` | yes | First relation/interest-like timestamp over the full window; audit-only and future-aware, exclude as a model feature. |
| `pre_t1_interaction_count` | `integer` | no | Count of user-user relation events at or before t1; used to verify cold-start membership. |

**Primary key:** `user_id`.
**Foreign keys:** `industry_id` → `industries.csv.industry_id`; `function_id` → `functions.csv.function_id`; `occupation_id` → `occupations.csv.occupation_id`; `university_id` → `universities.csv.university_id`; `company_id` → `companies.csv.company_id`; `geographic_region_id` → `locations.csv.geographic_region_id`.

### `companies.csv`

Synthetic organizations; sizes follow a heavy-tailed categorical distribution.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `company_id` | `string` | no | Unique synthetic company ID. |
| `industry_id` | `string` | no | FK to industries. |
| `company_size` | `string` | no | Synthetic company-size class. |
| `organization_type` | `string` | no | Synthetic organization-type token. |
| `geographic_region_id` | `string` | no | FK to locations. |
| `company_age_months` | `integer` | no | Synthetic organization age in months; not an individual's age. |

**Primary key:** `company_id`.
**Foreign keys:** `industry_id` → `industries.csv.industry_id`; `geographic_region_id` → `locations.csv.geographic_region_id`.

### `industries.csv`

24 synthetic industry categories.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `industry_id` | `string` | no | Synthetic industry ID. |
| `industry_name` | `string` | no | Tokenized synthetic label. |
| `supply_chain_relevance` | `integer` | no | Binary indicator 0/1 for synthetic scenario design. |

**Primary key:** `industry_id`.

### `functions.csv`

18 synthetic professional-function categories.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `function_id` | `string` | no | Synthetic function ID. |
| `function_name` | `string` | no | Tokenized synthetic label. |

**Primary key:** `function_id`.

### `occupations.csv`

96 synthetic occupations; each maps to one professional function.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `occupation_id` | `string` | no | Synthetic occupation ID. |
| `occupation_name` | `string` | no | Tokenized synthetic label. |
| `function_id` | `string` | no | FK to functions. |
| `seniority_scale_min` | `integer` | no | Synthetic minimum level; all are 1. |
| `seniority_scale_max` | `integer` | no | Synthetic maximum level; all are 7. |

**Primary key:** `occupation_id`.
**Foreign keys:** `function_id` → `functions.csv.function_id`.

### `skills.csv`

Synthetic skill vocabulary organized into semantic clusters.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `skill_id` | `string` | no | Synthetic skill ID. |
| `skill_name` | `string` | no | Synthetic skill token. |
| `skill_cluster` | `string` | no | Synthetic cluster identifier. |

**Primary key:** `skill_id`.

### `universities.csv`

Synthetic educational institution lookup including a valid unknown token.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `university_id` | `string` | no | Synthetic institution ID. |
| `university_name` | `string` | no | Synthetic tokenized label. |

**Primary key:** `university_id`.

### `locations.csv`

Synthetic geographic region lookup; no real addresses or coordinates.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `geographic_region_id` | `string` | no | Synthetic region ID. |
| `region_name` | `string` | no | Synthetic region token. |

**Primary key:** `geographic_region_id`.

### `user_skills.csv`

Timestamped skill records. Filter by timestamp <= decision time.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `user_id` | `string` | no | FK to users. |
| `skill_id` | `string` | no | FK to skills. |
| `proficiency` | `integer` | no | Synthetic level 1..5. |
| `timestamp` | `timestamp` | no | Skill-record timestamp within the observation window. |

**Primary key:** `user_id`, `skill_id`, `timestamp`.
**Foreign keys:** `user_id` → `users.csv.user_id`; `skill_id` → `skills.csv.skill_id`.

### `career_events.csv`

Chronological synthetic career-position records. First row per user matches the baseline profile; subsequent records are changes. end_time/previous fields are structurally null for active/first records.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `event_id` | `string` | no | Unique career event ID. |
| `user_id` | `string` | no | FK to users. |
| `company_id` | `string` | no | FK to companies. |
| `industry_id` | `string` | no | FK to industries. |
| `function_id` | `string` | no | FK to functions. |
| `occupation_id` | `string` | no | FK to occupations. |
| `skills` | `string` | no | Pipe-separated synthetic skill IDs. |
| `seniority_level` | `integer` | no | Current level 1..7. |
| `previous_seniority_level` | `integer` | yes | Previous level; structurally null for first trajectory event. |
| `previous_occupation_id` | `string` | yes | Previous occupation; structurally null for first trajectory event. |
| `start_time` | `timestamp` | no | Career-record/position start marker within the observation window. |
| `end_time` | `timestamp` | yes | Next career event timestamp, or null for final active position. |
| `duration_months` | `integer` | no | Synthetic position tenure 6..60; final active tenure can include time before the observation window. |
| `delta_seniority` | `integer` | no | Current minus previous seniority; 0 for first record. |
| `transition_type` | `string` | no | first_job for first record, otherwise promotion/lateral/demotion consistent with delta. |

**Primary key:** `event_id`.
**Foreign keys:** `user_id` → `users.csv.user_id`; `company_id` → `companies.csv.company_id`; `industry_id` → `industries.csv.industry_id`; `function_id` → `functions.csv.function_id`; `occupation_id` → `occupations.csv.occupation_id`; `previous_occupation_id` → `occupations.csv.occupation_id`.

### `user_user_relations.csv`

Unique undirected positive user-user relations stored exactly once with user_id_1 < user_id_2.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `relation_id` | `string` | no | Unique relation ID. |
| `user_id_1` | `string` | no | First endpoint, lexicographically smaller ID. |
| `user_id_2` | `string` | no | Second endpoint, lexicographically larger ID. |
| `timestamp` | `timestamp` | no | Relation formation timestamp. |
| `relation_type` | `string` | no | Synthetic relation category. |
| `is_reciprocal` | `integer` | no | Always 1; undirected relation semantics. |
| `interaction_strength` | `float` | no | Synthetic strength in [0,1]. |

**Primary key:** `relation_id`.
**Foreign keys:** `user_id_1` → `users.csv.user_id`; `user_id_2` → `users.csv.user_id`.

### `interest_titles.csv`

9,891 synthetic interest/topic titles.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `interest_title_id` | `string` | no | Unique synthetic title ID. |
| `title_text` | `string` | no | Synthetic token sequence. |
| `category` | `string` | no | Synthetic category token. |
| `related_skills` | `string` | no | Pipe-separated related skill IDs. |

**Primary key:** `interest_title_id`.

### `user_interest_interactions.csv`

Synthetic user-to-title likes, timestamped and unique by user/title pair.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `interaction_id` | `string` | no | Unique interest interaction ID. |
| `user_id` | `string` | no | FK to users. |
| `interest_title_id` | `string` | no | FK to interest_titles. |
| `timestamp` | `timestamp` | no | Interaction timestamp. |
| `interaction_type` | `string` | no | Always like. |

**Primary key:** `interaction_id`.
**Foreign keys:** `user_id` → `users.csv.user_id`; `interest_title_id` → `interest_titles.csv.interest_title_id`.

### `hyperedges.csv`

Exactly five timestamped user-centred hyperedges per user. Career and skills context is reconstructed as of the hyperedge timestamp.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `hyperedge_id` | `string` | no | Unique hyperedge ID. |
| `user_id` | `string` | no | FK to users. |
| `company_id` | `string` | no | FK to companies. |
| `industry_id` | `string` | no | FK to industries. |
| `function_id` | `string` | no | FK to functions. |
| `occupation_id` | `string` | no | FK to occupations. |
| `skill_ids` | `string` | no | Pipe-separated skill IDs. |
| `timestamp` | `timestamp` | no | Hyperedge creation timestamp. |
| `relation_type` | `string` | no | Synthetic higher-order relation category. |
| `supply_chain_specific` | `integer` | no | Binary flag; exactly 40% are 1. |
| `sc_role` | `string` | yes | Supply-chain role; structurally null if supply_chain_specific=0. |
| `sc_stage` | `string` | yes | Supply-chain stage; structurally null if supply_chain_specific=0. |
| `sc_path` | `string` | yes | Supply-chain path; structurally null if supply_chain_specific=0. |

**Primary key:** `hyperedge_id`.
**Foreign keys:** `user_id` → `users.csv.user_id`; `company_id` → `companies.csv.company_id`; `industry_id` → `industries.csv.industry_id`; `function_id` → `functions.csv.function_id`; `occupation_id` → `occupations.csv.occupation_id`.

### `temporal_events.csv`

Exactly 12,461 events: all career, relation and interest-like events plus 642 sampled hyperedge-creation events.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `event_id` | `string` | no | Unique temporal event ID. |
| `event_type` | `string` | no | career/relation/interest_like/hyperedge_creation. |
| `user_id` | `string` | no | FK to users. |
| `target_id` | `string` | no | Type-dependent synthetic target ID. |
| `timestamp` | `timestamp` | no | Source-event timestamp in the observation window. |
| `metadata` | `string` | no | Compact JSON object with synthetic IDs/categories only. |

**Primary key:** `event_id`.
**Foreign keys:** `user_id` → `users.csv.user_id`.

### `train_pairs.csv`

Positive training pairs and four uniformly sampled non-positive candidate IDs per positive. Training pairs contain warm users only.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `anchor_user_id` | `string` | no | Ranking query/anchor. |
| `positive_user_id` | `string` | no | Observed positive counterpart. |
| `negative_user_ids` | `string` | no | Four unique pipe-separated non-positive user IDs. |
| `split_time` | `timestamp` | no | Positive relation timestamp. |
| `label` | `integer` | no | Always 1 on positive pair rows. |

**Primary key:** `anchor_user_id`, `positive_user_id`.

### `val_pairs.csv`

Positive validation pairs and 99 uniformly sampled non-positive candidates; warm users only.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `anchor_user_id` | `string` | no | Ranking anchor. |
| `positive_user_id` | `string` | no | Observed positive counterpart. |
| `negative_user_ids` | `string` | no | 99 unique pipe-separated candidate IDs. |
| `split_time` | `timestamp` | no | Positive relation timestamp. |
| `label` | `integer` | no | Always 1 on positive pair rows. |

**Primary key:** `anchor_user_id`, `positive_user_id`.

### `test_pairs.csv`

Positive test pairs and 99 sampled non-positive candidates; includes inductive cold users that are absent from train/validation pairs.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `anchor_user_id` | `string` | no | Ranking anchor; cold users are anchors on designated cold-test edges. |
| `positive_user_id` | `string` | no | Observed positive counterpart. |
| `negative_user_ids` | `string` | no | 99 unique pipe-separated candidate IDs. |
| `split_time` | `timestamp` | no | Positive relation timestamp. |
| `label` | `integer` | no | Always 1 on positive pair rows. |

**Primary key:** `anchor_user_id`, `positive_user_id`.

### `train_negative_samples.csv`

Flat training non-positive samples: four per positive relation.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `pair_id` | `string` | no | Synthetic pair reference P_train_... |
| `anchor_user_id` | `string` | no | Ranking anchor. |
| `positive_user_id` | `string` | no | Positive counterpart for context. |
| `non_positive_user_id` | `string` | no | Sampled non-positive candidate. |
| `sample_index` | `integer` | no | 1..4 within pair. |
| `split_time` | `timestamp` | no | Positive relation time. |
| `label` | `integer` | no | Always 0. |

**Primary key:** `pair_id`, `sample_index`.

### `val_negative_samples.csv`

Flat validation non-positive samples: 99 per positive relation.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `pair_id` | `string` | no | Synthetic pair reference P_val_... |
| `anchor_user_id` | `string` | no | Ranking anchor. |
| `positive_user_id` | `string` | no | Positive counterpart for context. |
| `non_positive_user_id` | `string` | no | Sampled non-positive candidate. |
| `sample_index` | `integer` | no | 1..99 within pair. |
| `split_time` | `timestamp` | no | Positive relation time. |
| `label` | `integer` | no | Always 0. |

**Primary key:** `pair_id`, `sample_index`.

### `test_negative_samples.csv`

Flat test non-positive samples: 99 per positive relation.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `pair_id` | `string` | no | Synthetic pair reference P_test_... |
| `anchor_user_id` | `string` | no | Ranking anchor. |
| `positive_user_id` | `string` | no | Positive counterpart for context. |
| `non_positive_user_id` | `string` | no | Sampled non-positive candidate. |
| `sample_index` | `integer` | no | 1..99 within pair. |
| `split_time` | `timestamp` | no | Positive relation time. |
| `label` | `integer` | no | Always 0. |

**Primary key:** `pair_id`, `sample_index`.

### `graph_edges.csv`

Undirected graph edges stored once; construct an undirected graph or duplicate each row in memory if a framework requires directed message-passing arcs.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `source_user_id` | `string` | no | Lexicographically smaller endpoint. |
| `target_user_id` | `string` | no | Lexicographically larger endpoint. |
| `timestamp` | `timestamp` | no | Edge formation timestamp. |
| `edge_type` | `string` | no | Synthetic relation type. |

**Primary key:** `source_user_id`, `target_user_id`.
**Foreign keys:** `source_user_id` → `users.csv.user_id`; `target_user_id` → `users.csv.user_id`.

### `user_interest_edges.csv`

Auxiliary user-to-interest bipartite edges; time-filter before a decision timestamp.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `user_id` | `string` | no | FK to users. |
| `interest_title_id` | `string` | no | FK to interest_titles. |
| `timestamp` | `timestamp` | no | Like timestamp. |

**Primary key:** `user_id`, `interest_title_id`.
**Foreign keys:** `user_id` → `users.csv.user_id`; `interest_title_id` → `interest_titles.csv.interest_title_id`.

### `pair_context_features.csv`

Leakage-safe, event-time pair context for observed edges. Graph degree and mutual connections are computed strictly from relation timestamps earlier than the edge timestamp. Counterfactual sensitivity is a synthetic proxy, not a causal estimate.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `relation_id` | `string` | no | FK to user_user_relations. |
| `user_id_1` | `string` | no | Pair endpoint 1. |
| `user_id_2` | `string` | no | Pair endpoint 2. |
| `timestamp` | `timestamp` | no | Observed edge time. |
| `degree_1_pre_event` | `integer` | no | Degree in graph strictly before event timestamp. |
| `degree_2_pre_event` | `integer` | no | Degree in graph strictly before event timestamp. |
| `mutual_connections_pre_event` | `integer` | no | Common neighbors strictly before event timestamp. |
| `same_company_baseline` | `integer` | no | Same synthetic baseline company flag. |
| `same_industry_baseline` | `integer` | no | Same synthetic baseline industry flag. |
| `same_function_baseline` | `integer` | no | Same synthetic baseline function flag. |
| `seniority_gap_baseline` | `integer` | no | Absolute baseline seniority difference. |
| `supply_chain_compatibility` | `float` | no | Synthetic pair compatibility proxy in [0,1]. |
| `signed_counterfactual_sensitivity` | `float` | no | Synthetic signed delta proxy in [-0.4,0.4], not a causal effect. |
| `counterfactual_intervention` | `string` | no | Synthetic intervention label: remove_supply_chain_context. |

**Primary key:** `relation_id`.
**Foreign keys:** `relation_id` → `user_user_relations.csv.relation_id`; `user_id_1` → `users.csv.user_id`; `user_id_2` → `users.csv.user_id`.

### `pair_rating_targets.csv`

Synthetic 1..5 affinity targets attached to observed relation pairs for reproducible RMSE/MAE experiments. They are simulated labels, not real user ratings; non-observed pairs remain non-positive rather than assigned a low rating.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `relation_id` | `string` | no | FK to user_user_relations. |
| `user_id_1` | `string` | no | First undirected pair endpoint. |
| `user_id_2` | `string` | no | Second undirected pair endpoint. |
| `timestamp` | `timestamp` | no | Relation event timestamp. |
| `rating_target` | `float` | no | Synthetic ordinal-like affinity target in [1,5], rounded to two decimals. |
| `split` | `string` | no | Temporal split: train/val/test. |
| `rating_source` | `string` | no | Always synthetic_latent_affinity. |

**Primary key:** `relation_id`.
**Foreign keys:** `relation_id` → `user_user_relations.csv.relation_id`; `user_id_1` → `users.csv.user_id`; `user_id_2` → `users.csv.user_id`.

### `cold_start_users.csv`

Cold users reserved for the inductive test split; no user-user relations or interest likes before t1.

| Column | Type | Nullable | Description |
|---|---|:---:|---|
| `user_id` | `string` | no | FK to users. |
| `pre_t1_relation_count` | `integer` | no | Must be zero for designated cold test users. |
| `reserved_split` | `string` | no | Always test. |
| `cold_start_reason` | `string` | no | Synthetic split rule. |

**Primary key:** `user_id`.
**Foreign keys:** `user_id` → `users.csv.user_id`.

