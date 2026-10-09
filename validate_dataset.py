#!/usr/bin/env python3
"""Validate generated SC-CaTH synthetic data. Run: python validate_dataset.py"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = ROOT / "raw"
REPORTS = ROOT / "reports"
START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2025-03-01 00:00:00")
T1 = pd.Timestamp("2024-05-01 00:00:00")
T2 = pd.Timestamp("2024-11-01 00:00:00")
N_USERS = 1404
N_WARM = 1184
N_COLD = 220

EXPECTED = {
    "users.csv": 1404,
    "companies.csv": 318,
    "industries.csv": 24,
    "functions.csv": 18,
    "occupations.csv": 96,
    "skills.csv": 240,
    "universities.csv": 61,
    "locations.csv": 36,
    "career_events.csv": 2871,
    "user_user_relations.csv": 5612,
    "interest_titles.csv": 9891,
    "user_interest_interactions.csv": 3336,
    "hyperedges.csv": 7020,
    "temporal_events.csv": 12461,
    "train_pairs.csv": 3500,
    "val_pairs.csv": 1300,
    "test_pairs.csv": 812,
    "train_negative_samples.csv": 14000,
    "val_negative_samples.csv": 128700,
    "test_negative_samples.csv": 80388,
    "graph_edges.csv": 5612,
    "user_interest_edges.csv": 3336,
    "user_skills.csv": 11217,
    "pair_context_features.csv": 5612,
    "pair_rating_targets.csv": 5612,
    "cold_start_users.csv": 220,
}
STRUCTURAL_NULLS = {
    "career_events.csv": {"end_time", "previous_seniority_level", "previous_occupation_id"},
    "hyperedges.csv": {"sc_role", "sc_stage", "sc_path"},
}


def fail_if(condition: bool, message: str, errors: list[str]) -> None:
    if condition:
        errors.append(message)


def read_table(name: str, dtype: Any = None) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, dtype=dtype, low_memory=False)


def as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def validate() -> dict[str, Any]:
    errors: list[str] = []
    schema_path = ROOT / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    actual_csvs = {p.name for p in DATA.glob("*.csv")}
    schema_csvs = set(schema.get("files", {}))
    fail_if(actual_csvs != schema_csvs,
            f"Schema/data file mismatch: missing schemas={sorted(actual_csvs-schema_csvs)}, extra schemas={sorted(schema_csvs-actual_csvs)}", errors)

    tables: dict[str, pd.DataFrame] = {}
    for filename in sorted(actual_csvs):
        tables[filename] = read_table(filename)
    for filename, expected_n in EXPECTED.items():
        fail_if(filename not in tables, f"Missing expected CSV {filename}", errors)
        if filename in tables:
            fail_if(len(tables[filename]) != expected_n,
                    f"{filename}: expected {expected_n} rows, found {len(tables[filename])}", errors)

    # Table column contract: exact schema column set.
    for filename, spec in schema.get("files", {}).items():
        if filename not in tables:
            continue
        expected_cols = set(spec["fields"])
        actual_cols = set(tables[filename].columns)
        fail_if(expected_cols != actual_cols,
                f"{filename}: column mismatch; missing={sorted(expected_cols-actual_cols)}, extra={sorted(actual_cols-expected_cols)}", errors)
        frame = tables[filename]
        # Each non-structural field must be under the 6.4% null limit.
        structural = STRUCTURAL_NULLS.get(filename, set())
        for column in frame.columns:
            if column in structural:
                continue
            rate = float(frame[column].isna().mean())
            fail_if(rate >= 0.064, f"{filename}.{column}: null rate {rate:.4%} is not below 6.4%", errors)

    # Primary and foreign key validation from schema.json.
    for filename, spec in schema.get("files", {}).items():
        if filename not in tables:
            continue
        df = tables[filename]
        pk = spec.get("primary_key", [])
        if pk:
            fail_if(df.duplicated(pk).any(), f"{filename}: duplicate primary key on {pk}", errors)
        for fk in spec.get("foreign_keys", []):
            col = fk["field"]
            target = fk["references_file"]
            target_col = fk["references_field"]
            if target not in tables or col not in df or target_col not in tables.get(target, pd.DataFrame()).columns:
                errors.append(f"Broken FK schema definition: {filename}.{col} -> {target}.{target_col}")
                continue
            available = set(tables[target][target_col].dropna().astype(str))
            values = set(df[col].dropna().astype(str))
            invalid = sorted(values - available)
            fail_if(bool(invalid), f"{filename}.{col}: {len(invalid)} invalid FK values, examples={invalid[:5]}", errors)

    users = tables["users.csv"]
    companies = tables["companies.csv"]
    industries = tables["industries.csv"]
    functions = tables["functions.csv"]
    occupations = tables["occupations.csv"]
    career = tables["career_events.csv"]
    relations = tables["user_user_relations.csv"]
    titles = tables["interest_titles.csv"]
    likes = tables["user_interest_interactions.csv"]
    hyperedges = tables["hyperedges.csv"]
    temporal = tables["temporal_events.csv"]
    train_pairs = tables["train_pairs.csv"]
    val_pairs = tables["val_pairs.csv"]
    test_pairs = tables["test_pairs.csv"]
    train_neg = tables["train_negative_samples.csv"]
    val_neg = tables["val_negative_samples.csv"]
    test_neg = tables["test_negative_samples.csv"]
    graph_edges = tables["graph_edges.csv"]
    interest_edges = tables["user_interest_edges.csv"]
    context = tables["pair_context_features.csv"]
    ratings = tables["pair_rating_targets.csv"]
    cold_table = tables["cold_start_users.csv"]

    # User graph constraints.
    user_ids = set(users["user_id"].astype(str))
    fail_if(users["user_id"].duplicated().any(), "users.csv user_id must be unique", errors)
    fail_if(not (relations["user_id_1"].astype(str) < relations["user_id_2"].astype(str)).all(),
            "Relations must be stored exactly once with user_id_1 < user_id_2", errors)
    fail_if(relations.duplicated(["user_id_1", "user_id_2"]).any(), "Duplicate user-user pair in cleaned relation table", errors)
    fail_if(not (relations["is_reciprocal"] == 1).all(), "Every relation must have is_reciprocal=1", errors)
    fail_if(not relations["interaction_strength"].between(0, 1).all(), "interaction_strength outside [0,1]", errors)
    degree = pd.Series(0, index=list(user_ids), dtype=int)
    for col in ("user_id_1", "user_id_2"):
        degree = degree.add(relations[col].value_counts(), fill_value=0).astype(int)
    average_degree = float(2 * len(relations) / len(users))
    density = float(2 * len(relations) / (len(users) * (len(users)-1)))
    fail_if(abs(average_degree - 8.0) > 0.10, f"Average degree {average_degree:.6f} is not approximately 8", errors)
    fail_if(abs(density - 0.0057) > 0.0001, f"Graph density {density:.6f} is not approximately 0.0057", errors)

    # Interest matrix density and hyperedge counts.
    interest_density = float(len(likes) / (len(users) * len(titles)))
    fail_if(interest_density >= 0.0003, f"User-interest density {interest_density:.8f} is not below 0.0003", errors)
    fail_if(not (likes["interaction_type"] == "like").all(), "All user_interest_interactions must be 'like'", errors)
    fail_if(not (hyperedges.groupby("user_id").size() == 5).all(), "Every user must have exactly five hyperedges", errors)
    fail_if(int(hyperedges["supply_chain_specific"].sum()) != 2808, "Expected exactly 2,808 supply-chain-specific hyperedges", errors)
    for col in ("sc_role", "sc_stage", "sc_path"):
        fail_if(hyperedges.loc[hyperedges["supply_chain_specific"] == 1, col].isna().any(),
                f"{col} must be populated for supply_chain_specific=1", errors)
        fail_if(hyperedges.loc[hyperedges["supply_chain_specific"] == 0, col].notna().any(),
                f"{col} must be null for supply_chain_specific=0", errors)

    # Timestamps: every time-like field is within the 26-month observation window.
    datetime_cols = {"timestamp", "start_time", "end_time", "first_interaction_time", "split_time"}
    time_bounds: dict[str, dict[str, str | None]] = {}
    global_min: pd.Timestamp | None = None
    global_max: pd.Timestamp | None = None
    for filename, frame in tables.items():
        for col in frame.columns:
            if col not in datetime_cols:
                continue
            parsed = as_datetime(frame[col])
            nonnull = parsed.dropna()
            if frame[col].notna().sum() > 0 and nonnull.empty:
                errors.append(f"{filename}.{col}: all non-null timestamps failed to parse")
                continue
            if not nonnull.empty:
                lo, hi = nonnull.min(), nonnull.max()
                time_bounds.setdefault(filename, {})[col] = {"min": lo.isoformat(), "max": hi.isoformat()}
                global_min = lo if global_min is None or lo < global_min else global_min
                global_max = hi if global_max is None or hi > global_max else global_max
                bad = (nonnull < START) | (nonnull > END)
                fail_if(bool(bad.any()), f"{filename}.{col}: timestamp outside [{START}, {END}]", errors)
    fail_if(global_min is None or global_min > START, "No timestamp reaches the observation start boundary", errors)
    fail_if(global_max is None or global_max != END, f"The latest timestamp should equal inclusive endpoint {END}; got {global_max}", errors)

    # Split cutoffs are checked on relation events, positive pairs, and rating targets.
    relation_times = pd.to_datetime(relations["timestamp"])
    rel_train = relations.loc[relation_times <= T1]
    rel_val = relations.loc[(relation_times > T1) & (relation_times <= T2)]
    rel_test = relations.loc[relation_times > T2]
    fail_if((len(rel_train), len(rel_val), len(rel_test)) != (3500, 1300, 812),
            f"Split relation counts are {(len(rel_train), len(rel_val), len(rel_test))}, expected (3500,1300,812)", errors)
    for name, frame, low, high, condition in [
        ("train_pairs.csv", train_pairs, None, T1, "train"),
        ("val_pairs.csv", val_pairs, T1, T2, "val"),
        ("test_pairs.csv", test_pairs, T2, None, "test"),
    ]:
        time_col = pd.to_datetime(frame["split_time"])
        ok = time_col.le(high) if low is None else (time_col.gt(low) & time_col.le(high) if high is not None else time_col.gt(low))
        fail_if(not bool(ok.all()), f"{name} violates temporal split boundaries", errors)

    # Cold-start: all designated cold users have zero relation/like interactions before t1, and
    # are absent from training AND validation positive/negative folds.
    cold_ids = set(cold_table["user_id"].astype(str))
    fail_if(len(cold_ids) != N_COLD, f"Expected {N_COLD} cold test users, found {len(cold_ids)}", errors)
    fail_if(cold_ids != set(users.loc[users["is_cold"].astype(str).str.lower().isin(["true", "1"]), "user_id"].astype(str)),
            "users.is_cold does not match cold_start_users.csv", errors)
    fail_if(not (cold_table["pre_t1_relation_count"] == 0).all(), "Cold test users must have zero relations before t1", errors)
    cold_pre_rel = relations[(relations["user_id_1"].isin(cold_ids) | relations["user_id_2"].isin(cold_ids)) & (pd.to_datetime(relations["timestamp"]) <= T1)]
    cold_pre_likes = likes[likes["user_id"].isin(cold_ids) & (pd.to_datetime(likes["timestamp"]) <= T1)]
    fail_if(not cold_pre_rel.empty, "A cold test user has a relation at or before t1", errors)
    fail_if(not cold_pre_likes.empty, "A cold test user has an interest-like at or before t1", errors)
    fail_if(not (users.loc[~users["is_cold"].astype(str).str.lower().isin(["true", "1"]), "pre_t1_interaction_count"].astype(int) >= 5).all(),
            "Every non-cold user must have at least five pre-t1 relation events", errors)

    def parsed_negs(value: Any) -> list[str]:
        if pd.isna(value) or str(value) == "":
            return []
        return str(value).split("|")

    pair_file_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    flat_map = {"train": train_neg, "val": val_neg, "test": test_neg}
    negative_per_split = {"train": 4, "val": 99, "test": 99}
    for split, frame in pair_file_map.items():
        required_k = negative_per_split[split]
        for row in frame.itertuples(index=False):
            negs = parsed_negs(row.negative_user_ids)
            fail_if(len(negs) != required_k, f"{split} pair has {len(negs)} non-positive IDs, expected {required_k}", errors)
            fail_if(len(set(negs)) != len(negs), f"{split} pair has duplicate non-positive candidate IDs", errors)
            fail_if(row.anchor_user_id not in user_ids or row.positive_user_id not in user_ids or not set(negs).issubset(user_ids),
                    f"{split} pair has an unknown user ID", errors)
            fail_if(row.anchor_user_id == row.positive_user_id or row.positive_user_id in negs or row.anchor_user_id in negs,
                    f"{split} pair contains anchor/positive among non-positive candidates", errors)
            fail_if(int(row.label) != 1, f"{split} positive pair label must be 1", errors)
            if split in ("train", "val"):
                fail_if(row.anchor_user_id in cold_ids or row.positive_user_id in cold_ids or set(negs) & cold_ids,
                        f"Cold user present in {split} fold population", errors)
    for split, frame in flat_map.items():
        k = negative_per_split[split]
        positive_n = len(pair_file_map[split])
        fail_if(len(frame) != positive_n * k, f"{split} flat negative count should be {positive_n*k}, got {len(frame)}", errors)
        fail_if(not (frame["label"] == 0).all(), f"{split} negative rows must have label 0", errors)
        counts = frame.groupby("pair_id").size()
        fail_if(not (counts == k).all() or len(counts) != positive_n, f"{split} each pair_id must have exactly {k} flat samples", errors)
        if split in ("train", "val"):
            fail_if(frame["anchor_user_id"].isin(cold_ids).any() or frame["positive_user_id"].isin(cold_ids).any() or frame["non_positive_user_id"].isin(cold_ids).any(),
                    f"Cold user present in {split} flat negative samples", errors)
        fail_if((frame["non_positive_user_id"] == frame["anchor_user_id"]).any() or
                (frame["non_positive_user_id"] == frame["positive_user_id"]).any(),
                f"{split} flat sample includes anchor or positive as non-positive", errors)

    # Confirm causal negative sampling: a sampled non-positive was not already connected to the
    # anchor before the sample's decision timestamp. Training candidates are checked against all train edges.
    edge_times: dict[tuple[str, str], list[pd.Timestamp]] = defaultdict(list)
    for e in relations.itertuples(index=False):
        edge_times[(e.user_id_1, e.user_id_2)].append(pd.Timestamp(e.timestamp))
    def has_prior_edge(a: str, b: str, cutoff: pd.Timestamp) -> bool:
        key = (a, b) if a < b else (b, a)
        return any(t < cutoff for t in edge_times.get(key, []))
    for row in train_neg.itertuples(index=False):
        a, b, cutoff = row.anchor_user_id, row.non_positive_user_id, T1
        fail_if(any(t <= T1 for t in edge_times.get((a, b) if a < b else (b, a), [])),
                "Training non-positive candidate had a relation during training window", errors)
    for split, frame in [("val", val_neg), ("test", test_neg)]:
        for row in frame.itertuples(index=False):
            fail_if(has_prior_edge(row.anchor_user_id, row.non_positive_user_id, pd.Timestamp(row.split_time)),
                    f"{split} non-positive candidate was already linked before its decision time", errors)

    # Career chronology, transition labels, durations, and deltas.
    career["start_dt"] = pd.to_datetime(career["start_time"])
    career["end_dt"] = pd.to_datetime(career["end_time"], errors="coerce")
    career_sorted = career.sort_values(["user_id", "start_dt"]).reset_index(drop=True)
    career_errors = []
    for user_id, group in career_sorted.groupby("user_id", sort=False):
        records = list(group.itertuples(index=False))
        for idx, row in enumerate(records):
            fail_if(not (6 <= int(row.duration_months) <= 60), f"Career {row.event_id}: duration_months outside 6..60", errors)
            if idx == 0:
                fail_if(not pd.isna(row.previous_seniority_level) or not pd.isna(row.previous_occupation_id),
                        f"Career {row.event_id}: first event previous fields must be null", errors)
                fail_if(int(row.delta_seniority) != 0 or row.transition_type != "first_job",
                        f"Career {row.event_id}: first event must have delta=0 and transition_type=first_job", errors)
            else:
                prev = records[idx-1]
                expected_delta = int(row.seniority_level) - int(prev.seniority_level)
                fail_if(int(row.previous_seniority_level) != int(prev.seniority_level),
                        f"Career {row.event_id}: previous seniority is inconsistent", errors)
                fail_if(str(row.previous_occupation_id) != str(prev.occupation_id),
                        f"Career {row.event_id}: previous occupation is inconsistent", errors)
                fail_if(int(row.delta_seniority) != expected_delta,
                        f"Career {row.event_id}: delta_seniority is inconsistent", errors)
                expected_type = "promotion" if expected_delta > 0 else ("demotion" if expected_delta < 0 else "lateral")
                fail_if(row.transition_type != expected_type,
                        f"Career {row.event_id}: transition_type does not match delta_seniority", errors)
                fail_if(pd.Timestamp(prev.end_time) != pd.Timestamp(row.start_time),
                        f"Career {prev.event_id}: end_time must equal next start_time", errors)
            fail_if(int(row.seniority_level) not in range(1, 8), f"Career {row.event_id}: seniority outside 1..7", errors)
        if records:
            fail_if(not pd.isna(records[-1].end_time), f"Career final event for {user_id} must have null end_time", errors)
    fail_if(not set(users["seniority_level"].astype(int)).issubset(set(range(1, 8))), "User seniority_level not confined to 1..7", errors)
    occ_to_fn = dict(zip(occupations["occupation_id"], occupations["function_id"]))
    for row in career.itertuples(index=False):
        fail_if(occ_to_fn.get(row.occupation_id) != row.function_id, f"Career {row.event_id}: occupation/function mismatch", errors)

    # Temporal event composition.
    event_counts = temporal["event_type"].value_counts().to_dict()
    expected_event_counts = {"career": 2871, "relation": 5612, "interest_like": 3336, "hyperedge_creation": 642}
    fail_if(event_counts != expected_event_counts, f"Temporal event composition mismatch: {event_counts}", errors)

    # Graph/item baseline aliases should exactly preserve the corresponding relations/likes.
    graph_pairs = set(zip(graph_edges["source_user_id"], graph_edges["target_user_id"]))
    relation_pairs = set(zip(relations["user_id_1"], relations["user_id_2"]))
    fail_if(graph_pairs != relation_pairs, "graph_edges.csv does not exactly match relation pairs", errors)
    item_pairs = set(zip(interest_edges["user_id"], interest_edges["interest_title_id"]))
    like_pairs = set(zip(likes["user_id"], likes["interest_title_id"]))
    fail_if(item_pairs != like_pairs, "user_interest_edges.csv does not match interest likes", errors)

    # Event-time pair features are recomputed from only earlier timestamps to guard against future leakage.
    context_map = context.set_index("relation_id")
    history: dict[str, set[str]] = defaultdict(set)
    for stamp, group in relations.sort_values(["timestamp", "relation_id"]).groupby("timestamp", sort=True):
        pending = []
        for edge in group.itertuples(index=False):
            feat = context_map.loc[edge.relation_id]
            mutual = len(history[edge.user_id_1].intersection(history[edge.user_id_2]))
            fail_if(int(feat["degree_1_pre_event"]) != len(history[edge.user_id_1]) or
                    int(feat["degree_2_pre_event"]) != len(history[edge.user_id_2]) or
                    int(feat["mutual_connections_pre_event"]) != mutual,
                    f"Pair context for {edge.relation_id} contains future or inconsistent graph features", errors)
            pending.append((edge.user_id_1, edge.user_id_2))
        for a, b in pending:
            history[a].add(b)
            history[b].add(a)
    fail_if(not context["supply_chain_compatibility"].between(0, 1).all(), "Supply-chain compatibility outside [0,1]", errors)
    fail_if(not context["signed_counterfactual_sensitivity"].between(-0.4, 0.4).all(), "Signed counterfactual proxy outside [-0.4,0.4]", errors)
    fail_if(not ratings["rating_target"].between(1, 5).all(), "Synthetic rating targets must lie in [1,5]", errors)
    rating_split_counts = ratings["split"].value_counts().to_dict()
    fail_if(rating_split_counts != {"train": 3500, "val": 1300, "test": 812}, f"Rating target split counts mismatch: {rating_split_counts}", errors)

    # Cleaned data should have no duplicate relation rows; raw artifact should approximate 1.7% duplicates.
    raw_path = RAW / "user_user_relations_raw.csv"
    raw_duplicate_rate = None
    if not raw_path.exists():
        errors.append("Missing raw duplicate artifact raw/user_user_relations_raw.csv")
    else:
        raw = pd.read_csv(raw_path, low_memory=False)
        raw_duplicate_rate = float(raw.duplicated().sum() / len(raw))
        fail_if(abs(raw_duplicate_rate - 0.017) > 0.001, f"Raw duplicate rate {raw_duplicate_rate:.5%} is not approximately 1.7%", errors)
        fail_if(len(raw) != 5709, f"Raw duplicate artifact expected 5,709 rows, found {len(raw)}", errors)
        fail_if(relations.duplicated().any(), "Cleaned relation file contains duplicate full rows", errors)

    # Sample CSVs exist and contain no more than 20 rows per generated table.
    sample_dir = ROOT / "sample_data"
    for filename in actual_csvs:
        sample_path = sample_dir / filename
        fail_if(not sample_path.exists(), f"Missing sample_data/{filename}", errors)
        if sample_path.exists():
            sample = pd.read_csv(sample_path, low_memory=False)
            fail_if(len(sample) != min(20, len(tables[filename])), f"sample_data/{filename} should have first 20 rows", errors)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "record_counts": {name: int(len(df)) for name, df in tables.items()},
        "graph": {"average_degree": average_degree, "density": density, "min_degree": int(degree.min()), "max_degree": int(degree.max())},
        "user_interest_density": interest_density,
        "hyperedges_per_user": float(len(hyperedges)/len(users)),
        "cold_users": int(len(cold_ids)),
        "split_relation_counts": {"train": int(len(rel_train)), "val": int(len(rel_val)), "test": int(len(rel_test))},
        "flat_negative_counts": {"train": int(len(train_neg)), "val": int(len(val_neg)), "test": int(len(test_neg))},
        "temporal_event_counts": event_counts,
        "raw_duplicate_rate": raw_duplicate_rate,
        "global_timestamp_min": global_min.isoformat() if global_min is not None else None,
        "global_timestamp_max": global_max.isoformat() if global_max is not None else None,
        "timestamp_bounds_by_file": time_bounds,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} issue(s)")
        for err in errors[:100]:
            print(" -", err)
        raise SystemExit(1)
    print("VALIDATION PASSED")
    print(json.dumps({k: report[k] for k in ["record_counts", "graph", "user_interest_density", "hyperedges_per_user", "cold_users", "split_relation_counts", "flat_negative_counts", "temporal_event_counts", "raw_duplicate_rate", "global_timestamp_min", "global_timestamp_max"]}, indent=2))
    return report


if __name__ == "__main__":
    validate()
