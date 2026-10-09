#!/usr/bin/env python3
"""Generate a fully synthetic, reproducible SC-CaTH professional-network dataset.

No Faker, network access, real people, organizations, or scraped data are used.
Run: python generate_dataset.py
"""
from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 1404
N_COMPANIES = 318
N_INDUSTRIES = 24
N_FUNCTIONS = 18
N_OCCUPATIONS = 96
N_SKILLS = 240
N_UNIVERSITIES = 60
N_LOCATIONS = 36
N_INTEREST_TITLES = 9891
N_RELATIONS = 5612
N_INTEREST_LIKES = 3336
N_HYPEREDGES = 7020
N_CAREER_EVENTS = 2871
N_TEMPORAL_EVENTS = 12461
N_COLD = 220
N_WARM = N_USERS - N_COLD
START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2025-03-01 00:00:00")
T1 = pd.Timestamp("2024-05-01 00:00:00")  # month 16 from START
T2 = pd.Timestamp("2024-11-01 00:00:00")  # month 22 from START
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SAMPLES = ROOT / "sample_data"
RAW = ROOT / "raw"
REPORTS = ROOT / "reports"

ROLE_NAMES = [
    "Procurement", "Logistics", "Manufacturing", "Supplier Management",
    "SCM", "Distribution", "Retail",
]
STAGES = ["Upstream", "Midstream", "Downstream"]
PATHS = [f"Path_{chr(65+i)}" for i in range(12)]
COMPANY_SIZES = ["Micro", "Small", "Medium", "Large", "Enterprise"]
COMPANY_SIZE_P = np.array([0.42, 0.30, 0.16, 0.08, 0.04], dtype=float)
EDUCATION = ["Level_A", "Level_B", "Level_C", "Level_D", "Level_E"]
FIELDS = [f"Field_{chr(65+i)}" for i in range(16)]
ORG_TYPES = [f"OrgType_{chr(65+i)}" for i in range(8)]
REL_TYPES = ["colleague", "collaborator", "acquaintance"]
TRANSITION_TYPES = ["promotion", "lateral", "demotion", "first_job"]

# Synthetic role-complementarity dictionary. These are abstract domain categories,
# not records derived from any real company or person.
COMPLEMENTARY_ROLE_PAIRS = {
    frozenset(("Procurement", "Supplier Management")),
    frozenset(("Procurement", "Manufacturing")),
    frozenset(("Logistics", "Distribution")),
    frozenset(("Manufacturing", "Distribution")),
    frozenset(("SCM", "Procurement")),
    frozenset(("SCM", "Logistics")),
    frozenset(("Retail", "Distribution")),
    frozenset(("Supplier Management", "Manufacturing")),
    frozenset(("SCM", "Supplier Management")),
}


def uid(i: int) -> str:
    return f"U{i:04d}"


def cid(i: int) -> str:
    return f"C{i:03d}"


def iid(i: int) -> str:
    return f"I{i:03d}"


def fid(i: int) -> str:
    return f"F{i:03d}"


def oid(i: int) -> str:
    return f"O{i:03d}"


def sid(i: int) -> str:
    return f"S{i:03d}"


def univid(i: int) -> str:
    return f"V{i:03d}"


def locid(i: int) -> str:
    return f"L{i:03d}"


def hid(i: int) -> str:
    return f"H{i:05d}"


def ts_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")


def month_start(month_index: int) -> pd.Timestamp:
    return START + pd.DateOffset(months=int(month_index))


def random_timestamp(rng: np.random.Generator, month_index: int,
                      min_day: int = 1, max_day: int = 28) -> pd.Timestamp:
    """Random timestamp in a month, bounded by the agreed 26-month window."""
    base = month_start(month_index)
    last_day = min(max_day, int((base + pd.offsets.MonthEnd(0)).day))
    first_day = max(1, min_day)
    if first_day > last_day:
        first_day = last_day
    day = int(rng.integers(first_day, last_day + 1))
    hour = int(rng.integers(0, 24))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    stamp = base.replace(day=day, hour=hour, minute=minute, second=second)
    if stamp < START:
        stamp = START
    if stamp > END:
        stamp = END
    return stamp


def ordered_month_timestamp(month_index: int, rank: int, count: int,
                            min_day: int = 1, max_day: int = 28) -> pd.Timestamp:
    """Create nondecreasing timestamps for a chronologically generated edge batch."""
    base = month_start(month_index)
    last_day = min(max_day, int((base + pd.offsets.MonthEnd(0)).day))
    first_day = max(1, min_day)
    usable_days = max(1, last_day - first_day + 1)
    count = max(1, int(count))
    rank = int(rank)
    day_offset = min(usable_days - 1, (rank * usable_days) // count)
    day = first_day + day_offset
    # Seconds rise within each day's rank range; day_offset is monotone, so timestamps are too.
    within_day_rank = rank - math.floor(day_offset * count / usable_days)
    seconds = max(0, within_day_rank)
    stamp = base.replace(day=day, hour=0, minute=0, second=min(seconds, 86399))
    return min(max(stamp, START), END)


def write_csv(frame: pd.DataFrame, filename: str, index: bool = False) -> None:
    frame.to_csv(DATA / filename, index=index, encoding="utf-8", na_rep="")


def pack_ids(values: list[str] | tuple[str, ...] | np.ndarray) -> str:
    return "|".join(str(x) for x in values)


def build_schema() -> dict[str, Any]:
    """Machine-readable data dictionary. All IDs are synthetic opaque identifiers."""
    specs: dict[str, dict[str, Any]] = {
        "users.csv": {
            "description": "Baseline professional profile as of the beginning of the observation window; not a final/current snapshot.",
            "primary_key": ["user_id"],
            "fields": {
                "user_id": ("string", False, "Unique synthetic user identifier U0001..."),
                "experience_years": ("float", True, "Synthetic professional experience at baseline; 2% deliberately null."),
                "industry_id": ("string", False, "FK to industries.industry_id."),
                "function_id": ("string", False, "FK to functions.function_id."),
                "occupation_id": ("string", False, "FK to occupations.occupation_id."),
                "seniority_level": ("integer", False, "Synthetic ordinal level 1..7."),
                "education_level": ("string", False, "Synthetic category; unknown represents unrecorded information."),
                "field_of_study": ("string", False, "Synthetic category; unknown represents unrecorded information."),
                "university_id": ("string", False, "FK to universities; V_UNKNOWN is a valid synthetic lookup row."),
                "company_id": ("string", False, "FK to companies; company at baseline."),
                "company_size": ("string", False, "Baseline company-size category or unknown token."),
                "geographic_region_id": ("string", False, "FK to locations; synthetic region."),
                "supply_chain_role": ("string", False, "One of seven abstract supply-chain roles."),
                "supply_chain_stage": ("string", False, "Upstream, Midstream, or Downstream."),
                "supply_chain_path": ("string", False, "Synthetic path category."),
                "profile_text": ("string", True, "Synthetic baseline text built from tokens/IDs; 2% null."),
                "occupation_text": ("string", True, "Synthetic occupation text; 2% null."),
                "skills_text": ("string", True, "Baseline skill IDs separated by |; 2% null."),
                "is_cold": ("boolean", False, "True when user has fewer than 5 user-user relation events at or before t1; split metadata, not a model feature."),
                "first_interaction_time": ("timestamp", True, "First relation/interest-like timestamp over the full window; audit-only and future-aware, exclude as a model feature."),
                "pre_t1_interaction_count": ("integer", False, "Count of user-user relation events at or before t1; used to verify cold-start membership."),
            },
            "foreign_keys": [
                ["industry_id", "industries.csv", "industry_id"], ["function_id", "functions.csv", "function_id"],
                ["occupation_id", "occupations.csv", "occupation_id"], ["university_id", "universities.csv", "university_id"],
                ["company_id", "companies.csv", "company_id"], ["geographic_region_id", "locations.csv", "geographic_region_id"],
            ],
        },
        "companies.csv": {
            "description": "Synthetic organizations; sizes follow a heavy-tailed categorical distribution.",
            "primary_key": ["company_id"],
            "fields": {
                "company_id": ("string", False, "Unique synthetic company ID."),
                "industry_id": ("string", False, "FK to industries."),
                "company_size": ("string", False, "Synthetic company-size class."),
                "organization_type": ("string", False, "Synthetic organization-type token."),
                "geographic_region_id": ("string", False, "FK to locations."),
                "company_age_months": ("integer", False, "Synthetic organization age in months; not an individual's age."),
            },
            "foreign_keys": [["industry_id", "industries.csv", "industry_id"], ["geographic_region_id", "locations.csv", "geographic_region_id"]],
        },
        "industries.csv": {
            "description": "24 synthetic industry categories.", "primary_key": ["industry_id"],
            "fields": {"industry_id": ("string", False, "Synthetic industry ID."), "industry_name": ("string", False, "Tokenized synthetic label."), "supply_chain_relevance": ("integer", False, "Binary indicator 0/1 for synthetic scenario design.")}, "foreign_keys": []},
        "functions.csv": {
            "description": "18 synthetic professional-function categories.", "primary_key": ["function_id"],
            "fields": {"function_id": ("string", False, "Synthetic function ID."), "function_name": ("string", False, "Tokenized synthetic label.")}, "foreign_keys": []},
        "occupations.csv": {
            "description": "96 synthetic occupations; each maps to one professional function.", "primary_key": ["occupation_id"],
            "fields": {"occupation_id": ("string", False, "Synthetic occupation ID."), "occupation_name": ("string", False, "Tokenized synthetic label."), "function_id": ("string", False, "FK to functions."), "seniority_scale_min": ("integer", False, "Synthetic minimum level; all are 1."), "seniority_scale_max": ("integer", False, "Synthetic maximum level; all are 7.")}, "foreign_keys": [["function_id", "functions.csv", "function_id"]]},
        "skills.csv": {
            "description": "Synthetic skill vocabulary organized into semantic clusters.", "primary_key": ["skill_id"],
            "fields": {"skill_id": ("string", False, "Synthetic skill ID."), "skill_name": ("string", False, "Synthetic skill token."), "skill_cluster": ("string", False, "Synthetic cluster identifier.")}, "foreign_keys": []},
        "universities.csv": {
            "description": "Synthetic educational institution lookup including a valid unknown token.", "primary_key": ["university_id"],
            "fields": {"university_id": ("string", False, "Synthetic institution ID."), "university_name": ("string", False, "Synthetic tokenized label.")}, "foreign_keys": []},
        "locations.csv": {
            "description": "Synthetic geographic region lookup; no real addresses or coordinates.", "primary_key": ["geographic_region_id"],
            "fields": {"geographic_region_id": ("string", False, "Synthetic region ID."), "region_name": ("string", False, "Synthetic region token.")}, "foreign_keys": []},
        "user_skills.csv": {
            "description": "Timestamped skill records. Filter by timestamp <= decision time.", "primary_key": ["user_id", "skill_id", "timestamp"],
            "fields": {"user_id": ("string", False, "FK to users."), "skill_id": ("string", False, "FK to skills."), "proficiency": ("integer", False, "Synthetic level 1..5."), "timestamp": ("timestamp", False, "Skill-record timestamp within the observation window.")},
            "foreign_keys": [["user_id", "users.csv", "user_id"], ["skill_id", "skills.csv", "skill_id"]]},
        "career_events.csv": {
            "description": "Chronological synthetic career-position records. First row per user matches the baseline profile; subsequent records are changes. end_time/previous fields are structurally null for active/first records.", "primary_key": ["event_id"],
            "fields": {"event_id": ("string", False, "Unique career event ID."), "user_id": ("string", False, "FK to users."), "company_id": ("string", False, "FK to companies."), "industry_id": ("string", False, "FK to industries."), "function_id": ("string", False, "FK to functions."), "occupation_id": ("string", False, "FK to occupations."), "skills": ("string", False, "Pipe-separated synthetic skill IDs."), "seniority_level": ("integer", False, "Current level 1..7."), "previous_seniority_level": ("integer", True, "Previous level; structurally null for first trajectory event."), "previous_occupation_id": ("string", True, "Previous occupation; structurally null for first trajectory event."), "start_time": ("timestamp", False, "Career-record/position start marker within the observation window."), "end_time": ("timestamp", True, "Next career event timestamp, or null for final active position."), "duration_months": ("integer", False, "Synthetic position tenure 6..60; final active tenure can include time before the observation window."), "delta_seniority": ("integer", False, "Current minus previous seniority; 0 for first record."), "transition_type": ("string", False, "first_job for first record, otherwise promotion/lateral/demotion consistent with delta.")},
            "foreign_keys": [["user_id", "users.csv", "user_id"], ["company_id", "companies.csv", "company_id"], ["industry_id", "industries.csv", "industry_id"], ["function_id", "functions.csv", "function_id"], ["occupation_id", "occupations.csv", "occupation_id"], ["previous_occupation_id", "occupations.csv", "occupation_id"]]},
        "user_user_relations.csv": {
            "description": "Unique undirected positive user-user relations stored exactly once with user_id_1 < user_id_2.", "primary_key": ["relation_id"],
            "fields": {"relation_id": ("string", False, "Unique relation ID."), "user_id_1": ("string", False, "First endpoint, lexicographically smaller ID."), "user_id_2": ("string", False, "Second endpoint, lexicographically larger ID."), "timestamp": ("timestamp", False, "Relation formation timestamp."), "relation_type": ("string", False, "Synthetic relation category."), "is_reciprocal": ("integer", False, "Always 1; undirected relation semantics."), "interaction_strength": ("float", False, "Synthetic strength in [0,1].")},
            "foreign_keys": [["user_id_1", "users.csv", "user_id"], ["user_id_2", "users.csv", "user_id"]]},
        "interest_titles.csv": {
            "description": "9,891 synthetic interest/topic titles.", "primary_key": ["interest_title_id"],
            "fields": {"interest_title_id": ("string", False, "Unique synthetic title ID."), "title_text": ("string", False, "Synthetic token sequence."), "category": ("string", False, "Synthetic category token."), "related_skills": ("string", False, "Pipe-separated related skill IDs.")}, "foreign_keys": []},
        "user_interest_interactions.csv": {
            "description": "Synthetic user-to-title likes, timestamped and unique by user/title pair.", "primary_key": ["interaction_id"],
            "fields": {"interaction_id": ("string", False, "Unique interest interaction ID."), "user_id": ("string", False, "FK to users."), "interest_title_id": ("string", False, "FK to interest_titles."), "timestamp": ("timestamp", False, "Interaction timestamp."), "interaction_type": ("string", False, "Always like.")},
            "foreign_keys": [["user_id", "users.csv", "user_id"], ["interest_title_id", "interest_titles.csv", "interest_title_id"]]},
        "hyperedges.csv": {
            "description": "Exactly five timestamped user-centred hyperedges per user. Career and skills context is reconstructed as of the hyperedge timestamp.", "primary_key": ["hyperedge_id"],
            "fields": {"hyperedge_id": ("string", False, "Unique hyperedge ID."), "user_id": ("string", False, "FK to users."), "company_id": ("string", False, "FK to companies."), "industry_id": ("string", False, "FK to industries."), "function_id": ("string", False, "FK to functions."), "occupation_id": ("string", False, "FK to occupations."), "skill_ids": ("string", False, "Pipe-separated skill IDs."), "timestamp": ("timestamp", False, "Hyperedge creation timestamp."), "relation_type": ("string", False, "Synthetic higher-order relation category."), "supply_chain_specific": ("integer", False, "Binary flag; exactly 40% are 1."), "sc_role": ("string", True, "Supply-chain role; structurally null if supply_chain_specific=0."), "sc_stage": ("string", True, "Supply-chain stage; structurally null if supply_chain_specific=0."), "sc_path": ("string", True, "Supply-chain path; structurally null if supply_chain_specific=0.")},
            "foreign_keys": [["user_id", "users.csv", "user_id"], ["company_id", "companies.csv", "company_id"], ["industry_id", "industries.csv", "industry_id"], ["function_id", "functions.csv", "function_id"], ["occupation_id", "occupations.csv", "occupation_id"]]},
        "temporal_events.csv": {
            "description": "Exactly 12,461 events: all career, relation and interest-like events plus 642 sampled hyperedge-creation events.", "primary_key": ["event_id"],
            "fields": {"event_id": ("string", False, "Unique temporal event ID."), "event_type": ("string", False, "career/relation/interest_like/hyperedge_creation."), "user_id": ("string", False, "FK to users."), "target_id": ("string", False, "Type-dependent synthetic target ID."), "timestamp": ("timestamp", False, "Source-event timestamp in the observation window."), "metadata": ("string", False, "Compact JSON object with synthetic IDs/categories only.")},
            "foreign_keys": [["user_id", "users.csv", "user_id"]]},
        "train_pairs.csv": {
            "description": "Positive training pairs and four uniformly sampled non-positive candidate IDs per positive. Training pairs contain warm users only.", "primary_key": ["anchor_user_id", "positive_user_id"],
            "fields": {"anchor_user_id": ("string", False, "Ranking query/anchor."), "positive_user_id": ("string", False, "Observed positive counterpart."), "negative_user_ids": ("string", False, "Four unique pipe-separated non-positive user IDs."), "split_time": ("timestamp", False, "Positive relation timestamp."), "label": ("integer", False, "Always 1 on positive pair rows.")}, "foreign_keys": []},
        "val_pairs.csv": {
            "description": "Positive validation pairs and 99 uniformly sampled non-positive candidates; warm users only.", "primary_key": ["anchor_user_id", "positive_user_id"],
            "fields": {"anchor_user_id": ("string", False, "Ranking anchor."), "positive_user_id": ("string", False, "Observed positive counterpart."), "negative_user_ids": ("string", False, "99 unique pipe-separated candidate IDs."), "split_time": ("timestamp", False, "Positive relation timestamp."), "label": ("integer", False, "Always 1 on positive pair rows.")}, "foreign_keys": []},
        "test_pairs.csv": {
            "description": "Positive test pairs and 99 sampled non-positive candidates; includes inductive cold users that are absent from train/validation pairs.", "primary_key": ["anchor_user_id", "positive_user_id"],
            "fields": {"anchor_user_id": ("string", False, "Ranking anchor; cold users are anchors on designated cold-test edges."), "positive_user_id": ("string", False, "Observed positive counterpart."), "negative_user_ids": ("string", False, "99 unique pipe-separated candidate IDs."), "split_time": ("timestamp", False, "Positive relation timestamp."), "label": ("integer", False, "Always 1 on positive pair rows.")}, "foreign_keys": []},
        "train_negative_samples.csv": {
            "description": "Flat training non-positive samples: four per positive relation.", "primary_key": ["pair_id", "sample_index"],
            "fields": {"pair_id": ("string", False, "Synthetic pair reference P_train_..."), "anchor_user_id": ("string", False, "Ranking anchor."), "positive_user_id": ("string", False, "Positive counterpart for context."), "non_positive_user_id": ("string", False, "Sampled non-positive candidate."), "sample_index": ("integer", False, "1..4 within pair."), "split_time": ("timestamp", False, "Positive relation time."), "label": ("integer", False, "Always 0.")}, "foreign_keys": []},
        "val_negative_samples.csv": {
            "description": "Flat validation non-positive samples: 99 per positive relation.", "primary_key": ["pair_id", "sample_index"],
            "fields": {"pair_id": ("string", False, "Synthetic pair reference P_val_..."), "anchor_user_id": ("string", False, "Ranking anchor."), "positive_user_id": ("string", False, "Positive counterpart for context."), "non_positive_user_id": ("string", False, "Sampled non-positive candidate."), "sample_index": ("integer", False, "1..99 within pair."), "split_time": ("timestamp", False, "Positive relation time."), "label": ("integer", False, "Always 0.")}, "foreign_keys": []},
        "test_negative_samples.csv": {
            "description": "Flat test non-positive samples: 99 per positive relation.", "primary_key": ["pair_id", "sample_index"],
            "fields": {"pair_id": ("string", False, "Synthetic pair reference P_test_..."), "anchor_user_id": ("string", False, "Ranking anchor."), "positive_user_id": ("string", False, "Positive counterpart for context."), "non_positive_user_id": ("string", False, "Sampled non-positive candidate."), "sample_index": ("integer", False, "1..99 within pair."), "split_time": ("timestamp", False, "Positive relation time."), "label": ("integer", False, "Always 0.")}, "foreign_keys": []},
        "graph_edges.csv": {
            "description": "Undirected graph edges stored once; construct an undirected graph or duplicate each row in memory if a framework requires directed message-passing arcs.", "primary_key": ["source_user_id", "target_user_id"],
            "fields": {"source_user_id": ("string", False, "Lexicographically smaller endpoint."), "target_user_id": ("string", False, "Lexicographically larger endpoint."), "timestamp": ("timestamp", False, "Edge formation timestamp."), "edge_type": ("string", False, "Synthetic relation type.")},
            "foreign_keys": [["source_user_id", "users.csv", "user_id"], ["target_user_id", "users.csv", "user_id"]]},
        "user_interest_edges.csv": {
            "description": "Auxiliary user-to-interest bipartite edges; time-filter before a decision timestamp.", "primary_key": ["user_id", "interest_title_id"],
            "fields": {"user_id": ("string", False, "FK to users."), "interest_title_id": ("string", False, "FK to interest_titles."), "timestamp": ("timestamp", False, "Like timestamp.")},
            "foreign_keys": [["user_id", "users.csv", "user_id"], ["interest_title_id", "interest_titles.csv", "interest_title_id"]]},
        "pair_context_features.csv": {
            "description": "Leakage-safe, event-time pair context for observed edges. Graph degree and mutual connections are computed strictly from relation timestamps earlier than the edge timestamp. Counterfactual sensitivity is a synthetic proxy, not a causal estimate.", "primary_key": ["relation_id"],
            "fields": {"relation_id": ("string", False, "FK to user_user_relations."), "user_id_1": ("string", False, "Pair endpoint 1."), "user_id_2": ("string", False, "Pair endpoint 2."), "timestamp": ("timestamp", False, "Observed edge time."), "degree_1_pre_event": ("integer", False, "Degree in graph strictly before event timestamp."), "degree_2_pre_event": ("integer", False, "Degree in graph strictly before event timestamp."), "mutual_connections_pre_event": ("integer", False, "Common neighbors strictly before event timestamp."), "same_company_baseline": ("integer", False, "Same synthetic baseline company flag."), "same_industry_baseline": ("integer", False, "Same synthetic baseline industry flag."), "same_function_baseline": ("integer", False, "Same synthetic baseline function flag."), "seniority_gap_baseline": ("integer", False, "Absolute baseline seniority difference."), "supply_chain_compatibility": ("float", False, "Synthetic pair compatibility proxy in [0,1]."), "signed_counterfactual_sensitivity": ("float", False, "Synthetic signed delta proxy in [-0.4,0.4], not a causal effect."), "counterfactual_intervention": ("string", False, "Synthetic intervention label: remove_supply_chain_context.")},
            "foreign_keys": [["relation_id", "user_user_relations.csv", "relation_id"], ["user_id_1", "users.csv", "user_id"], ["user_id_2", "users.csv", "user_id"]]},
        "pair_rating_targets.csv": {
            "description": "Synthetic 1..5 affinity targets attached to observed relation pairs for reproducible RMSE/MAE experiments. They are simulated labels, not real user ratings; non-observed pairs remain non-positive rather than assigned a low rating.", "primary_key": ["relation_id"],
            "fields": {"relation_id": ("string", False, "FK to user_user_relations."), "user_id_1": ("string", False, "First undirected pair endpoint."), "user_id_2": ("string", False, "Second undirected pair endpoint."), "timestamp": ("timestamp", False, "Relation event timestamp."), "rating_target": ("float", False, "Synthetic ordinal-like affinity target in [1,5], rounded to two decimals."), "split": ("string", False, "Temporal split: train/val/test."), "rating_source": ("string", False, "Always synthetic_latent_affinity.")},
            "foreign_keys": [["relation_id", "user_user_relations.csv", "relation_id"], ["user_id_1", "users.csv", "user_id"], ["user_id_2", "users.csv", "user_id"]]},
        "cold_start_users.csv": {
            "description": "Cold users reserved for the inductive test split; no user-user relations or interest likes before t1.", "primary_key": ["user_id"],
            "fields": {"user_id": ("string", False, "FK to users."), "pre_t1_relation_count": ("integer", False, "Must be zero for designated cold test users."), "reserved_split": ("string", False, "Always test."), "cold_start_reason": ("string", False, "Synthetic split rule." )},
            "foreign_keys": [["user_id", "users.csv", "user_id"]]},
    }
    schema = {
        "dataset_name": "SC-CaTH Synthetic Professional Network",
        "version": "1.0.0",
        "synthetic_only": True,
        "seed": SEED,
        "time_window": {"start": ts_str(START), "end": ts_str(pd.Timestamp("2025-03-01 00:00:00")), "t1": ts_str(T1), "t2": ts_str(T2), "split_rule": "train: timestamp <= t1; validation: t1 < timestamp <= t2; test: timestamp > t2"},
        "structural_null_fields": {"career_events.csv": ["end_time", "previous_seniority_level", "previous_occupation_id"], "hyperedges.csv": ["sc_role", "sc_stage", "sc_path"]},
        "files": {},
    }
    for filename, spec in specs.items():
        fields = {}
        for field_name, (dtype, nullable, description) in spec["fields"].items():
            fields[field_name] = {"type": dtype, "nullable": nullable, "description": description}
        schema["files"][filename] = {
            "description": spec["description"], "primary_key": spec["primary_key"],
            "fields": fields, "foreign_keys": [
                {"field": f, "references_file": ref_file, "references_field": ref_field}
                for f, ref_file, ref_field in spec["foreign_keys"]
            ],
        }
    # Lookup schemas are completed here after their table specifications are included above.
    return schema


def create_lookups(rng: np.random.Generator) -> tuple[pd.DataFrame, ...]:
    industries = pd.DataFrame({
        "industry_id": [iid(i) for i in range(1, N_INDUSTRIES + 1)],
        "industry_name": [f"Industry_{chr(64+i)}" for i in range(1, N_INDUSTRIES + 1)],
        "supply_chain_relevance": [1 if i <= 17 else 0 for i in range(1, N_INDUSTRIES + 1)],
    })
    functions = pd.DataFrame({
        "function_id": [fid(i) for i in range(1, N_FUNCTIONS + 1)],
        "function_name": [f"Function_{chr(64+i)}" for i in range(1, N_FUNCTIONS + 1)],
    })
    occupations = pd.DataFrame([
        {"occupation_id": oid(i), "occupation_name": f"Occupation_{i:03d}",
         "function_id": fid(((((i - 1) * N_FUNCTIONS) // N_OCCUPATIONS) + 1)),
         "seniority_scale_min": 1, "seniority_scale_max": 7}
        for i in range(1, N_OCCUPATIONS + 1)
    ])
    skills = pd.DataFrame({
        "skill_id": [sid(i) for i in range(1, N_SKILLS + 1)],
        "skill_name": [f"Skill_{i:03d}" for i in range(1, N_SKILLS + 1)],
        "skill_cluster": [f"SkillCluster_{((i-1)//10)+1:02d}" for i in range(1, N_SKILLS + 1)],
    })
    universities = pd.DataFrame({
        "university_id": [univid(i) for i in range(1, N_UNIVERSITIES + 1)] + ["V_UNKNOWN"],
        "university_name": [f"UniversityToken_{i:03d}" for i in range(1, N_UNIVERSITIES + 1)] + ["unknown"],
    })
    locations = pd.DataFrame({
        "geographic_region_id": [locid(i) for i in range(1, N_LOCATIONS + 1)],
        "region_name": [f"RegionToken_{i:02d}" for i in range(1, N_LOCATIONS + 1)],
    })
    # Heavy-tailed organization sizes; industry is balanced with controlled variation.
    company_industry = np.resize(np.arange(1, N_INDUSTRIES + 1), N_COMPANIES)
    rng.shuffle(company_industry)
    company_sizes = rng.choice(COMPANY_SIZES, size=N_COMPANIES, p=COMPANY_SIZE_P)
    company_rows = []
    for i in range(1, N_COMPANIES + 1):
        company_rows.append({
            "company_id": cid(i), "industry_id": iid(int(company_industry[i-1])),
            "company_size": str(company_sizes[i-1]),
            "organization_type": str(rng.choice(ORG_TYPES)),
            "geographic_region_id": locid(int(rng.integers(1, N_LOCATIONS + 1))),
            "company_age_months": int(np.clip(rng.lognormal(mean=3.8, sigma=0.8), 1, 720)),
        })
    companies = pd.DataFrame(company_rows)
    return industries, functions, occupations, skills, universities, locations, companies


def choose_company_for_industry(rng: np.random.Generator, company_ids_by_industry: dict[str, list[str]], industry_id: str) -> str:
    choices = company_ids_by_industry[industry_id]
    return str(rng.choice(choices))


def make_user_profiles(rng: np.random.Generator, industries: pd.DataFrame, functions: pd.DataFrame,
                       occupations: pd.DataFrame, skills: pd.DataFrame, universities: pd.DataFrame,
                       locations: pd.DataFrame, companies: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    company_ids_by_industry: dict[str, list[str]] = defaultdict(list)
    for row in companies.itertuples(index=False):
        company_ids_by_industry[row.industry_id].append(row.company_id)
    # The balanced company generator gives every industry at least one company.
    all_skill_ids = skills["skill_id"].tolist()
    company_size_lookup = dict(zip(companies["company_id"], companies["company_size"]))
    company_region_lookup = dict(zip(companies["company_id"], companies["geographic_region_id"]))
    company_industry_lookup = dict(zip(companies["company_id"], companies["industry_id"]))
    skill_ids_by_user: dict[str, list[str]] = {}
    all_ids = [uid(i) for i in range(1, N_USERS + 1)]
    rows: list[dict[str, Any]] = []
    for index, user_id in enumerate(all_ids):
        industry_id = iid(int(rng.integers(1, N_INDUSTRIES + 1)))
        function_id = fid(int(rng.integers(1, N_FUNCTIONS + 1)))
        eligible_occ = occupations.loc[occupations["function_id"] == function_id, "occupation_id"].tolist()
        occupation_id = str(rng.choice(eligible_occ))
        company_id = choose_company_for_industry(rng, company_ids_by_industry, industry_id)
        role = str(rng.choice(ROLE_NAMES))
        stage = str(rng.choice(STAGES, p=[0.35, 0.35, 0.30]))
        path = str(rng.choice(PATHS))
        # Each profile has a baseline skill set, known at START; later additions are timestamped separately.
        baseline_skill_count = int(rng.integers(5, 9))
        baseline_skills = [str(x) for x in rng.choice(all_skill_ids, size=baseline_skill_count, replace=False)]
        skill_ids_by_user[user_id] = baseline_skills
        experience = round(float(np.clip(rng.gamma(shape=2.7, scale=3.5), 0.5, 38.0)), 2)
        education = str(rng.choice(EDUCATION + ["unknown"], p=[0.22, 0.29, 0.24, 0.15, 0.07, 0.03]))
        field = str(rng.choice(FIELDS + ["unknown"], p=[*[0.97/len(FIELDS)]*len(FIELDS), 0.03]))
        university = str(rng.choice([univid(i) for i in range(1, N_UNIVERSITIES + 1)] + ["V_UNKNOWN"],
                                    p=[*[0.97/N_UNIVERSITIES]*N_UNIVERSITIES, 0.03]))
        if rng.random() < 0.02:
            experience_value: float | None = np.nan
        else:
            experience_value = experience
        profile_text = (f"synthetic_profile {user_id} ; {industry_id} ; {function_id} ; {occupation_id} ; "
                        f"role_{ROLE_NAMES.index(role)+1:02d} ; stage_{STAGES.index(stage)+1:02d} ; "
                        f"skills {pack_ids(baseline_skills[:4])}")
        occupation_text = f"synthetic_occupation {occupation_id} ; function {function_id} ; seniority_token"
        skills_text = pack_ids(baseline_skills)
        if rng.random() < 0.02:
            profile_text = None
        if rng.random() < 0.02:
            occupation_text = None
        if rng.random() < 0.02:
            skills_text = None
        rows.append({
            "user_id": user_id, "experience_years": experience_value,
            "industry_id": industry_id, "function_id": function_id, "occupation_id": occupation_id,
            "seniority_level": int(rng.integers(1, 8)), "education_level": education,
            "field_of_study": field, "university_id": university, "company_id": company_id,
            "company_size": company_size_lookup[company_id],
            "geographic_region_id": company_region_lookup[company_id],
            "supply_chain_role": role, "supply_chain_stage": stage, "supply_chain_path": path,
            "profile_text": profile_text, "occupation_text": occupation_text, "skills_text": skills_text,
            # Set after relation graph is generated.
            "is_cold": False, "first_interaction_time": None, "pre_t1_interaction_count": 0,
        })
    users = pd.DataFrame(rows)
    # Validate lookup consistency of user baseline industry and company industry by design.
    assert all(company_industry_lookup[r.company_id] == r.industry_id for r in users.itertuples(index=False))
    return users, skill_ids_by_user, company_ids_by_industry


def make_skill_events(rng: np.random.Generator, users: pd.DataFrame,
                      skill_ids_by_user: dict[str, list[str]], skills: pd.DataFrame) -> pd.DataFrame:
    all_skill_ids = skills["skill_id"].tolist()
    rows: list[dict[str, Any]] = []
    interaction_idx = 1
    for user_id in users["user_id"].tolist():
        base_skills = skill_ids_by_user[user_id]
        # Baseline skills are available from START. Subsequent skill records are acquisitions/updates.
        for skill_id in base_skills:
            rows.append({"user_id": user_id, "skill_id": skill_id,
                         "proficiency": int(rng.integers(1, 6)), "timestamp": ts_str(START)})
        extras = int(rng.integers(0, 4))
        remaining = [s for s in all_skill_ids if s not in base_skills]
        if extras:
            for skill_id in rng.choice(remaining, size=extras, replace=False):
                month = int(rng.integers(1, 26))
                stamp = random_timestamp(rng, month)
                rows.append({"user_id": user_id, "skill_id": str(skill_id),
                             "proficiency": int(rng.integers(1, 6)), "timestamp": ts_str(stamp)})
    frame = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "skill_id", "timestamp"])
    return frame.sort_values(["timestamp", "user_id", "skill_id"]).reset_index(drop=True)


def career_count_schedule(rng: np.random.Generator) -> dict[str, int]:
    # Exact distribution: 300*1 + 800*2 + 250*3 + 49*4 + 5*5 = 2,871 events.
    counts = np.array([1] * 300 + [2] * 800 + [3] * 250 + [4] * 49 + [5] * 5, dtype=int)
    assert len(counts) == N_USERS and int(counts.sum()) == N_CAREER_EVENTS
    rng.shuffle(counts)
    return {uid(i): int(counts[i-1]) for i in range(1, N_USERS + 1)}


def schedule_career_months(n: int, rng: np.random.Generator) -> list[int]:
    if n == 1:
        return [0]
    if n == 2:
        first = int(rng.integers(0, 9))
        gap = int(rng.integers(6, min(13, 21-first)))
        return [first, first + gap]
    if n == 3:
        first = int(rng.integers(0, 3))
        gaps = [int(rng.integers(6, 10)), int(rng.integers(6, 10))]
        months = [first, first + gaps[0], first + sum(gaps)]
        if months[-1] > 20:
            months = [0, 7, 14]
        return months
    if n == 4:
        first = int(rng.integers(0, 3))
        gaps = [6, 6, 6]
        return [first, first+6, first+12, first+18]
    return [0, 6, 12, 18, 24]


def make_career_events(rng: np.random.Generator, users: pd.DataFrame,
                       skills_by_user: dict[str, list[str]], companies: pd.DataFrame,
                       occupations: pd.DataFrame, functions: pd.DataFrame,
                       company_ids_by_industry: dict[str, list[str]]) -> pd.DataFrame:
    user_lookup = users.set_index("user_id").to_dict(orient="index")
    company_industry = dict(zip(companies["company_id"], companies["industry_id"]))
    occupation_function = dict(zip(occupations["occupation_id"], occupations["function_id"]))
    occ_by_function = {fn: g["occupation_id"].tolist() for fn, g in occupations.groupby("function_id")}
    counts = career_count_schedule(rng)
    rows: list[dict[str, Any]] = []
    event_num = 1
    for user_id in users["user_id"].tolist():
        n_events = counts[user_id]
        user = user_lookup[user_id]
        months = schedule_career_months(n_events, rng)
        day = int(rng.integers(1, 26))
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        career_state: list[dict[str, Any]] = []
        current_industry = str(user["industry_id"])
        current_function = str(user["function_id"])
        current_occupation = str(user["occupation_id"])
        current_company = str(user["company_id"])
        current_seniority = int(user["seniority_level"])
        for k, month in enumerate(months):
            if k > 0:
                # Some transitions change function/industry/company, while others preserve them.
                if rng.random() < 0.26:
                    current_industry = iid(int(rng.integers(1, N_INDUSTRIES + 1)))
                if rng.random() < 0.34:
                    current_function = fid(int(rng.integers(1, N_FUNCTIONS + 1)))
                eligible = occ_by_function[current_function]
                if rng.random() < 0.70 or current_occupation not in eligible:
                    current_occupation = str(rng.choice(eligible))
                if rng.random() < 0.58:
                    current_company = choose_company_for_industry(rng, company_ids_by_industry, current_industry)
                # Usually promote, often lateral, occasionally demote; clamp to 1..7.
                draw = float(rng.random())
                if draw < 0.62 and current_seniority < 7:
                    current_seniority = int(rng.integers(current_seniority + 1, 8))
                elif draw < 0.90:
                    pass
                elif current_seniority > 1:
                    current_seniority = int(rng.integers(1, current_seniority))
            stamp = month_start(month).replace(day=min(day, int((month_start(month) + pd.offsets.MonthEnd(0)).day)),
                                              hour=hour, minute=minute, second=second)
            previous = career_state[-1] if career_state else None
            previous_seniority = int(previous["seniority_level"]) if previous else None
            previous_occupation = str(previous["occupation_id"]) if previous else None
            delta = 0 if previous is None else current_seniority - previous_seniority
            if previous is None:
                transition = "first_job"
            elif delta > 0:
                transition = "promotion"
            elif delta < 0:
                transition = "demotion"
            else:
                transition = "lateral"
            if k < n_events - 1:
                duration = int(months[k+1] - months[k])
                end_time = month_start(months[k+1]).replace(day=min(day, int((month_start(months[k+1]) + pd.offsets.MonthEnd(0)).day)),
                                                             hour=hour, minute=minute, second=second)
            else:
                duration = int(rng.integers(6, 61))
                end_time = None
            available_skills = skills_by_user[user_id]
            # Career skills evolve deterministically in the sense of using only known baseline skill IDs.
            skill_list = list(available_skills)
            if k > 0 and rng.random() < 0.55:
                extra = str(rng.choice([x for x in skills_by_user[user_id]]))
                if extra not in skill_list:
                    skill_list.append(extra)
            rec = {
                "event_id": f"CE{event_num:05d}", "user_id": user_id,
                "company_id": current_company, "industry_id": company_industry[current_company],
                "function_id": current_function, "occupation_id": current_occupation,
                "skills": pack_ids(skill_list), "seniority_level": int(current_seniority),
                "previous_seniority_level": previous_seniority,
                "previous_occupation_id": previous_occupation,
                "start_time": ts_str(stamp), "end_time": ts_str(end_time),
                "duration_months": duration, "delta_seniority": int(delta), "transition_type": transition,
                "_month_index": int(month),
            }
            # Occupation's function remains consistent by construction.
            assert occupation_function[current_occupation] == current_function
            career_state.append(rec)
            rows.append(rec)
            event_num += 1
    frame = pd.DataFrame(rows)
    assert len(frame) == N_CAREER_EVENTS
    # Remove generator-only month index; all exported timestamps stay in the defined window.
    frame = frame.drop(columns=["_month_index"])
    return frame.sort_values(["user_id", "start_time"]).reset_index(drop=True)


def role_compatibility(role_a: str, role_b: str, stage_a: str, stage_b: str) -> tuple[float, float]:
    pair = frozenset((role_a, role_b))
    if role_a != role_b and pair in COMPLEMENTARY_ROLE_PAIRS and stage_a != stage_b:
        signed_component = 0.80
    elif role_a != role_b and pair in COMPLEMENTARY_ROLE_PAIRS:
        signed_component = 0.60
    elif role_a == role_b:
        signed_component = -0.35
    elif stage_a == stage_b:
        signed_component = -0.12
    else:
        signed_component = -0.25
    compatibility = float(np.clip((signed_component + 1.0) / 2.0, 0.0, 1.0))
    sensitivity = float(np.clip(0.40 * signed_component, -0.4, 0.4))
    return compatibility, sensitivity


def make_pair_candidates(users: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(users)
    ii, jj = np.triu_indices(n, 1)
    u = users.set_index("user_id")
    ind = np.array([int(x[1:]) for x in u.index.to_list()], dtype=int)
    company = u["company_id"].to_numpy()
    industry = u["industry_id"].to_numpy()
    function = u["function_id"].to_numpy()
    seniority = u["seniority_level"].to_numpy(dtype=int)
    role = u["supply_chain_role"].to_numpy()
    stage = u["supply_chain_stage"].to_numpy()
    # Always-positive, interpretable synthetic affinity. Static profile similarities affect formation odds.
    weights = np.ones(len(ii), dtype=np.float64)
    weights += 1.70 * (company[ii] == company[jj])
    weights += 0.85 * (industry[ii] == industry[jj])
    weights += 1.05 * (function[ii] == function[jj])
    weights += 0.25 * (np.abs(seniority[ii] - seniority[jj]) <= 1)
    weights += 0.18 * (stage[ii] != stage[jj])
    comp = np.zeros(len(ii), dtype=bool)
    for pair in COMPLEMENTARY_ROLE_PAIRS:
        a, b = tuple(pair)
        comp |= ((role[ii] == a) & (role[jj] == b)) | ((role[ii] == b) & (role[jj] == a))
    weights += 1.15 * comp
    weights += 0.12 * (role[ii] == role[jj])
    keys = ii.astype(np.int64) * n + jj.astype(np.int64)
    return ii.astype(np.int32), jj.astype(np.int32), weights, keys, comp, ind


def make_seed_backbone(warm_nodes: list[int]) -> list[tuple[int, int]]:
    """Deterministic 5-regular backbone among a randomly assigned warm cohort."""
    n_warm = len(warm_nodes)
    edges: set[tuple[int, int]] = set()
    # User IDs remain opaque: cohort membership is randomly assigned rather than contiguous by ID.
    for offset in (1, 2):
        for k in range(n_warm):
            i, j = warm_nodes[k], warm_nodes[(k + offset) % n_warm]
            edges.add((min(i, j), max(i, j)))
    for k in range(n_warm // 2):
        i, j = warm_nodes[k], warm_nodes[k + n_warm // 2]
        edges.add((min(i, j), max(i, j)))
    assert len(edges) == (2 * n_warm + n_warm // 2)
    degree = {node: 0 for node in warm_nodes}
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    assert all(value == 5 for value in degree.values())
    return sorted(edges)


def make_relations(rng: np.random.Generator, users: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, int], set[str]]:
    ii, jj, weights, keys, _comp, _ind = make_pair_candidates(users)
    n = len(users)
    # IDs and integer positions share lexical order U0001...U1404.
    user_ids = users["user_id"].tolist()
    # Randomly assign the warm/cold cohort to opaque IDs to avoid ID-order leakage.
    node_permutation = rng.permutation(n).tolist()
    warm_nodes = sorted(int(x) for x in node_permutation[:N_WARM])
    cold_nodes = sorted(int(x) for x in node_permutation[N_WARM:])
    used: set[int] = set()
    adjacency: list[set[int]] = [set() for _ in range(n)]
    edge_records: list[dict[str, Any]] = []

    def add_edge(i: int, j: int, stamp: pd.Timestamp, source: str) -> None:
        if i > j:
            i, j = j, i
        key = i * n + j
        if key in used:
            raise RuntimeError("Attempted to add duplicate undirected edge")
        used.add(key)
        adjacency[i].add(j)
        adjacency[j].add(i)
        edge_records.append({"i": i, "j": j, "timestamp": stamp, "source": source})

    # 2,960 backbone edges: degree exactly 5 for all warm nodes; every cold node remains unseen.
    backbone = make_seed_backbone(warm_nodes)
    for rank, (i, j) in enumerate(backbone):
        month = int(rng.integers(0, 11))
        stamp = random_timestamp(rng, month)
        add_edge(i, j, stamp, "backbone")

    warm_mask = np.zeros(n, dtype=bool)
    warm_mask[np.array(warm_nodes, dtype=int)] = True
    warm_pool_mask = warm_mask[ii] & warm_mask[jj]
    warm_pool = np.flatnonzero(warm_pool_mask)
    all_pool = np.arange(len(ii), dtype=np.int64)
    warm_cdf = np.cumsum(weights[warm_pool], dtype=np.float64)
    warm_cdf /= warm_cdf[-1]
    all_cdf = np.cumsum(weights, dtype=np.float64)
    all_cdf /= all_cdf[-1]

    def select_edge(pool: np.ndarray, cdf: np.ndarray) -> tuple[int, int]:
        # Draw proposals by static affinity, then reweight them by already-observed mutual neighbours.
        for _attempt in range(40):
            proposal_pos = np.searchsorted(cdf, rng.random(96), side="left")
            proposal_idx = pool[np.minimum(proposal_pos, len(pool) - 1)]
            valid_idx = []
            for ci in proposal_idx.tolist():
                i, j = int(ii[ci]), int(jj[ci])
                if i * n + j not in used:
                    valid_idx.append(ci)
            if not valid_idx:
                continue
            valid_idx = list(dict.fromkeys(valid_idx))
            mutual = np.array([len(adjacency[int(ii[ci])].intersection(adjacency[int(jj[ci])])) for ci in valid_idx], dtype=float)
            local_weights = weights[np.array(valid_idx, dtype=int)] * (1.0 + 0.30 * np.minimum(mutual, 8.0))
            probabilities = local_weights / local_weights.sum()
            chosen = int(rng.choice(np.array(valid_idx, dtype=int), p=probabilities))
            return int(ii[chosen]), int(jj[chosen])
        # Practically unreachable fallback, but preserves termination and uniqueness.
        for ci in pool[rng.permutation(len(pool))]:
            i, j = int(ii[ci]), int(jj[ci])
            if i * n + j not in used:
                return i, j
        raise RuntimeError("No unused edge candidates remain")

    # Add 540 early edges chronologically after the fixed backbone, preserving causal mutual-neighbour scoring.
    extras_total = 540
    extras_by_month = [108, 108, 108, 108, 108]
    for month, count in zip(range(11, 16), extras_by_month):
        for rank in range(count):
            i, j = select_edge(warm_pool, warm_cdf)
            # month 15 is April 2024; all are <= t1 (2024-05-01).
            stamp = ordered_month_timestamp(month, rank, count, 1, 28)
            add_edge(i, j, stamp, "train_extra")

    # Validation: warm-only; no cold test user appears in train or validation positives.
    val_counts = [217, 217, 217, 217, 216, 216]  # sum 1,300
    for month, count in zip(range(16, 22), val_counts):
        min_day = 2 if month == 16 else 1  # strictly after t1
        for rank in range(count):
            i, j = select_edge(warm_pool, warm_cdf)
            stamp = ordered_month_timestamp(month, rank, count, min_day, 28)
            add_edge(i, j, stamp, "validation")

    # Reserve exactly one positive test edge for each cold user, anchored at the cold user later.
    # Cold users have no relations before the test window, and no likes before t1 (likes are arranged later).
    test_month_counts = {22: 160, 23: 200, 24: 200, 25: 251, 26: 1}
    cold_months = ([22] * 40) + ([23] * 55) + ([24] * 60) + ([25] * 65)
    rng.shuffle(cold_months)
    cold_test_edges_by_month: dict[int, list[tuple[int, int]]] = defaultdict(list)
    cold_partner_for_edge: dict[int, int] = {}
    # Candidate compatibility for a cold user is based on baseline profile, never future interactions.
    user_index = users.reset_index(drop=True)
    role = user_index["supply_chain_role"].to_numpy()
    stage = user_index["supply_chain_stage"].to_numpy()
    company = user_index["company_id"].to_numpy()
    industry = user_index["industry_id"].to_numpy()
    function = user_index["function_id"].to_numpy()
    seniority = user_index["seniority_level"].to_numpy(dtype=int)
    for cold_pos, month in zip(cold_nodes, cold_months):
        compat = np.ones(len(warm_nodes), dtype=float)
        warm_arr = np.array(warm_nodes, dtype=int)
        compat += 1.7 * (company[warm_arr] == company[cold_pos])
        compat += 0.85 * (industry[warm_arr] == industry[cold_pos])
        compat += 1.05 * (function[warm_arr] == function[cold_pos])
        compat += 0.25 * (np.abs(seniority[warm_arr] - seniority[cold_pos]) <= 1)
        for local_idx, k in enumerate(warm_nodes):
            if frozenset((role[k], role[cold_pos])) in COMPLEMENTARY_ROLE_PAIRS:
                compat[local_idx] += 1.15
        # A warm partner is selected proportionally to synthetic profile affinity.
        probs = compat / compat.sum()
        partner = int(rng.choice(warm_arr, p=probs))
        key = min(partner, cold_pos) * n + max(partner, cold_pos)
        if key in used:
            # Since cold users have degree zero before test, this only guards against unexpected changes.
            unused = [k for k in warm_nodes if min(k, cold_pos)*n + max(k, cold_pos) not in used]
            partner = int(rng.choice(unused))
        cold_test_edges_by_month[int(month)].append((min(partner, cold_pos), max(partner, cold_pos)))
        cold_partner_for_edge[min(partner, cold_pos)*n + max(partner, cold_pos)] = cold_pos

    for month in range(22, 27):
        count = test_month_counts.get(month, 0)
        forced = cold_test_edges_by_month.get(month, [])
        if month == 26:
            # One event on the exact inclusive end boundary makes the four-month test horizon explicit.
            i, j = select_edge(all_pool, all_cdf)
            stamp = pd.Timestamp("2025-03-01 00:00:00")
            add_edge(i, j, stamp, "test")
            continue
        # Forced cold edges precede generic edges inside each month, so mutual counts used in selection
        # only depend on earlier synthetic timestamps.
        total_for_month = count
        for rank, (i, j) in enumerate(forced):
            stamp = ordered_month_timestamp(month, rank, max(total_for_month, 1), 2 if month == 22 else 1, 28)
            add_edge(i, j, stamp, "test_cold")
        generic_count = count - len(forced)
        for rank in range(generic_count):
            i, j = select_edge(all_pool, all_cdf)
            global_rank = len(forced) + rank
            stamp = ordered_month_timestamp(month, global_rank, max(total_for_month, 1), 2 if month == 22 else 1, 28)
            add_edge(i, j, stamp, "test")

    assert len(edge_records) == N_RELATIONS, (len(edge_records), N_RELATIONS)
    edge_records.sort(key=lambda x: (x["timestamp"], x["i"], x["j"]))
    relation_rows = []
    key_to_relation: dict[int, dict[str, Any]] = {}
    for rel_num, edge in enumerate(edge_records, 1):
        i, j, stamp = edge["i"], edge["j"], edge["timestamp"]
        user_a, user_b = user_ids[i], user_ids[j]
        # Relation type probabilities are synthetic and slightly profile-dependent.
        if users.iloc[i]["company_id"] == users.iloc[j]["company_id"]:
            relation_type = str(rng.choice(REL_TYPES, p=[0.62, 0.27, 0.11]))
        else:
            relation_type = str(rng.choice(REL_TYPES, p=[0.18, 0.57, 0.25]))
        row = {"relation_id": f"R{rel_num:05d}", "user_id_1": user_a, "user_id_2": user_b,
               "timestamp": ts_str(stamp), "relation_type": relation_type, "is_reciprocal": 1,
               "interaction_strength": round(float(rng.random()), 6)}
        relation_rows.append(row)
        key_to_relation[i*n+j] = row
    relations = pd.DataFrame(relation_rows)

    # Training/cold membership is defined strictly from relation events <= t1.
    pre_t1_counts = np.zeros(n, dtype=int)
    for e in edge_records:
        if e["timestamp"] <= T1:
            pre_t1_counts[e["i"]] += 1
            pre_t1_counts[e["j"]] += 1
    # Structural backbone guarantees all warm users have at least five pre-t1 relation events;
    # cold users have exactly zero before t1.
    assert np.all(pre_t1_counts[np.array(warm_nodes, dtype=int)] >= 5)
    assert np.all(pre_t1_counts[np.array(cold_nodes, dtype=int)] == 0)
    pre_count_map = {user_ids[i]: int(pre_t1_counts[i]) for i in range(n)}
    cold_ids = {user_ids[i] for i in range(n) if pre_t1_counts[i] < 5}
    assert len(cold_ids) == N_COLD
    # This mapping is useful to orient test positives from each cold user as anchor.
    return relations, {user_ids[i]: {user_ids[j] for j in adjacency[i]} for i in range(n)}, pre_count_map, cold_ids


def build_interests(rng: np.random.Generator, users: pd.DataFrame, skills: pd.DataFrame,
                    skills_by_user: dict[str, list[str]], cold_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    skill_ids = skills["skill_id"].tolist()
    titles_rows = []
    categories = [f"Category_{chr(65+i)}" for i in range(20)]
    for i in range(1, N_INTEREST_TITLES + 1):
        n_related = int(rng.integers(2, 6))
        related = [str(s) for s in rng.choice(skill_ids, size=n_related, replace=False)]
        titles_rows.append({
            "interest_title_id": f"IT{i:05d}",
            "title_text": f"TopicToken_{i:05d} ProcessToken_{int(rng.integers(1, 201)):03d} PatternToken_{int(rng.integers(1, 101)):03d}",
            "category": str(rng.choice(categories)),
            "related_skills": pack_ids(related),
        })
    titles = pd.DataFrame(titles_rows)
    title_skill_sets = [set(x.split("|")) for x in titles["related_skills"].tolist()]
    title_ids = titles["interest_title_id"].tolist()
    user_ids = users["user_id"].tolist()
    # Sample unique user-title pairs, with a modest baseline-skill affinity preference.
    # Cold users may like titles only after t1.
    selected: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < N_INTEREST_LIKES:
        user_id = str(rng.choice(user_ids))
        title_idx = int(rng.integers(0, N_INTEREST_TITLES))
        title_id = title_ids[title_idx]
        if (user_id, title_id) in selected:
            continue
        base = set(skills_by_user[user_id])
        overlap = len(base.intersection(title_skill_sets[title_idx]))
        if rng.random() > min(0.98, 0.62 + 0.08 * overlap):
            continue
        if user_id in cold_ids:
            month = int(rng.integers(16, 26))
            stamp = random_timestamp(rng, month, min_day=2 if month in (16, 22) else 1)
        else:
            month = int(rng.integers(0, 26))
            min_day = 2 if month == 16 else 1
            stamp = random_timestamp(rng, month, min_day=min_day)
        # Cold users are deliberately held out of all network/interest interactions before t1.
        if user_id in cold_ids and stamp <= T1:
            stamp = random_timestamp(rng, 17, min_day=1)
        selected.add((user_id, title_id))
        rows.append({"interaction_id": f"LI{len(rows)+1:05d}", "user_id": user_id,
                     "interest_title_id": title_id, "timestamp": ts_str(stamp), "interaction_type": "like"})
    likes = pd.DataFrame(rows).sort_values(["timestamp", "interaction_id"]).reset_index(drop=True)
    assert len(likes) == N_INTEREST_LIKES
    return titles, likes


def make_hyperedges(rng: np.random.Generator, users: pd.DataFrame, career_events: pd.DataFrame,
                    skill_events: pd.DataFrame, skills_by_user: dict[str, list[str]]) -> pd.DataFrame:
    user_lookup = users.set_index("user_id").to_dict(orient="index")
    career_by_user: dict[str, pd.DataFrame] = {u: g.sort_values("start_time") for u, g in career_events.groupby("user_id")}
    skill_by_user: dict[str, pd.DataFrame] = {u: g.sort_values("timestamp") for u, g in skill_events.groupby("user_id")}
    # Exactly 2,808 (40%) supply-chain-specific hyperedges, distributed over all users.
    specific_flags = np.array([1] * int(N_HYPEREDGES * 0.40) + [0] * (N_HYPEREDGES - int(N_HYPEREDGES * 0.40)), dtype=int)
    rng.shuffle(specific_flags)
    rows = []
    h_idx = 1
    for user_id in users["user_id"].tolist():
        # Five evenly dispersed timestamps per user, with small random jitter.
        month_slots = np.array([0, 6, 12, 18, 24]) + rng.integers(0, 2, size=5)
        for slot in month_slots:
            month = int(min(slot, 25))
            stamp = random_timestamp(rng, month)
            career = career_by_user[user_id]
            known = career.loc[career["start_time"] <= ts_str(stamp)]
            if known.empty:
                known = career.iloc[[0]]
            state = known.iloc[-1]
            sk = skill_by_user[user_id]
            known_skills = sk.loc[sk["timestamp"] <= ts_str(stamp), "skill_id"].tolist()
            if not known_skills:
                known_skills = skills_by_user[user_id][:5]
            skill_selection = list(dict.fromkeys(known_skills))[:8]
            if len(skill_selection) < 3:
                skill_selection = skills_by_user[user_id][:max(3, len(skill_selection))]
            sc = int(specific_flags[h_idx - 1])
            user = user_lookup[user_id]
            rows.append({
                "hyperedge_id": hid(h_idx), "user_id": user_id,
                "company_id": state["company_id"], "industry_id": state["industry_id"],
                "function_id": state["function_id"], "occupation_id": state["occupation_id"],
                "skill_ids": pack_ids(skill_selection), "timestamp": ts_str(stamp),
                "relation_type": str(rng.choice(["work_context", "skill_context", "supply_context", "collaboration_context"])),
                "supply_chain_specific": sc,
                "sc_role": user["supply_chain_role"] if sc else None,
                "sc_stage": user["supply_chain_stage"] if sc else None,
                "sc_path": user["supply_chain_path"] if sc else None,
            })
            h_idx += 1
    frame = pd.DataFrame(rows).sort_values(["timestamp", "hyperedge_id"]).reset_index(drop=True)
    assert len(frame) == N_HYPEREDGES
    assert int(frame["supply_chain_specific"].sum()) == 2808
    return frame


def make_pair_context_features(relations: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    users_ix = users.set_index("user_id")
    adjacency: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    rels = relations.sort_values(["timestamp", "relation_id"]).reset_index(drop=True)
    # Group timestamps: pairs sharing a timestamp cannot observe each other before their event.
    for stamp, group in rels.groupby("timestamp", sort=True):
        pending_add = []
        for edge in group.itertuples(index=False):
            a, b = edge.user_id_1, edge.user_id_2
            ua, ub = users_ix.loc[a], users_ix.loc[b]
            mutual = len(adjacency[a].intersection(adjacency[b]))
            comp, sensitivity = role_compatibility(ua["supply_chain_role"], ub["supply_chain_role"],
                                                   ua["supply_chain_stage"], ub["supply_chain_stage"])
            rows.append({
                "relation_id": edge.relation_id, "user_id_1": a, "user_id_2": b,
                "timestamp": stamp, "degree_1_pre_event": len(adjacency[a]),
                "degree_2_pre_event": len(adjacency[b]), "mutual_connections_pre_event": mutual,
                "same_company_baseline": int(ua["company_id"] == ub["company_id"]),
                "same_industry_baseline": int(ua["industry_id"] == ub["industry_id"]),
                "same_function_baseline": int(ua["function_id"] == ub["function_id"]),
                "seniority_gap_baseline": abs(int(ua["seniority_level"]) - int(ub["seniority_level"])),
                "supply_chain_compatibility": comp,
                "signed_counterfactual_sensitivity": sensitivity,
                "counterfactual_intervention": "remove_supply_chain_context",
            })
            pending_add.append((a, b))
        for a, b in pending_add:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return pd.DataFrame(rows)


def make_pair_rating_targets(rng: np.random.Generator, relations: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Provide synthetic ordinal affinity targets for held-out RMSE/MAE evaluation.

    Non-observed user pairs are deliberately not assigned ratings: absence of an edge is not
    interpreted as dislike. The score is a simulated label derived from baseline profile affinity,
    supply-chain compatibility and the relation strength.
    """
    lookup = users.set_index("user_id")
    out = []
    for edge in relations.itertuples(index=False):
        a, b = lookup.loc[edge.user_id_1], lookup.loc[edge.user_id_2]
        compat, signed_cf = role_compatibility(a["supply_chain_role"], b["supply_chain_role"],
                                               a["supply_chain_stage"], b["supply_chain_stage"])
        latent = (2.55
                  + 0.62 * int(a["company_id"] == b["company_id"])
                  + 0.48 * int(a["industry_id"] == b["industry_id"])
                  + 0.52 * int(a["function_id"] == b["function_id"])
                  + 0.72 * (compat - 0.5)
                  + 0.35 * float(edge.interaction_strength)
                  + float(rng.normal(0.0, 0.58)))
        score = round(float(np.clip(latent, 1.0, 5.0)), 2)
        ts = pd.Timestamp(edge.timestamp)
        split = "train" if ts <= T1 else ("val" if ts <= T2 else "test")
        out.append({"relation_id": edge.relation_id, "user_id_1": edge.user_id_1,
                    "user_id_2": edge.user_id_2, "timestamp": edge.timestamp,
                    "rating_target": score, "split": split,
                    "rating_source": "synthetic_latent_affinity"})
    return pd.DataFrame(out)


def create_data_dictionary(schema: dict[str, Any]) -> None:
    lines = [
        "# SC-CaTH Synthetic Dataset — Data Dictionary", "",
        "All IDs, labels, profile snippets, companies, institutions, regions, titles, and interactions are synthetic. No real profiles or scraped data are used.", "",
        "## Temporal and leakage rules", "",
        f"- Observation window: `{ts_str(START)}` through `{ts_str(END)}` (26 months).",
        f"- `t1 = {ts_str(T1)}`; `t2 = {ts_str(T2)}`.",
        "- Train: `timestamp <= t1`; validation: `t1 < timestamp <= t2`; test: `timestamp > t2`.",
        "- User-user edges are undirected and stored once (`user_id_1 < user_id_2`).",
        "- Degree/mutual-neighbour features in `pair_context_features.csv` use only edges strictly earlier than the target edge timestamp.",
        "- `users.csv` profile attributes are the baseline snapshot. `is_cold`, `pre_t1_interaction_count`, and `first_interaction_time` are split/audit metadata and must not be used as unrestricted predictive features; `first_interaction_time` is full-window aware.",
        "- `career_events.csv`, `user_skills.csv`, `hyperedges.csv`, and `user_interest_interactions.csv` are timestamped. Filter each source to records at or before the decision time before feature construction.",
        "- A non-positive candidate means an unobserved pair at the relevant cutoff, not proof that a professional relationship does not exist.",
        "- `pair_rating_targets.csv` contains synthetic 1–5 affinity labels for observed positive relation pairs only; do not convert missing interactions into rating 1.",
        "- Nulls in `career_events.end_time`, `previous_seniority_level`, `previous_occupation_id` and `hyperedges.sc_*` are structural, as explained below.", "",
        "## File schemas", ""
    ]
    for filename, spec in schema["files"].items():
        lines.extend([f"### `{filename}`", "", spec["description"], "", "| Column | Type | Nullable | Description |", "|---|---|:---:|---|"])
        for field_name, field_spec in spec["fields"].items():
            desc = field_spec["description"].replace("|", r"\|")
            lines.append(f"| `{field_name}` | `{field_spec['type']}` | {'yes' if field_spec['nullable'] else 'no'} | {desc} |")
        if spec.get("primary_key"):
            lines.extend(["", "**Primary key:** " + ", ".join(f"`{x}`" for x in spec["primary_key"]) + "."])
        if spec.get("foreign_keys"):
            fks = "; ".join(f"`{x['field']}` → `{x['references_file']}.{x['references_field']}`" for x in spec["foreign_keys"])
            lines.extend(["**Foreign keys:** " + fks + "."])
        lines.append("")
    (ROOT / "data_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_temporal_events(career: pd.DataFrame, relations: pd.DataFrame, likes: pd.DataFrame,
                         hyperedges: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for e in career.itertuples(index=False):
        meta = {"career_event_id": e.event_id, "company_id": e.company_id,
                "industry_id": e.industry_id, "transition_type": e.transition_type}
        rows.append({"event_type": "career", "user_id": e.user_id, "target_id": e.company_id,
                     "timestamp": e.start_time, "metadata": json.dumps(meta, separators=(",", ":"), sort_keys=True)})
    for e in relations.itertuples(index=False):
        meta = {"relation_id": e.relation_id, "other_user_id": e.user_id_2,
                "relation_type": e.relation_type, "is_undirected": True}
        rows.append({"event_type": "relation", "user_id": e.user_id_1, "target_id": e.user_id_2,
                     "timestamp": e.timestamp, "metadata": json.dumps(meta, separators=(",", ":"), sort_keys=True)})
    for e in likes.itertuples(index=False):
        meta = {"interaction_id": e.interaction_id, "interaction_type": "like"}
        rows.append({"event_type": "interest_like", "user_id": e.user_id, "target_id": e.interest_title_id,
                     "timestamp": e.timestamp, "metadata": json.dumps(meta, separators=(",", ":"), sort_keys=True)})
    # Exactly 642 hyperedge creation events sampled without replacement.
    sample_idx = rng.choice(np.arange(len(hyperedges)), size=642, replace=False)
    sampled_hyperedges = hyperedges.iloc[np.sort(sample_idx)]
    for e in sampled_hyperedges.itertuples(index=False):
        meta = {"hyperedge_id": e.hyperedge_id, "relation_type": e.relation_type,
                "supply_chain_specific": int(e.supply_chain_specific)}
        rows.append({"event_type": "hyperedge_creation", "user_id": e.user_id, "target_id": e.hyperedge_id,
                     "timestamp": e.timestamp, "metadata": json.dumps(meta, separators=(",", ":"), sort_keys=True)})
    assert len(rows) == N_TEMPORAL_EVENTS, len(rows)
    frame = pd.DataFrame(rows).sort_values(["timestamp", "event_type", "user_id", "target_id"]).reset_index(drop=True)
    frame.insert(0, "event_id", [f"TE{i:05d}" for i in range(1, len(frame)+1)])
    return frame


def make_training_pairs(rng: np.random.Generator, relations: pd.DataFrame, users: pd.DataFrame,
                        cold_ids: set[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    rels = relations.copy()
    rels["timestamp_dt"] = pd.to_datetime(rels["timestamp"])
    train_rel = rels[rels["timestamp_dt"] <= T1].copy()
    val_rel = rels[(rels["timestamp_dt"] > T1) & (rels["timestamp_dt"] <= T2)].copy()
    test_rel = rels[rels["timestamp_dt"] > T2].copy()
    assert len(train_rel) == 3500 and len(val_rel) == 1300 and len(test_rel) == 812, (len(train_rel), len(val_rel), len(test_rel))
    assert not set(train_rel["user_id_1"]) & cold_ids
    assert not set(train_rel["user_id_2"]) & cold_ids
    assert not set(val_rel["user_id_1"]) & cold_ids
    assert not set(val_rel["user_id_2"]) & cold_ids
    warm_ids = [u for u in users["user_id"].tolist() if u not in cold_ids]
    all_ids = users["user_id"].tolist()
    # Full training-window seen adjacency is used for training candidate filtering.
    train_seen: dict[str, set[str]] = defaultdict(set)
    for e in train_rel.itertuples(index=False):
        train_seen[e.user_id_1].add(e.user_id_2)
        train_seen[e.user_id_2].add(e.user_id_1)

    flat_frames: dict[str, pd.DataFrame] = {}
    pair_frames: dict[str, pd.DataFrame] = {}

    def orient_pair(row: Any, split: str) -> tuple[str, str]:
        a, b = row.user_id_1, row.user_id_2
        if split == "test" and (a in cold_ids or b in cold_ids):
            return (a, b) if a in cold_ids else (b, a)
        return (a, b) if rng.random() < 0.5 else (b, a)

    def sample_ids(pool: list[str], k: int, forbidden: set[str], anchor: str, positive: str) -> list[str]:
        eligible = [x for x in pool if x != anchor and x != positive and x not in forbidden]
        if len(eligible) < k:
            raise RuntimeError(f"Not enough non-positive candidates for {anchor}: {len(eligible)} < {k}")
        return [str(x) for x in rng.choice(np.array(eligible, dtype=object), size=k, replace=False).tolist()]

    for split, split_rel, candidate_pool, k in [
        ("train", train_rel, warm_ids, 4),
        ("val", val_rel, warm_ids, 99),
    ]:
        pairs = []
        flat = []
        for pi, row in enumerate(split_rel.sort_values(["timestamp", "relation_id"]).itertuples(index=False), 1):
            anchor, positive = orient_pair(row, split)
            forbidden = set(train_seen[anchor])
            negs = sample_ids(candidate_pool, k, forbidden, anchor, positive)
            pair_id = f"P_{split}_{pi:05d}"
            pairs.append({"anchor_user_id": anchor, "positive_user_id": positive,
                          "negative_user_ids": pack_ids(negs), "split_time": row.timestamp, "label": 1})
            for si, neg in enumerate(negs, 1):
                flat.append({"pair_id": pair_id, "anchor_user_id": anchor, "positive_user_id": positive,
                             "non_positive_user_id": neg, "sample_index": si,
                             "split_time": row.timestamp, "label": 0})
        pair_frames[f"{split}_pairs.csv"] = pd.DataFrame(pairs)
        flat_frames[f"{split}_negative_samples.csv"] = pd.DataFrame(flat)

    # Evaluation negatives are sampled from candidates not connected to the anchor strictly before
    # the positive event timestamp. This is causal and allows a later future positive to have been
    # unknown at sampling time. Cold test users are excluded from validation and training candidates.
    test_pairs = []
    test_flat = []
    val_pairs = pair_frames["val_pairs.csv"].copy()
    val_flat = flat_frames["val_negative_samples.csv"].copy()
    # Rebuild validation samples with event-time history (still warm-only), overriding train-only pools.
    val_events = val_rel.sort_values(["timestamp", "relation_id"]).reset_index(drop=True)
    history: dict[str, set[str]] = defaultdict(set)
    test_edge_anchor_overrides: dict[tuple[str, str], str] = {}
    for row in test_rel.itertuples(index=False):
        if row.user_id_1 in cold_ids:
            test_edge_anchor_overrides[(row.user_id_1, row.user_id_2)] = row.user_id_1
        elif row.user_id_2 in cold_ids:
            test_edge_anchor_overrides[(row.user_id_1, row.user_id_2)] = row.user_id_2
    def rebuild_eval(split: str, split_rel: pd.DataFrame, pool: list[str], include_cold_pool: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        pairs: list[dict[str, Any]] = []
        flat: list[dict[str, Any]] = []
        hist: dict[str, set[str]] = defaultdict(set)
        # Seed history with all relation events before this split, then walk this split chronologically.
        earlier = rels[rels["timestamp_dt"] <= T1] if split == "val" else rels[rels["timestamp_dt"] <= T2]
        for e in earlier.itertuples(index=False):
            hist[e.user_id_1].add(e.user_id_2)
            hist[e.user_id_2].add(e.user_id_1)
        seen_times = split_rel.sort_values(["timestamp", "relation_id"])
        # Group by time so edges at same timestamp do not observe one another.
        pair_index = 1
        for stamp, group in seen_times.groupby("timestamp", sort=True):
            to_add = []
            for row in group.itertuples(index=False):
                key = (row.user_id_1, row.user_id_2)
                if split == "test" and key in test_edge_anchor_overrides:
                    anchor = test_edge_anchor_overrides[key]
                    positive = row.user_id_2 if anchor == row.user_id_1 else row.user_id_1
                else:
                    anchor, positive = orient_pair(row, split)
                forbidden = set(hist[anchor])
                sample_pool = pool
                # In test, cold users are in the candidate universe, but no already-known neighbour is sampled.
                negs = sample_ids(sample_pool, 99, forbidden, anchor, positive)
                pair_id = f"P_{split}_{pair_index:05d}"
                pairs.append({"anchor_user_id": anchor, "positive_user_id": positive,
                              "negative_user_ids": pack_ids(negs), "split_time": row.timestamp, "label": 1})
                for si, neg in enumerate(negs, 1):
                    flat.append({"pair_id": pair_id, "anchor_user_id": anchor, "positive_user_id": positive,
                                 "non_positive_user_id": neg, "sample_index": si,
                                 "split_time": row.timestamp, "label": 0})
                pair_index += 1
                to_add.append((row.user_id_1, row.user_id_2))
            for a, b in to_add:
                hist[a].add(b)
                hist[b].add(a)
        return pd.DataFrame(pairs), pd.DataFrame(flat)

    # Replace validation rows and negatives with fully time-aware samples.
    val_pairs, val_flat = rebuild_eval("val", val_rel, warm_ids, include_cold_pool=False)
    test_pairs, test_flat = rebuild_eval("test", test_rel, all_ids, include_cold_pool=True)
    pair_frames["val_pairs.csv"] = val_pairs
    pair_frames["test_pairs.csv"] = test_pairs
    flat_frames["val_negative_samples.csv"] = val_flat
    flat_frames["test_negative_samples.csv"] = test_flat
    # Training pair rows are already sampled against the complete training window; all sample candidates warm.
    return pair_frames, flat_frames


def create_manifest_and_schema(schema: dict[str, Any]) -> None:
    (ROOT / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate() -> None:
    rng = np.random.default_rng(SEED)
    for directory in (DATA, SAMPLES, RAW, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    # Remove stale generated files, preserving the scripts and docs.
    for f in DATA.glob("*.csv"):
        f.unlink()
    for f in SAMPLES.glob("*"):
        if f.is_file(): f.unlink()
    for f in RAW.glob("*"):
        if f.is_file(): f.unlink()
    for f in REPORTS.glob("*"):
        if f.is_file(): f.unlink()

    industries, functions, occupations, skills, universities, locations, companies = create_lookups(rng)
    users, skills_by_user, company_ids_by_industry = make_user_profiles(
        rng, industries, functions, occupations, skills, universities, locations, companies)
    skill_events = make_skill_events(rng, users, skills_by_user, skills)
    career = make_career_events(rng, users, skills_by_user, companies, occupations, functions, company_ids_by_industry)
    relations, _adjacency_final, pre_t1_count_map, cold_ids = make_relations(rng, users)

    # Derive split metadata from relation history. Cold users have zero pre-t1 relation events.
    users["pre_t1_interaction_count"] = users["user_id"].map(pre_t1_count_map).astype(int)
    users["is_cold"] = users["pre_t1_interaction_count"] < 5
    assert int(users["is_cold"].sum()) == N_COLD
    titles, likes = build_interests(rng, users, skills, skills_by_user, cold_ids)
    # Every cold user is required to have no user-user or interest-like interactions before t1.
    cold_like_early = likes[likes["user_id"].isin(cold_ids) & (pd.to_datetime(likes["timestamp"]) <= T1)]
    assert cold_like_early.empty
    hyperedges = make_hyperedges(rng, users, career, skill_events, skills_by_user)
    pair_features = make_pair_context_features(relations, users)
    pair_rating_targets = make_pair_rating_targets(rng, relations, users)
    temporal_events = make_temporal_events(career, relations, likes, hyperedges, rng)
    pair_frames, flat_frames = make_training_pairs(rng, relations, users, cold_ids)

    # first_interaction_time is audit-only and intentionally reflects the full generated window.
    first_times: dict[str, pd.Timestamp] = {}
    for row in relations.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        for u in (row.user_id_1, row.user_id_2):
            if u not in first_times or ts < first_times[u]:
                first_times[u] = ts
    for row in likes.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if row.user_id not in first_times or ts < first_times[row.user_id]:
            first_times[row.user_id] = ts
    users["first_interaction_time"] = users["user_id"].map(lambda u: ts_str(first_times.get(u)))

    # The undirected edge table stores each relationship once.
    graph_edges = relations[["user_id_1", "user_id_2", "timestamp", "relation_type"]].rename(
        columns={"user_id_1": "source_user_id", "user_id_2": "target_user_id", "relation_type": "edge_type"})
    user_interest_edges = likes[["user_id", "interest_title_id", "timestamp"]].copy()
    cold_table = users.loc[users["is_cold"], ["user_id", "pre_t1_interaction_count"]].copy().rename(columns={"pre_t1_interaction_count": "pre_t1_relation_count"})
    cold_table["reserved_split"] = "test"
    cold_table["cold_start_reason"] = "zero_user_user_relations_at_or_before_t1"

    tables: dict[str, pd.DataFrame] = {
        "users.csv": users,
        "companies.csv": companies,
        "industries.csv": industries,
        "functions.csv": functions,
        "occupations.csv": occupations,
        "skills.csv": skills,
        "universities.csv": universities,
        "locations.csv": locations,
        "user_skills.csv": skill_events,
        "career_events.csv": career,
        "user_user_relations.csv": relations,
        "interest_titles.csv": titles,
        "user_interest_interactions.csv": likes,
        "hyperedges.csv": hyperedges,
        "temporal_events.csv": temporal_events,
        "graph_edges.csv": graph_edges,
        "user_interest_edges.csv": user_interest_edges,
        "pair_context_features.csv": pair_features,
        "pair_rating_targets.csv": pair_rating_targets,
        "cold_start_users.csv": cold_table,
    }
    tables.update(pair_frames)
    tables.update(flat_frames)
    # Ensure no object leaks from generator internals and use stable column ordering based on construction.
    for filename, frame in tables.items():
        write_csv(frame, filename)
    # Raw duplicates: 97 appended exact duplicate relation rows => 97 / 5,709 = 1.70% duplicate-row rate.
    raw_relation_count = int(len(relations) + 97)
    duplicate_rows = relations.sample(n=97, random_state=SEED)
    raw_relations = pd.concat([relations, duplicate_rows], ignore_index=True)
    raw_relations.to_csv(RAW / "user_user_relations_raw.csv", index=False, encoding="utf-8", na_rep="")
    assert len(raw_relations) == raw_relation_count

    # Every main CSV (and the raw duplicate artifact) gets a 20-row sample.
    for path in sorted(DATA.glob("*.csv")):
        pd.read_csv(path, nrows=20).to_csv(SAMPLES / path.name, index=False, encoding="utf-8", na_rep="")
    pd.read_csv(RAW / "user_user_relations_raw.csv", nrows=20).to_csv(
        SAMPLES / "raw_user_user_relations_raw.csv", index=False, encoding="utf-8", na_rep="")

    schema = build_schema()
    # Add lookup/flat-file schemas which have concise, regular structures.
    create_manifest_and_schema(schema)
    create_data_dictionary(schema)
    stats = calculate_stats(tables, raw_relations, users, relations, titles, likes, hyperedges, temporal_events, pair_frames, flat_frames)
    (REPORTS / "summary_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nDataset written to: {DATA}")


def calculate_stats(tables: dict[str, pd.DataFrame], raw_relations: pd.DataFrame,
                    users: pd.DataFrame, relations: pd.DataFrame, titles: pd.DataFrame,
                    likes: pd.DataFrame, hyperedges: pd.DataFrame, temporal_events: pd.DataFrame,
                    pair_frames: dict[str, pd.DataFrame], flat_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    n = len(users)
    edge_count = len(relations)
    degrees = users["user_id"].map(relations["user_id_1"].value_counts().add(relations["user_id_2"].value_counts(), fill_value=0)).fillna(0)
    # The above mapping can produce floats from value_counts union; that's fine for summary stats.
    duplicate_rate = (len(raw_relations) - len(raw_relations.drop_duplicates())) / len(raw_relations)
    df_counts = {k: int(len(v)) for k, v in tables.items()}
    return {
        "dataset_name": "SC-CaTH Synthetic Professional Network",
        "seed": SEED,
        "synthetic_only": True,
        "time_window": {"start": ts_str(START), "inclusive_end": "2025-03-01T00:00:00", "t1": ts_str(T1), "t2": ts_str(T2), "months": 26},
        "record_counts": df_counts,
        "target_counts": {"users": N_USERS, "companies": N_COMPANIES, "industries": N_INDUSTRIES,
                           "functions": N_FUNCTIONS, "occupations": N_OCCUPATIONS,
                           "career_events": N_CAREER_EVENTS, "interest_titles": N_INTEREST_TITLES,
                           "relations": N_RELATIONS, "interest_likes": N_INTEREST_LIKES,
                           "hyperedges": N_HYPEREDGES, "temporal_events": N_TEMPORAL_EVENTS},
        "split_positive_counts": {k.replace("_pairs.csv", ""): int(len(v)) for k, v in pair_frames.items()},
        "negative_sample_counts": {k: int(len(v)) for k, v in flat_frames.items()},
        "graph": {"average_degree": float(2*edge_count/n), "density": float(2*edge_count/(n*(n-1))),
                  "max_degree": int(degrees.max()), "min_degree_all_window": int(degrees.min())},
        "user_interest_matrix": {"density": float(len(likes)/(len(users)*len(titles))), "interactions": int(len(likes)), "possible_pairs": int(len(users)*len(titles))},
        "hyperedges_per_user": {"mean": float(len(hyperedges)/len(users)), "supply_chain_specific_count": int(hyperedges["supply_chain_specific"].sum()), "supply_chain_specific_rate": float(hyperedges["supply_chain_specific"].mean())},
        "cold_start": {"cold_user_count": int(users["is_cold"].sum()), "cold_pre_t1_relation_count_min": int(users.loc[users["is_cold"], "pre_t1_interaction_count"].min()), "cold_pre_t1_relation_count_max": int(users.loc[users["is_cold"], "pre_t1_interaction_count"].max()), "train_pair_users_are_warm": True},
        "raw_duplicate_rate": float(duplicate_rate),
        "missingness": {filename: {column: float(frame[column].isna().mean()) for column in frame.columns if frame[column].isna().any()} for filename, frame in tables.items()},
    }


if __name__ == "__main__":
    generate()
