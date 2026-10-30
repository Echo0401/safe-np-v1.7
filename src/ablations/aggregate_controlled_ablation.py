#!/usr/bin/env python3
"""Aggregate the matched A0--A6 RELAX controlled ablation.

The script intentionally performs inference at participant level:

1. calculate each metric within each participant/seed/context/ablation;
2. average the metric across seeds within participant;
3. bootstrap and test the resulting paired participant values.

A0, A4 and A5 are deterministic counterfactuals reconstructed from the A6
query-level export.  They therefore use *the same trained A6 model and the same
queries*:

* A0: frozen Global Fusion probability;
* A4: full constrained residual with gamma fixed to 1;
* A5: full constrained residual with the raw source-selected gamma, before the
  minimum-improvement safety rule;
* A6: saved final probability after the minimum-improvement rule/fallback.

A1--A3 are loaded from their independently retrained matched-query exports.
No target-query outcome is used to choose gamma.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


ABLATION_LABELS = {
    "A0": "Frozen Global Fusion",
    "A1": "Direct attentive NP (no Global anchor)",
    "A2": "Anchored directional NP (no hard bound)",
    "A3": "Anchored bounded NP (free residual direction)",
    "A4": "Full constrained residual, fixed gamma=1",
    "A5": "Full constrained residual, raw source-selected gamma",
    "A6": "Safe NP: source-selected gamma + minimum-improvement fallback",
}

VARIANT_DEFINITIONS = {
    "A0": {
        "definition": "Frozen source Global Fusion; no target-personalization residual.",
        "role": "global baseline",
    },
    "A1": {
        "definition": (
            "Direct attentive Neural Process probability without a frozen Global anchor. "
            "The bounded residual, contrast-direction safety rule, source gamma activation, "
            "and exact Global fallback are consequently not applicable."
        ),
        "role": "broader unanchored architectural comparator, not a one-factor ablation",
    },
    "A2": {
        "definition": (
            "Frozen Global anchor plus contrast-direction residual, with no hard residual "
            "bound and gamma fixed to 1."
        ),
        "role": "hard-bound removal ablation",
    },
    "A3": {
        "definition": (
            "Frozen Global anchor plus bounded residual, but residual direction is learned "
            "freely rather than constrained by query contrast; gamma fixed to 1."
        ),
        "role": "direction-constraint removal ablation",
    },
    "A4": {
        "definition": (
            "Frozen Global anchor plus bounded, contrast-direction residual with gamma=1."
        ),
        "role": "full constrained residual before source strength selection",
    },
    "A5": {
        "definition": (
            "A4 with gamma selected on source-only validation episodes, before applying "
            "the minimum NLL-improvement activation threshold."
        ),
        "role": "source-only gamma-selection ablation",
    },
    "A6": {
        "definition": (
            "A5 plus the prespecified minimum source-NLL improvement threshold and exact "
            "gamma=0 fallback to the frozen Global probability."
        ),
        "role": "complete Safe NP",
    },
}

KEY_COLUMNS = [
    "seed",
    "target_subject",
    "query_model_row",
    "query_original_sample_index",
    "query_anchor_utc",
    "y_true",
    "k_per_class",
    "context_total",
    "context_original_sample_indices",
]

IDENTITY_COLUMNS = [
    "seed",
    "target_subject",
    "query_original_sample_index",
    "context_total",
]

EXPECTED_CONTEXTS = [2, 4, 6]
EXPECTED_SEEDS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_TARGETS = [13, 16, 18, 22, 25, 28, 38, 43, 45, 46, 48, 49, 56]
METRICS = ["balanced_accuracy", "nll", "brier"]


@dataclass
class KeyAudit:
    ablation_id: str
    n_rows: int
    n_unique_keys: int
    n_duplicate_keys: int
    n_seeds: int
    n_participants: int
    contexts: str
    key_sha256: str
    exact_match_to_A6: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict A0--A6 matched-query participant-level analysis"
    )
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--a6-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-mode",
        choices=["formal", "smoke"],
        default="formal",
        help=(
            "formal requires the prespecified 13 participants x 5 seeds. "
            "smoke restricts A6 to the common seed/participant domain of A1--A3 "
            "and marks every output non-publishable."
        ),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261013)
    parser.add_argument("--probability-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--reconstruction-tolerance",
        type=float,
        default=1e-6,
        help="Maximum A6 probability discrepancy allowed when reconstruction is audited.",
    )
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required final export not found: {path}")
    return pd.read_csv(path)


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # Exports legitimately mix whole-second and microsecond ISO timestamps.
    out["query_anchor_utc"] = pd.to_datetime(
        out["query_anchor_utc"], utc=True, errors="raise", format="mixed"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out["context_original_sample_indices"] = (
        out["context_original_sample_indices"].fillna("").astype(str)
    )
    for col in [
        "seed",
        "target_subject",
        "query_model_row",
        "query_original_sample_index",
        "y_true",
        "k_per_class",
        "context_total",
    ]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(np.int64)
    return out


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[KEY_COLUMNS], names=KEY_COLUMNS)


def _key_digest(frame: pd.DataFrame) -> str:
    ordered = frame[KEY_COLUMNS].sort_values(KEY_COLUMNS, kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_retrained_variant(run_dir: Path, expected_id: str) -> pd.DataFrame:
    path = run_dir / "matched_query_results.csv"
    all_rows = _read_csv(path)
    _require_columns(
        all_rows,
        KEY_COLUMNS + ["method", "p_stress", "p_nonstress", "y_pred"],
        path,
    )
    all_rows = _normalise_keys(all_rows)
    frame = all_rows.loc[all_rows["method"].eq("NeuralProcess")].copy()
    global_rows = all_rows.loc[all_rows["method"].eq("GlobalFusion")].copy()
    if frame.empty:
        raise ValueError(f"No NeuralProcess rows in {path}")
    if global_rows.empty:
        raise ValueError(f"No GlobalFusion control rows in {path}")
    _validate_probabilities(global_rows, path)
    if set(_key_index(frame)) != set(_key_index(global_rows)):
        raise ValueError(
            f"{expected_id} NeuralProcess and GlobalFusion query keys differ in {path}"
        )
    if "ablation_id" in frame.columns:
        observed = set(frame["ablation_id"].dropna().astype(str).unique())
        if observed != {expected_id}:
            raise ValueError(
                f"Expected only ablation_id={expected_id} in {path}, observed {observed}"
            )
    global_probability = global_rows[IDENTITY_COLUMNS + ["p_stress"]].rename(
        columns={"p_stress": "variant_global_p_stress"}
    )
    frame = frame.merge(
        global_probability,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    frame["ablation_id"] = expected_id
    frame["ablation_label"] = ABLATION_LABELS[expected_id]
    return _validate_probabilities(frame, path)


def _validate_probabilities(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    p = pd.to_numeric(frame["p_stress"], errors="raise").to_numpy(float)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"Invalid p_stress values in {source}")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        examples = frame.loc[frame.duplicated(IDENTITY_COLUMNS, keep=False), IDENTITY_COLUMNS]
        raise ValueError(
            f"Duplicate query identities in {source}:\n{examples.head(10).to_string(index=False)}"
        )
    return frame


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Could not parse boolean values: {series[mapped.isna()].unique()}")
    return mapped.astype(bool)


def _load_a6_components(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    query_path = run_dir / "matched_query_results.csv"
    gamma_path = run_dir / "source_gamma_calibration.csv"
    config_path = run_dir / "run_config.json"
    all_rows = _read_csv(query_path)
    _require_columns(
        all_rows,
        KEY_COLUMNS
        + [
            "method",
            "p_stress",
            "p_nonstress",
            "np_adaptation_gamma",
            "np_uncalibrated_applied_logodds_shift",
        ],
        query_path,
    )
    all_rows = _normalise_keys(all_rows)
    neural = all_rows.loc[all_rows["method"].eq("NeuralProcess")].copy()
    global_rows = all_rows.loc[all_rows["method"].eq("GlobalFusion")].copy()
    _validate_probabilities(neural, query_path)
    _validate_probabilities(global_rows, query_path)
    if set(_key_index(neural)) != set(_key_index(global_rows)):
        raise ValueError("A6 NeuralProcess and GlobalFusion query keys are not identical")

    global_probability = global_rows[IDENTITY_COLUMNS + ["p_stress"]].rename(
        columns={"p_stress": "global_p_stress"}
    )
    neural = neural.merge(
        global_probability,
        on=IDENTITY_COLUMNS,
        validate="one_to_one",
        how="left",
    )
    if neural["global_p_stress"].isna().any():
        raise AssertionError("Internal error merging A6 global probabilities")

    gamma = _read_csv(gamma_path)
    _require_columns(
        gamma,
        [
            "seed",
            "target_subject",
            "gamma",
            "raw_selected",
            "selected",
            "final_selected_gamma",
        ],
        gamma_path,
    )
    gamma["seed"] = pd.to_numeric(gamma["seed"], errors="raise").astype(np.int64)
    gamma["target_subject"] = pd.to_numeric(
        gamma["target_subject"], errors="raise"
    ).astype(np.int64)
    raw = gamma.loc[_as_bool(gamma["raw_selected"])].copy()
    if raw.duplicated(["seed", "target_subject"]).any():
        raise ValueError("More than one raw-selected gamma for at least one A6 fold")
    fold_domain = neural[["seed", "target_subject"]].drop_duplicates()
    raw_domain = raw[["seed", "target_subject"]].drop_duplicates()
    if set(map(tuple, fold_domain.to_numpy())) != set(map(tuple, raw_domain.to_numpy())):
        raise ValueError("Raw source-gamma table does not cover exactly the A6 folds")

    if not config_path.is_file():
        raise FileNotFoundError(f"Required A6 run configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    np_config = config.get("neural_process", {})
    hard_bound = np_config.get("max_applied_logodds_shift")
    if hard_bound is None:
        hard_bound = float(np_config["alpha_max"]) * float(
            np_config["residual_max_logodds"]
        )
    hard_bound = float(hard_bound)
    calibration = np_config.get("adaptation_strength_calibration", {})
    if not calibration.get("hard_bound_after_gamma", False):
        raise ValueError("A6 configuration does not assert a hard bound after gamma")
    if calibration.get("outer_target_used_for_selection", True) is not False:
        raise ValueError("A6 configuration does not assert source-only gamma selection")
    return neural, raw, hard_bound, config


def _apply_logodds_shift(
    base_probability: np.ndarray,
    uncalibrated_shift: np.ndarray,
    gamma: np.ndarray | float,
    hard_bound: float,
    epsilon: float,
) -> np.ndarray:
    base = np.asarray(base_probability, dtype=float)
    residual = np.asarray(uncalibrated_shift, dtype=float)
    gain = np.broadcast_to(np.asarray(gamma, dtype=float), base.shape)
    if base.shape != residual.shape:
        raise ValueError("Base probability and residual arrays must have identical shapes")
    if not np.isfinite(base).all() or not np.isfinite(residual).all() or not np.isfinite(gain).all():
        raise ValueError("Non-finite value encountered during probability reconstruction")
    if np.any((base < 0.0) | (base > 1.0)):
        raise ValueError("Base probabilities must lie in [0, 1]")

    # Preserve the exact fallback contract for gamma=0.  For positive gamma,
    # reproduce the implementation's stable binary log-softmax update.
    answer = base.copy()
    active = gain != 0.0
    if np.any(active):
        p = np.clip(base[active], epsilon, 1.0 - epsilon)
        shift = np.clip(gain[active] * residual[active], -hard_bound, hard_bound)
        logit = np.log(p) - np.log1p(-p)
        z = logit + shift
        answer[active] = np.where(
            z >= 0.0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )
    return answer


def _minimal_query_frame(source: pd.DataFrame, ablation_id: str) -> pd.DataFrame:
    columns = KEY_COLUMNS + ["p_stress"]
    out = source[columns].copy()
    out["p_nonstress"] = 1.0 - out["p_stress"].to_numpy(float)
    out["y_pred"] = (out["p_stress"].to_numpy(float) >= 0.5).astype(np.int64)
    out["ablation_id"] = ablation_id
    out["ablation_label"] = ABLATION_LABELS[ablation_id]
    return out


def _reconstruct_a0_a4_a5_a6(
    a6_neural: pd.DataFrame,
    raw_gamma: pd.DataFrame,
    hard_bound: float,
    epsilon: float,
    tolerance: float,
) -> tuple[dict[str, pd.DataFrame], dict]:
    base = a6_neural["global_p_stress"].to_numpy(float)
    residual = a6_neural["np_uncalibrated_applied_logodds_shift"].to_numpy(float)

    raw_map = raw_gamma[["seed", "target_subject", "gamma"]].rename(
        columns={"gamma": "raw_source_gamma"}
    )
    enriched = a6_neural.merge(
        raw_map, on=["seed", "target_subject"], validate="many_to_one", how="left"
    )
    if enriched["raw_source_gamma"].isna().any():
        raise ValueError("Missing raw source-selected gamma for an A6 query")

    a0 = a6_neural.copy()
    a0["p_stress"] = base.copy()
    a4 = a6_neural.copy()
    a4["p_stress"] = _apply_logodds_shift(base, residual, 1.0, hard_bound, epsilon)
    a5 = enriched.copy()
    a5["p_stress"] = _apply_logodds_shift(
        base,
        residual,
        a5["raw_source_gamma"].to_numpy(float),
        hard_bound,
        epsilon,
    )
    a6 = a6_neural.copy()

    reconstructed_a6 = _apply_logodds_shift(
        base,
        residual,
        a6["np_adaptation_gamma"].to_numpy(float),
        hard_bound,
        epsilon,
    )
    discrepancy = np.abs(reconstructed_a6 - a6["p_stress"].to_numpy(float))
    if float(np.max(discrepancy, initial=0.0)) > tolerance:
        raise ValueError(
            "Saved A6 probabilities do not reproduce from saved gamma/residual: "
            f"max_abs_error={float(discrepancy.max()):.9g}, tolerance={tolerance:.9g}"
        )

    # The exact fallback is stronger than numerical closeness.
    fallback = a6["np_adaptation_gamma"].to_numpy(float) == 0.0
    exact_fallback = bool(
        np.array_equal(
            a6.loc[fallback, "p_stress"].to_numpy(float),
            a6.loc[fallback, "global_p_stress"].to_numpy(float),
        )
    )
    if not exact_fallback:
        raise ValueError("A6 gamma=0 rows are not bitwise identical to Global Fusion")

    result = {
        "A0": _minimal_query_frame(a0, "A0"),
        "A4": _minimal_query_frame(a4, "A4"),
        "A5": _minimal_query_frame(a5, "A5"),
        "A6": _minimal_query_frame(a6, "A6"),
    }
    audit = {
        "hard_bound_logodds": hard_bound,
        "n_a6_queries": int(len(a6)),
        "n_a6_fallback_queries": int(fallback.sum()),
        "n_a6_active_queries": int((~fallback).sum()),
        "gamma0_exact_probability_identity": exact_fallback,
        "a6_reconstruction_max_abs_probability_error": float(
            np.max(discrepancy, initial=0.0)
        ),
        "a6_reconstruction_mean_abs_probability_error": float(discrepancy.mean()),
        "raw_gamma_fold_counts": {
            str(float(gamma)): int(count)
            for gamma, count in raw_gamma["gamma"].value_counts().sort_index().items()
        },
        "final_gamma_fold_counts": {
            str(float(gamma)): int(count)
            for gamma, count in a6_neural[
                ["seed", "target_subject", "np_adaptation_gamma"]
            ]
            .drop_duplicates()["np_adaptation_gamma"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return result, audit


def _restrict_smoke_domain(
    variants: dict[str, pd.DataFrame], a6_neural: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[tuple[int, int]]]:
    domains = []
    for ablation_id in ["A1", "A2", "A3"]:
        pairs = set(
            map(
                tuple,
                variants[ablation_id][["seed", "target_subject"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        domains.append(pairs)
    common = set.intersection(*domains)
    if not common:
        raise ValueError("A1--A3 have no common seed/participant domain for smoke mode")
    if any(domain != common for domain in domains):
        raise ValueError(
            "Smoke mode still requires identical seed/participant domains for A1--A3"
        )
    ordered = sorted((int(seed), int(subject)) for seed, subject in common)

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        mask = pd.MultiIndex.from_frame(frame[["seed", "target_subject"]]).isin(ordered)
        return frame.loc[mask].copy()

    variants = {key: keep(value) for key, value in variants.items()}
    return variants, keep(a6_neural), ordered


def _strict_key_audit(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reference = variants["A6"]
    reference_index = set(_key_index(reference))
    audits: list[KeyAudit] = []
    for ablation_id in sorted(variants):
        frame = variants[ablation_id]
        duplicate_count = int(frame.duplicated(KEY_COLUMNS).sum())
        if duplicate_count:
            raise ValueError(f"{ablation_id} has {duplicate_count} duplicate full query keys")
        observed_index = set(_key_index(frame))
        exact = observed_index == reference_index
        if not exact:
            missing = list(reference_index - observed_index)[:3]
            extra = list(observed_index - reference_index)[:3]
            raise ValueError(
                f"{ablation_id} query keys differ from A6; "
                f"missing examples={missing}, extra examples={extra}"
            )
        audits.append(
            KeyAudit(
                ablation_id=ablation_id,
                n_rows=int(len(frame)),
                n_unique_keys=int(len(observed_index)),
                n_duplicate_keys=duplicate_count,
                n_seeds=int(frame["seed"].nunique()),
                n_participants=int(frame["target_subject"].nunique()),
                contexts=";".join(
                    map(str, sorted(frame["context_total"].astype(int).unique()))
                ),
                key_sha256=_key_digest(frame),
                exact_match_to_A6=exact,
            )
        )
    return pd.DataFrame([asdict(row) for row in audits])


def _audit_global_invariance(variants: dict[str, pd.DataFrame]) -> dict:
    """Verify that A1--A3 changed only the NP branch, not the source encoder."""
    reference = variants["A0"][IDENTITY_COLUMNS + ["p_stress"]].rename(
        columns={"p_stress": "a6_global_p_stress"}
    )
    result: dict[str, dict[str, float | bool]] = {}
    for ablation_id in ["A1", "A2", "A3"]:
        frame = variants[ablation_id]
        if "variant_global_p_stress" not in frame.columns:
            raise ValueError(f"{ablation_id} lacks its GlobalFusion control probabilities")
        observed = frame[IDENTITY_COLUMNS + ["variant_global_p_stress"]]
        merged = reference.merge(
            observed,
            on=IDENTITY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(reference):
            raise ValueError(f"{ablation_id} GlobalFusion controls do not cover all A6 queries")
        delta = np.abs(
            merged["variant_global_p_stress"].to_numpy(float)
            - merged["a6_global_p_stress"].to_numpy(float)
        )
        exact = bool(
            np.array_equal(
                merged["variant_global_p_stress"].to_numpy(float),
                merged["a6_global_p_stress"].to_numpy(float),
            )
        )
        if not exact:
            raise ValueError(
                f"{ablation_id} source GlobalFusion probabilities differ from A6; "
                f"max_abs_error={float(delta.max()):.9g}. This would confound the ablation."
            )
        result[ablation_id] = {
            "bitwise_identical_to_A6_global": exact,
            "max_abs_probability_error": float(np.max(delta, initial=0.0)),
        }
    return result


def _check_formal_design(variants: dict[str, pd.DataFrame]) -> None:
    reference = variants["A6"]
    seeds = sorted(reference["seed"].astype(int).unique())
    targets = sorted(reference["target_subject"].astype(int).unique())
    contexts = sorted(reference["context_total"].astype(int).unique())
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Formal analysis requires seeds {EXPECTED_SEEDS}; observed {seeds}")
    if targets != EXPECTED_TARGETS:
        raise ValueError(
            f"Formal analysis requires targets {EXPECTED_TARGETS}; observed {targets}"
        )
    if contexts != EXPECTED_CONTEXTS:
        raise ValueError(
            f"Formal analysis requires contexts {EXPECTED_CONTEXTS}; observed {contexts}"
        )
    expected_folds = {
        (seed, subject) for seed in EXPECTED_SEEDS for subject in EXPECTED_TARGETS
    }
    expected_queries_per_context = 171 * len(EXPECTED_SEEDS)
    for ablation_id in ["A1", "A2", "A3"]:
        frame = variants[ablation_id]
        observed_folds = set(
            map(
                tuple,
                frame[["seed", "target_subject"]].drop_duplicates().to_numpy(),
            )
        )
        if observed_folds != expected_folds:
            missing = sorted(expected_folds - observed_folds)[:5]
            extra = sorted(observed_folds - expected_folds)[:5]
            raise ValueError(
                f"{ablation_id} formal fold domain is not exactly 13 targets x 5 seeds; "
                f"missing={missing}, extra={extra}"
            )
        context_counts = (
            frame.groupby("context_total", sort=True).size().astype(int).to_dict()
        )
        expected_counts = {
            context: expected_queries_per_context for context in EXPECTED_CONTEXTS
        }
        if context_counts != expected_counts:
            raise ValueError(
                f"{ablation_id} requires exactly 171 x 5 = "
                f"{expected_queries_per_context} query records per context; "
                f"observed {context_counts}"
            )
        per_seed_context = frame.groupby(["seed", "context_total"], sort=True).size()
        bad = per_seed_context.loc[per_seed_context.ne(171)]
        if not bad.empty:
            raise ValueError(
                f"{ablation_id} requires exactly 171 queries in every seed/context cell; "
                f"violations={bad.to_dict()}"
            )


def _metric_values(y: np.ndarray, p: np.ndarray, epsilon: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot score an empty query set")
    pred = (p >= 0.5).astype(int)
    if set(np.unique(y)) == {0, 1}:
        sensitivity = float(np.mean(pred[y == 1] == 1))
        specificity = float(np.mean(pred[y == 0] == 0))
        balanced_accuracy = 0.5 * (sensitivity + specificity)
    else:
        balanced_accuracy = math.nan
    p_clip = np.clip(p, epsilon, 1.0 - epsilon)
    nll = float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log1p(-p_clip)))
    brier = float(np.mean((p - y) ** 2))
    return {
        "balanced_accuracy": balanced_accuracy,
        "nll": nll,
        "brier": brier,
    }


def _seed_subject_metrics(all_queries: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    records = []
    groups = all_queries.groupby(
        ["ablation_id", "ablation_label", "context_total", "target_subject", "seed"],
        sort=True,
        observed=True,
    )
    for keys, group in groups:
        ablation_id, label, context, subject, seed = keys
        y = group["y_true"].to_numpy(int)
        p = group["p_stress"].to_numpy(float)
        record = {
            "ablation_id": ablation_id,
            "ablation_label": label,
            "context_total": int(context),
            "target_subject": int(subject),
            "seed": int(seed),
            "n_query": int(len(group)),
            "n_nonstress": int((y == 0).sum()),
            "n_stress": int((y == 1).sum()),
            "has_both_query_classes": bool(set(np.unique(y)) == {0, 1}),
        }
        record.update(_metric_values(y, p, epsilon))
        records.append(record)
    return pd.DataFrame(records)


def _participant_seed_average(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "ablation_id",
        "ablation_label",
        "context_total",
        "target_subject",
    ]
    metric_mean = (
        seed_metrics.groupby(id_columns, sort=True, observed=True)[METRICS]
        .mean()
        .reset_index()
    )
    counts = (
        seed_metrics.groupby(id_columns, sort=True, observed=True)
        .agg(
            n_seeds=("seed", "nunique"),
            n_query_per_seed_min=("n_query", "min"),
            n_query_per_seed_max=("n_query", "max"),
            has_both_query_classes=("has_both_query_classes", "all"),
        )
        .reset_index()
    )
    return counts.merge(metric_mean, on=id_columns, validate="one_to_one")


def _bootstrap_mean_ci(
    values: np.ndarray, reps: int, rng: np.random.Generator
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return math.nan, math.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    indices = rng.integers(0, len(clean), size=(reps, len(clean)))
    draws = clean[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _summary_table(
    participant: pd.DataFrame, reps: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for keys, group in participant.groupby(
        ["ablation_id", "ablation_label", "context_total"],
        sort=True,
        observed=True,
    ):
        ablation_id, label, context = keys
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            low, high = _bootstrap_mean_ci(values, reps, rng)
            records.append(
                {
                    "ablation_id": ablation_id,
                    "ablation_label": label,
                    "context_total": int(context),
                    "context_label": f"C{int(context)}",
                    "metric": metric,
                    "n_participants": int(len(values)),
                    "mean": float(np.mean(values)) if len(values) else math.nan,
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                    "median": float(np.median(values)) if len(values) else math.nan,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(records)


def _rank_biserial(differences: np.ndarray) -> float:
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0.0)]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff), method="average")
    total = float(ranks.sum())
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / total)


def _holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted
    valid_indices = np.flatnonzero(valid)
    order_local = np.argsort(p[valid])
    ordered_indices = valid_indices[order_local]
    m = len(ordered_indices)
    running = 0.0
    for rank, original_index in enumerate(ordered_indices):
        candidate = min(1.0, (m - rank) * float(p[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def _paired_comparisons(
    participant: pd.DataFrame,
    baseline_id: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = participant.loc[
        participant["ablation_id"].eq(baseline_id),
        ["context_total", "target_subject"] + METRICS,
    ].copy()
    records = []
    candidate_ids = [a for a in sorted(ABLATION_LABELS) if a != baseline_id]
    for context in sorted(participant["context_total"].astype(int).unique()):
        base_context = baseline.loc[baseline["context_total"].eq(context)]
        for metric in METRICS:
            base_values = base_context[["target_subject", metric]].rename(
                columns={metric: "baseline_value"}
            )
            for ablation_id in candidate_ids:
                candidate_values = participant.loc[
                    participant["ablation_id"].eq(ablation_id)
                    & participant["context_total"].eq(context),
                    ["target_subject", metric],
                ].rename(columns={metric: "candidate_value"})
                paired = base_values.merge(
                    candidate_values,
                    on="target_subject",
                    validate="one_to_one",
                    how="inner",
                ).dropna()
                raw_difference = (
                    paired["candidate_value"].to_numpy(float)
                    - paired["baseline_value"].to_numpy(float)
                )
                # Positive oriented differences always mean the candidate is better.
                oriented = (
                    raw_difference
                    if metric == "balanced_accuracy"
                    else -raw_difference
                )
                low, high = _bootstrap_mean_ci(oriented, reps, rng)
                if len(oriented) == 0:
                    statistic = p_value = math.nan
                elif np.all(oriented == 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = stats.wilcoxon(
                        oriented,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                sd = float(np.std(oriented, ddof=1)) if len(oriented) > 1 else math.nan
                mean = float(np.mean(oriented)) if len(oriented) else math.nan
                records.append(
                    {
                        "baseline_id": baseline_id,
                        "candidate_id": ablation_id,
                        "candidate_label": ABLATION_LABELS[ablation_id],
                        "context_total": int(context),
                        "context_label": f"C{int(context)}",
                        "metric": metric,
                        "difference_definition": (
                            "candidate - baseline"
                            if metric == "balanced_accuracy"
                            else "baseline - candidate"
                        ),
                        "n_paired_participants": int(len(oriented)),
                        "mean_oriented_improvement": mean,
                        "median_oriented_improvement": (
                            float(np.median(oriented)) if len(oriented) else math.nan
                        ),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "paired_cohen_dz": (
                            mean / sd if np.isfinite(sd) and sd > 0.0 else math.nan
                        ),
                        "matched_rank_biserial": _rank_biserial(oriented),
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_value,
                        "holm_family": f"{baseline_id}|C{int(context)}|{metric}",
                    }
                )
    result = pd.DataFrame(records)
    result["p_holm"] = np.nan
    for _, indices in result.groupby("holm_family", sort=True).groups.items():
        result.loc[list(indices), "p_holm"] = _holm_adjust(
            result.loc[list(indices), "p_raw"].to_numpy(float)
        )
    return result


def _paper_tables(
    summary: pd.DataFrame,
    vs_global: pd.DataFrame,
    vs_a6: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = summary.copy()
    for baseline_id, comparisons in [("A0", vs_global), ("A6", vs_a6)]:
        fields = comparisons[
            [
                "candidate_id",
                "context_total",
                "metric",
                "n_paired_participants",
                "mean_oriented_improvement",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "p_raw",
                "p_holm",
            ]
        ].rename(
            columns={
                "candidate_id": "ablation_id",
                "n_paired_participants": f"n_vs_{baseline_id}",
                "mean_oriented_improvement": f"improvement_vs_{baseline_id}",
                "bootstrap_ci_low": f"improvement_vs_{baseline_id}_ci_low",
                "bootstrap_ci_high": f"improvement_vs_{baseline_id}_ci_high",
                "p_raw": f"p_raw_vs_{baseline_id}",
                "p_holm": f"p_holm_vs_{baseline_id}",
            }
        )
        table = table.merge(
            fields,
            on=["ablation_id", "context_total", "metric"],
            how="left",
            validate="one_to_one",
        )

    formatted = table.copy()
    formatted["estimate_95ci"] = formatted.apply(
        lambda row: (
            f"{row['mean']:.3f} [{row['bootstrap_ci_low']:.3f}, "
            f"{row['bootstrap_ci_high']:.3f}]"
        ),
        axis=1,
    )
    for baseline_id in ["A0", "A6"]:
        formatted[f"improvement_vs_{baseline_id}_95ci"] = formatted.apply(
            lambda row, b=baseline_id: (
                ""
                if pd.isna(row.get(f"improvement_vs_{b}"))
                else (
                    f"{row[f'improvement_vs_{b}']:+.3f} "
                    f"[{row[f'improvement_vs_{b}_ci_low']:+.3f}, "
                    f"{row[f'improvement_vs_{b}_ci_high']:+.3f}]"
                )
            ),
            axis=1,
        )
        formatted[f"Holm_p_vs_{baseline_id}"] = formatted[
            f"p_holm_vs_{baseline_id}"
        ].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    formatted = formatted[
        [
            "ablation_id",
            "ablation_label",
            "context_label",
            "metric",
            "n_participants",
            "estimate_95ci",
            "improvement_vs_A0_95ci",
            "Holm_p_vs_A0",
            "improvement_vs_A6_95ci",
            "Holm_p_vs_A6",
        ]
    ]
    return table, formatted


def _write_readme(output_dir: Path, mode: str, reps: int, n_queries: int) -> None:
    status = (
        "FORMAL: passed the prespecified 13-participant, five-seed design checks."
        if mode == "formal"
        else "SMOKE ONLY: non-publishable; used only to validate the analysis pipeline."
    )
    text = f"""# A0--A6 controlled ablation aggregation

Status: **{status}**

The pipeline verified exact matched-query identity across A0--A6 before
calculating any result. It analysed {n_queries} query predictions per ablation.
In formal mode it additionally requires A1, A2 and A3 each to contain the exact
13-participant x five-seed Cartesian fold set, 171 queries in every seed/context
cell, and therefore 171 x 5 = 855 query records at each of C2, C4 and C6.

Inference order is fixed: metrics are calculated for each participant and seed,
then averaged across seeds within participant, and only then compared across
participants. Percentile bootstrap confidence intervals use {reps:,} paired
participant resamples. Two-sided paired Wilcoxon tests are Holm-corrected within
each baseline x context x metric family (six candidate comparisons per family).

For BA, a positive reported improvement means candidate minus baseline. For NLL
and Brier, a positive reported improvement means baseline minus candidate, so
positive always means the candidate is better. BA is missing for participants
whose future query set contains only one class; `n_participants` therefore
reports the exact inferential denominator for every cell.

A0/A4/A5 are deterministic reconstructions from A6's frozen Global probability
and saved uncalibrated log-odds shift. A5 uses the raw source-validation gamma;
A6 uses the final safety-thresholded gamma. No outer-target label is used for
gamma selection.

Variant interpretation:

- A0 is frozen Global Fusion without personalization.
- A1 is a direct attentive NP without the frozen Global anchor. Because the
  residual bound, contrast-direction rule, source-gamma activation and exact
  fallback are then not applicable, **A1 is a broader architectural comparator,
  not a one-factor ablation**. Its contrast with A6 must not be attributed only
  to removal of the anchor.
- A2 removes the hard bound while retaining the anchor and contrast direction;
  gamma is fixed to 1.
- A3 removes the direction constraint while retaining the anchor and hard bound;
  gamma is fixed to 1.
- A4 is the full anchored, bounded and direction-constrained residual at gamma=1.
- A5 adds raw source-only gamma selection but no minimum-improvement threshold.
- A6 adds the minimum source-NLL improvement rule and exact Global fallback.

Main files:

- `paper_table_formatted.csv`: compact manuscript-facing table.
- `paper_table_numeric.csv`: unrounded values behind that table.
- `participant_metrics_seed_averaged.csv`: participant-level inferential data.
- `paired_vs_Global_A0.csv` and `paired_vs_SafeNP_A6.csv`: paired tests/effects.
- `query_key_audit.csv`: strict identity hashes and row counts.
- `reconstruction_audit.json`: gamma/fallback and numerical reconstruction checks.
- `analysis_manifest.json`: analysis settings and publishability flag.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.bootstrap_reps < 1000:
        raise ValueError("Use at least 1,000 bootstrap replicates")
    if not (0.0 < args.probability_epsilon < 0.01):
        raise ValueError("--probability-epsilon must lie between 0 and 0.01")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "A1": _load_retrained_variant(args.a1_dir, "A1"),
        "A2": _load_retrained_variant(args.a2_dir, "A2"),
        "A3": _load_retrained_variant(args.a3_dir, "A3"),
    }
    a6_neural, raw_gamma, hard_bound, a6_config = _load_a6_components(args.a6_dir)

    smoke_domain: list[tuple[int, int]] | None = None
    if args.analysis_mode == "smoke":
        variants, a6_neural, smoke_domain = _restrict_smoke_domain(variants, a6_neural)
        keep_folds = set(smoke_domain)
        raw_gamma = raw_gamma.loc[
            raw_gamma.apply(
                lambda row: (int(row["seed"]), int(row["target_subject"])) in keep_folds,
                axis=1,
            )
        ].copy()

    derived, reconstruction_audit = _reconstruct_a0_a4_a5_a6(
        a6_neural,
        raw_gamma,
        hard_bound,
        args.probability_epsilon,
        args.reconstruction_tolerance,
    )
    variants.update(derived)

    key_audit = _strict_key_audit(variants)
    reconstruction_audit["source_global_invariance"] = _audit_global_invariance(
        variants
    )
    if args.analysis_mode == "formal":
        _check_formal_design(variants)

    all_queries = pd.concat(
        [variants[key] for key in sorted(variants)], ignore_index=True, sort=False
    )
    all_queries = all_queries.sort_values(
        ["ablation_id"] + IDENTITY_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    seed_metrics = _seed_subject_metrics(all_queries, args.probability_epsilon)
    participant = _participant_seed_average(seed_metrics)
    summary = _summary_table(participant, args.bootstrap_reps, args.bootstrap_seed)
    vs_global = _paired_comparisons(
        participant, "A0", args.bootstrap_reps, args.bootstrap_seed + 1
    )
    vs_a6 = _paired_comparisons(
        participant, "A6", args.bootstrap_reps, args.bootstrap_seed + 2
    )
    paper_numeric, paper_formatted = _paper_tables(summary, vs_global, vs_a6)

    all_queries.to_csv(args.output_dir / "all_query_predictions.csv", index=False)
    key_audit.to_csv(args.output_dir / "query_key_audit.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "seed_subject_metrics.csv", index=False)
    participant.to_csv(
        args.output_dir / "participant_metrics_seed_averaged.csv", index=False
    )
    summary.to_csv(args.output_dir / "ablation_summary_long.csv", index=False)
    vs_global.to_csv(args.output_dir / "paired_vs_Global_A0.csv", index=False)
    vs_a6.to_csv(args.output_dir / "paired_vs_SafeNP_A6.csv", index=False)
    paper_numeric.to_csv(args.output_dir / "paper_table_numeric.csv", index=False)
    paper_formatted.to_csv(args.output_dir / "paper_table_formatted.csv", index=False)

    with (args.output_dir / "reconstruction_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(reconstruction_audit, handle, ensure_ascii=False, indent=2)

    manifest = {
        "analysis_mode": args.analysis_mode,
        "publishable": args.analysis_mode == "formal",
        "bootstrap_replicates": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "probability_epsilon": args.probability_epsilon,
        "reconstruction_tolerance": args.reconstruction_tolerance,
        "inference_unit": "participant after within-participant averaging across seeds",
        "test": "two-sided paired Wilcoxon signed-rank",
        "multiplicity": (
            "Holm within each baseline x context x metric family; "
            "six candidate comparisons per family"
        ),
        "bootstrap": "percentile paired participant bootstrap",
        "positive_improvement_definition": {
            "balanced_accuracy": "candidate - baseline",
            "nll": "baseline - candidate",
            "brier": "baseline - candidate",
        },
        "variant_definitions": VARIANT_DEFINITIONS,
        "interpretation_limits": {
            "A1": (
                "A1 is not a single-parameter anchor-removal ablation. It is a broader "
                "direct/unanchored NP architectural comparator for which the bounded "
                "residual, direction rule, source-gamma activation and exact fallback "
                "are not applicable. Do not attribute A1-vs-A6 differences solely to "
                "the frozen anchor."
            )
        },
        "expected_formal_design": {
            "seeds": EXPECTED_SEEDS,
            "target_subjects": EXPECTED_TARGETS,
            "contexts": EXPECTED_CONTEXTS,
            "folds_per_retrained_variant": len(EXPECTED_SEEDS)
            * len(EXPECTED_TARGETS),
            "queries_per_seed_context": 171,
            "query_records_per_context": 171 * len(EXPECTED_SEEDS),
        },
        "observed": {
            "rows_per_ablation": int(len(variants["A6"])),
            "seeds": sorted(map(int, variants["A6"]["seed"].unique())),
            "target_subjects": sorted(
                map(int, variants["A6"]["target_subject"].unique())
            ),
            "contexts": sorted(map(int, variants["A6"]["context_total"].unique())),
            "variant_query_records_by_context": {
                ablation_id: {
                    str(int(context)): int(count)
                    for context, count in frame.groupby("context_total").size().items()
                }
                for ablation_id, frame in sorted(variants.items())
            },
            "retrained_variant_fold_counts": {
                ablation_id: int(
                    variants[ablation_id][["seed", "target_subject"]]
                    .drop_duplicates()
                    .shape[0]
                )
                for ablation_id in ["A1", "A2", "A3"]
            },
        },
        "smoke_domain": smoke_domain,
        "inputs": {
            "A1": str(args.a1_dir.resolve()),
            "A2": str(args.a2_dir.resolve()),
            "A3": str(args.a3_dir.resolve()),
            "A6": str(args.a6_dir.resolve()),
        },
        "a6_source_only_gamma_asserted_by_run_config": bool(
            a6_config.get("neural_process", {})
            .get("adaptation_strength_calibration", {})
            .get("outer_target_used_for_selection")
            is False
        ),
    }
    with (args.output_dir / "analysis_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    if args.analysis_mode == "smoke":
        (args.output_dir / "SMOKE_ONLY_DO_NOT_REPORT.txt").write_text(
            "This directory validates code paths only. It is not a formal result and "
            "must not be reported in the manuscript.\n",
            encoding="utf-8",
        )
    _write_readme(
        args.output_dir,
        args.analysis_mode,
        args.bootstrap_reps,
        int(len(variants["A6"])),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "analysis_mode": args.analysis_mode,
                "publishable": args.analysis_mode == "formal",
                "output_dir": str(args.output_dir.resolve()),
                "rows_per_ablation": int(len(variants["A6"])),
                "participants": int(variants["A6"]["target_subject"].nunique()),
                "seeds": int(variants["A6"]["seed"].nunique()),
                "all_query_keys_identical": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
