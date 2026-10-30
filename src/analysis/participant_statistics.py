from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


SOURCE = Path(
    'data/restricted/polar/raw_runs'
)
OUT = Path(
    'runs/polar_statistics'
)
OUT.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N = 20_000
BOOTSTRAP_SEED = 20_261_013
METRICS = ["balanced_accuracy", "macro_f1", "auroc", "brier", "nll"]
HIGHER_IS_BETTER = {
    "balanced_accuracy": True,
    "macro_f1": True,
    "auroc": True,
    "brier": False,
    "nll": False,
}
METHOD_LABELS = {
    "GlobalFusion": "GlobalFusion",
    "GrowingPrototype": "GrowingPrototype",
    "FineTuning": "FineTuning",
    "NeuralProcess": "SafeNeuralProcessV17",
}
EXPECTED_SEEDS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_PARTICIPANTS = list(range(1, 14))
ANALYSIS_STATUS = "external_pilot_exploratory_viewed_cohort"
RUN_LABEL = "Polar v1.7"
OUTPUT_PREFIX = "polar_v17"
CONFIRMATORY_WARNING = (
    "The Polar cohort and earlier S07 results had already been inspected; "
    "this v1.7 run is external/pipeline evidence, not an independent confirmatory test."
)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    draws = rng.choice(x, size=(BOOTSTRAP_N, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return out
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, adjusted)
        out[idx] = running
    return out


def validate_matched_predictions(df: pd.DataFrame) -> dict:
    required = {
        "seed",
        "target_subject",
        "query_original_sample_index",
        "query_anchor_utc",
        "y_true",
        "context_total",
        "method",
        "p_stress",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    key = [
        "seed",
        "target_subject",
        "query_original_sample_index",
        "query_anchor_utc",
        "y_true",
        "context_total",
    ]
    if df.duplicated(key + ["method"]).any():
        raise ValueError("Duplicate method predictions found for the same query key")

    counts = df.groupby(key, sort=False)["method"].nunique()
    method_n = df["method"].nunique()
    if not (counts == method_n).all():
        raise ValueError("Methods are not evaluated on exactly the same query keys")

    signatures = {
        method: set(map(tuple, g[key].itertuples(index=False, name=None)))
        for method, g in df.groupby("method", sort=True)
    }
    signature_values = list(signatures.values())
    if any(s != signature_values[0] for s in signature_values[1:]):
        raise ValueError("Method query signatures differ")

    probability = pd.to_numeric(df["p_stress"], errors="coerce")
    if probability.isna().any() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid stress probabilities")

    expected_methods = set(METHOD_LABELS)
    observed_methods = set(df["method"].astype(str).unique())
    if observed_methods != expected_methods:
        raise ValueError(
            f"Expected methods {sorted(expected_methods)}, observed {sorted(observed_methods)}"
        )

    return {
        "row_n": int(len(df)),
        "method_n": int(method_n),
        "query_key_n_per_method": int(len(signature_values[0])),
        "seeds": sorted(df["seed"].astype(int).unique().tolist()),
        "participants": sorted(df["target_subject"].astype(int).unique().tolist()),
        "contexts": sorted(df["context_total"].astype(int).unique().tolist()),
        "methods": sorted(observed_methods),
    }


def subject_metrics_from_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["seed", "target_subject", "context_total", "method"]
    for (seed, subject, context_total, method), g in df.groupby(keys, sort=True):
        y = g["y_true"].to_numpy(dtype=int)
        p = np.clip(g["p_stress"].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
        pred = (p >= 0.5).astype(int)
        both = np.unique(y).size == 2
        rows.append(
            {
                "seed": int(seed),
                "target_subject": int(subject),
                "method": METHOD_LABELS[str(method)],
                "context_total": int(context_total),
                "query_n": int(len(g)),
                "query_nonstress_n": int((y == 0).sum()),
                "query_stress_n": int((y == 1).sum()),
                "has_both_query_classes": bool(both),
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": (
                    float(balanced_accuracy_score(y, pred)) if both else np.nan
                ),
                "macro_f1": (
                    float(f1_score(y, pred, average="macro", zero_division=0))
                    if both
                    else np.nan
                ),
                "auroc": float(roc_auc_score(y, p)) if both else np.nan,
                "brier": float(brier_score_loss(y, p)),
                "nll": float(
                    log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])
                ),
            }
        )
    return pd.DataFrame(rows)


def seed_average(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    ids = ["target_subject", "method", "context_total"]
    numeric = [
        "query_n",
        "query_nonstress_n",
        "query_stress_n",
        "accuracy",
        *METRICS,
    ]
    out = subject_metrics.groupby(ids, sort=True, as_index=False)[numeric].mean()
    seed_counts = (
        subject_metrics.groupby(ids, sort=True)["seed"]
        .nunique()
        .rename("seed_n")
        .reset_index()
    )
    out = out.merge(seed_counts, on=ids, how="left", validate="one_to_one")
    cols = ids + ["seed_n"] + numeric
    return out[cols]


def summarize_methods(seed_avg: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    for (method, context_total), g in seed_avg.groupby(
        ["method", "context_total"], sort=True
    ):
        for metric in METRICS:
            x = g[metric].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            lo, hi = bootstrap_mean_ci(x, rng)
            rows.append(
                {
                    "method": str(method),
                    "context_total": int(context_total),
                    "metric": metric,
                    "participant_n": int(x.size),
                    "mean": float(np.mean(x)) if x.size else np.nan,
                    "sd": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
                    "median": float(np.median(x)) if x.size else np.nan,
                    "bootstrap_mean_ci_low": lo,
                    "bootstrap_mean_ci_high": hi,
                    "bootstrap_resamples": BOOTSTRAP_N,
                    "analysis_status": ANALYSIS_STATUS,
                }
            )
    return pd.DataFrame(rows)


def paired_test_rows(seed_avg: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        (
            "GrowingPrototype_C6_vs_GlobalFusion_C6",
            "GrowingPrototype",
            6,
            "GlobalFusion",
            6,
        ),
        (
            "SafeNeuralProcessV17_C6_vs_GlobalFusion_C6",
            "SafeNeuralProcessV17",
            6,
            "GlobalFusion",
            6,
        ),
        (
            "SafeNeuralProcessV17_C6_vs_GrowingPrototype_C6",
            "SafeNeuralProcessV17",
            6,
            "GrowingPrototype",
            6,
        ),
        (
            "SafeNeuralProcessV17_C6_vs_FineTuning_C6",
            "SafeNeuralProcessV17",
            6,
            "FineTuning",
            6,
        ),
        (
            "SafeNeuralProcessV17_C6_vs_SafeNeuralProcessV17_C2",
            "SafeNeuralProcessV17",
            6,
            "SafeNeuralProcessV17",
            2,
        ),
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows: list[dict] = []
    for comp, method_a, context_a, method_b, context_b in comparisons:
        a = seed_avg[
            (seed_avg["method"] == method_a)
            & (seed_avg["context_total"] == context_a)
        ][["target_subject", *METRICS]].copy()
        b = seed_avg[
            (seed_avg["method"] == method_b)
            & (seed_avg["context_total"] == context_b)
        ][["target_subject", *METRICS]].copy()
        merged = a.merge(b, on="target_subject", suffixes=("_a", "_b"))
        for metric in METRICS:
            pair = merged[[f"{metric}_a", f"{metric}_b"]].dropna()
            raw = pair[f"{metric}_a"].to_numpy(float) - pair[
                f"{metric}_b"
            ].to_numpy(float)
            benefit = raw if HIGHER_IS_BETTER[metric] else -raw
            lo, hi = bootstrap_mean_ci(benefit, rng)
            if benefit.size == 0:
                stat = p_raw = np.nan
            elif np.allclose(benefit, 0.0, rtol=0.0, atol=1e-15):
                stat, p_raw = 0.0, 1.0
            else:
                test = wilcoxon(
                    benefit,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                stat, p_raw = float(test.statistic), float(test.pvalue)
            sd = float(np.std(benefit, ddof=1)) if benefit.size > 1 else np.nan
            rows.append(
                {
                    "comparison": comp,
                    "method_a": method_a,
                    "context_a": context_a,
                    "method_b": method_b,
                    "context_b": context_b,
                    "metric": metric,
                    "participant_n": int(benefit.size),
                    "mean_raw_delta_a_minus_b": (
                        float(np.mean(raw)) if raw.size else np.nan
                    ),
                    "mean_benefit_delta_positive_favors_a": (
                        float(np.mean(benefit)) if benefit.size else np.nan
                    ),
                    "median_benefit_delta": (
                        float(np.median(benefit)) if benefit.size else np.nan
                    ),
                    "bootstrap_mean_benefit_ci_low": lo,
                    "bootstrap_mean_benefit_ci_high": hi,
                    "cohen_dz_benefit": (
                        float(np.mean(benefit) / sd)
                        if benefit.size > 1 and np.isfinite(sd) and sd > 0
                        else np.nan
                    ),
                    "wilcoxon_statistic": stat,
                    "p_raw": p_raw,
                    "analysis_unit": "participant_after_averaging_5_seeds",
                    "bootstrap_resamples": BOOTSTRAP_N,
                    "analysis_status": ANALYSIS_STATUS,
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm_all_25_tests"] = holm_adjust(out["p_raw"].to_numpy())
    out["p_holm_within_metric"] = np.nan
    for _, idx in out.groupby("metric").groups.items():
        idx = list(idx)
        out.loc[idx, "p_holm_within_metric"] = holm_adjust(
            out.loc[idx, "p_raw"].to_numpy()
        )
    return out


def gamma_audit(fold_summary: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seed",
        "target_subject",
        "source_selected_adaptation_gamma",
        "source_raw_best_adaptation_gamma",
        "source_gamma0_balanced_nll",
        "source_gamma_raw_best_balanced_nll",
        "source_gamma_nll_improvement_vs_gamma0",
        "source_gamma_minimum_required_nll_improvement",
        "source_gamma_fallback_to_global",
        "source_gamma_fallback_reason",
        "source_gamma_calibration_subject_n",
        "source_gamma_calibration_episode_n",
    ]
    audit = fold_summary[cols].copy()
    audit["source_gamma_fallback_to_global"] = (
        audit["source_gamma_fallback_to_global"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(audit["source_gamma_fallback_to_global"])
        .astype(bool)
    )

    key = [
        "seed",
        "target_subject",
        "query_original_sample_index",
        "query_anchor_utc",
        "y_true",
        "context_total",
    ]
    global_p = matched[matched["method"] == "GlobalFusion"][key + ["p_stress"]].rename(
        columns={"p_stress": "p_global"}
    )
    np_p = matched[matched["method"] == "NeuralProcess"][
        key + ["p_stress", "np_adaptation_gamma"]
    ].rename(columns={"p_stress": "p_np"})
    check = np_p.merge(global_p, on=key, how="inner", validate="one_to_one")
    check = check.merge(
        audit[["seed", "target_subject", "source_gamma_fallback_to_global"]],
        on=["seed", "target_subject"],
        validate="many_to_one",
    )
    fallback = check[check["source_gamma_fallback_to_global"]]
    max_diff = (
        float(np.max(np.abs(fallback["p_np"] - fallback["p_global"])))
        if len(fallback)
        else 0.0
    )
    if max_diff > 1e-15:
        raise AssertionError(
            f"gamma=0 fallback is not exactly GlobalFusion; max difference={max_diff}"
        )
    audit["gamma0_exact_global_verified"] = True
    audit["gamma0_max_abs_probability_difference"] = max_diff
    audit["outer_target_used_for_gamma_selection"] = False
    audit["analysis_status"] = ANALYSIS_STATUS
    return audit


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    c6 = summary[summary["context_total"] == 6]
    rows: list[dict] = []
    order = [
        "GlobalFusion",
        "GrowingPrototype",
        "FineTuning",
        "SafeNeuralProcessV17",
    ]
    for method in order:
        record: dict[str, str | int] = {"method": method, "context_total": 6}
        for metric in METRICS:
            r = c6[(c6["method"] == method) & (c6["metric"] == metric)].iloc[0]
            record[metric] = (
                f"{r['mean']:.3f} [{r['bootstrap_mean_ci_low']:.3f}, "
                f"{r['bootstrap_mean_ci_high']:.3f}] (n={int(r['participant_n'])})"
            )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    matched_path = SOURCE / "matched_query_results.csv"
    fold_path = SOURCE / "fold_training_summary.csv"
    if not matched_path.exists() or not fold_path.exists():
        raise FileNotFoundError(
            "The complete Polar v1.7 run has not finished: final CSV files are absent."
        )

    matched = pd.read_csv(matched_path)
    fold_summary = pd.read_csv(fold_path)
    validation = validate_matched_predictions(matched)
    if validation["seeds"] != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seed set: {validation['seeds']}")
    if validation["participants"] != EXPECTED_PARTICIPANTS:
        raise ValueError(f"Unexpected participant set: {validation['participants']}")

    subject_metrics = subject_metrics_from_predictions(matched)
    averaged = seed_average(subject_metrics)
    summary = summarize_methods(averaged)
    paired = paired_test_rows(averaged)
    audit = gamma_audit(fold_summary, matched)
    table = paper_table(summary)

    outputs = {
        f"{OUTPUT_PREFIX}_subject_seed_metrics.csv": subject_metrics,
        f"{OUTPUT_PREFIX}_participant_seed_averaged_metrics.csv": averaged,
        f"{OUTPUT_PREFIX}_participant_method_summary.csv": summary,
        f"{OUTPUT_PREFIX}_participant_paired_tests.csv": paired,
        f"{OUTPUT_PREFIX}_gamma_safety_audit.csv": audit,
        f"{OUTPUT_PREFIX}_paper_table_c6.csv": table,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    report = {
        **validation,
        "bootstrap_resamples": BOOTSTRAP_N,
        "safe_gamma_active_fold_n": int(
            (~audit["source_gamma_fallback_to_global"]).sum()
        ),
        "safe_gamma_fallback_fold_n": int(
            audit["source_gamma_fallback_to_global"].sum()
        ),
        "safe_gamma_fold_n": int(len(audit)),
        "gamma0_max_abs_probability_difference_from_global": float(
            audit["gamma0_max_abs_probability_difference"].max()
        ),
        "analysis_unit": "participant_after_averaging_5_seeds",
        "run_label": RUN_LABEL,
        "analysis_status": ANALYSIS_STATUS,
        "confirmatory_warning": CONFIRMATORY_WARNING,
    }
    (OUT / f"{OUTPUT_PREFIX}_analysis_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
