from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.stats import wilcoxon


FEATURE_NAMES = [
    "mean_rr",
    "median_rr",
    "min_rr",
    "max_rr",
    "range_rr",
    "sdnn",
    "rmssd",
    "sdsd",
    "pnn20",
    "pnn50",
    "mean_hr",
    "std_hr",
    "cv_rr",
    "iqr_rr",
    "mad_rr",
    "skew_rr",
    "kurtosis_rr",
    "q10_rr",
    "q25_rr",
    "q75_rr",
    "q90_rr",
    "vlf_power",
    "lf_power",
    "hf_power",
    "total_power",
    "lf_nu",
    "hf_nu",
    "lf_hf_ratio",
    "sd1",
    "sd2",
    "sd1_sd2_ratio",
    "sample_entropy",
]

REQUIRED_FILES = {
    "rr.csv",
    "events.csv",
    "questionnaire.csv",
    "arithmetic_trials.csv",
    "stroop_trials.csv",
    "session_info.json",
    "session_summary.json",
}
REQUIRED_RR_COLUMNS = {
    "timestamp",
    "heart_rate_bpm",
    "rr_ms",
    "phase",
    "rr_sequence",
    "rr_elapsed_ms",
    "packet_timestamp",
    "estimated_packet_based_timestamp",
    "rr_index_in_packet",
    "rr_count_in_packet",
    "timestamp_method",
    "qc_flag",
}
EXPECTED_PHASES = ["ADAPTATION", "BASELINE", "REST", "STRESS", "RECOVERY"]
EXPECTED_PHASE_SECONDS = {
    "ADAPTATION": 120.0,
    "BASELINE": 300.0,
    "REST": 60.0,
    "STRESS": 360.0,
    "RECOVERY": 300.0,
}
PRIMARY_PHASE_LABEL = {"BASELINE": 0, "STRESS": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and standardize repeated-session Polar H10 RR data."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--rr-min-ms", type=float, default=300.0)
    parser.add_argument("--rr-max-ms", type=float, default=2000.0)
    parser.add_argument("--rolling-median-relative-deviation", type=float, default=0.30)
    parser.add_argument("--max-gap-sec", type=float, default=10.0)
    parser.add_argument("--max-window-span-sec", type=float, default=120.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--future-query-target-per-class", type=int, default=10)
    return parser.parse_args()


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, adjusted)
        result[idx] = running
    return result


def bootstrap_mean_ci(
    values: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_resamples, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def trapz_band(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freqs[mask]))
    return float(np.trapz(psd[mask], freqs[mask]))


def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) <= m + 1:
        return 0.0
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    threshold = 0.2 * sd if r is None else r

    def count_matches(mm: int) -> int:
        count = 0
        templates = len(x) - mm + 1
        for i in range(templates):
            xi = x[i : i + mm]
            for j in range(i + 1, templates):
                if np.max(np.abs(xi - x[j : j + mm])) <= threshold:
                    count += 1
        return count

    b = count_matches(m)
    a = count_matches(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else 0.0


def frequency_features(rr_ms: np.ndarray) -> tuple[float, ...]:
    rr_sec = np.asarray(rr_ms, dtype=float) / 1000.0
    if len(rr_sec) < 8 or np.any(~np.isfinite(rr_sec)):
        return (0.0,) * 7
    t = np.cumsum(rr_sec)
    t -= t[0]
    if t[-1] <= 2.5:
        return (0.0,) * 7
    uniform_t = np.arange(0.0, t[-1], 0.25)
    if len(uniform_t) < 16:
        return (0.0,) * 7
    interpolated = np.interp(uniform_t, t, rr_sec)
    interpolated = signal.detrend(interpolated, type="constant")
    nperseg = min(256, len(interpolated))
    freqs, psd = signal.welch(
        interpolated,
        fs=4.0,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    vlf = trapz_band(freqs, psd, 0.0033, 0.04)
    lf = trapz_band(freqs, psd, 0.04, 0.15)
    hf = trapz_band(freqs, psd, 0.15, 0.40)
    total = trapz_band(freqs, psd, 0.0033, 0.40)
    denom = lf + hf
    lf_nu = lf / denom * 100.0 if denom > 0 else 0.0
    hf_nu = hf / denom * 100.0 if denom > 0 else 0.0
    lf_hf = lf / hf if hf > 1e-12 else 0.0
    return vlf, lf, hf, total, lf_nu, hf_nu, lf_hf


def compute_hrv32(rr_ms: np.ndarray) -> np.ndarray:
    rr = np.asarray(rr_ms, dtype=float)
    d = np.diff(rr)
    mean_rr = float(np.mean(rr))
    median_rr = float(np.median(rr))
    min_rr = float(np.min(rr))
    max_rr = float(np.max(rr))
    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(d**2))) if len(d) else 0.0
    sdsd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    pnn20 = float(np.mean(np.abs(d) > 20) * 100) if len(d) else 0.0
    pnn50 = float(np.mean(np.abs(d) > 50) * 100) if len(d) else 0.0
    hr = 60000.0 / rr
    mean_hr = float(np.mean(hr))
    std_hr = float(np.std(hr, ddof=1)) if len(hr) > 1 else 0.0
    cv_rr = sdnn / mean_rr if mean_rr else 0.0
    q10, q25, q75, q90 = [float(v) for v in np.quantile(rr, [0.1, 0.25, 0.75, 0.9])]
    iqr_rr = q75 - q25
    mad_rr = float(np.median(np.abs(rr - np.median(rr))))
    skew_rr = float(stats.skew(rr, bias=False)) if len(rr) >= 3 else 0.0
    kurtosis_rr = (
        float(stats.kurtosis(rr, fisher=True, bias=False)) if len(rr) >= 4 else 0.0
    )
    if not np.isfinite(skew_rr):
        skew_rr = 0.0
    if not np.isfinite(kurtosis_rr):
        kurtosis_rr = 0.0
    vlf, lf, hf, total, lf_nu, hf_nu, lf_hf = frequency_features(rr)
    sd1 = float(np.sqrt(0.5) * np.std(d, ddof=1)) if len(d) > 1 else 0.0
    sd2 = float(np.sqrt(max(0.0, 2.0 * sdnn**2 - 0.5 * sdsd**2)))
    ratio = sd1 / sd2 if sd2 > 1e-12 else 0.0
    sampen = sample_entropy(rr, m=2, r=0.2 * sdnn if sdnn > 0 else None)
    values = np.array(
        [
            mean_rr,
            median_rr,
            min_rr,
            max_rr,
            max_rr - min_rr,
            sdnn,
            rmssd,
            sdsd,
            pnn20,
            pnn50,
            mean_hr,
            std_hr,
            cv_rr,
            iqr_rr,
            mad_rr,
            skew_rr,
            kurtosis_rr,
            q10,
            q25,
            q75,
            q90,
            vlf,
            lf,
            hf,
            total,
            lf_nu,
            hf_nu,
            lf_hf,
            sd1,
            sd2,
            ratio,
            sampen,
        ],
        dtype=np.float32,
    )
    values[~np.isfinite(values)] = 0.0
    assert len(values) == 32
    return values


def select_latest_window(
    phase_df: pd.DataFrame,
    window_len: int,
    rr_min_ms: float,
    rr_max_ms: float,
    relative_deviation: float,
    max_gap_sec: float,
    max_span_sec: float,
) -> tuple[pd.DataFrame | None, dict]:
    df = phase_df.sort_values("rr_sequence").copy()
    raw_n = len(df)
    plausible = df["rr_ms"].between(rr_min_ms, rr_max_ms, inclusive="both")
    df = df[plausible].copy()
    plausible_n = len(df)
    if df.empty:
        return None, {"raw_n": raw_n, "plausible_n": 0, "artifact_filtered_n": 0}
    median = df["rr_ms"].rolling(11, center=True, min_periods=1).median()
    rel = (df["rr_ms"] - median).abs() / median.clip(lower=1e-9)
    df = df[rel <= relative_deviation].copy()
    artifact_filtered_n = len(df)
    if len(df) < window_len:
        return None, {
            "raw_n": raw_n,
            "plausible_n": plausible_n,
            "artifact_filtered_n": artifact_filtered_n,
        }
    gap = df["timestamp_utc"].diff().dt.total_seconds().fillna(0)
    df["segment_id"] = (gap > max_gap_sec).cumsum()
    for _, segment in reversed(list(df.groupby("segment_id", sort=True))):
        if len(segment) < window_len:
            continue
        chosen = segment.tail(window_len).copy()
        span = (
            chosen["timestamp_utc"].iloc[-1] - chosen["timestamp_utc"].iloc[0]
        ).total_seconds()
        if span <= max_span_sec:
            return chosen, {
                "raw_n": raw_n,
                "plausible_n": plausible_n,
                "artifact_filtered_n": artifact_filtered_n,
                "window_span_sec": float(span),
            }
    return None, {
        "raw_n": raw_n,
        "plausible_n": plausible_n,
        "artifact_filtered_n": artifact_filtered_n,
    }


def count_strict_c6_queries(frame: pd.DataFrame) -> tuple[int, int]:
    """Count queries with at least three strictly earlier samples per class."""
    visible = {0: 0, 1: 0}
    query_counts = {0: 0, 1: 0}
    ordered = frame.sort_values(["anchor_utc", "label"], kind="stable")
    for _, block in ordered.groupby("anchor_utc", sort=True):
        if visible[0] >= 3 and visible[1] >= 3:
            for label, count in block["label"].astype(int).value_counts().items():
                if label in query_counts:
                    query_counts[label] += int(count)
        # Reveal the whole equal-time block only after its query eligibility is set.
        for label, count in block["label"].astype(int).value_counts().items():
            if label in visible:
                visible[label] += int(count)
    return query_counts[0], query_counts[1]


def build_participant_coverage(
    frame: pd.DataFrame, target_per_class: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    gap_rows: list[dict] = []
    for user, g in frame.groupby("user", sort=True):
        counts = g.groupby("label").size().to_dict()
        nonstress_query_n, stress_query_n = count_strict_c6_queries(g)
        participant_code = g["participant_code"].iloc[0]
        summary_rows.append(
            {
                "participant_code": participant_code,
                "user": int(user),
                "session_n": int(g["session_id"].nunique()),
                "nonstress_window_n": int(counts.get(0, 0)),
                "stress_window_n": int(counts.get(1, 0)),
                "C6_future_nonstress_query_n_prior_session_only": nonstress_query_n,
                "C6_future_stress_query_n_prior_session_only": stress_query_n,
            }
        )
        gap_rows.append(
            {
                "participant_code": participant_code,
                "current_sessions": int(g["session_id"].nunique()),
                "current_C6_future_nonstress_queries": nonstress_query_n,
                "current_C6_future_stress_queries": stress_query_n,
                "target_future_queries_per_class": target_per_class,
                "nonstress_query_deficit": max(
                    0, target_per_class - nonstress_query_n
                ),
                "stress_query_deficit": max(0, target_per_class - stress_query_n),
                "additional_sessions_needed_if_one_window_per_class_per_session": max(
                    max(0, target_per_class - nonstress_query_n),
                    max(0, target_per_class - stress_query_n),
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(gap_rows)


def safe_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def paired_summary(
    wide: pd.DataFrame,
    outcomes: list[str],
    phase_a: str,
    phase_b: str,
    n_resamples: int,
    family: str,
) -> list[dict]:
    rows = []
    for index, outcome in enumerate(outcomes):
        a_col = f"{outcome}_{phase_a}"
        b_col = f"{outcome}_{phase_b}"
        pair = wide[[a_col, b_col]].dropna()
        diff = pair[a_col].to_numpy(float) - pair[b_col].to_numpy(float)
        if not len(diff):
            statistic = p_value = np.nan
        elif np.allclose(diff, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        lo, hi = bootstrap_mean_ci(diff, n_resamples, 20_261_100 + index)
        rows.append(
            {
                "family": family,
                "outcome": outcome,
                "phase_a": phase_a,
                "phase_b": phase_b,
                "participant_n": int(len(diff)),
                "mean_a": float(pair[a_col].mean()) if len(pair) else np.nan,
                "mean_b": float(pair[b_col].mean()) if len(pair) else np.nan,
                "mean_paired_delta_a_minus_b": float(diff.mean()) if len(diff) else np.nan,
                "bootstrap_delta_ci_low": lo,
                "bootstrap_delta_ci_high": hi,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    session_rows: list[dict] = []
    phase_rows: list[dict] = []
    questionnaire_rows: list[dict] = []
    trial_rows: list[dict] = []
    window_rows: list[dict] = []
    rr_arrays: list[np.ndarray] = []
    hrv_arrays: list[np.ndarray] = []

    participant_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for participant_dir in participant_dirs:
        participant_code = participant_dir.name
        participant_numeric = int(participant_code.lstrip("P"))
        for session_dir in sorted(path for path in participant_dir.iterdir() if path.is_dir()):
            session_id = session_dir.name
            session_order = int(session_id.lstrip("S"))
            present_files = {path.name for path in session_dir.iterdir() if path.is_file()}
            missing_files = sorted(REQUIRED_FILES - present_files)
            if missing_files:
                session_rows.append(
                    {
                        "participant_code": participant_code,
                        "participant_numeric": participant_numeric,
                        "session_id": session_id,
                        "session_order": session_order,
                        "missing_required_files": ";".join(missing_files),
                        "signal_qc_pass": False,
                    }
                )
                continue

            info = safe_json(session_dir / "session_info.json")
            summary = safe_json(session_dir / "session_summary.json")
            rr = pd.read_csv(session_dir / "rr.csv")
            missing_columns = sorted(REQUIRED_RR_COLUMNS - set(rr.columns))
            if missing_columns:
                raise ValueError(f"{session_dir}: missing RR columns {missing_columns}")
            rr["timestamp_utc"] = pd.to_datetime(rr["timestamp"], utc=True, errors="coerce")
            rr["rr_ms"] = pd.to_numeric(rr["rr_ms"], errors="coerce")
            rr["rr_sequence"] = pd.to_numeric(rr["rr_sequence"], errors="coerce")
            rr["rr_elapsed_ms"] = pd.to_numeric(rr["rr_elapsed_ms"], errors="coerce")
            rr["qc_flag"] = rr["qc_flag"].fillna("").astype(str)

            questionnaire = pd.read_csv(session_dir / "questionnaire.csv")
            questionnaire["participant_code"] = participant_code
            questionnaire["participant_numeric"] = participant_numeric
            questionnaire["session_id"] = session_id
            questionnaire["session_order"] = session_order
            questionnaire_rows.extend(questionnaire.to_dict("records"))

            arithmetic = pd.read_csv(session_dir / "arithmetic_trials.csv")
            stroop = pd.read_csv(session_dir / "stroop_trials.csv")
            for task, trials in [("arithmetic", arithmetic), ("stroop", stroop)]:
                trial_rows.append(
                    {
                        "participant_code": participant_code,
                        "session_id": session_id,
                        "session_order": session_order,
                        "task": task,
                        "trial_n": int(len(trials)),
                        "correct_n": int(pd.to_numeric(trials["is_correct"], errors="coerce").fillna(0).sum()),
                        "timeout_n": int(pd.to_numeric(trials["is_timeout"], errors="coerce").fillna(0).sum()),
                        "accuracy": float(pd.to_numeric(trials["is_correct"], errors="coerce").mean()),
                        "median_reaction_time_ms": float(
                            pd.to_numeric(trials["reaction_time_ms"], errors="coerce").median()
                        ),
                    }
                )

            sequence_diff = rr["rr_sequence"].diff().dropna()
            elapsed_diff = rr["rr_elapsed_ms"].diff()
            elapsed_expected = rr["rr_ms"]
            elapsed_error = (elapsed_diff - elapsed_expected).abs().iloc[1:]
            timestamp_diff = rr["timestamp_utc"].diff().dt.total_seconds()
            plausible = rr["rr_ms"].between(args.rr_min_ms, args.rr_max_ms)
            core_coverage = []
            for phase in EXPECTED_PHASES:
                phase_df = rr[rr["phase"] == phase].copy()
                rr_sum_sec = float(phase_df["rr_ms"].sum() / 1000.0)
                expected_sec = EXPECTED_PHASE_SECONDS[phase]
                coverage = rr_sum_sec / expected_sec if expected_sec else np.nan
                core_coverage.append(coverage)
                phase_rows.append(
                    {
                        "participant_code": participant_code,
                        "participant_numeric": participant_numeric,
                        "session_id": session_id,
                        "session_order": session_order,
                        "phase": phase,
                        "rr_rows": int(len(phase_df)),
                        "rr_sum_sec": rr_sum_sec,
                        "expected_sec": expected_sec,
                        "coverage_ratio": coverage,
                        "rr_outside_300_2000_n": int(
                            (~phase_df["rr_ms"].between(args.rr_min_ms, args.rr_max_ms)).sum()
                        ),
                        "collector_flagged_n": int((phase_df["qc_flag"] != "").sum()),
                    }
                )

            session_start = pd.to_datetime(info.get("session_start"), utc=True)
            structural_qc_pass = bool(
                len(rr) == int(summary.get("rr_rows_total", -1))
                and rr["timestamp_utc"].notna().all()
                and rr["rr_ms"].notna().all()
                and (sequence_diff == 1).all()
                and (timestamp_diff.dropna() > 0).all()
                and (elapsed_error <= 0.01).all()
                and all(0.95 <= value <= 1.05 for value in core_coverage)
                and len(questionnaire) == 3
                and bool(summary.get("basic_acquisition_checks", {}).get("normal_complete", False))
            )
            session_rows.append(
                {
                    "participant_code": participant_code,
                    "participant_numeric": participant_numeric,
                    "session_id": session_id,
                    "session_order": session_order,
                    "session_start_utc": session_start,
                    "protocol_version": info.get("protocol_version"),
                    "software_version": info.get("software_version"),
                    "protocol_locked_for_formal_collection": bool(
                        info.get("protocol_locked_for_formal_collection", False)
                    ),
                    "end_reason": summary.get("end_reason"),
                    "rr_rows_total": int(len(rr)),
                    "rr_outside_300_2000_n": int((~plausible).sum()),
                    "collector_flagged_rr_n": int((rr["qc_flag"] != "").sum()),
                    "sequence_gap_or_duplicate_n": int((sequence_diff != 1).sum()),
                    "nonmonotonic_analysis_timestamp_n": int((timestamp_diff.dropna() <= 0).sum()),
                    "analysis_gap_gt10s_n": int((timestamp_diff.dropna() > args.max_gap_sec).sum()),
                    "elapsed_increment_error_gt_0_01ms_n": int((elapsed_error > 0.01).sum()),
                    "duplicate_timestamp_n": int(rr["timestamp_utc"].duplicated().sum()),
                    "questionnaire_rows": int(len(questionnaire)),
                    "arithmetic_trial_n": int(len(arithmetic)),
                    "stroop_trial_n": int(len(stroop)),
                    "missing_required_files": "",
                    "structural_qc_pass": structural_qc_pass,
                    "signal_qc_pass": structural_qc_pass,
                    "confirmatory_protocol_ready": bool(
                        structural_qc_pass
                        and info.get("protocol_locked_for_formal_collection", False)
                    ),
                }
            )

            q_by_phase = questionnaire.set_index("after_phase")
            for phase in ["BASELINE", "STRESS", "RECOVERY"]:
                selected, qc = select_latest_window(
                    rr[rr["phase"] == phase],
                    window_len=args.window_len,
                    rr_min_ms=args.rr_min_ms,
                    rr_max_ms=args.rr_max_ms,
                    relative_deviation=args.rolling_median_relative_deviation,
                    max_gap_sec=args.max_gap_sec,
                    max_span_sec=args.max_window_span_sec,
                )
                if selected is None:
                    window_rows.append(
                        {
                            "participant_code": participant_code,
                            "user": participant_numeric,
                            "session_id": session_id,
                            "session_order": session_order,
                            "phase": phase,
                            "window_qc_pass": False,
                            **qc,
                        }
                    )
                    continue
                rr64 = selected["rr_ms"].to_numpy(dtype=np.float32)
                hrv32 = compute_hrv32(rr64)
                physiology_anchor = selected["timestamp_utc"].iloc[-1]
                q = q_by_phase.loc[phase] if phase in q_by_phase.index else pd.Series(dtype=float)
                row = {
                    "participant_code": participant_code,
                    "user": participant_numeric,
                    "session_id": session_id,
                    "session_order": session_order,
                    "session_date": session_start.date().isoformat(),
                    "phase": phase,
                    "label": PRIMARY_PHASE_LABEL.get(phase, 2),
                    "label_name": (
                        "nonstress" if phase == "BASELINE" else "stress" if phase == "STRESS" else "recovery"
                    ),
                    "anchor_utc": session_start,
                    "physiology_anchor_utc": physiology_anchor,
                    "rr_start_utc": selected["timestamp_utc"].iloc[0],
                    "rr_end_utc": selected["timestamp_utc"].iloc[-1],
                    "rr_len": args.window_len,
                    "window_method": "latest_analysis_qc_64_within_phase",
                    "window_qc_pass": True,
                    "collector_flagged_in_selected_window_n": int((selected["qc_flag"] != "").sum()),
                    "collector_flag_free_window": bool((selected["qc_flag"] == "").all()),
                    "stress_nrs_0_10": pd.to_numeric(q.get("stress_nrs_0_10"), errors="coerce"),
                    "arousal_sam_1_9": pd.to_numeric(q.get("arousal_sam_1_9"), errors="coerce"),
                    "valence_sam_1_9": pd.to_numeric(q.get("valence_sam_1_9"), errors="coerce"),
                    "mental_demand_0_10": pd.to_numeric(q.get("mental_demand_0_10"), errors="coerce"),
                    **qc,
                }
                row.update({name: float(value) for name, value in zip(FEATURE_NAMES, hrv32)})
                window_rows.append(row)
                if phase in PRIMARY_PHASE_LABEL:
                    rr_arrays.append(rr64)
                    hrv_arrays.append(hrv32)

    sessions = pd.DataFrame(session_rows).sort_values(
        ["participant_numeric", "session_order"]
    )
    phases = pd.DataFrame(phase_rows).sort_values(
        ["participant_numeric", "session_order", "phase"]
    )
    questionnaire = pd.DataFrame(questionnaire_rows).sort_values(
        ["participant_numeric", "session_order", "after_phase"]
    )
    trials = pd.DataFrame(trial_rows).sort_values(
        ["participant_code", "session_order", "task"]
    )
    windows = pd.DataFrame(window_rows).sort_values(
        ["user", "session_order", "phase"]
    )

    primary = windows[
        windows["phase"].isin(PRIMARY_PHASE_LABEL)
        & windows["window_qc_pass"].fillna(False)
    ].copy()
    primary = primary.sort_values(["user", "anchor_utc", "label"]).reset_index(drop=True)
    primary["sample_index"] = np.arange(len(primary), dtype=int)
    primary["subject_event_order"] = primary.groupby("user").cumcount() + 1
    primary["subject_prev_anchor_utc"] = primary.groupby("user")["anchor_utc"].shift(1)
    primary["minutes_since_prev_sample"] = (
        (primary["anchor_utc"] - primary["subject_prev_anchor_utc"]).dt.total_seconds()
        / 60.0
    )
    primary["prev_rr_end_utc"] = primary.groupby("user")["rr_end_utc"].shift(1)
    primary["overlaps_previous_sample_rr_window"] = (
        primary["prev_rr_end_utc"].notna()
        & (primary["rr_start_utc"] <= primary["prev_rr_end_utc"])
    )

    if len(primary) != len(rr_arrays):
        raise AssertionError("Primary manifest and array counts differ")
    array_lookup = {
        (row["participant_code"], row["session_id"], row["phase"]): (rr_values, hrv_values)
        for row, rr_values, hrv_values in zip(
            [r for r in window_rows if r.get("phase") in PRIMARY_PHASE_LABEL and r.get("window_qc_pass")],
            rr_arrays,
            hrv_arrays,
        )
    }
    ordered_arrays = [
        array_lookup[(r.participant_code, r.session_id, r.phase)]
        for r in primary.itertuples(index=False)
    ]
    X_rr = np.stack([item[0] for item in ordered_arrays]).astype(np.float32)
    X_hrv = np.stack([item[1] for item in ordered_arrays]).astype(np.float32)
    y = primary["label"].to_numpy(dtype=np.int64)
    subject_ids = primary["user"].to_numpy(dtype=np.int64)
    anchor_unix_ns = pd.to_datetime(primary["anchor_utc"], utc=True).astype("int64").to_numpy()

    np.savez_compressed(
        out / "polar_longitudinal_strict.npz",
        X_rr=X_rr,
        X_hrv=X_hrv,
        y=y,
        subject_ids=subject_ids,
        anchor_unix_ns=anchor_unix_ns,
        session_order=primary["session_order"].to_numpy(dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U"),
    )

    participant_summary, collection_gap = build_participant_coverage(
        primary, args.future_query_target_per_class
    )
    flag_free_primary = primary[
        primary["collector_flagged_in_selected_window_n"].fillna(0).eq(0)
    ].copy()
    # Preserve the row identity in the complete primary manifest for traceability,
    # while assigning a dense index that is aligned exactly with the filtered NPZ.
    flag_free_primary["original_main_sample_index"] = flag_free_primary["sample_index"]
    flag_free_primary = flag_free_primary.reset_index(drop=True)
    flag_free_primary["sample_index"] = np.arange(len(flag_free_primary), dtype=int)
    flag_free_ordered_arrays = [
        array_lookup[(r.participant_code, r.session_id, r.phase)]
        for r in flag_free_primary.itertuples(index=False)
    ]
    np.savez_compressed(
        out / "polar_longitudinal_collector_flag_free_sensitivity.npz",
        X_rr=np.stack([item[0] for item in flag_free_ordered_arrays]).astype(np.float32),
        X_hrv=np.stack([item[1] for item in flag_free_ordered_arrays]).astype(np.float32),
        y=flag_free_primary["label"].to_numpy(dtype=np.int64),
        subject_ids=flag_free_primary["user"].to_numpy(dtype=np.int64),
        anchor_unix_ns=pd.to_datetime(
            flag_free_primary["anchor_utc"], utc=True
        ).astype("int64").to_numpy(),
        session_order=flag_free_primary["session_order"].to_numpy(dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U"),
    )
    flag_free_summary, _ = build_participant_coverage(
        flag_free_primary, args.future_query_target_per_class
    )

    q_numeric = [
        "stress_nrs_0_10",
        "arousal_sam_1_9",
        "valence_sam_1_9",
    ]
    q_participant = (
        questionnaire.groupby(["participant_code", "after_phase"], as_index=False)[q_numeric]
        .mean()
        .pivot(index="participant_code", columns="after_phase", values=q_numeric)
    )
    q_participant.columns = [f"{metric}_{phase}" for metric, phase in q_participant.columns]
    q_participant = q_participant.reset_index()
    manipulation_rows = paired_summary(
        q_participant,
        q_numeric,
        "STRESS",
        "BASELINE",
        args.bootstrap_resamples,
        "questionnaire_stress_vs_baseline",
    )
    manipulation_rows += paired_summary(
        q_participant,
        q_numeric,
        "RECOVERY",
        "BASELINE",
        args.bootstrap_resamples,
        "questionnaire_recovery_vs_baseline",
    )

    physiological_metrics = ["mean_hr", "rmssd", "sdnn"]
    phys_participant = (
        windows[windows["window_qc_pass"].fillna(False)]
        .groupby(["participant_code", "phase"], as_index=False)[physiological_metrics]
        .mean()
        .pivot(index="participant_code", columns="phase", values=physiological_metrics)
    )
    phys_participant.columns = [
        f"{metric}_{phase}" for metric, phase in phys_participant.columns
    ]
    phys_participant = phys_participant.reset_index()
    manipulation_rows += paired_summary(
        phys_participant,
        physiological_metrics,
        "STRESS",
        "BASELINE",
        args.bootstrap_resamples,
        "physiology_stress_vs_baseline",
    )
    manipulation_rows += paired_summary(
        phys_participant,
        physiological_metrics,
        "RECOVERY",
        "BASELINE",
        args.bootstrap_resamples,
        "physiology_recovery_vs_baseline",
    )
    manipulation = pd.DataFrame(manipulation_rows)
    manipulation["p_holm_within_family"] = np.nan
    for _, idx in manipulation.groupby("family").groups.items():
        idx = list(idx)
        manipulation.loc[idx, "p_holm_within_family"] = holm_adjust(
            manipulation.loc[idx, "p_raw"]
        )

    sessions.to_csv(out / "polar_session_qc.csv", index=False, encoding="utf-8-sig")
    phases.to_csv(out / "polar_phase_qc.csv", index=False, encoding="utf-8-sig")
    questionnaire.to_csv(
        out / "polar_questionnaire_long.csv", index=False, encoding="utf-8-sig"
    )
    trials.to_csv(out / "polar_task_summary.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(out / "polar_all_phase_windows.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(
        out / "polar_longitudinal_strict_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    flag_free_primary.to_csv(
        out / "polar_collector_flag_free_sensitivity_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    flag_free_summary.to_csv(
        out / "polar_collector_flag_free_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    participant_summary.to_csv(
        out / "polar_subject_summary.csv", index=False, encoding="utf-8-sig"
    )
    collection_gap.to_csv(
        out / "polar_collection_gap.csv", index=False, encoding="utf-8-sig"
    )
    q_participant.to_csv(
        out / "polar_questionnaire_participant_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    phys_participant.to_csv(
        out / "polar_physiology_participant_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manipulation.to_csv(
        out / "polar_manipulation_checks.csv", index=False, encoding="utf-8-sig"
    )

    protocol_locked_n = int(sessions["protocol_locked_for_formal_collection"].sum())
    manifest = {
        "participant_n": int(sessions["participant_code"].nunique()),
        "session_n": int(len(sessions)),
        "sessions_per_participant": sorted(
            sessions.groupby("participant_code")["session_id"].nunique().unique().tolist()
        ),
        "structural_qc_pass_session_n": int(sessions["structural_qc_pass"].sum()),
        "structural_qc_fail_session_n": int((~sessions["structural_qc_pass"]).sum()),
        "signal_qc_pass_session_n_legacy_alias": int(sessions["signal_qc_pass"].sum()),
        "protocol_locked_session_n": protocol_locked_n,
        "protocol_not_locked_session_n": int(len(sessions) - protocol_locked_n),
        "primary_window_n": int(len(primary)),
        "primary_nonstress_window_n": int((primary["label"] == 0).sum()),
        "primary_stress_window_n": int((primary["label"] == 1).sum()),
        "primary_window_with_collector_flag_n": int(
            (primary["collector_flagged_in_selected_window_n"] > 0).sum()
        ),
        "collector_flagged_rr_in_primary_windows_n": int(
            primary["collector_flagged_in_selected_window_n"].sum()
        ),
        "collector_flag_free_sensitivity_window_n": int(len(flag_free_primary)),
        "collector_flag_free_sensitivity_participant_with_both_c6_query_classes_n": int(
            (
                (flag_free_summary["C6_future_nonstress_query_n_prior_session_only"] > 0)
                & (flag_free_summary["C6_future_stress_query_n_prior_session_only"] > 0)
            ).sum()
        ),
        "collector_flag_free_sensitivity_C6_future_query_n": int(
            flag_free_summary[
                [
                    "C6_future_nonstress_query_n_prior_session_only",
                    "C6_future_stress_query_n_prior_session_only",
                ]
            ].to_numpy().sum()
        ),
        "C6_future_query_n_prior_session_only": int(
            participant_summary[
                [
                    "C6_future_nonstress_query_n_prior_session_only",
                    "C6_future_stress_query_n_prior_session_only",
                ]
            ].to_numpy().sum()
        ),
        "C6_future_nonstress_query_n_prior_session_only": int(
            participant_summary["C6_future_nonstress_query_n_prior_session_only"].sum()
        ),
        "C6_future_stress_query_n_prior_session_only": int(
            participant_summary["C6_future_stress_query_n_prior_session_only"].sum()
        ),
        "total_additional_sessions_for_10_future_queries_per_class": int(
            collection_gap[
                "additional_sessions_needed_if_one_window_per_class_per_session"
            ].sum()
        ),
        "primary_binary_definition": "BASELINE=0, STRESS=1; RECOVERY excluded from classifier",
        "window_definition": (
            "latest 64 RR passing analysis-specific plausibility, local-deviation, "
            "gap and span criteria within each phase; collector flags are audited "
            "separately rather than treated as an automatic exclusion"
        ),
        "chronology_definition": (
            "both phase samples in a session use session_start as the model order anchor, "
            "which prevents same-session context and enforces prior-session-only C2/C4/C6"
        ),
        "privacy": (
            "outputs contain pseudonymous participant/session codes only; device names and "
            "Bluetooth addresses are excluded"
        ),
        "confirmatory_caution": (
            "all session_info records report protocol_locked_for_formal_collection=false; "
            "treat this collection as external pilot validation unless an independently "
            "documented protocol freeze predates enrollment"
        ),
    }
    (out / "polar_qc_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
