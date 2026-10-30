#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WESAD ECG/HRV subject-wise preprocessing + LOSO Random Forest baseline
=====================================================================

功能
----
1. 自动读取 WESAD/S2/S2.pkl ... WESAD/S17/S17.pkl
2. 使用胸部 RespiBAN ECG（700 Hz）
3. 使用 WESAD 官方同步标签：
      1 = baseline
      2 = stress
      3 = amusement
   默认二分类：
      0 = non-stress（baseline + amusement）
      1 = stress
   忽略 0、4、5、6、7
4. 按“每名受试者、每个连续状态区间”处理，保留 subject_id
5. ECG -> R 峰 -> RR 间期
6. RR 异常过滤
7. 按 64 个 RR 间期、步长 32 生成窗口
8. 为每个窗口提取 32 维 HRV 特征
9. 输出：
      wesad_windows.npz
      wesad_manifest.csv
      wesad_subject_counts.csv
      preprocessing_summary.json
10. 可选：运行严格 LOSO Random Forest 基线，输出：
      loso_rf_subject_results.csv
      loso_rf_summary.json

重要
----
- 不会把不同受试者拼接成一个长数组。
- 不会在预处理阶段对所有受试者统一 StandardScaler。
- LOSO 中 StandardScaler 只在训练受试者上拟合，避免数据泄漏。
- 这份代码默认只处理胸部 ECG，不使用腕部 E4。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 全局常量
# =============================================================================

CHEST_FS = 700

# WESAD 官方 protocol labels
ORIGINAL_STATE_NAMES: Dict[int, str] = {
    0: "undefined_or_transition",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignored_5",
    6: "ignored_6",
    7: "ignored_7",
}

# 默认二分类映射
BINARY_LABEL_MAP: Dict[int, int] = {
    1: 0,  # baseline -> non-stress
    2: 1,  # stress -> stress
    3: 0,  # amusement -> non-stress
}

BINARY_LABEL_NAMES: Dict[int, str] = {
    0: "non_stress",
    1: "stress",
}


HRV_FEATURE_NAMES: Tuple[str, ...] = (
    # 时域与统计
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
    # 频域
    "vlf_power",
    "lf_power",
    "hf_power",
    "total_power",
    "lf_nu",
    "hf_nu",
    "lf_hf_ratio",
    # 非线性
    "sd1",
    "sd2",
    "sd1_sd2_ratio",
    "sample_entropy",
)

assert len(HRV_FEATURE_NAMES) == 32


# =============================================================================
# 配置
# =============================================================================

@dataclass(frozen=True)
class PrepareConfig:
    wesad_root: str
    output_dir: str

    chest_fs: int = CHEST_FS

    # RR 窗口
    rr_window_size: int = 64
    rr_stride: int = 32

    # 至少保留多长的连续状态区间
    minimum_state_seconds: int = 30

    # RR 生理范围
    rr_min_ms: float = 350.0
    rr_max_ms: float = 1800.0

    # 相邻 RR 突变过滤
    max_relative_rr_change: float = 0.30

    # 状态边界两侧丢弃秒数，减少过渡影响
    transition_trim_seconds: int = 5

    # R 峰提取方式：auto / neurokit / scipy
    rr_method: str = "auto"

    # scipy R 峰检测参数
    scipy_min_peak_distance_seconds: float = 0.30
    scipy_prominence_mad_multiplier: float = 1.0
    scipy_height_mad_multiplier: float = 3.0

    # LOSO RF
    run_loso_rf: bool = False
    random_seed: int = 42
    rf_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 2
    rf_class_weight: str = "balanced_subsample"


# =============================================================================
# 日志与工具
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("wesad_prepare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        output_dir / "prepare_wesad.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def subject_number(subject_id: str) -> int:
    match = re.fullmatch(r"S(\d+)", subject_id)
    if match is None:
        raise ValueError(f"无法解析受试者编号：{subject_id!r}")
    return int(match.group(1))


def discover_subject_pickles(wesad_root: Path) -> List[Path]:
    """
    寻找 WESAD/S2/S2.pkl 等文件。
    只接受父文件夹名与 pkl 文件名一致的文件。
    """
    candidates: List[Path] = []

    for path in wesad_root.glob("S*/S*.pkl"):
        parent_name = path.parent.name
        if path.stem == parent_name and re.fullmatch(r"S\d+", parent_name):
            candidates.append(path)

    candidates.sort(key=lambda p: subject_number(p.parent.name))
    return candidates


def load_pickle_compat(path: Path) -> dict:
    """
    WESAD 的 pkl 可能由 Python 2 保存，因此兼容 latin1。
    """
    with path.open("rb") as file:
        try:
            return pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pickle.load(file, encoding="latin1")


def iter_contiguous_runs(labels: np.ndarray) -> Iterator[Tuple[int, int, int]]:
    """
    返回连续标签区间：
        (start_index, end_index_exclusive, original_label)
    """
    labels = np.asarray(labels).reshape(-1)

    if labels.size == 0:
        return

    start = 0
    current = int(labels[0])

    for index in range(1, len(labels)):
        value = int(labels[index])
        if value != current:
            yield start, index, current
            start = index
            current = value

    yield start, len(labels), current


# =============================================================================
# ECG -> RR
# =============================================================================

def bandpass_ecg(ecg: np.ndarray, fs: int) -> np.ndarray:
    """
    ECG 带通滤波，用于 R 峰检测。
    """
    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    if len(ecg) < fs * 2:
        return ecg

    nyquist = fs / 2.0
    low = 0.5 / nyquist
    high = 40.0 / nyquist

    b, a = scipy_signal.butter(
        N=3,
        Wn=[low, high],
        btype="bandpass",
    )

    try:
        return scipy_signal.filtfilt(b, a, ecg)
    except ValueError:
        return scipy_signal.lfilter(b, a, ecg)


def detect_r_peaks_neurokit(ecg: np.ndarray, fs: int) -> np.ndarray:
    """
    使用 neurokit2 检测 R 峰。
    """
    import neurokit2 as nk  # type: ignore

    cleaned = nk.ecg_clean(
        ecg,
        sampling_rate=fs,
        method="neurokit",
    )
    _, info = nk.ecg_peaks(
        cleaned,
        sampling_rate=fs,
    )

    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    return peaks


def detect_r_peaks_scipy(
    ecg: np.ndarray,
    fs: int,
    config: PrepareConfig,
) -> np.ndarray:
    """
    SciPy 回退方案。准确性通常不如 NeuroKit2，
    建议正式实验优先安装 neurokit2。
    """
    filtered = bandpass_ecg(ecg, fs)

    centered = filtered - np.median(filtered)
    mad = float(np.median(np.abs(centered - np.median(centered))))
    robust_scale = max(1.4826 * mad, float(np.std(centered)) * 0.1, 1e-8)

    height = (
        float(np.median(centered))
        + config.scipy_height_mad_multiplier * robust_scale
    )
    prominence = (
        config.scipy_prominence_mad_multiplier * robust_scale
    )

    peaks, _ = scipy_signal.find_peaks(
        centered,
        height=height,
        prominence=prominence,
        distance=max(
            1,
            int(config.scipy_min_peak_distance_seconds * fs),
        ),
    )

    return peaks.astype(np.int64)


def filter_rr_intervals(
    rr_ms: np.ndarray,
    config: PrepareConfig,
) -> np.ndarray:
    """
    基础 RR 清洗：
    1. 去 NaN/Inf
    2. 生理范围过滤
    3. 相邻 RR 相对突变过滤
    """
    rr = np.asarray(rr_ms, dtype=np.float64).reshape(-1)
    rr = rr[np.isfinite(rr)]

    rr = rr[
        (rr > config.rr_min_ms)
        & (rr < config.rr_max_ms)
    ]

    if len(rr) < 2:
        return rr.astype(np.float32)

    relative_change = (
        np.abs(np.diff(rr))
        / np.maximum(rr[:-1], 1e-8)
    )

    keep_mask = np.append(
        True,
        relative_change < config.max_relative_rr_change,
    )

    return rr[keep_mask].astype(np.float32)


def extract_rr_intervals(
    ecg_segment: np.ndarray,
    config: PrepareConfig,
) -> Tuple[np.ndarray, str]:
    """
    返回：
        rr_ms, method_used
    """
    ecg = np.asarray(ecg_segment, dtype=np.float64).reshape(-1)

    if len(ecg) < config.chest_fs * 5:
        return np.empty(0, dtype=np.float32), "too_short"

    method = config.rr_method.lower().strip()

    if method not in {"auto", "neurokit", "scipy"}:
        raise ValueError(
            "rr_method 必须是 auto、neurokit 或 scipy"
        )

    r_peaks: Optional[np.ndarray] = None
    method_used = ""

    if method in {"auto", "neurokit"}:
        try:
            r_peaks = detect_r_peaks_neurokit(
                ecg,
                config.chest_fs,
            )
            method_used = "neurokit"
        except Exception:
            if method == "neurokit":
                raise
            r_peaks = None

    if r_peaks is None or len(r_peaks) < 3:
        r_peaks = detect_r_peaks_scipy(
            ecg,
            config.chest_fs,
            config,
        )
        method_used = "scipy"

    if len(r_peaks) < 3:
        return np.empty(0, dtype=np.float32), method_used

    rr_ms = (
        np.diff(r_peaks)
        / float(config.chest_fs)
        * 1000.0
    )

    rr_ms = filter_rr_intervals(rr_ms, config)
    return rr_ms, method_used


# =============================================================================
# HRV 特征
# =============================================================================

def sample_entropy(
    values: np.ndarray,
    embedding_dimension: int = 2,
) -> float:
    """
    简单样本熵实现。
    64 个 RR 的窗口较短，因此此特征需谨慎解释。
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n < embedding_dimension + 2:
        return 0.0

    std_value = float(np.std(values, ddof=1))
    tolerance = 0.2 * std_value

    if tolerance <= 1e-12:
        return 0.0

    def count_matches(m: int) -> int:
        count = 0

        for i in range(n - m):
            template = values[i : i + m]

            for j in range(i + 1, n - m + 1):
                candidate = values[j : j + m]

                if np.max(
                    np.abs(template - candidate)
                ) <= tolerance:
                    count += 1

        return count

    matches_m = count_matches(embedding_dimension)
    matches_m_plus_1 = count_matches(
        embedding_dimension + 1
    )

    if matches_m <= 0 or matches_m_plus_1 <= 0:
        return 0.0

    return float(
        -math.log(matches_m_plus_1 / matches_m)
    )


def frequency_domain_features(
    rr_ms: np.ndarray,
) -> Tuple[float, ...]:
    """
    RR tachogram -> 4 Hz 插值 -> Welch PSD。
    返回 8 个值，最后一个 hf_peak 当前不进入 32 维向量，
    仅为函数完整性保留。
    """
    rr_ms = np.asarray(rr_ms, dtype=np.float64)

    if len(rr_ms) < 32:
        return (0.0,) * 8

    rr_seconds = rr_ms / 1000.0

    # 每个 RR 对应相邻心搏间隔，构造累计时间轴
    time_axis = np.cumsum(rr_seconds)
    time_axis = time_axis - time_axis[0]

    if len(time_axis) < 2 or time_axis[-1] <= 0:
        return (0.0,) * 8

    interpolation_fs = 4.0
    uniform_time = np.arange(
        0.0,
        time_axis[-1],
        1.0 / interpolation_fs,
    )

    if len(uniform_time) < 16:
        return (0.0,) * 8

    try:
        interpolator = interpolate.interp1d(
            time_axis,
            rr_seconds,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        tachogram = interpolator(uniform_time)
        tachogram = scipy_signal.detrend(tachogram)

        frequencies, psd = scipy_signal.welch(
            tachogram,
            fs=interpolation_fs,
            nperseg=min(256, len(tachogram)),
            scaling="density",
        )

    except Exception:
        return (0.0,) * 8

    def band_power(
        low_frequency: float,
        high_frequency: float,
    ) -> float:
        mask = (
            (frequencies >= low_frequency)
            & (frequencies < high_frequency)
        )

        if np.count_nonzero(mask) < 2:
            return 0.0

        return float(
            np.trapezoid(
                psd[mask],
                frequencies[mask],
            )
        )

    vlf = band_power(0.0033, 0.04)
    lf = band_power(0.04, 0.15)
    hf = band_power(0.15, 0.40)

    total = vlf + lf + hf
    lf_hf_sum = lf + hf

    lf_nu = (
        100.0 * lf / lf_hf_sum
        if lf_hf_sum > 1e-12
        else 0.0
    )
    hf_nu = (
        100.0 * hf / lf_hf_sum
        if lf_hf_sum > 1e-12
        else 0.0
    )
    lf_hf_ratio = (
        lf / hf
        if hf > 1e-12
        else 0.0
    )

    hf_mask = (
        (frequencies >= 0.15)
        & (frequencies < 0.40)
    )

    hf_peak = (
        float(
            frequencies[hf_mask][
                np.argmax(psd[hf_mask])
            ]
        )
        if np.any(hf_mask)
        else 0.0
    )

    return (
        vlf,
        lf,
        hf,
        total,
        lf_nu,
        hf_nu,
        lf_hf_ratio,
        hf_peak,
    )


def extract_hrv_features(
    rr_ms: Sequence[float],
) -> np.ndarray:
    rr = np.asarray(
        rr_ms,
        dtype=np.float64,
    ).reshape(-1)

    rr = rr[np.isfinite(rr)]

    if len(rr) < 4:
        return np.zeros(
            len(HRV_FEATURE_NAMES),
            dtype=np.float32,
        )

    differences = np.diff(rr)
    absolute_differences = np.abs(differences)
    heart_rate = 60000.0 / np.clip(
        rr,
        1e-8,
        None,
    )

    mean_rr = float(np.mean(rr))
    median_rr = float(np.median(rr))
    min_rr = float(np.min(rr))
    max_rr = float(np.max(rr))
    range_rr = max_rr - min_rr

    sdnn = (
        float(np.std(rr, ddof=1))
        if len(rr) > 1
        else 0.0
    )

    rmssd = (
        float(
            np.sqrt(
                np.mean(differences ** 2)
            )
        )
        if len(differences)
        else 0.0
    )

    sdsd = (
        float(np.std(differences, ddof=1))
        if len(differences) > 1
        else 0.0
    )

    pnn20 = (
        float(
            100.0
            * np.mean(
                absolute_differences > 20.0
            )
        )
        if len(differences)
        else 0.0
    )

    pnn50 = (
        float(
            100.0
            * np.mean(
                absolute_differences > 50.0
            )
        )
        if len(differences)
        else 0.0
    )

    mean_hr = float(np.mean(heart_rate))
    std_hr = (
        float(np.std(heart_rate, ddof=1))
        if len(heart_rate) > 1
        else 0.0
    )

    cv_rr = (
        float(sdnn / mean_rr)
        if abs(mean_rr) > 1e-12
        else 0.0
    )

    q10, q25, q75, q90 = np.percentile(
        rr,
        [10, 25, 75, 90],
    )

    iqr_rr = float(q75 - q25)
    mad_rr = float(
        np.median(
            np.abs(rr - median_rr)
        )
    )

    skew_rr = (
        float(skew(rr, bias=False))
        if len(rr) >= 3
        else 0.0
    )

    kurtosis_rr = (
        float(
            kurtosis(
                rr,
                fisher=True,
                bias=False,
            )
        )
        if len(rr) >= 4
        else 0.0
    )

    (
        vlf,
        lf,
        hf,
        total,
        lf_nu,
        hf_nu,
        lf_hf_ratio,
        _hf_peak,
    ) = frequency_domain_features(rr)

    difference_variance = (
        float(
            np.var(
                differences,
                ddof=1,
            )
        )
        if len(differences) > 1
        else 0.0
    )

    rr_variance = (
        float(np.var(rr, ddof=1))
        if len(rr) > 1
        else 0.0
    )

    sd1 = float(
        np.sqrt(
            max(
                0.5 * difference_variance,
                0.0,
            )
        )
    )

    sd2_squared = max(
        2.0 * rr_variance
        - 0.5 * difference_variance,
        0.0,
    )

    sd2 = float(np.sqrt(sd2_squared))

    sd1_sd2_ratio = (
        float(sd1 / sd2)
        if sd2 > 1e-12
        else 0.0
    )

    sampen = sample_entropy(rr)

    feature_vector = np.asarray(
        [
            mean_rr,
            median_rr,
            min_rr,
            max_rr,
            range_rr,
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
            float(q10),
            float(q25),
            float(q75),
            float(q90),
            vlf,
            lf,
            hf,
            total,
            lf_nu,
            hf_nu,
            lf_hf_ratio,
            sd1,
            sd2,
            sd1_sd2_ratio,
            sampen,
        ],
        dtype=np.float64,
    )

    feature_vector = np.nan_to_num(
        feature_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return feature_vector.astype(np.float32)


# =============================================================================
# 单个受试者处理
# =============================================================================

def validate_subject_data(
    data: dict,
    fallback_subject_id: str,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    从 SX.pkl 取出 subject_id、胸部 ECG、逐采样标签。
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"{fallback_subject_id}: pkl 顶层不是字典"
        )

    subject_id = str(
        data.get(
            "subject",
            fallback_subject_id,
        )
    )

    try:
        ecg = np.asarray(
            data["signal"]["chest"]["ECG"]
        ).reshape(-1)

        protocol_labels = np.asarray(
            data["label"]
        ).reshape(-1)

    except KeyError as error:
        raise KeyError(
            f"{subject_id}: pkl 缺少字段 {error}"
        ) from error

    if len(ecg) != len(protocol_labels):
        raise ValueError(
            f"{subject_id}: ECG 长度 {len(ecg)} "
            f"与 label 长度 {len(protocol_labels)} 不一致"
        )

    if len(ecg) == 0:
        raise ValueError(
            f"{subject_id}: ECG 为空"
        )

    return (
        subject_id,
        ecg.astype(np.float64),
        protocol_labels.astype(np.int16),
    )


def process_subject(
    pkl_path: Path,
    config: PrepareConfig,
    logger: logging.Logger,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[int],
    List[str],
    List[dict],
    dict,
]:
    """
    返回：
      rr_windows
      hrv_features
      binary_labels
      subject_ids
      manifest_rows
      subject_summary
    """
    fallback_subject_id = pkl_path.parent.name
    data = load_pickle_compat(pkl_path)

    subject_id, ecg, labels = validate_subject_data(
        data,
        fallback_subject_id,
    )

    logger.info(
        "%s | ECG=%d samples | %.1f min",
        subject_id,
        len(ecg),
        len(ecg) / config.chest_fs / 60.0,
    )

    rr_windows: List[np.ndarray] = []
    hrv_features: List[np.ndarray] = []
    binary_labels: List[int] = []
    subject_ids: List[str] = []
    manifest_rows: List[dict] = []

    run_counter = 0
    accepted_run_counter = 0
    skipped_short_runs = 0
    failed_rr_runs = 0
    rr_method_counter: Counter = Counter()
    state_window_counts: Counter = Counter()

    trim_samples = (
        config.transition_trim_seconds
        * config.chest_fs
    )

    minimum_samples = (
        config.minimum_state_seconds
        * config.chest_fs
    )

    for (
        run_start,
        run_end,
        original_label,
    ) in iter_contiguous_runs(labels):

        run_counter += 1

        if original_label not in BINARY_LABEL_MAP:
            continue

        # 丢弃状态边界附近数据
        trimmed_start = run_start + trim_samples
        trimmed_end = run_end - trim_samples

        if trimmed_end <= trimmed_start:
            skipped_short_runs += 1
            continue

        if (
            trimmed_end - trimmed_start
        ) < minimum_samples:
            skipped_short_runs += 1
            continue

        state_ecg = ecg[
            trimmed_start:trimmed_end
        ]

        rr_ms, method_used = extract_rr_intervals(
            state_ecg,
            config,
        )

        rr_method_counter[method_used] += 1

        if len(rr_ms) < config.rr_window_size:
            failed_rr_runs += 1
            logger.warning(
                "%s | %s | RR不足：%d",
                subject_id,
                ORIGINAL_STATE_NAMES[original_label],
                len(rr_ms),
            )
            continue

        accepted_run_counter += 1

        binary_label = BINARY_LABEL_MAP[
            original_label
        ]
        state_name = ORIGINAL_STATE_NAMES[
            original_label
        ]

        window_index_within_run = 0

        for rr_start in range(
            0,
            len(rr_ms)
            - config.rr_window_size
            + 1,
            config.rr_stride,
        ):
            rr_end = (
                rr_start
                + config.rr_window_size
            )

            rr_window = rr_ms[
                rr_start:rr_end
            ].astype(np.float32)

            features = extract_hrv_features(
                rr_window
            )

            sample_index = len(rr_windows)

            rr_windows.append(rr_window)
            hrv_features.append(features)
            binary_labels.append(binary_label)
            subject_ids.append(subject_id)

            state_window_counts[state_name] += 1

            # 近似时间：以状态段起点 + RR累计时长估算
            rr_elapsed_seconds = float(
                np.sum(
                    rr_ms[:rr_start]
                ) / 1000.0
            )
            rr_window_seconds = float(
                np.sum(rr_window) / 1000.0
            )

            approximate_start_sample = int(
                trimmed_start
                + rr_elapsed_seconds
                * config.chest_fs
            )
            approximate_end_sample = int(
                approximate_start_sample
                + rr_window_seconds
                * config.chest_fs
            )

            manifest_rows.append(
                {
                    "local_sample_index": sample_index,
                    "subject_id": subject_id,
                    "source_file": str(pkl_path),
                    "run_index": accepted_run_counter - 1,
                    "window_index_within_run": window_index_within_run,
                    "original_protocol_label": original_label,
                    "state": state_name,
                    "binary_label": binary_label,
                    "binary_label_name": BINARY_LABEL_NAMES[
                        binary_label
                    ],
                    "original_run_start_sample": run_start,
                    "original_run_end_sample": run_end,
                    "trimmed_run_start_sample": trimmed_start,
                    "trimmed_run_end_sample": trimmed_end,
                    "approx_window_start_sample": approximate_start_sample,
                    "approx_window_end_sample": approximate_end_sample,
                    "rr_start_index_within_run": rr_start,
                    "rr_end_index_within_run": rr_end,
                    "rr_count": config.rr_window_size,
                    "rr_window_duration_seconds": rr_window_seconds,
                    "rr_detection_method": method_used,
                    "mean_hr_bpm": float(
                        60000.0 / np.mean(rr_window)
                    ),
                    "rmssd_ms": float(
                        np.sqrt(
                            np.mean(
                                np.diff(
                                    rr_window
                                ) ** 2
                            )
                        )
                    ),
                }
            )

            window_index_within_run += 1

    subject_summary = {
        "subject_id": subject_id,
        "source_file": str(pkl_path),
        "total_ecg_samples": int(len(ecg)),
        "duration_minutes": float(
            len(ecg)
            / config.chest_fs
            / 60.0
        ),
        "all_contiguous_runs": int(run_counter),
        "accepted_state_runs": int(
            accepted_run_counter
        ),
        "skipped_short_runs": int(
            skipped_short_runs
        ),
        "failed_rr_runs": int(
            failed_rr_runs
        ),
        "rr_detection_methods": dict(
            rr_method_counter
        ),
        "window_counts": dict(
            state_window_counts
        ),
        "total_windows": int(
            len(rr_windows)
        ),
    }

    logger.info(
        "%s | windows=%d | baseline=%d | amusement=%d | stress=%d",
        subject_id,
        len(rr_windows),
        state_window_counts.get(
            "baseline",
            0,
        ),
        state_window_counts.get(
            "amusement",
            0,
        ),
        state_window_counts.get(
            "stress",
            0,
        ),
    )

    return (
        rr_windows,
        hrv_features,
        binary_labels,
        subject_ids,
        manifest_rows,
        subject_summary,
    )


# =============================================================================
# 全数据集处理与保存
# =============================================================================

def save_processed_dataset(
    output_dir: Path,
    rr_windows: np.ndarray,
    hrv_features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    manifest: pd.DataFrame,
    subject_summaries: List[dict],
    config: PrepareConfig,
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    npz_path = (
        output_dir
        / "wesad_windows.npz"
    )

    np.savez_compressed(
        npz_path,
        rr_windows=rr_windows,
        hrv_features=hrv_features,
        labels=labels,
        subject_ids=subject_ids,
        hrv_feature_names=np.asarray(
            HRV_FEATURE_NAMES
        ),
        rr_window_size=np.asarray(
            config.rr_window_size
        ),
        rr_stride=np.asarray(
            config.rr_stride
        ),
        chest_fs=np.asarray(
            config.chest_fs
        ),
        label_map_json=np.asarray(
            json.dumps(
                BINARY_LABEL_MAP,
                ensure_ascii=False,
            )
        ),
    )

    manifest_path = (
        output_dir
        / "wesad_manifest.csv"
    )
    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    counts = (
        manifest
        .groupby(
            [
                "subject_id",
                "state",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    counts = counts.reindex(
        sorted(
            counts.index,
            key=subject_number,
        )
    )

    for column in (
        "baseline",
        "amusement",
        "stress",
    ):
        if column not in counts.columns:
            counts[column] = 0

    counts = counts[
        [
            "baseline",
            "amusement",
            "stress",
        ]
    ]

    counts["non_stress_total"] = (
        counts["baseline"]
        + counts["amusement"]
    )
    counts["total"] = (
        counts[
            [
                "baseline",
                "amusement",
                "stress",
            ]
        ]
        .sum(axis=1)
    )

    counts_path = (
        output_dir
        / "wesad_subject_counts.csv"
    )
    counts.to_csv(
        counts_path,
        encoding="utf-8-sig",
    )

    class_values, class_counts = np.unique(
        labels,
        return_counts=True,
    )

    summary = {
        "created_at_unix": time.time(),
        "config": asdict(config),
        "n_subjects": int(
            len(
                np.unique(
                    subject_ids
                )
            )
        ),
        "subject_ids": [
            str(x)
            for x in sorted(
                np.unique(
                    subject_ids
                ),
                key=subject_number,
            )
        ],
        "rr_windows_shape": list(
            rr_windows.shape
        ),
        "hrv_features_shape": list(
            hrv_features.shape
        ),
        "class_counts": {
            BINARY_LABEL_NAMES[
                int(label)
            ]: int(count)
            for label, count
            in zip(
                class_values,
                class_counts,
            )
        },
        "subject_summaries": (
            subject_summaries
        ),
        "outputs": {
            "npz": str(npz_path),
            "manifest_csv": str(
                manifest_path
            ),
            "subject_counts_csv": str(
                counts_path
            ),
        },
    }

    summary_path = (
        output_dir
        / "preprocessing_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "保存完成：%s",
        npz_path,
    )
    logger.info(
        "保存完成：%s",
        manifest_path,
    )
    logger.info(
        "保存完成：%s",
        counts_path,
    )
    logger.info(
        "保存完成：%s",
        summary_path,
    )


def prepare_all_subjects(
    config: PrepareConfig,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    wesad_root = Path(
        config.wesad_root
    ).expanduser().resolve()

    output_dir = Path(
        config.output_dir
    ).expanduser().resolve()

    logger = setup_logging(
        output_dir
    )

    set_random_seed(
        config.random_seed
    )

    logger.info(
        "WESAD root: %s",
        wesad_root,
    )
    logger.info(
        "Output dir: %s",
        output_dir,
    )

    subject_pickles = (
        discover_subject_pickles(
            wesad_root
        )
    )

    if not subject_pickles:
        raise FileNotFoundError(
            "未找到任何 SX.pkl。\n"
            f"当前根目录：{wesad_root}\n"
            "期望结构示例："
            "WESAD/S2/S2.pkl"
        )

    logger.info(
        "找到 %d 个受试者 pkl：%s",
        len(subject_pickles),
        ", ".join(
            path.parent.name
            for path in subject_pickles
        ),
    )

    if len(subject_pickles) != 15:
        logger.warning(
            "标准 WESAD 应有 15 个有效受试者。"
            "当前找到 %d 个。",
            len(subject_pickles),
        )

    all_rr_windows: List[np.ndarray] = []
    all_hrv_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_subject_ids: List[str] = []
    all_manifest_rows: List[dict] = []
    subject_summaries: List[dict] = []

    global_sample_index = 0

    for pkl_path in subject_pickles:
        (
            rr_windows,
            hrv_features,
            labels,
            subject_ids,
            manifest_rows,
            subject_summary,
        ) = process_subject(
            pkl_path,
            config,
            logger,
        )

        # 转为全局 sample index
        for row in manifest_rows:
            row["sample_index"] = (
                global_sample_index
                + row.pop(
                    "local_sample_index"
                )
            )

        global_sample_index += len(
            rr_windows
        )

        all_rr_windows.extend(
            rr_windows
        )
        all_hrv_features.extend(
            hrv_features
        )
        all_labels.extend(
            labels
        )
        all_subject_ids.extend(
            subject_ids
        )
        all_manifest_rows.extend(
            manifest_rows
        )
        subject_summaries.append(
            subject_summary
        )

    if not all_rr_windows:
        raise RuntimeError(
            "没有生成有效 RR 窗口。"
            "请检查 ECG、标签或 R 峰检测。"
        )

    rr_array = np.stack(
        all_rr_windows
    ).astype(np.float32)

    hrv_array = np.stack(
        all_hrv_features
    ).astype(np.float32)

    labels_array = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    subject_ids_array = np.asarray(
        all_subject_ids,
        dtype=str,
    )

    manifest = pd.DataFrame(
        all_manifest_rows
    )

    # 核对索引
    manifest = manifest.sort_values(
        "sample_index"
    ).reset_index(drop=True)

    if not np.array_equal(
        manifest["sample_index"].to_numpy(),
        np.arange(len(manifest)),
    ):
        raise RuntimeError(
            "manifest sample_index 不连续"
        )

    if len(manifest) != len(
        rr_array
    ):
        raise RuntimeError(
            "manifest 与数组样本数不一致"
        )

    save_processed_dataset(
        output_dir=output_dir,
        rr_windows=rr_array,
        hrv_features=hrv_array,
        labels=labels_array,
        subject_ids=subject_ids_array,
        manifest=manifest,
        subject_summaries=subject_summaries,
        config=config,
        logger=logger,
    )

    logger.info(
        "最终 RR shape: %s",
        rr_array.shape,
    )
    logger.info(
        "最终 HRV shape: %s",
        hrv_array.shape,
    )
    logger.info(
        "类别计数: %s",
        dict(
            zip(
                *np.unique(
                    labels_array,
                    return_counts=True,
                )
            )
        ),
    )

    return (
        rr_array,
        hrv_array,
        labels_array,
        subject_ids_array,
        manifest,
    )


# =============================================================================
# 严格 LOSO Random Forest 基线
# =============================================================================

def run_loso_random_forest(
    hrv_features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    output_dir: Path,
    config: PrepareConfig,
) -> pd.DataFrame:
    """
    严格 LOSO：
      每轮完整留出 1 名受试者测试。
      StandardScaler 只在训练受试者上拟合。
    """
    logger = logging.getLogger(
        "wesad_prepare"
    )

    logo = LeaveOneGroupOut()
    results: List[dict] = []

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        logo.split(
            hrv_features,
            labels,
            groups=subject_ids,
        ),
        start=1,
    ):
        test_subject_values = np.unique(
            subject_ids[test_indices]
        )

        if len(test_subject_values) != 1:
            raise RuntimeError(
                "LOSO 每轮测试集应只有一名受试者"
            )

        test_subject = str(
            test_subject_values[0]
        )

        x_train = hrv_features[
            train_indices
        ]
        y_train = labels[
            train_indices
        ]

        x_test = hrv_features[
            test_indices
        ]
        y_test = labels[
            test_indices
        ]

        scaler = StandardScaler()
        x_train_scaled = (
            scaler.fit_transform(
                x_train
            )
        )
        x_test_scaled = (
            scaler.transform(
                x_test
            )
        )

        model = RandomForestClassifier(
            n_estimators=(
                config.rf_estimators
            ),
            max_depth=(
                config.rf_max_depth
            ),
            min_samples_leaf=(
                config.rf_min_samples_leaf
            ),
            class_weight=(
                config.rf_class_weight
            ),
            random_state=(
                config.random_seed
                + fold_index
            ),
            n_jobs=-1,
        )

        start_time = time.perf_counter()

        model.fit(
            x_train_scaled,
            y_train,
        )

        train_seconds = (
            time.perf_counter()
            - start_time
        )

        predict_start = time.perf_counter()
        y_pred = model.predict(
            x_test_scaled
        )
        predict_seconds = (
            time.perf_counter()
            - predict_start
        )

        if hasattr(
            model,
            "predict_proba",
        ):
            y_probability = (
                model.predict_proba(
                    x_test_scaled
                )[:, 1]
            )
        else:
            y_probability = None

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )
        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred,
            )
        )
        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        macro_precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        macro_recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        if (
            y_probability is not None
            and len(np.unique(y_test)) == 2
        ):
            auroc = roc_auc_score(
                y_test,
                y_probability,
            )
        else:
            auroc = float("nan")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        average_latency_ms = (
            predict_seconds
            / max(len(y_test), 1)
            * 1000.0
        )

        result = {
            "fold": fold_index,
            "test_subject": test_subject,
            "n_train": int(
                len(train_indices)
            ),
            "n_test": int(
                len(test_indices)
            ),
            "test_non_stress": int(
                np.sum(y_test == 0)
            ),
            "test_stress": int(
                np.sum(y_test == 1)
            ),
            "accuracy": float(
                accuracy
            ),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
            "macro_precision": float(
                macro_precision
            ),
            "macro_recall": float(
                macro_recall
            ),
            "macro_f1": float(
                macro_f1
            ),
            "auroc": float(
                auroc
            ),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "train_seconds": float(
                train_seconds
            ),
            "predict_seconds": float(
                predict_seconds
            ),
            "average_latency_ms_per_window": float(
                average_latency_ms
            ),
        }

        results.append(result)

        logger.info(
            "LOSO %02d | test=%s | "
            "BA=%.4f | Macro-F1=%.4f | AUROC=%.4f",
            fold_index,
            test_subject,
            balanced_accuracy,
            macro_f1,
            auroc,
        )

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        output_dir
        / "loso_rf_subject_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "auroc",
        "average_latency_ms_per_window",
    ]

    summary = {
        "model": "RandomForestClassifier",
        "validation": "Leave-One-Subject-Out",
        "n_folds": int(
            len(results_df)
        ),
        "metrics": {},
        "config": {
            "n_estimators": (
                config.rf_estimators
            ),
            "max_depth": (
                config.rf_max_depth
            ),
            "min_samples_leaf": (
                config.rf_min_samples_leaf
            ),
            "class_weight": (
                config.rf_class_weight
            ),
            "random_seed": (
                config.random_seed
            ),
        },
    }

    for column in metric_columns:
        values = results_df[
            column
        ].to_numpy(
            dtype=np.float64
        )

        summary["metrics"][column] = {
            "mean": float(
                np.nanmean(values)
            ),
            "std": float(
                np.nanstd(
                    values,
                    ddof=1,
                )
            ),
            "min": float(
                np.nanmin(values)
            ),
            "max": float(
                np.nanmax(values)
            ),
        }

    summary_path = (
        output_dir
        / "loso_rf_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "LOSO 逐受试者结果：%s",
        results_path,
    )
    logger.info(
        "LOSO 汇总：%s",
        summary_path,
    )
    logger.info(
        "LOSO Macro-F1 = %.4f ± %.4f",
        summary["metrics"]["macro_f1"][
            "mean"
        ],
        summary["metrics"]["macro_f1"][
            "std"
        ],
    )

    return results_df


# =============================================================================
# 命令行
# =============================================================================

def parse_optional_int(
    value: str,
) -> Optional[int]:
    text = value.strip().lower()

    if text in {
        "none",
        "null",
        "",
    }:
        return None

    return int(text)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WESAD 胸部 ECG -> RR -> 64点窗口 "
            "-> 32维HRV -> subject-wise 数据集"
        )
    )

    parser.add_argument(
        "--wesad-root",
        type=str,
        required=True,
        help=(
            "WESAD 根目录。"
            "该目录下应有 S2、S3、...、S17 文件夹。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./wesad_processed",
        help="输出目录",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="每个窗口包含的 RR 间期数量",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="RR 窗口步长",
    )

    parser.add_argument(
        "--minimum-state-seconds",
        type=int,
        default=30,
        help="最短连续状态区间长度（秒）",
    )

    parser.add_argument(
        "--transition-trim-seconds",
        type=int,
        default=5,
        help="状态区间两端各丢弃的秒数",
    )

    parser.add_argument(
        "--rr-method",
        type=str,
        choices=[
            "auto",
            "neurokit",
            "scipy",
        ],
        default="auto",
        help=(
            "R峰检测方式。"
            "auto 会优先 NeuroKit2，失败后使用 SciPy。"
        ),
    )

    parser.add_argument(
        "--run-loso-rf",
        action="store_true",
        help=(
            "处理完成后运行严格 LOSO "
            "Random Forest 基线"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=500,
        help="Random Forest 树数量",
    )

    parser.add_argument(
        "--rf-max-depth",
        type=parse_optional_int,
        default=None,
        help=(
            "Random Forest 最大深度。"
            "可填整数或 none"
        ),
    )

    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=2,
        help="Random Forest min_samples_leaf",
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.window_size <= 3:
        parser.error(
            "--window-size 必须大于 3"
        )

    if args.stride <= 0:
        parser.error(
            "--stride 必须大于 0"
        )

    config = PrepareConfig(
        wesad_root=args.wesad_root,
        output_dir=args.output_dir,
        rr_window_size=args.window_size,
        rr_stride=args.stride,
        minimum_state_seconds=(
            args.minimum_state_seconds
        ),
        transition_trim_seconds=(
            args.transition_trim_seconds
        ),
        rr_method=args.rr_method,
        run_loso_rf=args.run_loso_rf,
        random_seed=args.seed,
        rf_estimators=(
            args.rf_estimators
        ),
        rf_max_depth=(
            args.rf_max_depth
        ),
        rf_min_samples_leaf=(
            args.rf_min_samples_leaf
        ),
    )

    try:
        (
            _rr_windows,
            hrv_features,
            labels,
            subject_ids,
            _manifest,
        ) = prepare_all_subjects(
            config
        )

        if config.run_loso_rf:
            run_loso_random_forest(
                hrv_features=hrv_features,
                labels=labels,
                subject_ids=subject_ids,
                output_dir=Path(
                    config.output_dir
                ).expanduser().resolve(),
                config=config,
            )

    except Exception as error:
        logging.getLogger(
            "wesad_prepare"
        ).exception(
            "运行失败：%s",
            error,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())