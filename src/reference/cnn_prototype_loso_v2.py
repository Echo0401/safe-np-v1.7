#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_prototype_loso.py

WESAD: CNN/HRV Fusion Encoder + Few-shot Prototype under strict LOSO
====================================================================

研究目的
--------
第三步实验：把原来“直接在 32D HRV 空间做 Prototype”升级为：

    64 RR sequence -> 1D CNN ---------\
                                       -> Fusion -> 64D embedding
    32D HRV --------> MLP ------------/

然后在 64D learned embedding 空间中做：
    0-shot Global CNN classifier
    1-shot CNN + Prototype
    3-shot CNN + Prototype
    5-shot CNN + Prototype

严格协议
--------
1. 外层 LOSO：每轮 1 个 target subject，其他 14 个 source subjects。
2. target subject 完全不参与 encoder/classifier 训练。
3. target 内部：
      每个连续状态 run 前 calibration_fraction 作为 calibration region；
      后面作为 query/test region；
      两者之间保留 gap_rr；
      calibration 支持样本彼此不允许 RR 区间重叠。
4. StandardScaler（HRV）只在 source 数据上 fit。
5. RR 标准化均值/标准差只由 source 数据计算。
6. CNN 训练 epoch 的选择只使用 source subjects 内部的 validation subjects；
   target subject 不参与 early stopping / epoch 选择。
7. 选出 best epoch 后，重新初始化模型并在全部 14 名 source subjects
   上训练 best_epoch 个 epoch，再对 target 测试。
8. Prototype 的 1/3/5-shot 每个 K 重复多次，默认 20 次。

输入
----
prepare_wesad_and_loso.py 生成的：
    wesad_windows.npz
    wesad_manifest.csv

NPZ 至少包含：
    rr_windows       shape = (N, 64)
    hrv_features     shape = (N, 32)
    labels           shape = (N,)
    subject_ids      shape = (N,)

输出
----
cnn_global_subject_results.csv
cnn_prototype_episode_results.csv
cnn_prototype_kshot_summary.csv
cnn_prototype_subject_results.csv
cnn_matched_comparison_summary.csv
cnn_subject_comparison.csv
cnn_training_history.csv
cnn_fold_training_summary.csv
cnn_split_diagnostics.csv
cnn_config.json

默认模型
--------
RR branch:
    Conv1d(1->32, k=5)
    Conv1d(32->64, k=3)
    AdaptiveAvgPool1d(1)
    Linear(64->32)

HRV branch:
    Linear(32->64)
    Linear(64->32)

Fusion:
    concat(32+32)
    Linear(64->64) -> embedding
    Linear(64->2)  -> global classifier

可用输入模式
------------
--input-mode fusion   默认，RR + HRV
--input-mode rr       仅 RR 1D-CNN
--input-mode hrv      仅 HRV-MLP

说明
----
这一步暂时不加入 Neural Process，也不加入 Margin Gate。
目标是先验证 learned embedding 是否比原始 32D HRV 空间更适合 Prototype。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "本脚本需要 PyTorch。请先在当前虚拟环境安装 torch。"
    ) from exc


# =============================================================================
# 配置
# =============================================================================

@dataclass(frozen=True)
class Config:
    data_npz: str
    manifest_csv: str
    output_dir: str

    # 输入和 embedding
    input_mode: str = "fusion"  # fusion / rr / hrv
    embedding_dim: int = 64

    # Few-shot
    k_values: Tuple[int, ...] = (1, 3, 5)
    repeats: int = 20
    random_seed: int = 2026

    # Target subject 时间切分
    calibration_fraction: float = 0.50
    gap_rr: int = 64
    balance_nonstress_states: bool = True
    require_both_test_classes: bool = True

    # Prototype
    prototype_metric: str = "cosine"  # cosine / euclidean
    prototype_temperature: float = 1.0

    # CNN training
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 12
    min_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.20
    n_val_subjects: int = 2
    num_workers: int = 0

    # 若 best_epoch 太小，正式全 source 重训至少这么多 epoch
    minimum_retrain_epochs: int = 1

    # 可复现
    deterministic: bool = True


# =============================================================================
# 可复现与设备
# =============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(device_text: str) -> torch.device:
    text = device_text.lower().strip()

    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "你指定了 --device cuda，但当前 PyTorch 检测不到 CUDA。"
            )
        return torch.device("cuda")

    if text == "cpu":
        return torch.device("cpu")

    raise ValueError("--device 必须是 auto / cpu / cuda")


def subject_number(subject_id: str) -> int:
    text = str(subject_id).strip()

    if text.startswith("S"):
        try:
            return int(text[1:])
        except ValueError:
            pass

    raise ValueError(f"无法解析受试者编号: {subject_id!r}")


# =============================================================================
# 读取和核对数据
# =============================================================================

def load_processed_data(
    npz_path: Path,
    manifest_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到 NPZ: {npz_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest: {manifest_path}")

    data = np.load(npz_path, allow_pickle=False)

    required_keys = {
        "rr_windows",
        "hrv_features",
        "labels",
        "subject_ids",
    }

    missing = required_keys - set(data.files)

    if missing:
        raise KeyError(
            "wesad_windows.npz 缺少字段: "
            + ", ".join(sorted(missing))
        )

    rr = np.asarray(
        data["rr_windows"],
        dtype=np.float32,
    )

    hrv = np.asarray(
        data["hrv_features"],
        dtype=np.float32,
    )

    labels = np.asarray(
        data["labels"],
        dtype=np.int64,
    ).reshape(-1)

    subject_ids = np.asarray(
        data["subject_ids"]
    ).astype(str).reshape(-1)

    manifest = pd.read_csv(manifest_path)

    required_manifest = {
        "sample_index",
        "subject_id",
        "run_index",
        "state",
        "binary_label",
        "rr_start_index_within_run",
        "rr_end_index_within_run",
    }

    missing_manifest = required_manifest - set(manifest.columns)

    if missing_manifest:
        raise KeyError(
            "wesad_manifest.csv 缺少字段: "
            + ", ".join(sorted(missing_manifest))
        )

    if rr.ndim != 2:
        raise ValueError(
            f"rr_windows 应为二维矩阵，当前 shape={rr.shape}"
        )

    if hrv.ndim != 2:
        raise ValueError(
            f"hrv_features 应为二维矩阵，当前 shape={hrv.shape}"
        )

    n = len(labels)

    if not (
        len(rr)
        == len(hrv)
        == len(subject_ids)
        == len(manifest)
        == n
    ):
        raise ValueError(
            "样本数不一致："
            f"RR={len(rr)}, HRV={len(hrv)}, y={len(labels)}, "
            f"subject_ids={len(subject_ids)}, manifest={len(manifest)}"
        )

    if set(np.unique(labels)) - {0, 1}:
        raise ValueError(
            f"当前脚本只支持二分类 0/1，实际={np.unique(labels)}"
        )

    manifest = manifest.sort_values(
        "sample_index"
    ).reset_index(drop=True)

    if not np.array_equal(
        manifest["sample_index"].to_numpy(dtype=np.int64),
        np.arange(n),
    ):
        raise ValueError(
            "manifest 的 sample_index 与 NPZ 数组顺序不一致。"
        )

    if not np.array_equal(
        manifest["subject_id"].astype(str).to_numpy(),
        subject_ids,
    ):
        raise ValueError(
            "manifest subject_id 与 NPZ subject_ids 不一致。"
        )

    if not np.array_equal(
        manifest["binary_label"].to_numpy(dtype=np.int64),
        labels,
    ):
        raise ValueError(
            "manifest binary_label 与 NPZ labels 不一致。"
        )

    if not np.all(np.isfinite(rr)):
        raise ValueError("rr_windows 中存在 NaN/Inf。")

    if not np.all(np.isfinite(hrv)):
        raise ValueError("hrv_features 中存在 NaN/Inf。")

    return rr, hrv, labels, subject_ids, manifest


# =============================================================================
# Source-only 标准化
# =============================================================================

@dataclass
class FoldPreprocessor:
    hrv_scaler: StandardScaler
    rr_mean: float
    rr_std: float

    @classmethod
    def fit(
        cls,
        rr: np.ndarray,
        hrv: np.ndarray,
        fit_indices: np.ndarray,
    ) -> "FoldPreprocessor":
        hrv_scaler = StandardScaler()
        hrv_scaler.fit(hrv[fit_indices])

        rr_train = rr[fit_indices].astype(np.float64)

        rr_mean = float(np.mean(rr_train))
        rr_std = float(np.std(rr_train))

        if rr_std < 1e-8:
            rr_std = 1.0

        return cls(
            hrv_scaler=hrv_scaler,
            rr_mean=rr_mean,
            rr_std=rr_std,
        )

    def transform_rr(
        self,
        rr: np.ndarray,
    ) -> np.ndarray:
        transformed = (
            rr.astype(np.float64) - self.rr_mean
        ) / self.rr_std

        return transformed.astype(np.float32)

    def transform_hrv(
        self,
        hrv: np.ndarray,
    ) -> np.ndarray:
        return self.hrv_scaler.transform(
            hrv
        ).astype(np.float32)


# =============================================================================
# PyTorch Dataset
# =============================================================================

class RRHRVDataset(Dataset):
    def __init__(
        self,
        rr: np.ndarray,
        hrv: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
    ) -> None:
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.rr = torch.from_numpy(
            rr[self.indices].astype(np.float32)
        )

        self.hrv = torch.from_numpy(
            hrv[self.indices].astype(np.float32)
        )

        self.labels = torch.from_numpy(
            labels[self.indices].astype(np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        item: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.rr[item],
            self.hrv[item],
            self.labels[item],
        )


# =============================================================================
# 模型
# =============================================================================

class RRBranch(nn.Module):
    def __init__(
        self,
        output_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(
        self,
        rr: torch.Tensor,
    ) -> torch.Tensor:
        x = rr.unsqueeze(1)
        x = self.features(x)
        return self.project(x)


class HRVBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(
        self,
        hrv: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(hrv)


class CNNPrototypeNet(nn.Module):
    def __init__(
        self,
        hrv_dim: int,
        embedding_dim: int = 64,
        input_mode: str = "fusion",
        dropout: float = 0.20,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        input_mode = input_mode.lower().strip()

        if input_mode not in {"fusion", "rr", "hrv"}:
            raise ValueError(
                "input_mode 必须是 fusion / rr / hrv"
            )

        self.input_mode = input_mode

        if input_mode in {"fusion", "rr"}:
            rr_output_dim = 32 if input_mode == "fusion" else 64

            self.rr_branch = RRBranch(
                output_dim=rr_output_dim,
                dropout=dropout,
            )
        else:
            self.rr_branch = None

        if input_mode in {"fusion", "hrv"}:
            hrv_output_dim = 32 if input_mode == "fusion" else 64

            self.hrv_branch = HRVBranch(
                input_dim=hrv_dim,
                output_dim=hrv_output_dim,
                dropout=dropout,
            )
        else:
            self.hrv_branch = None

        if input_mode == "fusion":
            fusion_input_dim = 64
        else:
            fusion_input_dim = 64

        self.embedding_head = nn.Sequential(
            nn.Linear(
                fusion_input_dim,
                embedding_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def encode(
        self,
        rr: torch.Tensor,
        hrv: torch.Tensor,
    ) -> torch.Tensor:
        if self.input_mode == "fusion":
            rr_feature = self.rr_branch(rr)
            hrv_feature = self.hrv_branch(hrv)
            fused = torch.cat(
                [rr_feature, hrv_feature],
                dim=1,
            )

        elif self.input_mode == "rr":
            fused = self.rr_branch(rr)

        elif self.input_mode == "hrv":
            fused = self.hrv_branch(hrv)

        else:
            raise RuntimeError("未知 input_mode")

        embedding = self.embedding_head(fused)
        return embedding

    def forward(
        self,
        rr: torch.Tensor,
        hrv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(rr, hrv)
        logits = self.classifier(embedding)

        return logits, embedding


# =============================================================================
# 指标
# =============================================================================

METRIC_NAMES = (
    "balanced_accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "auroc",
)


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stress_probability: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    stress_probability = np.asarray(
        stress_probability,
        dtype=np.float64,
    )

    result = {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(
            roc_auc_score(
                y_true,
                stress_probability,
            )
        )
    else:
        result["auroc"] = float("nan")

    return result


# =============================================================================
# 训练辅助
# =============================================================================

def compute_class_weights(
    labels: np.ndarray,
    indices: np.ndarray,
) -> torch.Tensor:
    y = labels[indices]

    counts = np.bincount(
        y,
        minlength=2,
    ).astype(np.float64)

    total = float(np.sum(counts))

    weights = np.ones(
        2,
        dtype=np.float32,
    )

    for class_id in (0, 1):
        if counts[class_id] > 0:
            weights[class_id] = (
                total / (2.0 * counts[class_id])
            )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def make_loader(
    rr_scaled: np.ndarray,
    hrv_scaled: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = RRHRVDataset(
        rr=rr_scaled,
        hrv=hrv_scaled,
        labels=labels,
        indices=indices,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def run_model_on_indices(
    model: nn.Module,
    rr_scaled: np.ndarray,
    hrv_scaled: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    loader = make_loader(
        rr_scaled=rr_scaled,
        hrv_scaled=hrv_scaled,
        labels=labels,
        indices=indices,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )

    model.eval()

    all_logits: List[np.ndarray] = []
    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for rr_batch, hrv_batch, y_batch in loader:
            rr_batch = rr_batch.to(device)
            hrv_batch = hrv_batch.to(device)

            logits, embedding = model(
                rr_batch,
                hrv_batch,
            )

            all_logits.append(
                logits.detach().cpu().numpy()
            )

            all_embeddings.append(
                embedding.detach().cpu().numpy()
            )

            all_labels.append(
                y_batch.numpy()
            )

    logits_np = np.concatenate(
        all_logits,
        axis=0,
    )

    embeddings_np = np.concatenate(
        all_embeddings,
        axis=0,
    )

    labels_np = np.concatenate(
        all_labels,
        axis=0,
    )

    probabilities = torch.softmax(
        torch.from_numpy(logits_np),
        dim=1,
    ).numpy()

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    stress_probability = probabilities[
        :, 1
    ].astype(np.float64)

    return (
        labels_np,
        predictions,
        stress_probability,
        embeddings_np.astype(np.float32),
    )


def evaluate_model_indices(
    model: nn.Module,
    rr_scaled: np.ndarray,
    hrv_scaled: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Dict[str, float]:
    (
        y_true,
        y_pred,
        stress_probability,
        _,
    ) = run_model_on_indices(
        model=model,
        rr_scaled=rr_scaled,
        hrv_scaled=hrv_scaled,
        labels=labels,
        indices=indices,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    return evaluate_binary(
        y_true=y_true,
        y_pred=y_pred,
        stress_probability=stress_probability,
    )


def choose_validation_subjects(
    source_subjects: List[str],
    outer_fold_index: int,
    n_val_subjects: int,
) -> List[str]:
    if n_val_subjects <= 0:
        return []

    if n_val_subjects >= len(source_subjects):
        raise ValueError(
            "n_val_subjects 必须小于 source subject 数量"
        )

    ordered = sorted(
        source_subjects,
        key=subject_number,
    )

    start = (
        outer_fold_index * n_val_subjects
    ) % len(ordered)

    selected: List[str] = []

    cursor = start

    while len(selected) < n_val_subjects:
        candidate = ordered[
            cursor % len(ordered)
        ]

        if candidate not in selected:
            selected.append(candidate)

        cursor += 1

    return selected


def train_with_inner_validation(
    rr: np.ndarray,
    hrv: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    source_indices: np.ndarray,
    source_subjects: List[str],
    outer_fold_index: int,
    config: Config,
    device: torch.device,
    fold_seed: int,
) -> Tuple[int, List[dict], List[str]]:
    """
    只使用 source subjects 内部做 epoch 选择。
    返回：
        best_epoch
        training_history_rows
        validation_subjects
    """

    validation_subjects = choose_validation_subjects(
        source_subjects=source_subjects,
        outer_fold_index=outer_fold_index,
        n_val_subjects=config.n_val_subjects,
    )

    val_mask = np.isin(
        subject_ids,
        validation_subjects,
    )

    source_mask = np.zeros(
        len(labels),
        dtype=bool,
    )
    source_mask[source_indices] = True

    val_indices = np.where(
        source_mask & val_mask
    )[0]

    train_indices = np.where(
        source_mask & (~val_mask)
    )[0]

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise RuntimeError(
            "inner train/validation 划分为空。"
        )

    preprocessor = FoldPreprocessor.fit(
        rr=rr,
        hrv=hrv,
        fit_indices=train_indices,
    )

    rr_scaled = preprocessor.transform_rr(rr)
    hrv_scaled = preprocessor.transform_hrv(hrv)

    train_loader = make_loader(
        rr_scaled=rr_scaled,
        hrv_scaled=hrv_scaled,
        labels=labels,
        indices=train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        seed=fold_seed,
        num_workers=config.num_workers,
    )

    model = CNNPrototypeNet(
        hrv_dim=hrv.shape[1],
        embedding_dim=config.embedding_dim,
        input_mode=config.input_mode,
        dropout=config.dropout,
    ).to(device)

    class_weights = compute_class_weights(
        labels=labels,
        indices=train_indices,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_macro_f1 = -np.inf
    best_epoch = 1
    patience_counter = 0

    history: List[dict] = []

    for epoch in range(
        1,
        config.max_epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for (
            rr_batch,
            hrv_batch,
            y_batch,
        ) in train_loader:
            rr_batch = rr_batch.to(device)
            hrv_batch = hrv_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, _ = model(
                rr_batch,
                hrv_batch,
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_n = int(
                y_batch.shape[0]
            )

            running_loss += (
                float(loss.item())
                * batch_n
            )

            n_seen += batch_n

        train_loss = (
            running_loss / max(n_seen, 1)
        )

        val_metrics = evaluate_model_indices(
            model=model,
            rr_scaled=rr_scaled,
            hrv_scaled=hrv_scaled,
            labels=labels,
            indices=val_indices,
            batch_size=config.batch_size,
            device=device,
            num_workers=config.num_workers,
        )

        current_f1 = val_metrics[
            "macro_f1"
        ]

        history.append(
            {
                "stage": "inner_validation",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_balanced_accuracy": val_metrics[
                    "balanced_accuracy"
                ],
                "val_macro_f1": val_metrics[
                    "macro_f1"
                ],
                "val_auroc": val_metrics[
                    "auroc"
                ],
                "is_best": False,
                "validation_subjects": ",".join(
                    validation_subjects
                ),
                "n_train": int(
                    len(train_indices)
                ),
                "n_val": int(
                    len(val_indices)
                ),
            }
        )

        improved = (
            current_f1
            > best_macro_f1 + 1e-8
        )

        if improved:
            best_macro_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0

            history[-1]["is_best"] = True

        else:
            patience_counter += 1

        if (
            epoch >= config.min_epochs
            and patience_counter
            >= config.patience
        ):
            break

    return (
        best_epoch,
        history,
        validation_subjects,
    )


def train_final_source_model(
    rr: np.ndarray,
    hrv: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    best_epoch: int,
    config: Config,
    device: torch.device,
    fold_seed: int,
) -> Tuple[
    CNNPrototypeNet,
    FoldPreprocessor,
    List[dict],
]:
    """
    用 inner validation 得到的 best_epoch，
    重新初始化模型，在全部 14 名 source subjects 上训练。
    """

    retrain_epochs = int(best_epoch)

    preprocessor = FoldPreprocessor.fit(
        rr=rr,
        hrv=hrv,
        fit_indices=source_indices,
    )

    rr_scaled = preprocessor.transform_rr(rr)
    hrv_scaled = preprocessor.transform_hrv(hrv)

    train_loader = make_loader(
        rr_scaled=rr_scaled,
        hrv_scaled=hrv_scaled,
        labels=labels,
        indices=source_indices,
        batch_size=config.batch_size,
        shuffle=True,
        seed=fold_seed + 999,
        num_workers=config.num_workers,
    )

    model = CNNPrototypeNet(
        hrv_dim=hrv.shape[1],
        embedding_dim=config.embedding_dim,
        input_mode=config.input_mode,
        dropout=config.dropout,
    ).to(device)

    class_weights = compute_class_weights(
        labels=labels,
        indices=source_indices,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: List[dict] = []

    for epoch in range(
        1,
        retrain_epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for (
            rr_batch,
            hrv_batch,
            y_batch,
        ) in train_loader:
            rr_batch = rr_batch.to(device)
            hrv_batch = hrv_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, _ = model(
                rr_batch,
                hrv_batch,
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            batch_n = int(
                y_batch.shape[0]
            )

            running_loss += (
                float(loss.item())
                * batch_n
            )

            n_seen += batch_n

        history.append(
            {
                "stage": "final_source_retrain",
                "epoch": epoch,
                "train_loss": (
                    running_loss
                    / max(n_seen, 1)
                ),
                "n_train": int(
                    len(source_indices)
                ),
            }
        )

    return (
        model,
        preprocessor,
        history,
    )


# =============================================================================
# Target calibration / test 时间切分
# =============================================================================

def split_one_run_by_time(
    run_df: pd.DataFrame,
    calibration_fraction: float,
    gap_rr: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    run_df = run_df.sort_values(
        [
            "rr_start_index_within_run",
            "sample_index",
        ]
    ).copy()

    if run_df.empty:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
            {},
        )

    min_rr = int(
        run_df[
            "rr_start_index_within_run"
        ].min()
    )

    max_rr = int(
        run_df[
            "rr_end_index_within_run"
        ].max()
    )

    span = max_rr - min_rr

    if span <= 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
            {},
        )

    boundary = (
        min_rr
        + int(
            math.floor(
                span
                * calibration_fraction
            )
        )
    )

    test_start = (
        boundary
        + int(gap_rr)
    )

    rr_end = run_df[
        "rr_end_index_within_run"
    ].to_numpy(dtype=np.int64)

    rr_start = run_df[
        "rr_start_index_within_run"
    ].to_numpy(dtype=np.int64)

    cal_mask = (
        rr_end <= boundary
    )

    test_mask = (
        rr_start >= test_start
    )

    cal_indices = run_df.loc[
        cal_mask,
        "sample_index",
    ].to_numpy(dtype=np.int64)

    test_indices = run_df.loc[
        test_mask,
        "sample_index",
    ].to_numpy(dtype=np.int64)

    diag = {
        "run_index": int(
            run_df[
                "run_index"
            ].iloc[0]
        ),
        "state": str(
            run_df[
                "state"
            ].iloc[0]
        ),
        "binary_label": int(
            run_df[
                "binary_label"
            ].iloc[0]
        ),
        "run_rr_min": min_rr,
        "run_rr_max": max_rr,
        "boundary_rr": boundary,
        "test_start_rr": test_start,
        "n_windows_total": int(
            len(run_df)
        ),
        "n_calibration_pool": int(
            len(cal_indices)
        ),
        "n_test_pool": int(
            len(test_indices)
        ),
        "n_dropped_boundary": int(
            len(run_df)
            - len(cal_indices)
            - len(test_indices)
        ),
    }

    return (
        cal_indices,
        test_indices,
        diag,
    )


def build_target_pools(
    target_subject: str,
    manifest: pd.DataFrame,
    calibration_fraction: float,
    gap_rr: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    List[dict],
]:
    target_df = manifest[
        manifest[
            "subject_id"
        ].astype(str)
        == str(target_subject)
    ].copy()

    if target_df.empty:
        raise ValueError(
            f"manifest 找不到 {target_subject}"
        )

    calibration_indices: List[int] = []
    test_indices: List[int] = []
    diagnostics: List[dict] = []

    for _, run_df in target_df.groupby(
        "run_index",
        sort=True,
    ):
        (
            cal_idx,
            test_idx,
            diag,
        ) = split_one_run_by_time(
            run_df=run_df,
            calibration_fraction=calibration_fraction,
            gap_rr=gap_rr,
        )

        calibration_indices.extend(
            cal_idx.tolist()
        )

        test_indices.extend(
            test_idx.tolist()
        )

        if diag:
            diag[
                "subject_id"
            ] = str(
                target_subject
            )

            diagnostics.append(
                diag
            )

    calibration_pool = manifest[
        manifest[
            "sample_index"
        ].isin(
            calibration_indices
        )
    ].copy()

    test_pool = manifest[
        manifest[
            "sample_index"
        ].isin(
            test_indices
        )
    ].copy()

    intersection = (
        set(
            calibration_pool[
                "sample_index"
            ]
        )
        & set(
            test_pool[
                "sample_index"
            ]
        )
    )

    if intersection:
        raise RuntimeError(
            f"{target_subject}: calibration/test 样本交叉"
        )

    return (
        calibration_pool,
        test_pool,
        diagnostics,
    )


# =============================================================================
# Calibration 内部非重叠 K-shot
# =============================================================================

def rows_overlap(
    a: pd.Series,
    b: pd.Series,
) -> bool:
    if int(
        a["run_index"]
    ) != int(
        b["run_index"]
    ):
        return False

    a_start = int(
        a[
            "rr_start_index_within_run"
        ]
    )

    a_end = int(
        a[
            "rr_end_index_within_run"
        ]
    )

    b_start = int(
        b[
            "rr_start_index_within_run"
        ]
    )

    b_end = int(
        b[
            "rr_end_index_within_run"
        ]
    )

    return (
        max(
            a_start,
            b_start,
        )
        < min(
            a_end,
            b_end,
        )
    )


def max_nonoverlap_count(
    pool_df: pd.DataFrame,
) -> int:
    if pool_df.empty:
        return 0

    total = 0

    for _, run_df in pool_df.groupby(
        "run_index",
        sort=False,
    ):
        run_df = run_df.sort_values(
            [
                "rr_end_index_within_run",
                "rr_start_index_within_run",
            ]
        )

        last_end = -10**18

        for _, row in run_df.iterrows():
            start = int(
                row[
                    "rr_start_index_within_run"
                ]
            )

            end = int(
                row[
                    "rr_end_index_within_run"
                ]
            )

            if start >= last_end:
                total += 1
                last_end = end

    return total


def random_nonoverlap_sample(
    pool_df: pd.DataFrame,
    k: int,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> Optional[np.ndarray]:
    if k <= 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    if len(pool_df) < k:
        return None

    pool_df = pool_df.reset_index(
        drop=True
    )

    n = len(pool_df)

    for _ in range(
        max_attempts
    ):
        order = rng.permutation(
            n
        )

        chosen_rows: List[
            pd.Series
        ] = []

        for pos in order:
            row = pool_df.iloc[
                int(pos)
            ]

            valid = all(
                not rows_overlap(
                    row,
                    chosen,
                )
                for chosen
                in chosen_rows
            )

            if valid:
                chosen_rows.append(
                    row
                )

                if len(
                    chosen_rows
                ) == k:
                    return np.asarray(
                        [
                            int(
                                chosen[
                                    "sample_index"
                                ]
                            )
                            for chosen
                            in chosen_rows
                        ],
                        dtype=np.int64,
                    )

    return None


def allocation_candidates(
    k: int,
) -> List[Tuple[int, int]]:
    desired_baseline = (
        k + 1
    ) // 2

    candidates = []

    for baseline_n in range(
        k + 1
    ):
        amusement_n = (
            k - baseline_n
        )

        distance = abs(
            baseline_n
            - desired_baseline
        )

        candidates.append(
            (
                distance,
                baseline_n,
                amusement_n,
            )
        )

    candidates.sort()

    return [
        (
            baseline_n,
            amusement_n,
        )
        for (
            _,
            baseline_n,
            amusement_n,
        )
        in candidates
    ]


def sample_nonstress_nonoverlap(
    calibration_pool: pd.DataFrame,
    k: int,
    rng: np.random.Generator,
    balance_states: bool,
) -> Optional[np.ndarray]:
    nonstress = calibration_pool[
        calibration_pool[
            "binary_label"
        ].astype(int) == 0
    ].copy()

    if not balance_states:
        return random_nonoverlap_sample(
            nonstress,
            k,
            rng,
        )

    baseline = nonstress[
        nonstress[
            "state"
        ].astype(str)
        == "baseline"
    ].copy()

    amusement = nonstress[
        nonstress[
            "state"
        ].astype(str)
        == "amusement"
    ].copy()

    if (
        baseline.empty
        or amusement.empty
    ):
        return random_nonoverlap_sample(
            nonstress,
            k,
            rng,
        )

    for (
        baseline_n,
        amusement_n,
    ) in allocation_candidates(k):
        if (
            max_nonoverlap_count(
                baseline
            )
            < baseline_n
        ):
            continue

        if (
            max_nonoverlap_count(
                amusement
            )
            < amusement_n
        ):
            continue

        baseline_idx = random_nonoverlap_sample(
            baseline,
            baseline_n,
            rng,
        )

        amusement_idx = random_nonoverlap_sample(
            amusement,
            amusement_n,
            rng,
        )

        if (
            baseline_idx is None
            or amusement_idx is None
        ):
            continue

        selected = np.concatenate(
            [
                baseline_idx,
                amusement_idx,
            ]
        ).astype(np.int64)

        if len(selected) == k:
            rng.shuffle(selected)
            return selected

    return None


def sample_kshot_calibration(
    calibration_pool: pd.DataFrame,
    k: int,
    rng: np.random.Generator,
    balance_nonstress_states: bool,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    stress_pool = calibration_pool[
        calibration_pool[
            "binary_label"
        ].astype(int) == 1
    ].copy()

    stress_indices = (
        random_nonoverlap_sample(
            stress_pool,
            k,
            rng,
        )
    )

    if stress_indices is None:
        return None

    nonstress_indices = (
        sample_nonstress_nonoverlap(
            calibration_pool=calibration_pool,
            k=k,
            rng=rng,
            balance_states=balance_nonstress_states,
        )
    )

    if (
        nonstress_indices
        is None
    ):
        return None

    return (
        nonstress_indices,
        stress_indices,
    )


# =============================================================================
# Prototype in learned embedding space
# =============================================================================

def normalize_rows(
    x: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    return (
        x
        / np.maximum(
            norms,
            1e-12,
        )
    )


def stable_softmax(
    scores: np.ndarray,
    temperature: float,
) -> np.ndarray:
    temperature = max(
        float(temperature),
        1e-8,
    )

    scaled = (
        scores
        / temperature
    )

    scaled = (
        scaled
        - np.max(
            scaled,
            axis=1,
            keepdims=True,
        )
    )

    exp_values = np.exp(
        scaled
    )

    return (
        exp_values
        / np.maximum(
            np.sum(
                exp_values,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )
    )


def compute_prototypes(
    embeddings: np.ndarray,
    nonstress_indices: np.ndarray,
    stress_indices: np.ndarray,
) -> Dict[int, np.ndarray]:
    return {
        0: np.mean(
            embeddings[
                nonstress_indices
            ],
            axis=0,
        ).astype(np.float64),
        1: np.mean(
            embeddings[
                stress_indices
            ],
            axis=0,
        ).astype(np.float64),
    }


def prototype_predict(
    query_embeddings: np.ndarray,
    prototypes: Dict[int, np.ndarray],
    metric: str,
    temperature: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    prototype_matrix = np.stack(
        [
            prototypes[0],
            prototypes[1],
        ],
        axis=0,
    )

    metric = metric.lower().strip()

    if metric == "cosine":
        scores = (
            normalize_rows(
                query_embeddings
            )
            @ normalize_rows(
                prototype_matrix
            ).T
        )

    elif metric == "euclidean":
        diff = (
            query_embeddings[
                :, None, :
            ]
            - prototype_matrix[
                None, :, :
            ]
        )

        scores = -np.linalg.norm(
            diff,
            axis=2,
        )

    else:
        raise ValueError(
            "prototype_metric 必须是 cosine 或 euclidean"
        )

    probabilities = stable_softmax(
        scores=scores,
        temperature=temperature,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    stress_probability = probabilities[
        :, 1
    ].astype(np.float64)

    return (
        predictions,
        stress_probability,
    )


# =============================================================================
# 汇总
# =============================================================================

def summarize_across_subjects(
    df: pd.DataFrame,
    method: str,
    k_shot: int,
) -> dict:
    row = {
        "method": method,
        "k_shot_per_class": int(
            k_shot
        ),
        "n_subjects": int(
            df[
                "subject_id"
            ].nunique()
        ),
    }

    for metric in METRIC_NAMES:
        values = df[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        row[
            f"{metric}_mean"
        ] = float(
            np.nanmean(
                values
            )
        )

        row[
            f"{metric}_sd_subject"
        ] = float(
            np.nanstd(
                values,
                ddof=1,
            )
            if len(
                values
            ) > 1
            else 0.0
        )

    return row


# =============================================================================
# 主实验
# =============================================================================

def run_experiment(
    config: Config,
    device: torch.device,
) -> None:
    set_seed(
        config.random_seed,
        deterministic=config.deterministic,
    )

    npz_path = Path(
        config.data_npz
    ).expanduser().resolve()

    manifest_path = Path(
        config.manifest_csv
    ).expanduser().resolve()

    output_dir = Path(
        config.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        rr,
        hrv,
        labels,
        subject_ids,
        manifest,
    ) = load_processed_data(
        npz_path=npz_path,
        manifest_path=manifest_path,
    )

    subjects = sorted(
        np.unique(
            subject_ids
        ).tolist(),
        key=subject_number,
    )

    print("=" * 88)
    print("CNN + Prototype strict LOSO")
    print("=" * 88)
    print(f"NPZ        : {npz_path}")
    print(f"Manifest   : {manifest_path}")
    print(f"Output     : {output_dir}")
    print(f"Device     : {device}")
    print(f"Input mode : {config.input_mode}")
    print(f"RR shape   : {rr.shape}")
    print(f"HRV shape  : {hrv.shape}")
    print(f"Subjects   : {subjects}")
    print(f"K          : {config.k_values}")
    print(f"Repeats    : {config.repeats}")
    print(
        f"Cal frac   : {config.calibration_fraction:.2f}"
    )
    print(f"Gap RR     : {config.gap_rr}")
    print()

    global_rows: List[dict] = []
    episode_rows: List[dict] = []
    split_rows: List[dict] = []
    training_history_rows: List[dict] = []
    fold_training_rows: List[dict] = []

    for (
        outer_fold_index,
        target_subject,
    ) in enumerate(
        subjects
    ):
        print("-" * 88)
        print(
            f"Fold {outer_fold_index + 1:02d}/"
            f"{len(subjects)} | target={target_subject}"
        )

        target_mask = (
            subject_ids
            == target_subject
        )

        source_mask = (
            ~target_mask
        )

        source_indices = np.where(
            source_mask
        )[0]

        target_indices = np.where(
            target_mask
        )[0]

        source_subjects = sorted(
            np.unique(
                subject_ids[
                    source_indices
                ]
            ).tolist(),
            key=subject_number,
        )

        fold_seed = (
            config.random_seed
            + subject_number(
                target_subject
            ) * 1000
        )

        set_seed(
            fold_seed,
            deterministic=config.deterministic,
        )

        # ---------------------------------------------------------------------
        # 1) 仅 source subjects 内部选择 best epoch
        # ---------------------------------------------------------------------
        start_train = time.perf_counter()

        (
            best_epoch,
            inner_history,
            validation_subjects,
        ) = train_with_inner_validation(
            rr=rr,
            hrv=hrv,
            labels=labels,
            subject_ids=subject_ids,
            source_indices=source_indices,
            source_subjects=source_subjects,
            outer_fold_index=outer_fold_index,
            config=config,
            device=device,
            fold_seed=fold_seed,
        )

        for row in inner_history:
            row.update(
                {
                    "target_subject": target_subject,
                    "outer_fold": (
                        outer_fold_index
                        + 1
                    ),
                }
            )

            training_history_rows.append(
                row
            )

        # ---------------------------------------------------------------------
        # 2) 全部 14 source subjects 按 best epoch 重训
        # ---------------------------------------------------------------------
        set_seed(
            fold_seed + 999,
            deterministic=config.deterministic,
        )

        (
            model,
            preprocessor,
            final_history,
        ) = train_final_source_model(
            rr=rr,
            hrv=hrv,
            labels=labels,
            source_indices=source_indices,
            best_epoch=best_epoch,
            config=config,
            device=device,
            fold_seed=fold_seed,
        )

        for row in final_history:
            row.update(
                {
                    "target_subject": target_subject,
                    "outer_fold": (
                        outer_fold_index
                        + 1
                    ),
                    "validation_subjects": ",".join(
                        validation_subjects
                    ),
                }
            )

            training_history_rows.append(
                row
            )

        train_seconds = (
            time.perf_counter()
            - start_train
        )

        retrain_epochs = int(best_epoch)

        fold_training_rows.append(
            {
                "outer_fold": (
                    outer_fold_index
                    + 1
                ),
                "target_subject": target_subject,
                "source_subjects": ",".join(
                    source_subjects
                ),
                "inner_validation_subjects": ",".join(
                    validation_subjects
                ),
                "best_epoch": int(
                    best_epoch
                ),
                "final_retrain_epochs": int(
                    retrain_epochs
                ),
                "training_seconds_total": float(
                    train_seconds
                ),
                "device": str(
                    device
                ),
                "input_mode": config.input_mode,
            }
        )

        print(
            f"  inner val={validation_subjects} | "
            f"best epoch={best_epoch} | "
            f"retrain={retrain_epochs} epochs | "
            f"time={train_seconds:.1f}s"
        )

        # ---------------------------------------------------------------------
        # 3) 用 source-only scaler transform 全部样本
        # ---------------------------------------------------------------------
        rr_scaled = preprocessor.transform_rr(
            rr
        )

        hrv_scaled = preprocessor.transform_hrv(
            hrv
        )

        # ---------------------------------------------------------------------
        # 4) target 内部固定 calibration/test pool
        # ---------------------------------------------------------------------
        (
            calibration_pool,
            test_pool,
            diagnostics,
        ) = build_target_pools(
            target_subject=target_subject,
            manifest=manifest,
            calibration_fraction=(
                config.calibration_fraction
            ),
            gap_rr=config.gap_rr,
        )

        for diag in diagnostics:
            diag.update(
                {
                    "outer_fold": (
                        outer_fold_index
                        + 1
                    )
                }
            )

            split_rows.append(
                diag
            )

        test_sample_indices = test_pool[
            "sample_index"
        ].to_numpy(
            dtype=np.int64
        )

        y_test = labels[
            test_sample_indices
        ]

        if (
            config.require_both_test_classes
            and len(
                np.unique(
                    y_test
                )
            ) < 2
        ):
            print(
                "  SKIP: target test pool 不足两个类别"
            )
            continue

        # ---------------------------------------------------------------------
        # 5) 一次性抽取 target calibration/test 的 learned embeddings
        #
        # 为方便使用原始 sample_index，创建全数据 embedding 数组，
        # 但只有真正需要的 target calibration + test 被推理。
        # ---------------------------------------------------------------------
        target_needed_indices = np.unique(
            np.concatenate(
                [
                    calibration_pool[
                        "sample_index"
                    ].to_numpy(
                        dtype=np.int64
                    ),
                    test_sample_indices,
                ]
            )
        )

        (
            _,
            _,
            _,
            target_embeddings_compact,
        ) = run_model_on_indices(
            model=model,
            rr_scaled=rr_scaled,
            hrv_scaled=hrv_scaled,
            labels=labels,
            indices=target_needed_indices,
            batch_size=config.batch_size,
            device=device,
            num_workers=config.num_workers,
        )

        embeddings_by_sample = np.full(
            (
                len(labels),
                config.embedding_dim,
            ),
            np.nan,
            dtype=np.float32,
        )

        embeddings_by_sample[
            target_needed_indices
        ] = target_embeddings_compact

        # ---------------------------------------------------------------------
        # 6) 0-shot Global CNN：和 Prototype 完全相同的 test pool
        # ---------------------------------------------------------------------
        infer_start = time.perf_counter()

        (
            global_y_true,
            global_y_pred,
            global_stress_prob,
            _,
        ) = run_model_on_indices(
            model=model,
            rr_scaled=rr_scaled,
            hrv_scaled=hrv_scaled,
            labels=labels,
            indices=test_sample_indices,
            batch_size=config.batch_size,
            device=device,
            num_workers=config.num_workers,
        )

        infer_seconds = (
            time.perf_counter()
            - infer_start
        )

        global_metrics = evaluate_binary(
            y_true=global_y_true,
            y_pred=global_y_pred,
            stress_probability=global_stress_prob,
        )

        global_rows.append(
            {
                "outer_fold": (
                    outer_fold_index
                    + 1
                ),
                "subject_id": target_subject,
                "method": (
                    f"Global {config.input_mode.upper()} Encoder"
                ),
                "k_shot_per_class": 0,
                "n_source": int(
                    len(source_indices)
                ),
                "n_target_total": int(
                    len(target_indices)
                ),
                "n_calibration_pool": int(
                    len(calibration_pool)
                ),
                "n_test": int(
                    len(test_sample_indices)
                ),
                "n_test_nonstress": int(
                    np.sum(
                        y_test == 0
                    )
                ),
                "n_test_stress": int(
                    np.sum(
                        y_test == 1
                    )
                ),
                "best_epoch": int(
                    best_epoch
                ),
                "inference_seconds": float(
                    infer_seconds
                ),
                "average_latency_ms_per_window": float(
                    infer_seconds
                    / max(
                        len(
                            test_sample_indices
                        ),
                        1,
                    )
                    * 1000.0
                ),
                **global_metrics,
            }
        )

        print(
            "  0-shot Global CNN: "
            f"BA={global_metrics['balanced_accuracy']:.4f} | "
            f"F1={global_metrics['macro_f1']:.4f} | "
            f"AUC={global_metrics['auroc']:.4f}"
        )

        # ---------------------------------------------------------------------
        # 7) K-shot Prototype in learned embedding
        # ---------------------------------------------------------------------
        cal_nonstress = calibration_pool[
            calibration_pool[
                "binary_label"
            ].astype(int) == 0
        ]

        cal_stress = calibration_pool[
            calibration_pool[
                "binary_label"
            ].astype(int) == 1
        ]

        max_nonstress = max_nonoverlap_count(
            cal_nonstress
        )

        max_stress = max_nonoverlap_count(
            cal_stress
        )

        query_embeddings = embeddings_by_sample[
            test_sample_indices
        ]

        if not np.all(
            np.isfinite(
                query_embeddings
            )
        ):
            raise RuntimeError(
                f"{target_subject}: query embedding 存在 NaN/Inf"
            )

        for k in config.k_values:
            if (
                max_nonstress < k
                or max_stress < k
            ):
                print(
                    f"  K={k}: 不可行，"
                    f"non-overlap capacity "
                    f"nonstress={max_nonstress}, "
                    f"stress={max_stress}"
                )

                episode_rows.append(
                    {
                        "subject_id": target_subject,
                        "k_shot_per_class": int(
                            k
                        ),
                        "repeat": -1,
                        "status": (
                            "insufficient_nonoverlap_calibration"
                        ),
                        "max_nonoverlap_nonstress": int(
                            max_nonstress
                        ),
                        "max_nonoverlap_stress": int(
                            max_stress
                        ),
                        "n_test": int(
                            len(
                                test_sample_indices
                            )
                        ),
                    }
                )

                continue

            valid_repeats = 0

            for repeat in range(
                config.repeats
            ):
                episode_seed = (
                    config.random_seed
                    + subject_number(
                        target_subject
                    ) * 100_000
                    + int(k) * 1_000
                    + repeat
                )

                rng = np.random.default_rng(
                    episode_seed
                )

                sampled = sample_kshot_calibration(
                    calibration_pool=calibration_pool,
                    k=k,
                    rng=rng,
                    balance_nonstress_states=(
                        config.balance_nonstress_states
                    ),
                )

                if sampled is None:
                    episode_rows.append(
                        {
                            "subject_id": target_subject,
                            "k_shot_per_class": int(
                                k
                            ),
                            "repeat": int(
                                repeat
                            ),
                            "episode_seed": int(
                                episode_seed
                            ),
                            "status": "sampling_failed",
                            "n_test": int(
                                len(
                                    test_sample_indices
                                )
                            ),
                        }
                    )

                    continue

                (
                    nonstress_idx,
                    stress_idx,
                ) = sampled

                if not np.all(
                    np.isfinite(
                        embeddings_by_sample[
                            np.concatenate(
                                [
                                    nonstress_idx,
                                    stress_idx,
                                ]
                            )
                        ]
                    )
                ):
                    raise RuntimeError(
                        f"{target_subject} K={k}: calibration embedding 缺失"
                    )

                prototypes = compute_prototypes(
                    embeddings=embeddings_by_sample,
                    nonstress_indices=nonstress_idx,
                    stress_indices=stress_idx,
                )

                proto_pred, proto_stress_prob = prototype_predict(
                    query_embeddings=query_embeddings,
                    prototypes=prototypes,
                    metric=config.prototype_metric,
                    temperature=(
                        config.prototype_temperature
                    ),
                )

                metrics = evaluate_binary(
                    y_true=y_test,
                    y_pred=proto_pred,
                    stress_probability=proto_stress_prob,
                )

                selected_rows = manifest[
                    manifest[
                        "sample_index"
                    ].isin(
                        np.concatenate(
                            [
                                nonstress_idx,
                                stress_idx,
                            ]
                        )
                    )
                ]

                episode_rows.append(
                    {
                        "subject_id": target_subject,
                        "outer_fold": (
                            outer_fold_index
                            + 1
                        ),
                        "k_shot_per_class": int(
                            k
                        ),
                        "repeat": int(
                            repeat
                        ),
                        "episode_seed": int(
                            episode_seed
                        ),
                        "status": "ok",
                        "method": (
                            f"{config.input_mode.upper()} Encoder + Prototype"
                        ),
                        "input_mode": config.input_mode,
                        "embedding_dim": int(
                            config.embedding_dim
                        ),
                        "prototype_metric": (
                            config.prototype_metric
                        ),
                        "n_source": int(
                            len(
                                source_indices
                            )
                        ),
                        "n_target_total": int(
                            len(
                                target_indices
                            )
                        ),
                        "n_calibration_pool": int(
                            len(
                                calibration_pool
                            )
                        ),
                        "max_nonoverlap_nonstress": int(
                            max_nonstress
                        ),
                        "max_nonoverlap_stress": int(
                            max_stress
                        ),
                        "n_calibration_selected_total": int(
                            2 * k
                        ),
                        "n_selected_baseline": int(
                            np.sum(
                                selected_rows[
                                    "state"
                                ].astype(str)
                                == "baseline"
                            )
                        ),
                        "n_selected_amusement": int(
                            np.sum(
                                selected_rows[
                                    "state"
                                ].astype(str)
                                == "amusement"
                            )
                        ),
                        "n_selected_stress": int(
                            np.sum(
                                selected_rows[
                                    "state"
                                ].astype(str)
                                == "stress"
                            )
                        ),
                        "n_test": int(
                            len(
                                test_sample_indices
                            )
                        ),
                        "n_test_nonstress": int(
                            np.sum(
                                y_test == 0
                            )
                        ),
                        "n_test_stress": int(
                            np.sum(
                                y_test == 1
                            )
                        ),
                        **metrics,
                    }
                )

                valid_repeats += 1

            print(
                f"  K={k}: valid repeats="
                f"{valid_repeats}/{config.repeats}"
            )

    # =========================================================================
    # 保存原始结果
    # =========================================================================

    global_df = pd.DataFrame(
        global_rows
    )

    episodes_df = pd.DataFrame(
        episode_rows
    )

    split_df = pd.DataFrame(
        split_rows
    )

    training_history_df = pd.DataFrame(
        training_history_rows
    )

    fold_training_df = pd.DataFrame(
        fold_training_rows
    )

    global_path = (
        output_dir
        / "cnn_global_subject_results.csv"
    )

    episodes_path = (
        output_dir
        / "cnn_prototype_episode_results.csv"
    )

    split_path = (
        output_dir
        / "cnn_split_diagnostics.csv"
    )

    history_path = (
        output_dir
        / "cnn_training_history.csv"
    )

    fold_training_path = (
        output_dir
        / "cnn_fold_training_summary.csv"
    )

    global_df.to_csv(
        global_path,
        index=False,
        encoding="utf-8-sig",
    )

    episodes_df.to_csv(
        episodes_path,
        index=False,
        encoding="utf-8-sig",
    )

    split_df.to_csv(
        split_path,
        index=False,
        encoding="utf-8-sig",
    )

    training_history_df.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    fold_training_df.to_csv(
        fold_training_path,
        index=False,
        encoding="utf-8-sig",
    )

    ok = episodes_df[
        episodes_df[
            "status"
        ].astype(str)
        == "ok"
    ].copy()

    if ok.empty:
        raise RuntimeError(
            "没有任何成功的 CNN Prototype episode。"
        )

    # =========================================================================
    # 每个 subject/K：先对 repeats 求均值
    # =========================================================================

    subject_k_mean = (
        ok.groupby(
            [
                "subject_id",
                "k_shot_per_class",
            ],
            as_index=False,
        )[
            list(
                METRIC_NAMES
            )
        ]
        .mean()
    )

    subject_k_sd = (
        ok.groupby(
            [
                "subject_id",
                "k_shot_per_class",
            ],
            as_index=False,
        )[
            list(
                METRIC_NAMES
            )
        ]
        .std()
    )

    subject_k_sd = (
        subject_k_sd.rename(
            columns={
                metric: (
                    f"{metric}_sd_repeat"
                )
                for metric
                in METRIC_NAMES
            }
        )
    )

    subject_results = (
        subject_k_mean.merge(
            subject_k_sd,
            on=[
                "subject_id",
                "k_shot_per_class",
            ],
            how="left",
        )
    )

    subject_results[
        "subject_sort"
    ] = subject_results[
        "subject_id"
    ].map(
        subject_number
    )

    subject_results = (
        subject_results.sort_values(
            [
                "subject_sort",
                "k_shot_per_class",
            ]
        )
        .drop(
            columns=[
                "subject_sort"
            ]
        )
    )

    subject_results_path = (
        output_dir
        / "cnn_prototype_subject_results.csv"
    )

    subject_results.to_csv(
        subject_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # K-shot summary across subjects
    # =========================================================================

    kshot_rows: List[dict] = []

    for (
        k,
        group,
    ) in subject_k_mean.groupby(
        "k_shot_per_class",
        sort=True,
    ):
        kshot_rows.append(
            summarize_across_subjects(
                df=group,
                method=(
                    f"{config.input_mode.upper()} Encoder + Prototype"
                ),
                k_shot=int(
                    k
                ),
            )
        )

    kshot_summary = pd.DataFrame(
        kshot_rows
    )

    kshot_summary_path = (
        output_dir
        / "cnn_prototype_kshot_summary.csv"
    )

    kshot_summary.to_csv(
        kshot_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # Matched comparison: 0-shot Global CNN + K-shot Prototype
    # =========================================================================

    comparison_rows: List[dict] = []

    comparison_rows.append(
        summarize_across_subjects(
            df=global_df,
            method=(
                f"Global {config.input_mode.upper()} Encoder"
            ),
            k_shot=0,
        )
    )

    comparison_rows.extend(
        kshot_summary.to_dict(
            orient="records"
        )
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        output_dir
        / "cnn_matched_comparison_summary.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # Subject-level Macro-F1 comparison
    # =========================================================================

    global_subject = global_df[
        [
            "subject_id",
            "macro_f1",
            "balanced_accuracy",
            "auroc",
        ]
    ].rename(
        columns={
            "macro_f1": (
                "cnn_0shot_macro_f1"
            ),
            "balanced_accuracy": (
                "cnn_0shot_balanced_accuracy"
            ),
            "auroc": (
                "cnn_0shot_auroc"
            ),
        }
    )

    proto_pivot = (
        subject_k_mean.pivot(
            index="subject_id",
            columns="k_shot_per_class",
            values="macro_f1",
        )
        .rename(
            columns={
                k: (
                    f"cnn_prototype_"
                    f"{int(k)}shot_macro_f1"
                )
                for k
                in subject_k_mean[
                    "k_shot_per_class"
                ].unique()
            }
        )
        .reset_index()
    )

    subject_comparison = (
        global_subject.merge(
            proto_pivot,
            on="subject_id",
            how="left",
        )
    )

    for k in config.k_values:
        proto_col = (
            f"cnn_prototype_"
            f"{k}shot_macro_f1"
        )

        if proto_col in subject_comparison.columns:
            subject_comparison[
                f"delta_{k}shot_vs_cnn"
            ] = (
                subject_comparison[
                    proto_col
                ]
                - subject_comparison[
                    "cnn_0shot_macro_f1"
                ]
            )

    subject_comparison[
        "subject_sort"
    ] = subject_comparison[
        "subject_id"
    ].map(
        subject_number
    )

    subject_comparison = (
        subject_comparison.sort_values(
            "subject_sort"
        )
        .drop(
            columns=[
                "subject_sort"
            ]
        )
    )

    subject_comparison_path = (
        output_dir
        / "cnn_subject_comparison.csv"
    )

    subject_comparison.to_csv(
        subject_comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # 保存配置
    # =========================================================================

    config_path = (
        output_dir
        / "cnn_config.json"
    )

    config_dict = asdict(
        config
    )

    config_dict[
        "device_used"
    ] = str(
        device
    )

    config_dict[
        "torch_version"
    ] = str(
        torch.__version__
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_dict,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================================
    # 控制台输出
    # =========================================================================

    print()
    print("=" * 88)
    print("CNN MATCHED-TEST SUMMARY")
    print("=" * 88)

    display_columns = [
        "method",
        "k_shot_per_class",
        "n_subjects",
        "balanced_accuracy_mean",
        "balanced_accuracy_sd_subject",
        "macro_f1_mean",
        "macro_f1_sd_subject",
        "auroc_mean",
        "auroc_sd_subject",
    ]

    print(
        comparison_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    for path in [
        global_path,
        episodes_path,
        kshot_summary_path,
        subject_results_path,
        comparison_path,
        subject_comparison_path,
        history_path,
        fold_training_path,
        split_path,
        config_path,
    ]:
        print(
            f"  {path}"
        )


# =============================================================================
# CLI
# =============================================================================

def parse_k_values(
    text: str,
) -> Tuple[int, ...]:
    values: List[int] = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        value = int(item)

        if value <= 0:
            raise argparse.ArgumentTypeError(
                "K 必须 > 0"
            )

        values.append(
            value
        )

    if not values:
        raise argparse.ArgumentTypeError(
            "至少需要一个 K"
        )

    return tuple(
        sorted(
            set(
                values
            )
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WESAD CNN/HRV Encoder + Few-shot Prototype strict LOSO"
        )
    )

    parser.add_argument(
        "--data-npz",
        type=str,
        required=True,
        help="wesad_windows.npz",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="wesad_manifest.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--input-mode",
        choices=[
            "fusion",
            "rr",
            "hrv",
        ],
        default="fusion",
        help="默认 fusion = RR-CNN + HRV-MLP",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=(1, 3, 5),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--gap-rr",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--prototype-metric",
        choices=[
            "cosine",
            "euclidean",
        ],
        default="cosine",
    )

    parser.add_argument(
        "--prototype-temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--n-val-subjects",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Windows/PyCharm 建议先保持 0",
    )

    parser.add_argument(
        "--minimum-retrain-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--no-balance-nonstress",
        action="store_true",
    )

    parser.add_argument(
        "--allow-single-class-test",
        action="store_true",
    )

    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="关闭 deterministic 模式，可换取部分 GPU 速度",
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    if args.embedding_dim <= 0:
        parser.error(
            "--embedding-dim 必须 > 0"
        )

    if args.repeats <= 0:
        parser.error(
            "--repeats 必须 > 0"
        )

    if not (
        0.05
        <= args.calibration_fraction
        <= 0.80
    ):
        parser.error(
            "--calibration-fraction 建议在 [0.05, 0.80]"
        )

    if args.gap_rr < 0:
        parser.error(
            "--gap-rr 不能 < 0"
        )

    if args.prototype_temperature <= 0:
        parser.error(
            "--prototype-temperature 必须 > 0"
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size 必须 > 0"
        )

    if args.max_epochs <= 0:
        parser.error(
            "--max-epochs 必须 > 0"
        )

    if args.patience <= 0:
        parser.error(
            "--patience 必须 > 0"
        )

    if args.min_epochs <= 0:
        parser.error(
            "--min-epochs 必须 > 0"
        )

    if args.learning_rate <= 0:
        parser.error(
            "--learning-rate 必须 > 0"
        )

    if args.n_val_subjects <= 0:
        parser.error(
            "--n-val-subjects 必须 > 0"
        )

    device = resolve_device(
        args.device
    )

    config = Config(
        data_npz=args.data_npz,
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        embedding_dim=args.embedding_dim,
        k_values=args.k_values,
        repeats=args.repeats,
        random_seed=args.seed,
        calibration_fraction=(
            args.calibration_fraction
        ),
        gap_rr=args.gap_rr,
        balance_nonstress_states=(
            not args.no_balance_nonstress
        ),
        require_both_test_classes=(
            not args.allow_single_class_test
        ),
        prototype_metric=(
            args.prototype_metric
        ),
        prototype_temperature=(
            args.prototype_temperature
        ),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        n_val_subjects=args.n_val_subjects,
        num_workers=args.num_workers,
        minimum_retrain_epochs=(
            args.minimum_retrain_epochs
        ),
        deterministic=(
            not args.non_deterministic
        ),
    )

    run_experiment(
        config=config,
        device=device,
    )


if __name__ == "__main__":
    main()
