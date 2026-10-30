#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
np_longitudinal_relax_loso_v1.py

Main experiment for RELAX longitudinal personalized stress recognition.

Goal
----
Evaluate whether predictions for the SAME future query improve as labeled
personal history grows, while keeping target-side model weights frozen for
Neural Process inference.

Main matched-query protocol
---------------------------
For each outer target subject:
  1) train a source-only Fusion Encoder + global classifier on all other subjects;
  2) freeze the final source model;
  3) meta-train a temporal attentive latent Neural Process on source-only chronological past->future episodes;
  4) sort target samples chronologically;
  5) keep only future queries that already have >= max_k examples PER CLASS
     in their past history;
  6) for the SAME query, build nested contexts from the most recent history:
         C2 = 1 nonstress + 1 stress
         C4 = 2 nonstress + 2 stress
         C6 = 3 nonstress + 3 stress
         C8 = 4 nonstress + 4 stress
     by default.
  7) evaluate:
         Global Fusion
         Growing Prototype
         Gradient Fine-tuning (linear head only)
         Neural Process (NO target-side gradient update)

Critical anti-confounding constraints
-------------------------------------
- identical query set for C2/C4/C6/C8;
- every context timestamp is strictly earlier than its query timestamp;
- contexts are nested;
- context is balanced by class;
- Global Fusion probabilities are identical across context sizes;
- NP target inference is wrapped in torch.no_grad() and never optimized on target.
- primary NP probability uses latent-prior mean; MC sampling is uncertainty-only.

Inputs
------
Prepared with prepare_relax_longitudinal_v2.py:

  relax_longitudinal_v2.npz
  relax_longitudinal_v2_manifest.csv

NPZ must contain:
  X_rr            (N, 64)
  X_hrv           (N, 32)
  y               (N,)
  subject_ids     (N,)
  anchor_unix_ns  (N,)

Manifest must contain at least:
  sample_index
  user
  label
  anchor_utc
  overlaps_previous_sample_rr_window  (optional)

Outputs
-------
  matched_query_results.csv
  context_audit.csv
  subject_metrics.csv
  pooled_metrics.csv
  progressive_gain_subject.csv
  paired_stats.csv
  fold_training_summary.csv
  global_invariance_audit.csv
  run_config.json

Notes
-----
- No preprocessing is fitted on the target subject.
- Inner validation uses source subjects only.
- After best epoch selection, encoder and NP are retrained on ALL source
  subjects for exactly the selected number of epochs.
- This v1 script focuses on the matched-query main experiment. A true
  prequential secondary deployment simulation should be added separately
  after this main experiment is stable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise SystemExit(
        "\nPyTorch is required.\n"
        "Install it first, then rerun this script.\n"
    ) from e

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise SystemExit(
        "\nscikit-learn is required.\n"
        "Install with:\n"
        "  pip install scikit-learn\n"
    ) from e

try:
    from scipy.stats import wilcoxon
except Exception as e:
    raise SystemExit(
        "\nSciPy is required.\n"
        "Install with:\n"
        "  pip install scipy\n"
    ) from e


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred over maximal speed for publication runs.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def parse_int_list(text: str) -> List[int]:
    vals = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    if not vals:
        raise ValueError(f"Empty integer list: {text!r}")
    return vals


def parse_float_list(text: str) -> List[float]:
    vals = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))

    if not vals:
        raise ValueError(
            f"Empty floating-point list: {text!r}"
        )

    if any(
        (not np.isfinite(v)) or v < 0.0
        for v in vals
    ):
        raise ValueError(
            "Calibration gamma candidates must be finite and >= 0."
        )

    return vals


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FoldScalers:
    rr_mean: float
    rr_std: float
    hrv_mean: np.ndarray
    hrv_scale: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(
        description="RELAX longitudinal matched-query LOSO experiment."
    )

    p.add_argument("--data-npz", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--ablation-id",
        choices=["A1", "A2", "A3", "A4", "A5", "A6"],
        default="A6",
        help=(
            "Matched ablation: A1 direct attentive NP without Global anchor; "
            "A2 anchored directional residual without a hard bound; A3 anchored "
            "bounded residual without a contrast-direction constraint; A4 full "
            "structural model with gamma=1; A5 source-selected gamma without the "
            "minimum-improvement gate; A6 complete safe method."
        ),
    )

    p.add_argument(
        "--context-per-class",
        default="1,2,3,4",
        help=(
            "Balanced target context sizes PER CLASS. "
            "Default 1,2,3,4 -> total C2,C4,C6,C8."
        ),
    )
    p.add_argument(
        "--target-subjects",
        default="auto",
        help=(
            "Comma-separated target IDs, or 'auto'. "
            "Auto keeps subjects with at least one matched future query."
        ),
    )
    p.add_argument(
        "--drop-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop samples flagged as overlapping the immediately previous "
            "RR window. Default: true."
        ),
    )

    p.add_argument(
        "--seeds",
        default="2026",
        help=(
            "Comma-separated model seeds. Use one seed for smoke tests; "
            "e.g. 11,22,33,44,55 for final repeated runs."
        ),
    )
    p.add_argument("--device", default="auto")

    # Source encoder.
    p.add_argument("--inner-val-fraction", type=float, default=0.20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--encoder-weight-decay", type=float, default=1e-4)
    p.add_argument("--encoder-max-epochs", type=int, default=80)
    p.add_argument("--encoder-patience", type=int, default=12)

    # NP meta-training.
    p.add_argument("--np-lr", type=float, default=5e-4)
    p.add_argument("--np-weight-decay", type=float, default=1e-5)
    p.add_argument("--np-max-epochs", type=int, default=80)
    p.add_argument("--np-patience", type=int, default=12)
    p.add_argument("--episodes-per-epoch", type=int, default=64)
    p.add_argument("--np-val-episodes", type=int, default=64)
    p.add_argument("--meta-query-per-class", type=int, default=1)
    p.add_argument("--np-latent-dim", type=int, default=32)
    p.add_argument("--np-hidden-dim", type=int, default=96)
    p.add_argument("--np-attention-dim", type=int, default=64)
    p.add_argument(
        "--np-context-aux-weight",
        type=float,
        default=0.10,
        help=(
            "Auxiliary query-specific context classification loss. "
            "V1.5 default 0.10."
        ),
    )
    p.add_argument(
        "--np-posterior-weight",
        type=float,
        default=0.50,
        help=(
            "Weight of posterior predictive CE. The deployment-matched "
            "prior predictive CE always has weight 1.0."
        ),
    )
    p.add_argument(
        "--np-alpha-max",
        type=float,
        default=0.50,
        help=(
            "Maximum adaptation gate. Final log-odds correction is bounded "
            "by alpha_max * residual_max. Default 0.50."
        ),
    )
    p.add_argument(
        "--np-residual-max",
        type=float,
        default=2.0,
        help=(
            "Maximum absolute raw residual log-odds before gating. "
            "With alpha_max=0.5, applied log-odds shift is at most 1.0."
        ),
    )
    p.add_argument(
        "--np-gate-init-fraction",
        type=float,
        default=0.10,
        help=(
            "Initial sigmoid fraction of alpha_max for the gate. "
            "Default 0.10 -> initial gate about 0.05 when alpha_max=0.5."
        ),
    )
    p.add_argument(
        "--np-gate-reg-weight",
        type=float,
        default=0.01,
        help=(
            "Small regularizer discouraging unnecessary departure from "
            "the source Global Fusion anchor."
        ),
    )
    p.add_argument(
        "--np-residual-reg-weight",
        type=float,
        default=0.02,
        help=(
            "V1.6 residual saturation penalty. Applied to the normalized "
            "pre-gate contrastive residual. Default 0.02."
        ),
    )
    p.add_argument(
        "--np-contrast-scale",
        type=float,
        default=2.0,
        help=(
            "Fixed scale for tanh(query-to-personal-history contrast). "
            "Default 2.0."
        ),
    )
    p.add_argument(
        "--np-residual-init-fraction",
        type=float,
        default=0.02,
        help=(
            "Initial residual magnitude fraction in (0,1). "
            "Default 0.02, so V1.6 begins very close to Global Fusion."
        ),
    )
    p.add_argument(
        "--np-beta-kl",
        type=float,
        default=5e-3,
        help=(
            "KL weight between posterior and prior latent distributions. "
            "V1.5 default 0.005."
        ),
    )
    p.add_argument("--mc-samples-eval", type=int, default=30)

    # Source-only adaptation-strength calibration.
    p.add_argument(
        "--adaptation-gamma-candidates",
        default="0,0.25,0.5,1,2,4,8",
        help=(
            "Fixed multipliers for the already-bounded V1.6 personal "
            "log-odds shift. Selection uses SOURCE-ONLY inner-validation "
            "chronological episodes. gamma=0 is the Global-Fusion fallback."
        ),
    )
    p.add_argument(
        "--adaptation-calibration-primary",
        choices=["nll", "brier"],
        default="nll",
        help=(
            "Primary SOURCE-ONLY calibration objective. "
            "Ties: secondary proper score, then smaller gamma."
        ),
    )
    p.add_argument(
        "--adaptation-min-nll-improvement",
        type=float,
        default=1e-3,
        help=(
            "Minimum SOURCE-ONLY balanced-NLL improvement relative to "
            "gamma=0 required before personal adaptation is activated. "
            "Otherwise the fold falls back exactly to Global Fusion. "
            "Default 0.001."
        ),
    )

    # Prototype.
    p.add_argument("--prototype-temperature", type=float, default=1.0)

    # Fine-tuning comparator.
    p.add_argument("--finetune-lr", type=float, default=5e-3)
    p.add_argument("--finetune-steps", type=int, default=60)
    p.add_argument(
        "--finetune-anchor-lambda",
        type=float,
        default=1e-3,
        help="L2 anchor to the source global head.",
    )

    # Runtime controls.
    p.add_argument(
        "--max-target-subjects",
        type=int,
        default=0,
        help="0 means no limit; useful for smoke testing 1-2 folds.",
    )
    p.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Device / data
# ---------------------------------------------------------------------------

ABLATION_SPECS = {
    "A1": {
        "label": "Direct attentive NP (no Global anchor)",
        "uses_global_anchor": False,
        "hard_bound_enabled": False,
        "direction_constraint_enabled": False,
        "source_gamma_mode": "not_applicable",
    },
    "A2": {
        "label": "Anchored directional NP (no hard bound)",
        "uses_global_anchor": True,
        "hard_bound_enabled": False,
        "direction_constraint_enabled": True,
        "source_gamma_mode": "fixed_gamma_1",
    },
    "A3": {
        "label": "Anchored bounded NP (free residual direction)",
        "uses_global_anchor": True,
        "hard_bound_enabled": True,
        "direction_constraint_enabled": False,
        "source_gamma_mode": "fixed_gamma_1",
    },
    "A4": {
        "label": "Full constrained residual (fixed gamma=1)",
        "uses_global_anchor": True,
        "hard_bound_enabled": True,
        "direction_constraint_enabled": True,
        "source_gamma_mode": "fixed_gamma_1",
    },
    "A5": {
        "label": "Full constrained residual + source-selected gamma",
        "uses_global_anchor": True,
        "hard_bound_enabled": True,
        "direction_constraint_enabled": True,
        "source_gamma_mode": "source_argmin_without_minimum",
    },
    "A6": {
        "label": "Safe NP (complete method)",
        "uses_global_anchor": True,
        "hard_bound_enabled": True,
        "direction_constraint_enabled": True,
        "source_gamma_mode": "source_argmin_with_minimum_and_exact_fallback",
    },
}


def get_ablation_spec(ablation_id: str) -> dict:
    try:
        return ABLATION_SPECS[str(ablation_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation id: {ablation_id!r}") from exc


def configured_hard_bound(args) -> float:
    spec = get_ablation_spec(args.ablation_id)
    if not spec["hard_bound_enabled"]:
        return float("inf")
    return float(args.np_alpha_max * args.np_residual_max)

def resolve_device(name: str) -> torch.device:
    name = str(name).lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    return torch.device(name)


def load_prepared_data(
    npz_path: Path,
    manifest_path: Path,
    drop_overlap: bool,
):
    z = np.load(npz_path, allow_pickle=False)

    required = {
        "X_rr",
        "X_hrv",
        "y",
        "subject_ids",
        "anchor_unix_ns",
    }
    missing = required.difference(z.files)
    if missing:
        raise ValueError(
            f"NPZ missing keys: {sorted(missing)}"
        )

    X_rr = np.asarray(z["X_rr"], dtype=np.float32)
    X_hrv = np.asarray(z["X_hrv"], dtype=np.float32)
    y = np.asarray(z["y"], dtype=np.int64)
    subject_ids = np.asarray(z["subject_ids"], dtype=np.int64)
    anchor_ns = np.asarray(z["anchor_unix_ns"], dtype=np.int64)

    n = len(y)
    if not (
        len(X_rr)
        == len(X_hrv)
        == len(subject_ids)
        == len(anchor_ns)
        == n
    ):
        raise ValueError("NPZ arrays have inconsistent lengths.")

    if X_rr.ndim != 2 or X_rr.shape[1] != 64:
        raise ValueError(
            f"Expected X_rr shape (N,64), got {X_rr.shape}"
        )

    if X_hrv.ndim != 2 or X_hrv.shape[1] != 32:
        raise ValueError(
            f"Expected X_hrv shape (N,32), got {X_hrv.shape}"
        )

    manifest = pd.read_csv(manifest_path)

    required_manifest = {
        "sample_index",
        "user",
        "label",
        "anchor_utc",
        "ifb_submission_utc",
        "rr_end_utc",
    }
    mm = required_manifest.difference(manifest.columns)
    if mm:
        raise ValueError(
            f"Manifest missing columns: {sorted(mm)}"
        )

    manifest["sample_index"] = (
        pd.to_numeric(
            manifest["sample_index"],
            errors="raise",
        )
        .astype(int)
    )
    manifest["user"] = (
        pd.to_numeric(
            manifest["user"],
            errors="raise",
        )
        .astype(int)
    )
    manifest["label"] = (
        pd.to_numeric(
            manifest["label"],
            errors="raise",
        )
        .astype(int)
    )
    # The manifest contains valid ISO-8601 timestamps in mixed textual forms:
    # some rows include fractional seconds (e.g. ...09.940000+00:00),
    # while others are whole-second timestamps (e.g. ...10:42:00+00:00).
    # Recent pandas versions may infer one strict format from the first row and
    # then reject the other form. format="mixed" parses each ISO timestamp
    # independently while preserving UTC.
    manifest["anchor_utc"] = pd.to_datetime(
        manifest["anchor_utc"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    manifest["ifb_submission_utc"] = pd.to_datetime(
        manifest["ifb_submission_utc"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    manifest["rr_end_utc"] = pd.to_datetime(
        manifest["rr_end_utc"],
        utc=True,
        errors="raise",
        format="mixed",
    )

    if len(manifest) != n:
        raise ValueError(
            f"Manifest N={len(manifest)} != NPZ N={n}"
        )

    manifest = manifest.sort_values(
        "sample_index"
    ).reset_index(drop=True)

    if not np.array_equal(
        manifest["sample_index"].to_numpy(),
        np.arange(n),
    ):
        raise ValueError(
            "Manifest sample_index must be exactly 0..N-1 "
            "and align with NPZ arrays."
        )

    if not np.array_equal(
        manifest["label"].to_numpy(dtype=int),
        y,
    ):
        raise ValueError(
            "Manifest labels do not align with NPZ y."
        )

    if not np.array_equal(
        manifest["user"].to_numpy(dtype=int),
        subject_ids,
    ):
        raise ValueError(
            "Manifest user does not align with NPZ subject_ids."
        )

    # Use manifest timestamp as the source of truth after alignment checks.
    mani_ns = manifest["anchor_utc"].astype("int64").to_numpy()
    if not np.array_equal(mani_ns, anchor_ns):
        max_diff = np.max(np.abs(mani_ns - anchor_ns))
        raise ValueError(
            "Manifest anchor timestamps do not align with NPZ. "
            f"Max ns difference={max_diff}"
        )

    keep = np.ones(n, dtype=bool)

    if (
        drop_overlap
        and "overlaps_previous_sample_rr_window" in manifest.columns
    ):
        overlap = (
            manifest["overlaps_previous_sample_rr_window"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .to_numpy()
        )
        keep &= ~overlap

    # Reindex after QC filtering, while keeping original sample index.
    work = manifest.loc[keep].copy()
    work["original_sample_index"] = work["sample_index"]
    work = work.reset_index(drop=True)
    work["model_row"] = np.arange(len(work), dtype=int)

    X_rr = X_rr[keep]
    X_hrv = X_hrv[keep]
    y = y[keep]
    subject_ids = subject_ids[keep]
    anchor_ns = anchor_ns[keep]

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels 0/1, got {np.unique(y)}"
        )

    return {
        "X_rr": X_rr,
        "X_hrv": X_hrv,
        "y": y,
        "subject_ids": subject_ids,
        "anchor_ns": anchor_ns,
        "manifest": work,
    }


# ---------------------------------------------------------------------------
# Source-only preprocessing
# ---------------------------------------------------------------------------

def fit_scalers(
    X_rr: np.ndarray,
    X_hrv: np.ndarray,
) -> FoldScalers:
    rr_mean = float(np.mean(X_rr))
    rr_std = float(np.std(X_rr))
    if not np.isfinite(rr_std) or rr_std < 1e-8:
        rr_std = 1.0

    scaler = StandardScaler()
    scaler.fit(X_hrv)

    return FoldScalers(
        rr_mean=rr_mean,
        rr_std=rr_std,
        hrv_mean=scaler.mean_.astype(np.float64),
        hrv_scale=np.where(
            scaler.scale_ < 1e-12,
            1.0,
            scaler.scale_,
        ).astype(np.float64),
    )


def apply_scalers(
    X_rr: np.ndarray,
    X_hrv: np.ndarray,
    scalers: FoldScalers,
) -> Tuple[np.ndarray, np.ndarray]:
    rr = (
        (X_rr.astype(np.float32) - scalers.rr_mean)
        / scalers.rr_std
    ).astype(np.float32)

    hrv = (
        (
            X_hrv.astype(np.float64)
            - scalers.hrv_mean
        )
        / scalers.hrv_scale
    ).astype(np.float32)

    return rr, hrv


# ---------------------------------------------------------------------------
# Fusion encoder
# ---------------------------------------------------------------------------

class FusionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.rr = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.rr_fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.hrv = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.fusion = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(
        self,
        rr: torch.Tensor,
        hrv: torch.Tensor,
    ) -> torch.Tensor:
        # rr: (B,64)
        x_rr = rr.unsqueeze(1)
        x_rr = self.rr(x_rr).squeeze(-1)
        x_rr = self.rr_fc(x_rr)

        x_hrv = self.hrv(hrv)

        emb = torch.cat(
            [x_rr, x_hrv],
            dim=1,
        )
        emb = self.fusion(emb)
        return emb


class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusionEncoder()
        self.classifier = nn.Linear(64, 2)

    def forward(
        self,
        rr: torch.Tensor,
        hrv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(rr, hrv)
        logits = self.classifier(emb)
        return logits, emb


class RRHRVDataset(Dataset):
    def __init__(
        self,
        rr: np.ndarray,
        hrv: np.ndarray,
        y: np.ndarray,
    ):
        self.rr = torch.as_tensor(
            rr,
            dtype=torch.float32,
        )
        self.hrv = torch.as_tensor(
            hrv,
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(
            y,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.rr[idx], self.hrv[idx], self.y[idx]


def balanced_class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    total = counts.sum()

    weights = np.ones(2, dtype=np.float32)
    for c in [0, 1]:
        if counts[c] > 0:
            weights[c] = total / (2.0 * counts[c])

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def predict_classifier(
    model: FusionClassifier,
    rr: np.ndarray,
    hrv: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
):
    model.eval()

    ds = RRHRVDataset(
        rr,
        hrv,
        np.zeros(len(rr), dtype=np.int64),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
    )

    probs = []
    embeddings = []

    with torch.no_grad():
        for xb_rr, xb_hrv, _ in loader:
            xb_rr = xb_rr.to(device)
            xb_hrv = xb_hrv.to(device)

            logits, emb = model(
                xb_rr,
                xb_hrv,
            )

            probs.append(
                torch.softmax(logits, dim=1)
                .cpu()
                .numpy()
            )
            embeddings.append(
                emb.cpu().numpy()
            )

    return (
        np.concatenate(probs, axis=0),
        np.concatenate(embeddings, axis=0),
    )


def macro_f1_np(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> float:
    pred = np.argmax(probs, axis=1)
    return float(
        f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0,
        )
    )


def train_one_encoder_pass(
    rr_train,
    hrv_train,
    y_train,
    rr_val,
    hrv_val,
    y_val,
    args,
    device,
    seed,
):
    seed_everything(seed)

    model = FusionClassifier().to(device)

    weights = balanced_class_weights(
        y_train
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
    )

    train_ds = RRHRVDataset(
        rr_train,
        hrv_train,
        y_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_score = -np.inf
    best_epoch = 1
    best_state = None
    patience_left = args.encoder_patience

    history = []

    for epoch in range(1, args.encoder_max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb_rr, xb_hrv, yb in train_loader:
            xb_rr = xb_rr.to(device)
            xb_hrv = xb_hrv.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            logits, _ = model(
                xb_rr,
                xb_hrv,
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()
            opt.step()

            train_loss_sum += (
                float(loss.item()) * len(yb)
            )
            train_n += len(yb)

        val_probs, _ = predict_classifier(
            model,
            rr_val,
            hrv_val,
            device=device,
        )
        val_f1 = macro_f1_np(
            y_val,
            val_probs,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_loss_sum / max(1, train_n)
                ),
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 > best_score + 1e-6:
            best_score = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_left = args.encoder_patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    if best_state is None:
        best_state = copy.deepcopy(
            model.state_dict()
        )

    return {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_score),
        "best_state": best_state,
        "history": pd.DataFrame(history),
    }


def retrain_encoder_exact_epochs(
    rr,
    hrv,
    y,
    args,
    device,
    seed,
    epochs,
):
    seed_everything(seed)

    model = FusionClassifier().to(device)

    weights = balanced_class_weights(y).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
    )

    ds = RRHRVDataset(
        rr,
        hrv,
        y,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    for _ in range(int(epochs)):
        model.train()

        for xb_rr, xb_hrv, yb in loader:
            xb_rr = xb_rr.to(device)
            xb_hrv = xb_hrv.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            logits, _ = model(
                xb_rr,
                xb_hrv,
            )

            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    return model


# ---------------------------------------------------------------------------
# Source inner subject split
# ---------------------------------------------------------------------------

def split_source_subjects(
    source_subjects: Sequence[int],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    subjects = sorted(
        [int(s) for s in source_subjects]
    )

    if len(subjects) < 3:
        raise ValueError(
            "Need at least 3 source subjects "
            "for inner subject validation."
        )

    rng = np.random.default_rng(seed)
    perm = np.array(subjects, dtype=int)
    rng.shuffle(perm)

    n_val = max(
        1,
        int(round(
            len(subjects) * val_fraction
        )),
    )
    n_val = min(
        n_val,
        len(subjects) - 2,
    )

    val = sorted(
        perm[:n_val].tolist()
    )
    train = sorted(
        perm[n_val:].tolist()
    )

    return train, val


# ---------------------------------------------------------------------------
# Contrastive Gated Residual Temporal Attentive Neural Process
# ---------------------------------------------------------------------------

class ContrastiveGatedResidualTemporalAttentiveNP(nn.Module):
    """
    V1.6: query-contrastive, source-anchored longitudinal personalization.

    What changes relative to V1.5
    -----------------------------
    1) The gate no longer receives context count. C2/C4/C6 cannot directly
       increase adaptation merely because the model knows there are more
       context samples.

    2) Residual DIRECTION is no longer a free MLP output. It is determined by
       query-specific contrastive evidence:

           evidence = cos(q, A_stress) - cos(q, A_nonstress)

       where A_stress / A_nonstress are time-aware, query-specific attended
       personal-history prototypes built from normalized embeddings.

       Thus:
         evidence > 0 -> residual may move toward stress
         evidence < 0 -> residual may move toward non-stress
         evidence ~=0 -> residual must remain small

    3) The neural network learns only residual MAGNITUDE (0..1), while the
       contrast determines direction. This blocks the V1.5 failure mode where
       a subject-level bias could push nearly every query in the same direction.

    4) A residual regularizer penalizes sustained saturation near residual_max.

    Preserved from V1.5
    -------------------
    - frozen source Global Fusion anchor;
    - hard log-odds bound;
    - normalized personalization embeddings;
    - chronological source episodes;
    - prior predictive loss;
    - posterior auxiliary loss + KL;
    - temporal attention / recency;
    - target-side zero-gradient inference.
    """

    def __init__(
        self,
        embedding_dim=64,
        hidden_dim=96,
        latent_dim=32,
        attention_dim=64,
        context_aux_weight=0.10,
        posterior_weight=0.50,
        alpha_max=0.50,
        residual_max=2.0,
        gate_init_fraction=0.10,
        gate_reg_weight=0.01,
        residual_reg_weight=0.02,
        contrast_scale=2.0,
        residual_init_fraction=0.02,
        ablation_id="A6",
    ):
        super().__init__()

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.attention_dim = int(attention_dim)

        self.context_aux_weight = float(context_aux_weight)
        self.posterior_weight = float(posterior_weight)
        self.alpha_max = float(alpha_max)
        self.residual_max = float(residual_max)
        self.gate_reg_weight = float(gate_reg_weight)
        self.residual_reg_weight = float(residual_reg_weight)
        self.contrast_scale = float(contrast_scale)
        self.ablation_id = str(ablation_id)
        self.ablation_spec = get_ablation_spec(self.ablation_id)

        if not (0.0 < self.alpha_max <= 1.0):
            raise ValueError("alpha_max must be in (0,1].")
        if self.residual_max <= 0:
            raise ValueError("residual_max must be > 0.")
        if self.contrast_scale <= 0:
            raise ValueError("contrast_scale must be > 0.")
        if not (0.0 < gate_init_fraction < 1.0):
            raise ValueError("gate_init_fraction must be in (0,1).")
        if not (0.0 < residual_init_fraction < 1.0):
            raise ValueError("residual_init_fraction must be in (0,1).")

        # normalized embedding + one-hot label + 2 temporal features.
        pair_dim = self.embedding_dim + 2 + 2

        self.context_value_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.global_pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.key_encoder = nn.Linear(
            hidden_dim,
            attention_dim,
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(
                embedding_dim,
                attention_dim,
            ),
            nn.LayerNorm(
                attention_dim
            ),
        )

        self.raw_recency_scale = nn.Parameter(
            torch.tensor(-1.0)
        )

        self.latent_stats = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                latent_dim * 2,
            ),
        )

        # Geometry:
        # cos(q,p0_mean), cos(q,p1_mean),
        # d0=(1-cos0)/2, d1=(1-cos1)/2.
        # Query-specific temporal contrast is supplied separately.
        base_geometry_dim = 4

        # Magnitude network: it may decide HOW MUCH to adapt, but not direction.
        magnitude_in = (
            embedding_dim
            + hidden_dim * 2
            + latent_dim
            + base_geometry_dim
            + 2  # signed contrast + |contrast|
        )

        self.residual_body = nn.Sequential(
            nn.Linear(
                magnitude_in,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_out = nn.Linear(
            hidden_dim,
            1,
        )

        # Small initial residual magnitude instead of an exactly-zero sigmoid.
        nn.init.zeros_(
            self.residual_out.weight
        )
        if self.ablation_id == "A1":
            # A direct binary log-odds head should start at an uninformative
            # 0.5 prediction, not inherit the bounded-magnitude bias.
            residual_bias = 0.0
        elif self.ablation_id == "A3":
            # A free signed residual must start exactly at the Global anchor;
            # a positive or negative bias would create an arbitrary class tilt.
            residual_bias = 0.0
        else:
            residual_bias = math.log(
                residual_init_fraction
                / (1.0 - residual_init_fraction)
            )
        nn.init.constant_(
            self.residual_out.bias,
            residual_bias,
        )

        # IMPORTANT V1.6:
        # gate does NOT receive context count.
        #
        # Inputs:
        #   normalized query
        #   attended hidden context
        #   bounded mean-prototype geometry
        #   signed contrast + |contrast|
        #   mean context age only
        gate_in = (
            embedding_dim
            + hidden_dim * 2
            + base_geometry_dim
            + 2
            + 1
        )

        self.gate_body = nn.Sequential(
            nn.Linear(
                gate_in,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate_out = nn.Linear(
            hidden_dim,
            1,
        )

        nn.init.zeros_(
            self.gate_out.weight
        )
        gate_bias = math.log(
            gate_init_fraction
            / (1.0 - gate_init_fraction)
        )
        nn.init.constant_(
            self.gate_out.bias,
            gate_bias,
        )

        self.context_aux_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 2
                + base_geometry_dim
                + 2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    @staticmethod
    def normalize_embedding(
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize(
            x,
            p=2,
            dim=-1,
            eps=1e-8,
        )

    @staticmethod
    def temporal_features(
        age_days: torch.Tensor,
    ) -> torch.Tensor:
        age = torch.clamp(
            age_days.float(),
            min=0.0,
            max=365.0,
        )

        return torch.stack(
            [
                torch.log1p(age),
                torch.exp(-age / 7.0),
            ],
            dim=-1,
        )

    def pair_input(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        age_days: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.normalize_embedding(
            x
        )

        onehot = F.one_hot(
            y.long(),
            num_classes=2,
        ).float()

        tf = self.temporal_features(
            age_days
        )

        return torch.cat(
            [
                x_norm,
                onehot,
                tf,
            ],
            dim=-1,
        )

    def global_aggregate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        age_days: torch.Tensor,
    ) -> torch.Tensor:
        pair = self.pair_input(
            x,
            y,
            age_days,
        )

        r = self.global_pair_encoder(
            pair
        )

        reps = []

        for c in [0, 1]:
            mask = (
                y == c
            )

            if mask.any():
                rep = r[
                    mask
                ].mean(
                    dim=0,
                    keepdim=True,
                )
            else:
                rep = torch.zeros(
                    (
                        1,
                        self.hidden_dim,
                    ),
                    dtype=r.dtype,
                    device=r.device,
                )

            reps.append(
                rep
            )

        return torch.cat(
            reps,
            dim=-1,
        )

    def latent_distribution(
        self,
        global_repr: torch.Tensor,
    ):
        stats = self.latent_stats(
            global_repr
        )

        mu, raw_scale = torch.chunk(
            stats,
            2,
            dim=-1,
        )

        std = (
            0.05
            + 0.95
            * F.softplus(
                raw_scale
            )
        )

        return mu, std

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        std: torch.Tensor,
    ):
        return (
            mu
            + torch.randn_like(
                std
            )
            * std
        )

    @staticmethod
    def kl_diag_gaussians(
        mu_q,
        std_q,
        mu_p,
        std_p,
    ):
        var_q = std_q.pow(2)
        var_p = std_p.pow(2)

        kl = (
            torch.log(
                std_p / std_q
            )
            + (
                var_q
                + (
                    mu_q - mu_p
                ).pow(2)
            )
            / (
                2.0 * var_p
            )
            - 0.5
        )

        return kl.sum(
            dim=-1
        ).mean()

    def attend(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        context_age_days: torch.Tensor,
        query_x: torch.Tensor,
    ):
        pair = self.pair_input(
            context_x,
            context_y,
            context_age_days,
        )

        values = self.context_value_encoder(
            pair
        )
        keys = self.key_encoder(
            values
        )

        context_norm = self.normalize_embedding(
            context_x
        )
        query_norm = self.normalize_embedding(
            query_x
        )
        queries = self.query_encoder(
            query_norm
        )

        scale = math.sqrt(
            float(
                self.attention_dim
            )
        )

        raw_scores = (
            queries
            @ keys.transpose(
                0,
                1,
            )
        ) / scale

        recency_scale = F.softplus(
            self.raw_recency_scale
        )

        recency_penalty = (
            recency_scale
            * torch.log1p(
                torch.clamp(
                    context_age_days,
                    min=0.0,
                    max=365.0,
                )
            )
        )

        raw_scores = (
            raw_scores
            - recency_penalty.unsqueeze(
                0
            )
        )

        hidden_reps = []
        attended_embedding_reps = []
        entropies = []

        for c in [0, 1]:
            mask = (
                context_y == c
            )

            if not mask.any():
                hidden_reps.append(
                    torch.zeros(
                        (
                            query_x.shape[0],
                            self.hidden_dim,
                        ),
                        dtype=query_x.dtype,
                        device=query_x.device,
                    )
                )
                attended_embedding_reps.append(
                    torch.zeros(
                        (
                            query_x.shape[0],
                            self.embedding_dim,
                        ),
                        dtype=query_x.dtype,
                        device=query_x.device,
                    )
                )
                entropies.append(
                    torch.zeros(
                        query_x.shape[0],
                        dtype=query_x.dtype,
                        device=query_x.device,
                    )
                )
                continue

            class_scores = raw_scores[
                :,
                mask,
            ]
            weights = torch.softmax(
                class_scores,
                dim=-1,
            )

            class_values = values[
                mask
            ]
            hidden_rep = (
                weights
                @ class_values
            )
            hidden_reps.append(
                hidden_rep
            )

            # Query-specific temporal personal prototype in normalized
            # embedding space.
            class_embeddings = context_norm[
                mask
            ]
            attended_emb = (
                weights
                @ class_embeddings
            )
            attended_emb = self.normalize_embedding(
                attended_emb
            )
            attended_embedding_reps.append(
                attended_emb
            )

            eps = 1e-8
            entropies.append(
                -(
                    weights
                    * torch.log(
                        weights.clamp_min(
                            eps
                        )
                    )
                ).sum(
                    dim=-1
                )
            )

        return {
            "attended": torch.cat(
                hidden_reps,
                dim=-1,
            ),
            "attended_embedding_class0": (
                attended_embedding_reps[
                    0
                ]
            ),
            "attended_embedding_class1": (
                attended_embedding_reps[
                    1
                ]
            ),
            "attention_entropy_class0": entropies[
                0
            ],
            "attention_entropy_class1": entropies[
                1
            ],
            "recency_scale": recency_scale,
        }

    def similarity_features(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        context_norm = self.normalize_embedding(
            context_x
        )
        query_norm = self.normalize_embedding(
            query_x
        )

        cosines = []
        bounded_distances = []

        for c in [0, 1]:
            mask = (
                context_y == c
            )

            if mask.any():
                proto = context_norm[
                    mask
                ].mean(
                    dim=0,
                    keepdim=True,
                )
                proto = self.normalize_embedding(
                    proto
                )
            else:
                proto = torch.zeros(
                    (
                        1,
                        context_norm.shape[
                            -1
                        ],
                    ),
                    dtype=context_norm.dtype,
                    device=context_norm.device,
                )

            cosine = (
                query_norm
                * proto
            ).sum(
                dim=-1,
                keepdim=True,
            )

            d_norm = (
                1.0
                - cosine
            ) / 2.0

            cosines.append(
                cosine
            )
            bounded_distances.append(
                d_norm
            )

        return torch.cat(
            [
                cosines[
                    0
                ],
                cosines[
                    1
                ],
                bounded_distances[
                    0
                ],
                bounded_distances[
                    1
                ],
            ],
            dim=-1,
        )

    def contrastive_evidence(
        self,
        query_x: torch.Tensor,
        attended_embedding_class0: torch.Tensor,
        attended_embedding_class1: torch.Tensor,
    ):
        q = self.normalize_embedding(
            query_x
        )

        p0 = self.normalize_embedding(
            attended_embedding_class0
        )
        p1 = self.normalize_embedding(
            attended_embedding_class1
        )

        cos0 = (
            q
            * p0
        ).sum(
            dim=-1,
            keepdim=True,
        )
        cos1 = (
            q
            * p1
        ).sum(
            dim=-1,
            keepdim=True,
        )

        signed_contrast = (
            cos1
            - cos0
        )

        direction = torch.tanh(
            self.contrast_scale
            * signed_contrast
        )

        return {
            "cos_attended_nonstress": cos0,
            "cos_attended_stress": cos1,
            "signed_contrast": signed_contrast,
            "abs_contrast": signed_contrast.abs(),
            "direction": direction,
        }

    def context_metadata(
        self,
        context_age_days: torch.Tensor,
        n_query: int,
    ) -> torch.Tensor:
        """
        V1.6 deliberately excludes context count.

        Only mean age remains, because longitudinal recency is a property of
        the history content rather than a shortcut for C2/C4/C6 identity.
        """
        mean_age = torch.mean(
            torch.clamp(
                context_age_days,
                min=0.0,
                max=365.0,
            )
        ) / 30.0

        meta = mean_age.reshape(
            1,
            1,
        ).to(
            dtype=context_age_days.dtype,
            device=context_age_days.device,
        )

        return meta.expand(
            n_query,
            -1,
        )

    def gate(
        self,
        query_x: torch.Tensor,
        attended_repr: torch.Tensor,
        similarity_features: torch.Tensor,
        contrast: dict,
        context_age_days: torch.Tensor,
    ) -> torch.Tensor:
        query_norm = self.normalize_embedding(
            query_x
        )

        meta = self.context_metadata(
            context_age_days,
            query_x.shape[
                0
            ],
        )

        gate_features = torch.cat(
            [
                query_norm,
                attended_repr,
                similarity_features,
                contrast[
                    "signed_contrast"
                ],
                contrast[
                    "abs_contrast"
                ],
                meta,
            ],
            dim=-1,
        )

        gate_hidden = self.gate_body(
            gate_features
        )

        gate_fraction = torch.sigmoid(
            self.gate_out(
                gate_hidden
            )
        )

        return (
            self.alpha_max
            * gate_fraction
        )

    def contrastive_residual_logodds(
        self,
        query_x: torch.Tensor,
        attended_repr: torch.Tensor,
        z: torch.Tensor,
        similarity_features: torch.Tensor,
        contrast: dict,
    ):
        """
        Direction is fixed by query-to-personal-history contrast.
        The network learns magnitude only.
        """
        query_norm = self.normalize_embedding(
            query_x
        )

        z_expand = z.expand(
            query_x.shape[
                0
            ],
            -1,
        )

        residual_features = torch.cat(
            [
                query_norm,
                attended_repr,
                z_expand,
                similarity_features,
                contrast[
                    "signed_contrast"
                ],
                contrast[
                    "abs_contrast"
                ],
            ],
            dim=-1,
        )

        hidden = self.residual_body(
            residual_features
        )

        raw_residual = self.residual_out(
            hidden
        )

        if self.ablation_id == "A1":
            # Direct NP comparator: the network predicts the binary log-odds
            # itself. The frozen Global anchor, gate, hard bound and imposed
            # contrast direction are all absent.
            bounded_delta = raw_residual
            magnitude_fraction = (
                raw_residual.abs()
                / self.residual_max
            )
        elif self.ablation_id == "A2":
            # Keep the contrast direction but remove the hard magnitude cap.
            magnitude_fraction = F.softplus(
                raw_residual
            )
            bounded_delta = (
                self.residual_max
                * contrast["direction"]
                * magnitude_fraction
            )
        elif self.ablation_id == "A3":
            # Keep the same symmetric hard bound while allowing the residual
            # sign to be learned freely.
            signed_fraction = torch.tanh(
                raw_residual
            )
            magnitude_fraction = signed_fraction.abs()
            bounded_delta = (
                self.residual_max
                * signed_fraction
            )
        else:
            magnitude_fraction = torch.sigmoid(
                raw_residual
            )
            bounded_delta = (
                self.residual_max
                * contrast["direction"]
                * magnitude_fraction
            )

        return {
            "bounded_residual_logodds": bounded_delta,
            "residual_magnitude_fraction": magnitude_fraction,
            "contrast_direction": contrast[
                "direction"
            ],
        }

    def apply_bounded_residual(
        self,
        base_log_probs: torch.Tensor,
        bounded_delta: torch.Tensor,
        gate: torch.Tensor,
    ):
        if self.ablation_id == "A1":
            applied_shift = bounded_delta
        else:
            applied_shift = (
                gate
                * bounded_delta
            )

        adjustment = torch.cat(
            [
                -0.5
                * applied_shift,
                0.5
                * applied_shift,
            ],
            dim=-1,
        )

        if self.ablation_id == "A1":
            final_logits = adjustment
        else:
            final_logits = (
                base_log_probs
                + adjustment
            )

        return {
            "final_logits": final_logits,
            "bounded_residual_logodds": bounded_delta,
            "applied_logodds_shift": applied_shift,
        }

    def context_path(
        self,
        context_x,
        context_y,
        context_age_days,
        query_x,
    ):
        attn = self.attend(
            context_x,
            context_y,
            context_age_days,
            query_x,
        )

        sim = self.similarity_features(
            context_x,
            context_y,
            query_x,
        )

        contrast = self.contrastive_evidence(
            query_x,
            attn[
                "attended_embedding_class0"
            ],
            attn[
                "attended_embedding_class1"
            ],
        )

        alpha = self.gate(
            query_x,
            attn[
                "attended"
            ],
            sim,
            contrast,
            context_age_days,
        )

        return {
            "attended": attn[
                "attended"
            ],
            "similarity": sim,
            "contrast": contrast,
            "gate": alpha,
            "attention_entropy_class0": attn[
                "attention_entropy_class0"
            ],
            "attention_entropy_class1": attn[
                "attention_entropy_class1"
            ],
            "recency_scale": attn[
                "recency_scale"
            ],
        }

    def training_episode(
        self,
        context_x,
        context_y,
        context_age_days,
        query_x,
        query_y,
        base_log_probs,
        beta_kl: float,
    ):
        # ----- Deployment-matched prior path -----
        r_context_global = self.global_aggregate(
            context_x,
            context_y,
            context_age_days,
        )
        mu_p, std_p = self.latent_distribution(
            r_context_global
        )

        path = self.context_path(
            context_x,
            context_y,
            context_age_days,
            query_x,
        )

        prior_res = self.contrastive_residual_logodds(
            query_x,
            path[
                "attended"
            ],
            mu_p,
            path[
                "similarity"
            ],
            path[
                "contrast"
            ],
        )

        prior_out = self.apply_bounded_residual(
            base_log_probs,
            prior_res[
                "bounded_residual_logodds"
            ],
            path[
                "gate"
            ],
        )

        prior_ce = F.cross_entropy(
            prior_out[
                "final_logits"
            ],
            query_y,
        )

        # ----- Source-only posterior auxiliary path -----
        all_x = torch.cat(
            [
                context_x,
                query_x,
            ],
            dim=0,
        )
        all_y = torch.cat(
            [
                context_y,
                query_y,
            ],
            dim=0,
        )
        all_age = torch.cat(
            [
                context_age_days,
                torch.zeros(
                    query_x.shape[
                        0
                    ],
                    dtype=context_age_days.dtype,
                    device=context_age_days.device,
                ),
            ],
            dim=0,
        )

        r_target_global = self.global_aggregate(
            all_x,
            all_y,
            all_age,
        )
        mu_q, std_q = self.latent_distribution(
            r_target_global
        )

        z_q = self.reparameterize(
            mu_q,
            std_q,
        )

        posterior_res = self.contrastive_residual_logodds(
            query_x,
            path[
                "attended"
            ],
            z_q,
            path[
                "similarity"
            ],
            path[
                "contrast"
            ],
        )

        posterior_out = self.apply_bounded_residual(
            base_log_probs,
            posterior_res[
                "bounded_residual_logodds"
            ],
            path[
                "gate"
            ],
        )

        posterior_ce = F.cross_entropy(
            posterior_out[
                "final_logits"
            ],
            query_y,
        )

        kl = self.kl_diag_gaussians(
            mu_q,
            std_q,
            mu_p,
            std_p,
        )

        aux_logits = self.context_aux_head(
            torch.cat(
                [
                    path[
                        "attended"
                    ],
                    path[
                        "similarity"
                    ],
                    path[
                        "contrast"
                    ][
                        "signed_contrast"
                    ],
                    path[
                        "contrast"
                    ][
                        "abs_contrast"
                    ],
                ],
                dim=-1,
            )
        )

        aux_ce = F.cross_entropy(
            aux_logits,
            query_y,
        )

        normalized_gate = (
            path[
                "gate"
            ]
            / self.alpha_max
        )
        gate_reg = normalized_gate.pow(
            2
        ).mean()

        # Directly penalize residual saturation BEFORE the gate. This is what
        # V1.5 lacked: V1.5 could sit near +/-residual_max and let the gate
        # perform all stabilization.
        normalized_residual = (
            prior_res[
                "bounded_residual_logodds"
            ]
            / self.residual_max
        )
        residual_reg = normalized_residual.pow(
            2
        ).mean()

        if self.ablation_id == "A1":
            gate_reg = torch.zeros_like(gate_reg)
            residual_reg = torch.zeros_like(residual_reg)

        loss = (
            prior_ce
            + self.posterior_weight
            * posterior_ce
            + beta_kl
            * kl
            + self.context_aux_weight
            * aux_ce
            + self.gate_reg_weight
            * gate_reg
            + self.residual_reg_weight
            * residual_reg
        )

        return {
            "loss": loss,
            "prior_ce": prior_ce.detach(),
            "posterior_ce": posterior_ce.detach(),
            "kl": kl.detach(),
            "aux_ce": aux_ce.detach(),
            "gate_reg": gate_reg.detach(),
            "residual_reg": residual_reg.detach(),
            "mean_gate": path[
                "gate"
            ].mean().detach(),
            "mean_abs_applied_shift": prior_out[
                "applied_logodds_shift"
            ].abs().mean().detach(),
            "mean_abs_bounded_residual": prior_res[
                "bounded_residual_logodds"
            ].abs().mean().detach(),
            "mean_abs_contrast": path[
                "contrast"
            ][
                "abs_contrast"
            ].mean().detach(),
            "mean_residual_magnitude_fraction": prior_res[
                "residual_magnitude_fraction"
            ].mean().detach(),
        }

    @torch.no_grad()
    def predict_mean(
        self,
        context_x,
        context_y,
        context_age_days,
        query_x,
        base_log_probs,
    ):
        self.eval()

        global_repr = self.global_aggregate(
            context_x,
            context_y,
            context_age_days,
        )
        mu_p, _ = self.latent_distribution(
            global_repr
        )

        path = self.context_path(
            context_x,
            context_y,
            context_age_days,
            query_x,
        )

        residual = self.contrastive_residual_logodds(
            query_x,
            path[
                "attended"
            ],
            mu_p,
            path[
                "similarity"
            ],
            path[
                "contrast"
            ],
        )

        adapted = self.apply_bounded_residual(
            base_log_probs,
            residual[
                "bounded_residual_logodds"
            ],
            path[
                "gate"
            ],
        )

        prob = torch.softmax(
            adapted[
                "final_logits"
            ],
            dim=-1,
        )

        return {
            "prob": prob,
            "gate": path[
                "gate"
            ],
            "bounded_residual_logodds": residual[
                "bounded_residual_logodds"
            ],
            "applied_logodds_shift": adapted[
                "applied_logodds_shift"
            ],
            "residual_magnitude_fraction": residual[
                "residual_magnitude_fraction"
            ],
            "contrast_direction": residual[
                "contrast_direction"
            ],
            "signed_contrast": path[
                "contrast"
            ][
                "signed_contrast"
            ],
            "abs_contrast": path[
                "contrast"
            ][
                "abs_contrast"
            ],
            "cos_attended_nonstress": path[
                "contrast"
            ][
                "cos_attended_nonstress"
            ],
            "cos_attended_stress": path[
                "contrast"
            ][
                "cos_attended_stress"
            ],
            "attention_entropy_class0": path[
                "attention_entropy_class0"
            ],
            "attention_entropy_class1": path[
                "attention_entropy_class1"
            ],
            "recency_scale": path[
                "recency_scale"
            ],
        }

    @torch.no_grad()
    def predict_mc(
        self,
        context_x,
        context_y,
        context_age_days,
        query_x,
        base_log_probs,
        mc_samples=30,
        adaptation_gamma=1.0,
    ):
        self.eval()

        global_repr = self.global_aggregate(
            context_x,
            context_y,
            context_age_days,
        )
        mu_p, std_p = self.latent_distribution(
            global_repr
        )

        path = self.context_path(
            context_x,
            context_y,
            context_age_days,
            query_x,
        )

        probs = []
        shifts = []

        for _ in range(
            int(
                mc_samples
            )
        ):
            z = self.reparameterize(
                mu_p,
                std_p,
            )

            residual = self.contrastive_residual_logodds(
                query_x,
                path[
                    "attended"
                ],
                z,
                path[
                    "similarity"
                ],
                path[
                    "contrast"
                ],
            )

            adapted = self.apply_bounded_residual(
                base_log_probs,
                residual[
                    "bounded_residual_logodds"
                ],
                path[
                    "gate"
                ],
            )

            uncalibrated_shift = adapted[
                "applied_logodds_shift"
            ]

            if self.ablation_id == "A1":
                calibrated_shift = uncalibrated_shift
            elif self.ablation_spec["hard_bound_enabled"]:
                hard_bound = (
                    self.alpha_max
                    * self.residual_max
                )
                calibrated_shift = torch.clamp(
                    float(adaptation_gamma)
                    * uncalibrated_shift,
                    min=-hard_bound,
                    max=hard_bound,
                )
            else:
                calibrated_shift = (
                    float(adaptation_gamma)
                    * uncalibrated_shift
                )

            calibrated_adjustment = torch.cat(
                [
                    -0.5 * calibrated_shift,
                    0.5 * calibrated_shift,
                ],
                dim=-1,
            )

            if self.ablation_id == "A1":
                calibrated_logits = calibrated_adjustment
            else:
                calibrated_logits = (
                    base_log_probs
                    + calibrated_adjustment
                )

            probs.append(
                torch.softmax(
                    calibrated_logits,
                    dim=-1,
                )
            )
            shifts.append(
                calibrated_shift
            )

        probs = torch.stack(
            probs,
            dim=0,
        )
        shifts = torch.stack(
            shifts,
            dim=0,
        )

        mean_prob = probs.mean(
            dim=0
        )

        eps = 1e-8
        predictive_entropy = -(
            mean_prob
            * torch.log(
                mean_prob.clamp_min(
                    eps
                )
            )
        ).sum(
            dim=-1
        )

        sample_entropy = -(
            probs
            * torch.log(
                probs.clamp_min(
                    eps
                )
            )
        ).sum(
            dim=-1
        )

        expected_entropy = sample_entropy.mean(
            dim=0
        )

        mutual_information = (
            predictive_entropy
            - expected_entropy
        )

        stress_var = probs[
            :,
            :,
            1,
        ].var(
            dim=0,
            unbiased=False,
        )

        return {
            "mean_prob": mean_prob,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_information,
            "stress_probability_variance": stress_var,
            "mc_applied_shift_std": shifts.squeeze(
                -1
            ).std(
                dim=0,
                unbiased=False,
            ),
            "attention_entropy_class0": path[
                "attention_entropy_class0"
            ],
            "attention_entropy_class1": path[
                "attention_entropy_class1"
            ],
            "recency_scale": path[
                "recency_scale"
            ],
        }


# ---------------------------------------------------------------------------
# Chronological source meta-episode sampling
# ---------------------------------------------------------------------------

def build_chronological_episode_catalog(
    subject_ids: np.ndarray,
    y: np.ndarray,
    anchor_ns: np.ndarray,
    allowed_subjects: Sequence[int],
    k_values: Sequence[int],
):
    """
    Build source meta-training episodes that mirror target deployment.

    Each episode contains:
      - one FUTURE query sample q_t;
      - a balanced context drawn only from the same subject's history;
      - most recent k non-stress + k stress samples before q_t.

    Catalog layout:
      catalog[k][query_class][subject] -> list[episode]

    Sampling later is balanced by:
      context size -> query class -> subject -> query episode.

    This prevents source subjects with many observations from dominating and,
    crucially, avoids the random support/query mismatch of v1.2.
    """
    catalog = {
        int(k): {
            0: {},
            1: {},
        }
        for k in k_values
    }

    for s in sorted(
        [int(v) for v in allowed_subjects]
    ):
        idx_s = np.flatnonzero(
            subject_ids == s
        )

        if len(idx_s) == 0:
            continue

        order = idx_s[
            np.argsort(
                anchor_ns[idx_s],
                kind="stable",
            )
        ]

        past = {
            0: [],
            1: [],
        }

        # Samples sharing an anchor belong to one acquisition/session block.
        # Evaluate every query in the block before revealing any label from
        # that block.  This preserves the strict "earlier timestamp" contract
        # for datasets with paired baseline/stress samples per session.
        ordered_times = anchor_ns[order]
        for current_time in np.unique(ordered_times):
            block = order[ordered_times == current_time]

            for qidx in block:
                qidx = int(qidx)
                qclass = int(y[qidx])

                for k in k_values:
                    k = int(k)

                    if (
                        len(past[0]) >= k
                        and len(past[1]) >= k
                    ):
                        context_idx = (
                            past[0][-k:]
                            + past[1][-k:]
                        )

                        # Chronological ordering of the context itself.
                        context_idx = sorted(
                            context_idx,
                            key=lambda j: int(
                                anchor_ns[int(j)]
                            ),
                        )

                        # Hard temporal audit.
                        if any(
                            int(anchor_ns[int(ci)])
                            >= int(anchor_ns[qidx])
                            for ci in context_idx
                        ):
                            raise AssertionError(
                                "Source chronological episode contains "
                                "context that is not strictly earlier than query."
                            )

                        ep = {
                            "subject": s,
                            "k_per_class": k,
                            "context_idx": np.asarray(
                                context_idx,
                                dtype=int,
                            ),
                            "query_idx": np.asarray(
                                [qidx],
                                dtype=int,
                            ),
                            "query_class": qclass,
                        }

                        catalog[k][qclass].setdefault(
                            s,
                            [],
                        ).append(ep)

            # Reveal the whole block only after all of its queries are built.
            for qidx in block:
                qidx = int(qidx)
                past[int(y[qidx])].append(qidx)

    return catalog


def chronological_catalog_subjects(
    catalog,
) -> List[int]:
    subjects = set()

    for by_class in catalog.values():
        for by_subject in by_class.values():
            subjects.update(
                int(s)
                for s in by_subject.keys()
            )

    return sorted(subjects)


def chronological_catalog_size(
    catalog,
) -> int:
    total = 0

    for by_class in catalog.values():
        for by_subject in by_class.values():
            for episodes in by_subject.values():
                total += len(episodes)

    return int(total)


def sample_chronological_episode(
    rng: np.random.Generator,
    catalog,
    k_values: Sequence[int],
):
    """
    Balanced sampling:
      1) choose a context size uniformly among feasible k;
      2) choose future-query class uniformly among feasible classes;
      3) choose subject uniformly;
      4) choose one chronological episode uniformly within that subject.
    """
    feasible_k = []

    for k in k_values:
        k = int(k)
        if any(
            len(catalog[k][c]) > 0
            for c in [0, 1]
        ):
            feasible_k.append(k)

    if not feasible_k:
        return None

    k = int(
        rng.choice(
            feasible_k
        )
    )

    feasible_classes = [
        c
        for c in [0, 1]
        if len(catalog[k][c]) > 0
    ]

    if not feasible_classes:
        return None

    query_class = int(
        rng.choice(
            feasible_classes
        )
    )

    subjects = sorted(
        catalog[k][query_class].keys()
    )

    if not subjects:
        return None

    subject = int(
        rng.choice(
            subjects
        )
    )

    episodes = catalog[k][
        query_class
    ][subject]

    ep = episodes[
        int(
            rng.integers(
                0,
                len(episodes),
            )
        )
    ]

    return ep


@torch.no_grad()
def context_age_days_from_indices(
    anchor_ns: np.ndarray,
    context_idx: np.ndarray,
    query_idx: np.ndarray,
) -> np.ndarray:
    q_ns = int(
        anchor_ns[
            int(query_idx[0])
        ]
    )

    ages = (
        q_ns
        - anchor_ns[
            np.asarray(
                context_idx,
                dtype=int,
            )
        ]
    ) / (
        86400.0 * 1e9
    )

    if np.any(
        ages <= 0
    ):
        raise AssertionError(
            "Context age must be strictly positive."
        )

    return ages.astype(
        np.float32
    )


@torch.no_grad()
def probs_to_log_probs_tensor(
    probs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    p = torch.as_tensor(
        probs,
        dtype=torch.float32,
        device=device,
    )

    return torch.log(
        p.clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
    )


@torch.no_grad()
def np_predict_latent_mean(
    model: ContrastiveGatedResidualTemporalAttentiveNP,
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    context_age_days: torch.Tensor,
    query_x: torch.Tensor,
    base_log_probs: torch.Tensor,
):
    return model.predict_mean(
        context_x,
        context_y,
        context_age_days,
        query_x,
        base_log_probs,
    )["prob"]


def eval_np_source_episodes(
    model,
    embeddings,
    global_probs,
    y,
    anchor_ns,
    catalog,
    k_values,
    n_episodes,
    device,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    y_true_all = []
    p_all = []

    for _ in range(
        int(
            n_episodes
        )
    ):
        ep = sample_chronological_episode(
            rng,
            catalog,
            k_values,
        )

        if ep is None:
            continue

        ci = ep[
            "context_idx"
        ]
        qi = ep[
            "query_idx"
        ]

        context_age_np = (
            context_age_days_from_indices(
                anchor_ns,
                ci,
                qi,
            )
        )

        cx = torch.as_tensor(
            embeddings[
                ci
            ],
            dtype=torch.float32,
            device=device,
        )
        cy = torch.as_tensor(
            y[
                ci
            ],
            dtype=torch.long,
            device=device,
        )
        ca = torch.as_tensor(
            context_age_np,
            dtype=torch.float32,
            device=device,
        )
        qx = torch.as_tensor(
            embeddings[
                qi
            ],
            dtype=torch.float32,
            device=device,
        )

        base_log_probs = (
            probs_to_log_probs_tensor(
                global_probs[
                    qi
                ],
                device,
            )
        )

        p = (
            np_predict_latent_mean(
                model,
                cx,
                cy,
                ca,
                qx,
                base_log_probs,
            )
            .cpu()
            .numpy()
        )

        y_true_all.extend(
            y[
                qi
            ].tolist()
        )
        p_all.extend(
            p[
                :,
                1,
            ].tolist()
        )

    if not y_true_all:
        return {
            "nll": np.inf,
            "macro_f1": 0.0,
        }

    y_true = np.asarray(
        y_true_all,
        dtype=int,
    )
    p1 = np.asarray(
        p_all,
        dtype=float,
    )

    probs = np.column_stack(
        [
            1.0 - p1,
            p1,
        ]
    )

    pred = (
        p1 >= 0.5
    ).astype(
        int
    )

    nll = float(
        log_loss(
            y_true,
            probs,
            labels=[
                0,
                1,
            ],
        )
    )

    macro = float(
        f1_score(
            y_true,
            pred,
            average="macro",
            labels=[
                0,
                1,
            ],
            zero_division=0,
        )
    )

    return {
        "nll": nll,
        "macro_f1": macro,
    }



def _calibrated_binary_probability_from_shift(
    base_prob: np.ndarray,
    uncalibrated_shift: float,
    gamma: float,
    hard_bound: float,
) -> np.ndarray:
    """
    Multiply the V1.6 applied log-odds shift by a SOURCE-selected gamma,
    then re-apply the original hard bound.
    """
    base_prob = np.asarray(
        base_prob,
        dtype=float,
    )

    # Safety contract: gamma=0 is an exact, bit-for-bit return to the
    # frozen Global Fusion anchor.  Avoid log/exp round trips here because
    # they can introduce tiny numerical differences even when the shift is 0.
    if float(gamma) == 0.0:
        return base_prob.copy()

    base_prob = np.clip(
        base_prob,
        1e-7,
        1.0 - 1e-7,
    )
    base_prob = (
        base_prob
        / np.sum(
            base_prob
        )
    )

    shift = float(
        np.clip(
            float(gamma)
            * float(
                uncalibrated_shift
            ),
            -float(
                hard_bound
            ),
            float(
                hard_bound
            ),
        )
    )

    logits = (
        np.log(
            base_prob
        )
        + np.asarray(
            [
                -0.5 * shift,
                0.5 * shift,
            ],
            dtype=float,
        )
    )

    logits = (
        logits
        - np.max(
            logits
        )
    )
    exp_logits = np.exp(
        logits
    )

    return (
        exp_logits
        / np.sum(
            exp_logits
        )
    )


@torch.no_grad()
def collect_source_calibration_records(
    model,
    embeddings,
    global_probs,
    y,
    anchor_ns,
    catalog,
    device,
):
    """
    Exhaustively collect deterministic prior-path shifts from the
    SOURCE-ONLY validation catalog.
    """
    rows = []
    model.eval()

    for k in sorted(
        catalog.keys()
    ):
        for query_class in [
            0,
            1,
        ]:
            by_subject = catalog[
                int(k)
            ][
                query_class
            ]

            for subject in sorted(
                by_subject.keys()
            ):
                episodes = by_subject[
                    subject
                ]

                for ep_i, ep in enumerate(
                    episodes
                ):
                    ci = np.asarray(
                        ep[
                            "context_idx"
                        ],
                        dtype=int,
                    )
                    qi = np.asarray(
                        ep[
                            "query_idx"
                        ],
                        dtype=int,
                    )

                    context_age_np = (
                        context_age_days_from_indices(
                            anchor_ns,
                            ci,
                            qi,
                        )
                    )

                    cx = torch.as_tensor(
                        embeddings[
                            ci
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    cy = torch.as_tensor(
                        y[
                            ci
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                    ca = torch.as_tensor(
                        context_age_np,
                        dtype=torch.float32,
                        device=device,
                    )
                    qx = torch.as_tensor(
                        embeddings[
                            qi
                        ],
                        dtype=torch.float32,
                        device=device,
                    )

                    base_prob = np.asarray(
                        global_probs[
                            qi[
                                0
                            ]
                        ],
                        dtype=float,
                    )

                    base_log_probs = (
                        probs_to_log_probs_tensor(
                            base_prob[
                                None,
                                :,
                            ],
                            device,
                        )
                    )

                    out = model.predict_mean(
                        cx,
                        cy,
                        ca,
                        qx,
                        base_log_probs,
                    )

                    rows.append(
                        {
                            "k_per_class": int(
                                k
                            ),
                            "context_total": int(
                                2 * int(
                                    k
                                )
                            ),
                            "query_class": int(
                                query_class
                            ),
                            "subject": int(
                                subject
                            ),
                            "episode_index_within_subject_stratum": int(
                                ep_i
                            ),
                            "query_model_row": int(
                                qi[
                                    0
                                ]
                            ),
                            "y_true": int(
                                y[
                                    qi[
                                        0
                                    ]
                                ]
                            ),
                            "base_p_stress": float(
                                base_prob[
                                    1
                                ]
                            ),
                            "uncalibrated_applied_logodds_shift": float(
                                out[
                                    "applied_logodds_shift"
                                ][
                                    0,
                                    0,
                                ]
                                .detach()
                                .cpu()
                                .item()
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def score_source_calibration_gamma(
    records: pd.DataFrame,
    gamma: float,
    hard_bound: float,
):
    """
    Hierarchical equal-weight score:
      episode -> subject within (k, class)
      -> (k, class) stratum
      -> all available strata.
    """
    if records.empty:
        return {
            "gamma": float(
                gamma
            ),
            "balanced_nll": np.inf,
            "balanced_brier": np.inf,
            "episode_nll": np.inf,
            "episode_brier": np.inf,
            "n_episode": 0,
            "n_subject": 0,
            "n_k_class_strata": 0,
        }

    tmp = records.copy()

    p1_values = []

    for row in tmp.itertuples(
        index=False
    ):
        base_p1 = float(
            row.base_p_stress
        )

        p = _calibrated_binary_probability_from_shift(
            np.asarray(
                [
                    1.0 - base_p1,
                    base_p1,
                ],
                dtype=float,
            ),
            row.uncalibrated_applied_logodds_shift,
            gamma,
            hard_bound,
        )

        p1_values.append(
            float(
                p[
                    1
                ]
            )
        )

    tmp[
        "p_stress_calibrated"
    ] = p1_values

    eps = 1e-7
    y_true = tmp[
        "y_true"
    ].to_numpy(
        dtype=int
    )
    p1 = np.clip(
        tmp[
            "p_stress_calibrated"
        ].to_numpy(
            dtype=float
        ),
        eps,
        1.0 - eps,
    )

    tmp[
        "episode_nll"
    ] = -(
        y_true
        * np.log(
            p1
        )
        + (
            1 - y_true
        )
        * np.log(
            1.0 - p1
        )
    )
    tmp[
        "episode_brier"
    ] = (
        p1
        - y_true
    ) ** 2

    subject_strata = (
        tmp
        .groupby(
            [
                "k_per_class",
                "query_class",
                "subject",
            ],
            sort=True,
            as_index=False,
        )[
            [
                "episode_nll",
                "episode_brier",
            ]
        ]
        .mean()
    )

    k_class = (
        subject_strata
        .groupby(
            [
                "k_per_class",
                "query_class",
            ],
            sort=True,
            as_index=False,
        )[
            [
                "episode_nll",
                "episode_brier",
            ]
        ]
        .mean()
    )

    return {
        "gamma": float(
            gamma
        ),
        "balanced_nll": float(
            k_class[
                "episode_nll"
            ].mean()
        ),
        "balanced_brier": float(
            k_class[
                "episode_brier"
            ].mean()
        ),
        "episode_nll": float(
            tmp[
                "episode_nll"
            ].mean()
        ),
        "episode_brier": float(
            tmp[
                "episode_brier"
            ].mean()
        ),
        "n_episode": int(
            len(
                tmp
            )
        ),
        "n_subject": int(
            tmp[
                "subject"
            ].nunique()
        ),
        "n_k_class_strata": int(
            len(
                k_class
            )
        ),
    }


def fixed_gamma_selection(ablation_id: str) -> dict:
    """Return a schema-compatible record for ablations without calibration."""
    if ablation_id == "A1":
        selected_gamma = np.nan
        reason = "not_applicable_direct_np"
        mode = "not_applicable"
    else:
        selected_gamma = 1.0
        reason = "prespecified_fixed_gamma_1"
        mode = "fixed_gamma_1"

    scores = pd.DataFrame(
        [
            {
                "gamma": selected_gamma,
                "balanced_nll": np.nan,
                "balanced_brier": np.nan,
                "episode_nll": np.nan,
                "episode_brier": np.nan,
                "n_episode": 0,
                "n_subject": 0,
                "n_k_class_strata": 0,
                "raw_selected": True,
                "selected": True,
                "gamma0_balanced_nll": np.nan,
                "raw_best_balanced_nll": np.nan,
                "nll_improvement_vs_gamma0": np.nan,
                "minimum_required_nll_improvement": np.nan,
                "final_selected_gamma": selected_gamma,
                "fallback_to_global": False,
                "fallback_reason": reason,
            }
        ]
    )

    return {
        "selected_gamma": selected_gamma,
        "raw_best_gamma": selected_gamma,
        "gamma0_balanced_nll": np.nan,
        "raw_best_balanced_nll": np.nan,
        "nll_improvement_vs_gamma0": np.nan,
        "minimum_required_nll_improvement": np.nan,
        "fallback_to_global": False,
        "fallback_reason": reason,
        "scores": scores,
        "records": pd.DataFrame(),
        "calibration_subjects": [],
        "primary_metric": mode,
        "secondary_metric": mode,
    }


def select_source_adaptation_gamma(
    model,
    embeddings,
    global_probs,
    y,
    subject_ids,
    anchor_ns,
    calibration_subjects,
    k_values,
    gamma_candidates,
    args,
    device,
):
    """
    Nested SOURCE-ONLY adaptation-strength calibration.
    """
    catalog = (
        build_chronological_episode_catalog(
            subject_ids,
            y,
            anchor_ns,
            calibration_subjects,
            k_values,
        )
    )

    records = (
        collect_source_calibration_records(
            model,
            embeddings,
            global_probs,
            y,
            anchor_ns,
            catalog,
            device,
        )
    )

    if records.empty:
        raise RuntimeError(
            "Source-only gamma calibration has no valid chronological "
            "episodes."
        )

    hard_bound = (
        args.np_alpha_max
        * args.np_residual_max
    )

    score_rows = []

    for gamma in gamma_candidates:
        score_rows.append(
            score_source_calibration_gamma(
                records,
                gamma=float(
                    gamma
                ),
                hard_bound=hard_bound,
            )
        )

    scores = pd.DataFrame(
        score_rows
    )

    primary = (
        "balanced_nll"
        if args.adaptation_calibration_primary
        == "nll"
        else "balanced_brier"
    )
    secondary = (
        "balanced_brier"
        if primary
        == "balanced_nll"
        else "balanced_nll"
    )

    scores = scores.sort_values(
        [
            primary,
            secondary,
            "gamma",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    raw_best_gamma = float(
        scores.iloc[
            0
        ][
            "gamma"
        ]
    )

    gamma0_rows = scores[
        np.isclose(
            scores["gamma"].to_numpy(dtype=float),
            0.0,
        )
    ]
    if len(gamma0_rows) != 1:
        raise RuntimeError(
            "Safe adaptation requires exactly one gamma=0 candidate."
        )

    gamma0_balanced_nll = float(
        gamma0_rows.iloc[0]["balanced_nll"]
    )
    raw_best_balanced_nll = float(
        scores.iloc[0]["balanced_nll"]
    )
    nll_improvement_vs_gamma0 = float(
        gamma0_balanced_nll - raw_best_balanced_nll
    )
    minimum_required_improvement = float(
        args.adaptation_min_nll_improvement
    )
    if (
        not np.isfinite(minimum_required_improvement)
        or minimum_required_improvement < 0.0
    ):
        raise ValueError(
            "--adaptation-min-nll-improvement must be finite and >= 0."
        )

    if args.ablation_id == "A5":
        # A5 deliberately omits the minimum-improvement activation gate.
        adaptation_activated = bool(raw_best_gamma > 0.0)
    else:
        adaptation_activated = bool(
            raw_best_gamma > 0.0
            and nll_improvement_vs_gamma0
            >= minimum_required_improvement
        )
    selected_gamma = (
        raw_best_gamma
        if adaptation_activated
        else 0.0
    )

    if raw_best_gamma == 0.0:
        fallback_reason = "raw_best_gamma_zero"
    elif not adaptation_activated:
        fallback_reason = "nll_improvement_below_minimum"
    elif args.ablation_id == "A5":
        fallback_reason = "source_argmin_selected_without_minimum"
    else:
        fallback_reason = "source_nll_improvement_met"

    scores[
        "raw_selected"
    ] = (
        scores[
            "gamma"
        ]
        == raw_best_gamma
    )

    scores[
        "selected"
    ] = (
        scores[
            "gamma"
        ]
        == selected_gamma
    )
    scores["gamma0_balanced_nll"] = gamma0_balanced_nll
    scores["raw_best_balanced_nll"] = raw_best_balanced_nll
    scores["nll_improvement_vs_gamma0"] = nll_improvement_vs_gamma0
    scores[
        "minimum_required_nll_improvement"
    ] = minimum_required_improvement
    scores["final_selected_gamma"] = selected_gamma
    scores["fallback_to_global"] = not adaptation_activated
    scores["fallback_reason"] = fallback_reason

    return {
        "selected_gamma": selected_gamma,
        "raw_best_gamma": raw_best_gamma,
        "gamma0_balanced_nll": gamma0_balanced_nll,
        "raw_best_balanced_nll": raw_best_balanced_nll,
        "nll_improvement_vs_gamma0": nll_improvement_vs_gamma0,
        "minimum_required_nll_improvement": minimum_required_improvement,
        "fallback_to_global": not adaptation_activated,
        "fallback_reason": fallback_reason,
        "scores": scores,
        "records": records,
        "calibration_subjects": sorted(
            [
                int(
                    s
                )
                for s in calibration_subjects
            ]
        ),
        "primary_metric": primary,
        "secondary_metric": secondary,
    }


def build_v16_np_model(
    embedding_dim,
    args,
    device,
):
    return (
        ContrastiveGatedResidualTemporalAttentiveNP(
            embedding_dim=embedding_dim,
            hidden_dim=args.np_hidden_dim,
            latent_dim=args.np_latent_dim,
            attention_dim=args.np_attention_dim,
            context_aux_weight=args.np_context_aux_weight,
            posterior_weight=args.np_posterior_weight,
            alpha_max=args.np_alpha_max,
            residual_max=args.np_residual_max,
            gate_init_fraction=args.np_gate_init_fraction,
            gate_reg_weight=args.np_gate_reg_weight,
            residual_reg_weight=args.np_residual_reg_weight,
            contrast_scale=args.np_contrast_scale,
            residual_init_fraction=args.np_residual_init_fraction,
            ablation_id=args.ablation_id,
        )
        .to(
            device
        )
    )


def train_np_with_early_stopping(
    embeddings,
    global_probs,
    y,
    subject_ids,
    anchor_ns,
    train_subjects,
    val_subjects,
    k_values,
    args,
    device,
    seed,
):
    seed_everything(
        seed
    )

    train_catalog = (
        build_chronological_episode_catalog(
            subject_ids,
            y,
            anchor_ns,
            train_subjects,
            k_values,
        )
    )

    val_catalog = (
        build_chronological_episode_catalog(
            subject_ids,
            y,
            anchor_ns,
            val_subjects,
            k_values,
        )
    )

    train_subjects_eligible = (
        chronological_catalog_subjects(
            train_catalog
        )
    )

    val_subjects_eligible = (
        chronological_catalog_subjects(
            val_catalog
        )
    )

    if not train_subjects_eligible:
        raise RuntimeError(
            "No source meta-train subject has a valid "
            "chronological context->future-query episode."
        )

    if (
        chronological_catalog_size(
            val_catalog
        )
        == 0
    ):
        # Source-only fallback; outer target remains untouched.
        val_catalog = train_catalog
        val_subjects_eligible = (
            train_subjects_eligible
        )

    model = build_v16_np_model(
        embeddings.shape[
            1
        ],
        args,
        device,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.np_lr,
        weight_decay=args.np_weight_decay,
    )

    rng = np.random.default_rng(
        seed
        + 9101
    )

    best_nll = np.inf
    best_epoch = 1
    best_state = None
    patience_left = (
        args.np_patience
    )

    history = []

    for epoch in range(
        1,
        args.np_max_epochs
        + 1,
    ):
        model.train()

        losses = []
        prior_ces = []
        posterior_ces = []
        kls = []
        aux_ces = []
        gate_regs = []
        residual_regs = []
        mean_gates = []
        mean_shifts = []
        mean_bounded_residuals = []
        mean_abs_contrasts = []
        mean_residual_magnitudes = []

        for _ in range(
            int(
                args.episodes_per_epoch
            )
        ):
            ep = sample_chronological_episode(
                rng,
                train_catalog,
                k_values,
            )

            if ep is None:
                continue

            ci = ep[
                "context_idx"
            ]
            qi = ep[
                "query_idx"
            ]

            context_age_np = (
                context_age_days_from_indices(
                    anchor_ns,
                    ci,
                    qi,
                )
            )

            cx = torch.as_tensor(
                embeddings[
                    ci
                ],
                dtype=torch.float32,
                device=device,
            )
            cy = torch.as_tensor(
                y[
                    ci
                ],
                dtype=torch.long,
                device=device,
            )
            ca = torch.as_tensor(
                context_age_np,
                dtype=torch.float32,
                device=device,
            )
            qx = torch.as_tensor(
                embeddings[
                    qi
                ],
                dtype=torch.float32,
                device=device,
            )
            qy = torch.as_tensor(
                y[
                    qi
                ],
                dtype=torch.long,
                device=device,
            )
            base_log_probs = (
                probs_to_log_probs_tensor(
                    global_probs[
                        qi
                    ],
                    device,
                )
            )

            opt.zero_grad(
                set_to_none=True
            )

            out = model.training_episode(
                cx,
                cy,
                ca,
                qx,
                qy,
                base_log_probs,
                beta_kl=args.np_beta_kl,
            )

            out[
                "loss"
            ].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            opt.step()

            losses.append(
                float(
                    out[
                        "loss"
                    ].item()
                )
            )
            prior_ces.append(
                float(
                    out[
                        "prior_ce"
                    ].item()
                )
            )
            posterior_ces.append(
                float(
                    out[
                        "posterior_ce"
                    ].item()
                )
            )
            kls.append(
                float(
                    out[
                        "kl"
                    ].item()
                )
            )
            aux_ces.append(
                float(
                    out[
                        "aux_ce"
                    ].item()
                )
            )
            gate_regs.append(
                float(
                    out[
                        "gate_reg"
                    ].item()
                )
            )
            residual_regs.append(
                float(
                    out[
                        "residual_reg"
                    ].item()
                )
            )
            mean_bounded_residuals.append(
                float(
                    out[
                        "mean_abs_bounded_residual"
                    ].item()
                )
            )
            mean_abs_contrasts.append(
                float(
                    out[
                        "mean_abs_contrast"
                    ].item()
                )
            )
            mean_residual_magnitudes.append(
                float(
                    out[
                        "mean_residual_magnitude_fraction"
                    ].item()
                )
            )
            mean_gates.append(
                float(
                    out[
                        "mean_gate"
                    ].item()
                )
            )
            mean_shifts.append(
                float(
                    out[
                        "mean_abs_applied_shift"
                    ].item()
                )
            )

        val = eval_np_source_episodes(
            model,
            embeddings,
            global_probs,
            y,
            anchor_ns,
            val_catalog,
            k_values,
            n_episodes=args.np_val_episodes,
            device=device,
            seed=seed
            + epoch
            * 17,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    float(
                        np.mean(
                            losses
                        )
                    )
                    if losses
                    else np.nan
                ),
                "train_prior_ce": (
                    float(
                        np.mean(
                            prior_ces
                        )
                    )
                    if prior_ces
                    else np.nan
                ),
                "train_posterior_ce": (
                    float(
                        np.mean(
                            posterior_ces
                        )
                    )
                    if posterior_ces
                    else np.nan
                ),
                "train_kl": (
                    float(
                        np.mean(
                            kls
                        )
                    )
                    if kls
                    else np.nan
                ),
                "train_context_aux_ce": (
                    float(
                        np.mean(
                            aux_ces
                        )
                    )
                    if aux_ces
                    else np.nan
                ),
                "train_gate_reg": (
                    float(
                        np.mean(
                            gate_regs
                        )
                    )
                    if gate_regs
                    else np.nan
                ),
                "train_residual_reg": (
                    float(
                        np.mean(
                            residual_regs
                        )
                    )
                    if residual_regs
                    else np.nan
                ),
                "train_mean_abs_bounded_residual": (
                    float(
                        np.mean(
                            mean_bounded_residuals
                        )
                    )
                    if mean_bounded_residuals
                    else np.nan
                ),
                "train_mean_abs_contrast": (
                    float(
                        np.mean(
                            mean_abs_contrasts
                        )
                    )
                    if mean_abs_contrasts
                    else np.nan
                ),
                "train_mean_residual_magnitude_fraction": (
                    float(
                        np.mean(
                            mean_residual_magnitudes
                        )
                    )
                    if mean_residual_magnitudes
                    else np.nan
                ),
                "train_mean_gate": (
                    float(
                        np.mean(
                            mean_gates
                        )
                    )
                    if mean_gates
                    else np.nan
                ),
                "train_mean_abs_applied_logodds_shift": (
                    float(
                        np.mean(
                            mean_shifts
                        )
                    )
                    if mean_shifts
                    else np.nan
                ),
                "val_nll": val[
                    "nll"
                ],
                "val_macro_f1": val[
                    "macro_f1"
                ],
            }
        )

        if (
            val[
                "nll"
            ]
            < best_nll
            - 1e-6
        ):
            best_nll = val[
                "nll"
            ]
            best_epoch = (
                epoch
            )
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_left = (
                args.np_patience
            )
        else:
            patience_left -= (
                1
            )

        if (
            patience_left
            <= 0
        ):
            break

    if best_state is None:
        best_state = copy.deepcopy(
            model.state_dict()
        )

    return {
        "best_epoch": int(
            best_epoch
        ),
        "best_val_nll": float(
            best_nll
        ),
        "best_state": best_state,
        "history": pd.DataFrame(
            history
        ),
        "eligible_meta_train_subjects": (
            train_subjects_eligible
        ),
        "eligible_meta_val_subjects": (
            val_subjects_eligible
        ),
        "meta_train_episode_catalog_n": (
            chronological_catalog_size(
                train_catalog
            )
        ),
        "meta_val_episode_catalog_n": (
            chronological_catalog_size(
                val_catalog
            )
        ),
    }


def retrain_np_exact_epochs(
    embeddings,
    global_probs,
    y,
    subject_ids,
    anchor_ns,
    source_subjects,
    k_values,
    args,
    device,
    seed,
    epochs,
):
    seed_everything(
        seed
    )

    catalog = (
        build_chronological_episode_catalog(
            subject_ids,
            y,
            anchor_ns,
            source_subjects,
            k_values,
        )
    )

    eligible = (
        chronological_catalog_subjects(
            catalog
        )
    )

    if not eligible:
        raise RuntimeError(
            "No source subject is eligible "
            "for chronological NP meta-training."
        )

    model = build_v16_np_model(
        embeddings.shape[
            1
        ],
        args,
        device,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.np_lr,
        weight_decay=args.np_weight_decay,
    )

    rng = np.random.default_rng(
        seed
        + 12345
    )

    for _ in range(
        int(
            epochs
        )
    ):
        model.train()

        for _ in range(
            int(
                args.episodes_per_epoch
            )
        ):
            ep = sample_chronological_episode(
                rng,
                catalog,
                k_values,
            )

            if ep is None:
                continue

            ci = ep[
                "context_idx"
            ]
            qi = ep[
                "query_idx"
            ]

            context_age_np = (
                context_age_days_from_indices(
                    anchor_ns,
                    ci,
                    qi,
                )
            )

            cx = torch.as_tensor(
                embeddings[
                    ci
                ],
                dtype=torch.float32,
                device=device,
            )
            cy = torch.as_tensor(
                y[
                    ci
                ],
                dtype=torch.long,
                device=device,
            )
            ca = torch.as_tensor(
                context_age_np,
                dtype=torch.float32,
                device=device,
            )
            qx = torch.as_tensor(
                embeddings[
                    qi
                ],
                dtype=torch.float32,
                device=device,
            )
            qy = torch.as_tensor(
                y[
                    qi
                ],
                dtype=torch.long,
                device=device,
            )
            base_log_probs = (
                probs_to_log_probs_tensor(
                    global_probs[
                        qi
                    ],
                    device,
                )
            )

            opt.zero_grad(
                set_to_none=True
            )

            out = model.training_episode(
                cx,
                cy,
                ca,
                qx,
                qy,
                base_log_probs,
                beta_kl=args.np_beta_kl,
            )

            out[
                "loss"
            ].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            opt.step()

    return (
        model,
        eligible,
        chronological_catalog_size(
            catalog
        ),
    )


# ---------------------------------------------------------------------------
# Target matched-query construction
# ---------------------------------------------------------------------------

def audit_label_availability_contract(
    manifest: pd.DataFrame,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Fail closed if any chronological episode reveals an unavailable label.

    This mirrors the source episode builder for every participant and every
    requested context size.  It is run once before training, saved as a CSV,
    and guarantees that anchor-only episode construction cannot silently use a
    label whose questionnaire was submitted after the later query.
    """
    required = {
        "sample_index",
        "user",
        "label",
        "anchor_utc",
        "ifb_submission_utc",
        "rr_end_utc",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(
            "Label-availability audit missing manifest columns: "
            f"{missing}"
        )

    rows = []
    for subject, group in manifest.groupby("user", sort=True):
        group = group.sort_values(
            ["anchor_utc", "model_row"],
            kind="stable",
        )
        past = {0: [], 1: []}

        for query_time, block in group.groupby("anchor_utc", sort=True):
            query_time = pd.Timestamp(query_time)
            for _, query in block.iterrows():
                for k in k_values:
                    k = int(k)
                    if len(past[0]) < k or len(past[1]) < k:
                        continue

                    selected = past[0][-k:] + past[1][-k:]
                    for context in selected:
                        label_available = pd.Timestamp(
                            context["ifb_submission_utc"]
                        ) < query_time
                        window_complete = pd.Timestamp(
                            context["rr_end_utc"]
                        ) < query_time
                        row = {
                            "subject": int(subject),
                            "query_original_sample_index": int(
                                query["original_sample_index"]
                            ),
                            "query_anchor_utc": query_time,
                            "k_per_class": k,
                            "context_original_sample_index": int(
                                context["original_sample_index"]
                            ),
                            "context_anchor_utc": pd.Timestamp(
                                context["anchor_utc"]
                            ),
                            "context_ifb_submission_utc": pd.Timestamp(
                                context["ifb_submission_utc"]
                            ),
                            "context_rr_end_utc": pd.Timestamp(
                                context["rr_end_utc"]
                            ),
                            "label_available_before_query": bool(
                                label_available
                            ),
                            "window_complete_before_query": bool(
                                window_complete
                            ),
                        }
                        rows.append(row)
                        if not (label_available and window_complete):
                            raise AssertionError(
                                "Chronological episode contains a context "
                                "whose label or RR window was unavailable "
                                "at query time."
                            )

            # Same-anchor samples remain hidden until the whole block has
            # been evaluated.
            for _, revealed in block.iterrows():
                past[int(revealed["label"])].append(revealed.to_dict())

    return pd.DataFrame(rows)

def target_chronological_rows(
    manifest: pd.DataFrame,
    target_subject: int,
) -> pd.DataFrame:
    g = (
        manifest[
            manifest["user"].astype(int)
            == int(target_subject)
        ]
        .copy()
        .sort_values(
            ["anchor_utc", "model_row"]
        )
        .reset_index(drop=True)
    )
    return g


def build_matched_queries(
    manifest: pd.DataFrame,
    target_subject: int,
    k_values: Sequence[int],
):
    """
    Fixed query set:
      keep a query only if its PAST already contains >= max(k_values)
      examples of BOTH classes.

    Then construct nested contexts from the most recent k/class history.
    """
    g = target_chronological_rows(
        manifest,
        target_subject,
    )

    max_k = max(k_values)

    query_defs = []
    audit_rows = []

    past_by_class = {
        0: [],
        1: [],
    }

    for _, row in g.iterrows():
        q_model_row = int(row["model_row"])
        q_original = int(
            row["original_sample_index"]
        )
        q_label = int(row["label"])
        q_time = pd.Timestamp(
            row["anchor_utc"]
        )

        # A dataset may provide more than one labeled sample from the same
        # acquisition/session block.  Keep those samples for later blocks,
        # but never allow an equal-timestamp sample into the current query's
        # context.
        # A historical label is usable only after the corresponding IFB was
        # submitted.  The physiological window must also have ended before
        # the query prediction anchor.  This is stricter than ordering only by
        # the event/physiology anchor, which can precede a retrospective label
        # submission by minutes or hours.
        strict_past_by_class = {
            c: [
                item
                for item in past_by_class[c]
                if pd.Timestamp(item["anchor_utc"])
                < q_time
                and pd.Timestamp(item["ifb_submission_utc"])
                < q_time
                and pd.Timestamp(item["rr_end_utc"])
                < q_time
            ]
            for c in [0, 1]
        }

        eligible = (
            len(strict_past_by_class[0]) >= max_k
            and len(strict_past_by_class[1]) >= max_k
        )

        if eligible:
            contexts = {}

            for k in k_values:
                ctx = []

                for c in [0, 1]:
                    # Most recent k samples of each class.
                    selected = (
                        strict_past_by_class[c][-k:]
                    )
                    ctx.extend(selected)

                # Sort context chronologically for auditing / NP input.
                ctx = sorted(
                    ctx,
                    key=lambda x: x["anchor_utc"],
                )

                contexts[int(k)] = ctx

                for citem in ctx:
                    if not (
                        pd.Timestamp(
                            citem["anchor_utc"]
                        )
                        < q_time
                        and pd.Timestamp(
                            citem["ifb_submission_utc"]
                        )
                        < q_time
                        and pd.Timestamp(
                            citem["rr_end_utc"]
                        )
                        < q_time
                    ):
                        raise AssertionError(
                            "Context anchor, label availability, or RR-window "
                            "end is not strictly earlier than query."
                        )

                labels = [
                    int(x["label"])
                    for x in ctx
                ]

                if labels.count(0) != k:
                    raise AssertionError(
                        "Context not balanced for class 0."
                    )
                if labels.count(1) != k:
                    raise AssertionError(
                        "Context not balanced for class 1."
                    )

                for citem in ctx:
                    audit_rows.append(
                        {
                            "target_subject": int(target_subject),
                            "query_model_row": q_model_row,
                            "query_original_sample_index": q_original,
                            "query_anchor_utc": q_time,
                            "query_label": q_label,
                            "k_per_class": int(k),
                            "context_total": int(2 * k),
                            "context_model_row": int(
                                citem["model_row"]
                            ),
                            "context_original_sample_index": int(
                                citem[
                                    "original_sample_index"
                                ]
                            ),
                            "context_anchor_utc": pd.Timestamp(
                                citem["anchor_utc"]
                            ),
                            "context_ifb_submission_utc": pd.Timestamp(
                                citem["ifb_submission_utc"]
                            ),
                            "context_rr_end_utc": pd.Timestamp(
                                citem["rr_end_utc"]
                            ),
                            "context_label": int(
                                citem["label"]
                            ),
                        }
                    )

            # Nested context assertion by original sample id.
            prev_set = set()
            for k in sorted(k_values):
                cur_set = {
                    int(x["original_sample_index"])
                    for x in contexts[int(k)]
                }

                if prev_set and not prev_set.issubset(
                    cur_set
                ):
                    raise AssertionError(
                        "Contexts are not nested."
                    )

                prev_set = cur_set

            query_defs.append(
                {
                    "target_subject": int(target_subject),
                    "query_model_row": q_model_row,
                    "query_original_sample_index": q_original,
                    "query_anchor_utc": q_time,
                    "query_label": q_label,
                    "contexts": contexts,
                }
            )

        # Reveal label AFTER deciding whether this row is a query.
        item = {
            "model_row": q_model_row,
            "original_sample_index": q_original,
            "label": q_label,
            "anchor_utc": q_time,
            "ifb_submission_utc": pd.Timestamp(
                row["ifb_submission_utc"]
            ),
            "rr_end_utc": pd.Timestamp(
                row["rr_end_utc"]
            ),
        }
        past_by_class[q_label].append(item)

    return query_defs, pd.DataFrame(audit_rows)


def auto_target_subjects(
    manifest: pd.DataFrame,
    k_values: Sequence[int],
) -> List[int]:
    subjects = sorted(
        manifest["user"]
        .astype(int)
        .unique()
        .tolist()
    )

    eligible = []

    for s in subjects:
        q, _ = build_matched_queries(
            manifest,
            s,
            k_values,
        )
        if len(q) > 0:
            eligible.append(int(s))

    return eligible


# ---------------------------------------------------------------------------
# Target methods
# ---------------------------------------------------------------------------

def softmax_np(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def prototype_probability(
    embeddings: np.ndarray,
    y: np.ndarray,
    context_rows: Sequence[int],
    query_row: int,
    temperature: float,
):
    ctx = embeddings[
        np.asarray(context_rows, dtype=int)
    ]
    cy = y[
        np.asarray(context_rows, dtype=int)
    ]

    q = embeddings[
        int(query_row)
    ]

    # Unit-normalized geometry improves robustness to embedding norm drift.
    ctx_n = ctx / np.clip(
        np.linalg.norm(
            ctx,
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )
    q_n = q / max(
        float(np.linalg.norm(q)),
        1e-8,
    )

    protos = []

    for c in [0, 1]:
        pc = ctx_n[cy == c].mean(
            axis=0
        )
        pc = pc / max(
            float(np.linalg.norm(pc)),
            1e-8,
        )
        protos.append(pc)

    d0 = float(
        np.sum(
            (q_n - protos[0]) ** 2
        )
    )
    d1 = float(
        np.sum(
            (q_n - protos[1]) ** 2
        )
    )

    temp = max(
        float(temperature),
        1e-6,
    )

    logits = np.array(
        [-d0 / temp, -d1 / temp],
        dtype=float,
    )

    p = softmax_np(logits)
    return p


def finetuned_head_probability(
    source_head: nn.Linear,
    embeddings: np.ndarray,
    y: np.ndarray,
    context_rows: Sequence[int],
    query_row: int,
    args,
    device,
    seed,
):
    seed_everything(seed)

    head = nn.Linear(
        source_head.in_features,
        source_head.out_features,
    ).to(device)

    head.load_state_dict(
        copy.deepcopy(
            source_head.state_dict()
        )
    )

    head.train()

    initial_weight = (
        source_head.weight.detach().clone().to(device)
    )
    initial_bias = (
        source_head.bias.detach().clone().to(device)
    )

    ci = np.asarray(
        context_rows,
        dtype=int,
    )

    x = torch.as_tensor(
        embeddings[ci],
        dtype=torch.float32,
        device=device,
    )
    yy = torch.as_tensor(
        y[ci],
        dtype=torch.long,
        device=device,
    )

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=args.finetune_lr,
        weight_decay=0.0,
    )

    for _ in range(int(args.finetune_steps)):
        opt.zero_grad(set_to_none=True)

        logits = head(x)

        ce = F.cross_entropy(
            logits,
            yy,
        )

        anchor = (
            (head.weight - initial_weight)
            .pow(2)
            .mean()
            + (
                head.bias - initial_bias
            )
            .pow(2)
            .mean()
        )

        loss = (
            ce
            + args.finetune_anchor_lambda
            * anchor
        )

        loss.backward()
        opt.step()

    head.eval()

    with torch.no_grad():
        q = torch.as_tensor(
            embeddings[
                int(query_row)
            ][None, :],
            dtype=torch.float32,
            device=device,
        )

        p = torch.softmax(
            head(q),
            dim=-1,
        )[0].cpu().numpy()

    return p


def np_probability(
    np_model,
    embeddings,
    global_probs,
    y,
    anchor_ns,
    context_rows,
    query_row,
    adaptation_gamma,
    args,
    device,
):
    """
    V1.6 target inference.

    The Global Fusion probability is the immutable source anchor.
    NP only proposes a bounded log-odds residual, with no target optimizer.
    """
    with torch.no_grad():
        ci = np.asarray(
            context_rows,
            dtype=int,
        )
        qi = np.asarray(
            [
                int(
                    query_row
                )
            ],
            dtype=int,
        )

        context_age_np = (
            context_age_days_from_indices(
                anchor_ns,
                ci,
                qi,
            )
        )

        cx = torch.as_tensor(
            embeddings[
                ci
            ],
            dtype=torch.float32,
            device=device,
        )
        cy = torch.as_tensor(
            y[
                ci
            ],
            dtype=torch.long,
            device=device,
        )
        ca = torch.as_tensor(
            context_age_np,
            dtype=torch.float32,
            device=device,
        )
        qx = torch.as_tensor(
            embeddings[
                int(
                    query_row
                )
            ][
                None,
                :,
            ],
            dtype=torch.float32,
            device=device,
        )

        base_prob_np = global_probs[
            int(
                query_row
            )
        ][
            None,
            :,
        ]

        base_log_probs = (
            probs_to_log_probs_tensor(
                base_prob_np,
                device,
            )
        )

        deterministic = (
            np_model.predict_mean(
                cx,
                cy,
                ca,
                qx,
                base_log_probs,
            )
        )

        uncalibrated_final_prob = (
            deterministic[
                "prob"
            ][
                0
            ]
            .cpu()
            .numpy()
        )

        gate = float(
            deterministic[
                "gate"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )
        bounded_residual = float(
            deterministic[
                "bounded_residual_logodds"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )
        uncalibrated_applied_shift = float(
            deterministic[
                "applied_logodds_shift"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )

        spec = get_ablation_spec(
            args.ablation_id
        )
        hard_bound = configured_hard_bound(
            args
        )

        if args.ablation_id == "A1":
            applied_shift = uncalibrated_applied_shift
            final_prob = uncalibrated_final_prob
        else:
            raw_scaled_shift = (
                float(adaptation_gamma)
                * uncalibrated_applied_shift
            )
            if spec["hard_bound_enabled"]:
                applied_shift = float(
                    np.clip(
                        raw_scaled_shift,
                        -hard_bound,
                        hard_bound,
                    )
                )
            else:
                applied_shift = float(
                    raw_scaled_shift
                )

            final_prob = (
                _calibrated_binary_probability_from_shift(
                    base_prob_np[0],
                    uncalibrated_applied_shift,
                    float(adaptation_gamma),
                    hard_bound,
                )
            )

        residual_magnitude_fraction = float(
            deterministic[
                "residual_magnitude_fraction"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )
        signed_contrast = float(
            deterministic[
                "signed_contrast"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )
        contrast_direction = float(
            deterministic[
                "contrast_direction"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )
        cos_attended_nonstress = float(
            deterministic[
                "cos_attended_nonstress"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )
        cos_attended_stress = float(
            deterministic[
                "cos_attended_stress"
            ][
                0,
                0
            ]
            .cpu()
            .item()
        )

        # V1.6 mechanistic audit:
        # except at numerical zero, residual direction must agree with the
        # query-to-history contrast.
        if (
            spec["direction_constraint_enabled"]
            and
            abs(signed_contrast) > 1e-8
            and abs(bounded_residual) > 1e-8
            and signed_contrast * bounded_residual < -1e-8
        ):
            raise AssertionError(
                "Contrastive residual direction disagrees with "
                "query-to-history evidence."
            )

        if (
            args.ablation_id != "A1"
            and
            gate
            < -1e-7
            or gate
            > args.np_alpha_max
            + 1e-6
        ):
            raise AssertionError(
                "NP gate exceeded configured bound."
            )

        if (
            spec["hard_bound_enabled"]
            and
            abs(
                applied_shift
            )
            > hard_bound
            + 1e-5
        ):
            raise AssertionError(
                "NP applied log-odds shift exceeded hard bound."
            )

        if not np.all(
            np.isfinite(
                final_prob
            )
        ):
            raise AssertionError(
                "NP produced non-finite probability."
            )

        mc = np_model.predict_mc(
            cx,
            cy,
            ca,
            qx,
            base_log_probs,
            mc_samples=args.mc_samples_eval,
            adaptation_gamma=float(
                adaptation_gamma
            ),
        )

        return {
            "prob": final_prob,
            "base_prob": base_prob_np[
                0
            ],
            "gate": gate,
            "gate_fraction_of_max": (
                np.nan
                if args.ablation_id == "A1"
                else gate / args.np_alpha_max
            ),
            "bounded_residual_logodds": (
                bounded_residual
            ),
            "adaptation_gamma": (
                np.nan
                if args.ablation_id == "A1"
                else float(adaptation_gamma)
            ),
            "uncalibrated_applied_logodds_shift": (
                uncalibrated_applied_shift
            ),
            "applied_logodds_shift": (
                applied_shift
            ),
            "max_abs_logodds_shift": (
                hard_bound
            ),
            "uses_global_anchor": bool(
                spec["uses_global_anchor"]
            ),
            "hard_bound_enabled": bool(
                spec["hard_bound_enabled"]
            ),
            "direction_constraint_enabled": bool(
                spec["direction_constraint_enabled"]
            ),
            "p_stress_minus_global": float(
                final_prob[
                    1
                ]
                - base_prob_np[
                    0,
                    1,
                ]
            ),
            "residual_magnitude_fraction": (
                residual_magnitude_fraction
            ),
            "signed_contrast": signed_contrast,
            "contrast_direction": contrast_direction,
            "cos_attended_nonstress": (
                cos_attended_nonstress
            ),
            "cos_attended_stress": (
                cos_attended_stress
            ),
            "predictive_entropy": float(
                mc[
                    "predictive_entropy"
                ][
                    0
                ]
                .cpu()
                .item()
            ),
            "expected_entropy": float(
                mc[
                    "expected_entropy"
                ][
                    0
                ]
                .cpu()
                .item()
            ),
            "mutual_information": float(
                mc[
                    "mutual_information"
                ][
                    0
                ]
                .cpu()
                .item()
            ),
            "stress_probability_variance": float(
                mc[
                    "stress_probability_variance"
                ][
                    0
                ]
                .cpu()
                .item()
            ),
            "mc_applied_shift_std": float(
                mc[
                    "mc_applied_shift_std"
                ][
                    0
                ]
                .cpu()
                .item()
            ),
            "attention_entropy_class0": float(
                deterministic[
                    "attention_entropy_class0"
                ][
                    0
                ]
                .cpu()
                .item()
            ),
            "attention_entropy_class1": float(
                deterministic[
                    "attention_entropy_class1"
                ][
                    0
                ]
                .cpu()
                .item()
            ),
            "recency_scale": float(
                deterministic[
                    "recency_scale"
                ]
                .detach()
                .cpu()
                .item()
            ),
        }


# ---------------------------------------------------------------------------
# Metrics / statistics
# ---------------------------------------------------------------------------

def safe_auc(
    y_true: np.ndarray,
    p1: np.ndarray,
) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(
            roc_auc_score(
                y_true,
                p1,
            )
        )
    except Exception:
        return np.nan


def compute_binary_metrics(
    y_true: np.ndarray,
    p1: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    p1 = np.asarray(
        p1,
        dtype=float,
    )

    p1 = np.clip(
        p1,
        1e-7,
        1.0 - 1e-7,
    )

    pred = (
        p1 >= 0.5
    ).astype(int)

    probs = np.column_stack(
        [1.0 - p1, p1]
    )

    has_both_classes = (
        len(np.unique(y_true)) == 2
    )

    # Accuracy and proper scoring rules remain well-defined for a one-class
    # query set. Balanced Accuracy / Macro-F1 / AUROC are not used as
    # subject-level discrimination metrics unless both true classes occur.
    if has_both_classes:
        balanced_acc = float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        )
        macro_f1 = float(
            f1_score(
                y_true,
                pred,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        )
        auc = safe_auc(
            y_true,
            p1,
        )
    else:
        balanced_acc = np.nan
        macro_f1 = np.nan
        auc = np.nan

    return {
        "n": int(len(y_true)),
        "n_nonstress": int(
            np.sum(y_true == 0)
        ),
        "n_stress": int(
            np.sum(y_true == 1)
        ),
        "has_both_query_classes": bool(
            has_both_classes
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "auroc": auc,
        "brier": float(
            brier_score_loss(
                y_true,
                p1,
            )
        ),
        "nll": float(
            log_loss(
                y_true,
                probs,
                labels=[0, 1],
            )
        ),
    }


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    pvals = np.asarray(
        pvals,
        dtype=float,
    )

    out = np.full(
        len(pvals),
        np.nan,
        dtype=float,
    )

    valid = np.flatnonzero(
        np.isfinite(pvals)
    )

    if len(valid) == 0:
        return out.tolist()

    order = valid[
        np.argsort(
            pvals[valid]
        )
    ]

    m = len(order)
    running = 0.0

    for rank, idx in enumerate(
        order,
        start=1,
    ):
        adjusted = (
            (m - rank + 1)
            * pvals[idx]
        )
        running = max(
            running,
            adjusted,
        )
        out[idx] = min(
            running,
            1.0,
        )

    return out.tolist()


def paired_wilcoxon(
    subject_metrics: pd.DataFrame,
    method_a: str,
    context_a: int,
    method_b: str,
    context_b: int,
    metric: str,
):
    a = (
        subject_metrics[
            (subject_metrics["method"] == method_a)
            & (
                subject_metrics["context_total"]
                == context_a
            )
        ][
            ["target_subject", metric]
        ]
        .rename(
            columns={metric: "a"}
        )
    )

    b = (
        subject_metrics[
            (subject_metrics["method"] == method_b)
            & (
                subject_metrics["context_total"]
                == context_b
            )
        ][
            ["target_subject", metric]
        ]
        .rename(
            columns={metric: "b"}
        )
    )

    m = a.merge(
        b,
        on="target_subject",
        how="inner",
    )

    m = m[
        np.isfinite(m["a"])
        & np.isfinite(m["b"])
    ].copy()

    if len(m) < 3:
        return {
            "n_subjects": len(m),
            "statistic": np.nan,
            "p_raw": np.nan,
            "median_a": (
                float(m["a"].median())
                if len(m) else np.nan
            ),
            "median_b": (
                float(m["b"].median())
                if len(m) else np.nan
            ),
            "median_a_minus_b": (
                float(
                    (m["a"] - m["b"])
                    .median()
                )
                if len(m) else np.nan
            ),
        }

    diff = (
        m["a"].to_numpy()
        - m["b"].to_numpy()
    )

    if np.allclose(
        diff,
        0.0,
        atol=1e-12,
    ):
        stat = 0.0
        p = 1.0
    else:
        try:
            res = wilcoxon(
                m["a"],
                m["b"],
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            stat = float(res.statistic)
            p = float(res.pvalue)
        except Exception:
            stat = np.nan
            p = np.nan

    return {
        "n_subjects": int(len(m)),
        "statistic": stat,
        "p_raw": p,
        "median_a": float(
            m["a"].median()
        ),
        "median_b": float(
            m["b"].median()
        ),
        "median_a_minus_b": float(
            (m["a"] - m["b"]).median()
        ),
    }


# ---------------------------------------------------------------------------
# Fold experiment
# ---------------------------------------------------------------------------

def run_fold_seed(
    data,
    target_subject,
    k_values,
    args,
    device,
    seed,
    fold_dir: Path,
):
    X_rr = data["X_rr"]
    X_hrv = data["X_hrv"]
    y = data["y"]
    subject_ids = data["subject_ids"]
    manifest = data["manifest"]

    source_subjects = sorted(
        [
            int(s)
            for s in np.unique(
                subject_ids
            )
            if int(s) != int(target_subject)
        ]
    )

    source_mask = (
        subject_ids != int(target_subject)
    )
    target_mask = (
        subject_ids == int(target_subject)
    )

    if target_mask.sum() == 0:
        raise RuntimeError(
            f"Target subject {target_subject} not found."
        )

    query_defs, context_audit = build_matched_queries(
        manifest,
        target_subject,
        k_values,
    )

    if len(query_defs) == 0:
        return None

    # -----------------------
    # Inner source split
    # -----------------------
    inner_train_subjects, inner_val_subjects = split_source_subjects(
        source_subjects,
        args.inner_val_fraction,
        seed=seed + int(target_subject) * 101,
    )

    inner_train_mask = np.isin(
        subject_ids,
        inner_train_subjects,
    )
    inner_val_mask = np.isin(
        subject_ids,
        inner_val_subjects,
    )

    # Fit inner-train-only preprocessing for early stopping.
    inner_scalers = fit_scalers(
        X_rr[inner_train_mask],
        X_hrv[inner_train_mask],
    )

    rr_inner_train, hrv_inner_train = apply_scalers(
        X_rr[inner_train_mask],
        X_hrv[inner_train_mask],
        inner_scalers,
    )
    rr_inner_val, hrv_inner_val = apply_scalers(
        X_rr[inner_val_mask],
        X_hrv[inner_val_mask],
        inner_scalers,
    )

    enc_select = train_one_encoder_pass(
        rr_inner_train,
        hrv_inner_train,
        y[inner_train_mask],
        rr_inner_val,
        hrv_inner_val,
        y[inner_val_mask],
        args,
        device,
        seed=seed + 1,
    )

    # -----------------------
    # Clean source-only NP model selection.
    #
    # IMPORTANT V1.6 preserved fix:
    # NP early stopping uses embeddings/global predictions from the encoder
    # trained ONLY on inner-train subjects. Therefore the inner validation
    # subjects remain genuinely unseen to the representation learner used for
    # NP epoch selection.
    # -----------------------
    rr_inner_all, hrv_inner_all = apply_scalers(
        X_rr,
        X_hrv,
        inner_scalers,
    )

    selection_model = FusionClassifier().to(
        device
    )
    selection_model.load_state_dict(
        enc_select[
            "best_state"
        ]
    )

    (
        selection_global_probs,
        selection_embeddings,
    ) = predict_classifier(
        selection_model,
        rr_inner_all,
        hrv_inner_all,
        device=device,
    )

    np_select = train_np_with_early_stopping(
        selection_embeddings,
        selection_global_probs,
        y,
        subject_ids,
        data["anchor_ns"],
        inner_train_subjects,
        inner_val_subjects,
        k_values,
        args,
        device,
        seed=seed + 3,
    )

    # ---------------------------------------------------------------
    # SOURCE-ONLY adaptation-strength calibration.
    # The outer target subject is NEVER used here.
    # ---------------------------------------------------------------
    gamma_candidates = parse_float_list(
        args.adaptation_gamma_candidates
    )

    if args.ablation_id in {"A5", "A6"}:
        gamma_model = build_v16_np_model(
            selection_embeddings.shape[1],
            args,
            device,
        )
        gamma_model.load_state_dict(
            np_select["best_state"]
        )

        gamma_calibration_subjects = list(
            inner_val_subjects
        )
        gamma_catalog_probe = (
            build_chronological_episode_catalog(
                subject_ids,
                y,
                data["anchor_ns"],
                gamma_calibration_subjects,
                k_values,
            )
        )

        if chronological_catalog_size(
            gamma_catalog_probe
        ) == 0:
            # Rare SOURCE-only fallback.
            gamma_calibration_subjects = list(
                inner_train_subjects
            )

        gamma_select = select_source_adaptation_gamma(
            gamma_model,
            selection_embeddings,
            selection_global_probs,
            y,
            subject_ids,
            data["anchor_ns"],
            gamma_calibration_subjects,
            k_values,
            gamma_candidates,
            args,
            device,
        )
        del gamma_model
    else:
        gamma_select = fixed_gamma_selection(
            args.ablation_id
        )

    selected_gamma = float(
        gamma_select[
            "selected_gamma"
        ]
    )

    del selection_model

    # -----------------------
    # Exact encoder retrain on ALL outer-source subjects.
    # -----------------------
    final_scalers = fit_scalers(
        X_rr[source_mask],
        X_hrv[source_mask],
    )

    rr_all, hrv_all = apply_scalers(
        X_rr,
        X_hrv,
        final_scalers,
    )

    final_model = retrain_encoder_exact_epochs(
        rr_all[source_mask],
        hrv_all[source_mask],
        y[source_mask],
        args,
        device,
        seed=seed + 2,
        epochs=enc_select["best_epoch"],
    )

    # Frozen source global anchor for every sample, including outer target.
    global_probs, embeddings = predict_classifier(
        final_model,
        rr_all,
        hrv_all,
        device=device,
    )

    # Exact NP retrain on ALL outer-source subjects for the selected epoch
    # count. Target remains completely excluded.
    (
        np_model,
        meta_source_subjects,
        meta_source_episode_catalog_n,
    ) = retrain_np_exact_epochs(
        embeddings,
        global_probs,
        y,
        subject_ids,
        data["anchor_ns"],
        source_subjects,
        k_values,
        args,
        device,
        seed=seed + 4,
        epochs=np_select["best_epoch"],
    )

    # -----------------------
    # Target evaluation
    # -----------------------
    rows = []

    max_context_total = 2 * max(k_values)

    for q_idx, qdef in enumerate(query_defs):
        qrow = int(
            qdef["query_model_row"]
        )
        y_true = int(
            qdef["query_label"]
        )

        p_global = global_probs[
            qrow
        ]

        global_p1_values = []

        for k in k_values:
            context_items = qdef["contexts"][
                int(k)
            ]
            context_rows = [
                int(x["model_row"])
                for x in context_items
            ]
            context_original = [
                int(
                    x["original_sample_index"]
                )
                for x in context_items
            ]

            context_total = int(
                2 * k
            )

            # Global - identical by construction for same query.
            global_p1_values.append(
                float(p_global[1])
            )

            rows.append(
                {
                    "seed": seed,
                    "target_subject": int(target_subject),
                    "query_model_row": qrow,
                    "query_original_sample_index": int(
                        qdef[
                            "query_original_sample_index"
                        ]
                    ),
                    "query_anchor_utc": qdef["query_anchor_utc"],
                    "y_true": y_true,
                    "k_per_class": int(k),
                    "context_total": context_total,
                    "method": "GlobalFusion",
                    "p_nonstress": float(
                        p_global[0]
                    ),
                    "p_stress": float(
                        p_global[1]
                    ),
                    "y_pred": int(
                        np.argmax(
                            p_global
                        )
                    ),
                    "predictive_entropy": np.nan,
                    "expected_entropy": np.nan,
                    "mutual_information": np.nan,
                    "stress_probability_variance": np.nan,
                    "context_original_sample_indices": ";".join(
                        map(str, context_original)
                    ),
                }
            )

            # Growing prototype.
            p_proto = prototype_probability(
                embeddings,
                y,
                context_rows,
                qrow,
                temperature=args.prototype_temperature,
            )

            rows.append(
                {
                    "seed": seed,
                    "target_subject": int(target_subject),
                    "query_model_row": qrow,
                    "query_original_sample_index": int(
                        qdef[
                            "query_original_sample_index"
                        ]
                    ),
                    "query_anchor_utc": qdef["query_anchor_utc"],
                    "y_true": y_true,
                    "k_per_class": int(k),
                    "context_total": context_total,
                    "method": "GrowingPrototype",
                    "p_nonstress": float(
                        p_proto[0]
                    ),
                    "p_stress": float(
                        p_proto[1]
                    ),
                    "y_pred": int(
                        np.argmax(
                            p_proto
                        )
                    ),
                    "predictive_entropy": np.nan,
                    "expected_entropy": np.nan,
                    "mutual_information": np.nan,
                    "stress_probability_variance": np.nan,
                    "context_original_sample_indices": ";".join(
                        map(str, context_original)
                    ),
                }
            )

            # Gradient fine-tuning comparator.
            p_ft = finetuned_head_probability(
                final_model.classifier,
                embeddings,
                y,
                context_rows,
                qrow,
                args,
                device,
                seed=(
                    seed
                    + q_idx * 1009
                    + int(k) * 31
                    + int(target_subject)
                ),
            )

            rows.append(
                {
                    "seed": seed,
                    "target_subject": int(target_subject),
                    "query_model_row": qrow,
                    "query_original_sample_index": int(
                        qdef[
                            "query_original_sample_index"
                        ]
                    ),
                    "query_anchor_utc": qdef["query_anchor_utc"],
                    "y_true": y_true,
                    "k_per_class": int(k),
                    "context_total": context_total,
                    "method": "FineTuning",
                    "p_nonstress": float(
                        p_ft[0]
                    ),
                    "p_stress": float(
                        p_ft[1]
                    ),
                    "y_pred": int(
                        np.argmax(
                            p_ft
                        )
                    ),
                    "predictive_entropy": np.nan,
                    "expected_entropy": np.nan,
                    "mutual_information": np.nan,
                    "stress_probability_variance": np.nan,
                    "context_original_sample_indices": ";".join(
                        map(str, context_original)
                    ),
                }
            )

            # Neural Process - NO target-side optimizer.
            np_out = np_probability(
                np_model,
                embeddings,
                global_probs,
                y,
                data["anchor_ns"],
                context_rows,
                qrow,
                selected_gamma,
                args,
                device,
            )
            p_np = np_out["prob"]

            rows.append(
                {
                    "seed": seed,
                    "target_subject": int(target_subject),
                    "query_model_row": qrow,
                    "query_original_sample_index": int(
                        qdef[
                            "query_original_sample_index"
                        ]
                    ),
                    "query_anchor_utc": qdef["query_anchor_utc"],
                    "y_true": y_true,
                    "k_per_class": int(k),
                    "context_total": context_total,
                    "method": "NeuralProcess",
                    "p_nonstress": float(
                        p_np[0]
                    ),
                    "p_stress": float(
                        p_np[1]
                    ),
                    "y_pred": int(
                        np.argmax(
                            p_np
                        )
                    ),
                    "predictive_entropy": np_out[
                        "predictive_entropy"
                    ],
                    "expected_entropy": np_out[
                        "expected_entropy"
                    ],
                    "mutual_information": np_out[
                        "mutual_information"
                    ],
                    "stress_probability_variance": np_out[
                        "stress_probability_variance"
                    ],
                    "attention_entropy_class0": np_out[
                        "attention_entropy_class0"
                    ],
                    "attention_entropy_class1": np_out[
                        "attention_entropy_class1"
                    ],
                    "np_recency_scale": np_out[
                        "recency_scale"
                    ],
                    "np_adaptation_gamma": np_out[
                        "adaptation_gamma"
                    ],
                    "np_gate": np_out[
                        "gate"
                    ],
                    "np_gate_fraction_of_max": np_out[
                        "gate_fraction_of_max"
                    ],
                    "np_bounded_residual_logodds": np_out[
                        "bounded_residual_logodds"
                    ],
                    "np_uncalibrated_applied_logodds_shift": np_out[
                        "uncalibrated_applied_logodds_shift"
                    ],
                    "np_applied_logodds_shift": np_out[
                        "applied_logodds_shift"
                    ],
                    "np_max_abs_logodds_shift": np_out[
                        "max_abs_logodds_shift"
                    ],
                    "np_p_stress_minus_global": np_out[
                        "p_stress_minus_global"
                    ],
                    "np_mc_applied_shift_std": np_out[
                        "mc_applied_shift_std"
                    ],
                    "np_residual_magnitude_fraction": np_out[
                        "residual_magnitude_fraction"
                    ],
                    "np_signed_contrast": np_out[
                        "signed_contrast"
                    ],
                    "np_contrast_direction": np_out[
                        "contrast_direction"
                    ],
                    "np_cos_attended_nonstress": np_out[
                        "cos_attended_nonstress"
                    ],
                    "np_cos_attended_stress": np_out[
                        "cos_attended_stress"
                    ],
                    "context_original_sample_indices": ";".join(
                        map(str, context_original)
                    ),
                }
            )

        # Hard global-invariance assertion for the SAME query.
        if not np.allclose(
            global_p1_values,
            global_p1_values[0],
            atol=0.0,
            rtol=0.0,
        ):
            raise AssertionError(
                "Global Fusion changed across context sizes "
                "for the same query."
            )

    result_df = pd.DataFrame(rows)
    spec = get_ablation_spec(args.ablation_id)
    result_df.insert(0, "ablation_id", args.ablation_id)
    result_df.insert(1, "ablation_label", spec["label"])
    result_df["uses_global_anchor"] = bool(
        spec["uses_global_anchor"]
    )
    result_df["hard_bound_enabled"] = bool(
        spec["hard_bound_enabled"]
    )
    result_df["direction_constraint_enabled"] = bool(
        spec["direction_constraint_enabled"]
    )
    result_df["source_gamma_mode"] = spec[
        "source_gamma_mode"
    ]

    context_audit["seed"] = seed
    context_audit["ablation_id"] = args.ablation_id

    # Additional global invariance audit table.
    global_rows = result_df[
        result_df["method"] == "GlobalFusion"
    ]

    invariance = (
        global_rows
        .groupby(
            [
                "seed",
                "target_subject",
                "query_original_sample_index",
            ]
        )["p_stress"]
        .agg(["min", "max"])
        .reset_index()
    )
    invariance["max_abs_range"] = (
        invariance["max"]
        - invariance["min"]
    ).abs()
    invariance["ablation_id"] = args.ablation_id

    if (
        invariance["max_abs_range"]
        .max()
        > 1e-12
    ):
        raise AssertionError(
            "Global invariance audit failed."
        )

    # Save optional checkpoints.
    if args.save_checkpoints:
        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        ckpt = {
            "ablation_id": args.ablation_id,
            "ablation_spec": get_ablation_spec(args.ablation_id),
            "target_subject": int(target_subject),
            "seed": int(seed),
            "encoder_state_dict": final_model.state_dict(),
            "np_state_dict": np_model.state_dict(),
            "scalers": {
                "rr_mean": final_scalers.rr_mean,
                "rr_std": final_scalers.rr_std,
                "hrv_mean": final_scalers.hrv_mean,
                "hrv_scale": final_scalers.hrv_scale,
            },
            "encoder_best_epoch": enc_select["best_epoch"],
            "np_best_epoch": np_select["best_epoch"],
            "source_selected_adaptation_gamma": selected_gamma,
            "source_raw_best_adaptation_gamma": gamma_select[
                "raw_best_gamma"
            ],
            "source_gamma0_balanced_nll": gamma_select[
                "gamma0_balanced_nll"
            ],
            "source_gamma_nll_improvement_vs_gamma0": gamma_select[
                "nll_improvement_vs_gamma0"
            ],
            "source_gamma_minimum_required_nll_improvement": gamma_select[
                "minimum_required_nll_improvement"
            ],
            "source_gamma_fallback_to_global": gamma_select[
                "fallback_to_global"
            ],
            "source_gamma_fallback_reason": gamma_select[
                "fallback_reason"
            ],
            "source_gamma_candidates": gamma_candidates,
        }

        torch.save(
            ckpt,
            fold_dir
            / (
                f"target_{target_subject}"
                f"_seed_{seed}.pt"
            ),
        )

        enc_select["history"].to_csv(
            fold_dir
            / (
                f"encoder_history_target_"
                f"{target_subject}_seed_{seed}.csv"
            ),
            index=False,
        )

        np_select["history"].to_csv(
            fold_dir
            / (
                f"np_history_target_"
                f"{target_subject}_seed_{seed}.csv"
            ),
            index=False,
        )

    summary = {
        "ablation_id": args.ablation_id,
        "ablation_label": spec["label"],
        "uses_global_anchor": bool(spec["uses_global_anchor"]),
        "hard_bound_enabled": bool(spec["hard_bound_enabled"]),
        "direction_constraint_enabled": bool(
            spec["direction_constraint_enabled"]
        ),
        "source_gamma_mode": spec["source_gamma_mode"],
        "seed": seed,
        "target_subject": int(target_subject),
        "source_subject_n": len(source_subjects),
        "source_sample_n": int(
            source_mask.sum()
        ),
        "target_sample_n": int(
            target_mask.sum()
        ),
        "matched_query_n": len(query_defs),
        "query_stress_n": int(
            sum(
                int(q["query_label"]) == 1
                for q in query_defs
            )
        ),
        "query_nonstress_n": int(
            sum(
                int(q["query_label"]) == 0
                for q in query_defs
            )
        ),
        "query_has_both_classes": bool(
            any(
                int(q["query_label"]) == 0
                for q in query_defs
            )
            and any(
                int(q["query_label"]) == 1
                for q in query_defs
            )
        ),
        "encoder_best_epoch": enc_select[
            "best_epoch"
        ],
        "encoder_best_val_macro_f1": enc_select[
            "best_val_macro_f1"
        ],
        "np_best_epoch": np_select[
            "best_epoch"
        ],
        "np_best_val_nll": np_select[
            "best_val_nll"
        ],
        "source_selected_adaptation_gamma": selected_gamma,
        "source_raw_best_adaptation_gamma": float(
            gamma_select["raw_best_gamma"]
        ),
        "source_gamma0_balanced_nll": float(
            gamma_select["gamma0_balanced_nll"]
        ),
        "source_gamma_raw_best_balanced_nll": float(
            gamma_select["raw_best_balanced_nll"]
        ),
        "source_gamma_nll_improvement_vs_gamma0": float(
            gamma_select["nll_improvement_vs_gamma0"]
        ),
        "source_gamma_minimum_required_nll_improvement": float(
            gamma_select["minimum_required_nll_improvement"]
        ),
        "source_gamma_fallback_to_global": bool(
            gamma_select["fallback_to_global"]
        ),
        "source_gamma_fallback_reason": str(
            gamma_select["fallback_reason"]
        ),
        "source_gamma_calibration_primary": gamma_select[
            "primary_metric"
        ],
        "source_gamma_calibration_subject_n": int(
            len(
                gamma_select[
                    "calibration_subjects"
                ]
            )
        ),
        "source_gamma_calibration_episode_n": int(
            len(
                gamma_select[
                    "records"
                ]
            )
        ),
        "source_gamma_calibration_selected_balanced_nll": float(
            gamma_select[
                "scores"
            ]
            .loc[
                gamma_select[
                    "scores"
                ][
                    "selected"
                ],
                "balanced_nll",
            ]
            .iloc[0]
        ),
        "source_gamma_calibration_selected_balanced_brier": float(
            gamma_select[
                "scores"
            ]
            .loc[
                gamma_select[
                    "scores"
                ][
                    "selected"
                ],
                "balanced_brier",
            ]
            .iloc[0]
        ),
        "np_meta_source_subject_n": len(
            meta_source_subjects
        ),
        "np_meta_source_episode_catalog_n": int(
            meta_source_episode_catalog_n
        ),
        "np_meta_train_episode_catalog_n": int(
            np_select[
                "meta_train_episode_catalog_n"
            ]
        ),
        "np_meta_val_episode_catalog_n": int(
            np_select[
                "meta_val_episode_catalog_n"
            ]
        ),
    }

    gamma_scores = gamma_select[
        "scores"
    ].copy()
    gamma_scores.insert(
        0,
        "ablation_id",
        args.ablation_id,
    )
    gamma_scores.insert(
        1,
        "target_subject",
        int(target_subject),
    )
    gamma_scores.insert(
        2,
        "seed",
        int(seed),
    )

    return {
        "results": result_df,
        "context_audit": context_audit,
        "global_invariance": invariance,
        "training_summary": summary,
        "source_gamma_calibration": gamma_scores,
    }


# ---------------------------------------------------------------------------
# Aggregate outputs
# ---------------------------------------------------------------------------

def make_subject_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "seed",
        "target_subject",
        "method",
        "context_total",
    ]

    for keys, g in results.groupby(
        group_cols,
        sort=True,
    ):
        seed, subject, method, context_total = keys

        m = compute_binary_metrics(
            g["y_true"].to_numpy(),
            g["p_stress"].to_numpy(),
        )

        rows.append(
            {
                "seed": int(seed),
                "target_subject": int(subject),
                "method": str(method),
                "context_total": int(context_total),
                **m,
            }
        )

    return pd.DataFrame(rows)


def make_pooled_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Report each seed separately.
    for keys, g in results.groupby(
        ["seed", "method", "context_total"],
        sort=True,
    ):
        seed, method, context_total = keys

        m = compute_binary_metrics(
            g["y_true"].to_numpy(),
            g["p_stress"].to_numpy(),
        )

        rows.append(
            {
                "seed": int(seed),
                "method": str(method),
                "context_total": int(context_total),
                **m,
            }
        )

    return pd.DataFrame(rows)


def make_progressive_gain_subject(
    subject_metrics: pd.DataFrame,
    min_context_total: int,
    max_context_total: int,
) -> pd.DataFrame:
    rows = []

    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "brier",
        "nll",
    ]

    for (
        seed,
        subject,
        method,
    ), g in subject_metrics.groupby(
        ["seed", "target_subject", "method"],
        sort=True,
    ):
        a = g[
            g["context_total"]
            == min_context_total
        ]
        b = g[
            g["context_total"]
            == max_context_total
        ]

        if len(a) != 1 or len(b) != 1:
            continue

        rec = {
            "seed": int(seed),
            "target_subject": int(subject),
            "method": str(method),
            "context_from": int(
                min_context_total
            ),
            "context_to": int(
                max_context_total
            ),
        }

        for metric in metrics:
            va = float(a.iloc[0][metric])
            vb = float(b.iloc[0][metric])

            rec[f"{metric}_from"] = va
            rec[f"{metric}_to"] = vb
            rec[f"{metric}_delta"] = (
                vb - va
                if (
                    np.isfinite(va)
                    and np.isfinite(vb)
                )
                else np.nan
            )

        rows.append(rec)

    return pd.DataFrame(rows)


def make_paired_stats(
    subject_metrics: pd.DataFrame,
    min_context_total: int,
    max_context_total: int,
) -> pd.DataFrame:
    """
    Paired statistics are done separately by seed, then written out.
    For the final manuscript, aggregate repeated-seed behavior should be
    summarized carefully rather than treating seeds as independent subjects.
    """
    rows = []

    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "brier",
        "nll",
    ]

    seeds = sorted(
        subject_metrics["seed"]
        .astype(int)
        .unique()
        .tolist()
    )

    methods = [
        "GlobalFusion",
        "GrowingPrototype",
        "FineTuning",
        "NeuralProcess",
    ]

    for seed in seeds:
        sm = subject_metrics[
            subject_metrics["seed"] == seed
        ]

        # Within-method context growth.
        for method in methods:
            for metric in metrics:
                res = paired_wilcoxon(
                    sm,
                    method,
                    max_context_total,
                    method,
                    min_context_total,
                    metric,
                )

                rows.append(
                    {
                        "seed": seed,
                        "family": "context_growth",
                        "comparison": (
                            f"{method}:"
                            f"C{max_context_total}"
                            f"_vs_C{min_context_total}"
                        ),
                        "metric": metric,
                        "method_a": method,
                        "context_a": max_context_total,
                        "method_b": method,
                        "context_b": min_context_total,
                        **res,
                    }
                )

        # NP vs alternatives at largest context.
        for competitor in [
            "GlobalFusion",
            "GrowingPrototype",
            "FineTuning",
        ]:
            for metric in metrics:
                res = paired_wilcoxon(
                    sm,
                    "NeuralProcess",
                    max_context_total,
                    competitor,
                    max_context_total,
                    metric,
                )

                rows.append(
                    {
                        "seed": seed,
                        "family": "method_at_max_context",
                        "comparison": (
                            "NeuralProcess_vs_"
                            f"{competitor}_"
                            f"C{max_context_total}"
                        ),
                        "metric": metric,
                        "method_a": "NeuralProcess",
                        "context_a": max_context_total,
                        "method_b": competitor,
                        "context_b": max_context_total,
                        **res,
                    }
                )

    df = pd.DataFrame(rows)

    if len(df):
        df["p_holm_within_seed_family"] = np.nan

        for (
            seed,
            family,
        ), idx in df.groupby(
            ["seed", "family"]
        ).groups.items():
            idx = list(idx)
            adjusted = holm_adjust(
                df.loc[
                    idx,
                    "p_raw",
                ].to_numpy()
            )
            df.loc[
                idx,
                "p_holm_within_seed_family",
            ] = adjusted

    return df


def make_context_sensitivity_audit(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        seed,
        subject,
        query_id,
        method,
    ), g in results.groupby(
        [
            "seed",
            "target_subject",
            "query_original_sample_index",
            "method",
        ],
        sort=True,
    ):
        g = g.sort_values(
            "context_total"
        )

        if len(g) < 2:
            continue

        p = g[
            "p_stress"
        ].to_numpy(
            dtype=float
        )

        contexts = g[
            "context_total"
        ].to_numpy(
            dtype=int
        )

        rows.append(
            {
                "seed": int(seed),
                "target_subject": int(subject),
                "query_original_sample_index": int(query_id),
                "method": str(method),
                "context_min": int(
                    contexts[0]
                ),
                "context_max": int(
                    contexts[-1]
                ),
                "p_stress_first": float(
                    p[0]
                ),
                "p_stress_last": float(
                    p[-1]
                ),
                "signed_p_change": float(
                    p[-1] - p[0]
                ),
                "abs_p_change": float(
                    abs(
                        p[-1] - p[0]
                    )
                ),
                "p_range": float(
                    np.max(p)
                    - np.min(p)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_np_safety_audit(
    results: pd.DataFrame,
) -> pd.DataFrame:
    np_rows = results[
        results[
            "method"
        ]
        == "NeuralProcess"
    ].copy()

    wanted = [
        "seed",
        "target_subject",
        "query_original_sample_index",
        "query_anchor_utc",
        "y_true",
        "context_total",
        "p_stress",
        "np_adaptation_gamma",
        "np_gate",
        "np_gate_fraction_of_max",
        "np_bounded_residual_logodds",
        "np_uncalibrated_applied_logodds_shift",
        "np_applied_logodds_shift",
        "np_max_abs_logodds_shift",
        "np_p_stress_minus_global",
        "np_mc_applied_shift_std",
        "np_residual_magnitude_fraction",
        "np_signed_contrast",
        "np_contrast_direction",
        "np_cos_attended_nonstress",
        "np_cos_attended_stress",
        "attention_entropy_class0",
        "attention_entropy_class1",
        "np_recency_scale",
    ]

    keep = [
        c
        for c in wanted
        if c in np_rows.columns
    ]

    return np_rows[
        keep
    ].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(
        args.output_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        out_dir / "checkpoints"
    )

    device = resolve_device(
        args.device
    )

    seeds = parse_int_list(
        args.seeds
    )
    k_values = sorted(
        set(
            parse_int_list(
                args.context_per_class
            )
        )
    )

    if min(k_values) < 1:
        raise ValueError(
            "Context per class must be >=1."
        )

    print("Loading prepared RELAX data...")
    data = load_prepared_data(
        Path(args.data_npz),
        Path(args.manifest),
        drop_overlap=args.drop_overlap,
    )

    manifest = data["manifest"]

    label_availability_audit = audit_label_availability_contract(
        manifest,
        k_values,
    )
    label_availability_audit.to_csv(
        out_dir / "label_availability_audit.csv",
        index=False,
    )
    print(
        "Label-availability audit:",
        len(label_availability_audit),
        "context-query links; 0 violations.",
    )

    all_subjects = sorted(
        manifest["user"]
        .astype(int)
        .unique()
        .tolist()
    )

    if str(args.target_subjects).lower() == "auto":
        targets = auto_target_subjects(
            manifest,
            k_values,
        )
    else:
        targets = parse_int_list(
            args.target_subjects
        )
        targets = [
            s
            for s in targets
            if s in all_subjects
        ]

    if args.max_target_subjects > 0:
        targets = targets[
            :args.max_target_subjects
        ]

    if not targets:
        raise SystemExit(
            "No eligible target subjects."
        )

    print("\n========== RUN CONFIG ==========")
    print("Device:", device)
    print(
        "Ablation:",
        args.ablation_id,
        "-",
        get_ablation_spec(args.ablation_id)["label"],
    )
    print("Samples after overlap QC:", len(manifest))
    print("Subjects:", len(all_subjects))
    print("Target subjects:", targets)
    print(
        "Context totals:",
        [2 * k for k in k_values],
    )
    print("Seeds:", seeds)
    print(
        "Source-only gamma candidates:",
        parse_float_list(
            args.adaptation_gamma_candidates
        ),
    )
    print(
        "Source-only gamma objective:",
        args.adaptation_calibration_primary,
    )
    print(
        "Minimum source balanced-NLL improvement for activation:",
        args.adaptation_min_nll_improvement,
    )

    config = {
        "script": "controlled_ablation_v17.py",
        "ablation_id": args.ablation_id,
        "ablation_spec": get_ablation_spec(args.ablation_id),
        "data_npz": str(
            Path(args.data_npz).resolve()
        ),
        "manifest": str(
            Path(args.manifest).resolve()
        ),
        "device": str(device),
        "seeds": seeds,
        "target_subjects": targets,
        "context_per_class": k_values,
        "context_total": [
            2 * k for k in k_values
        ],
        "drop_overlap": bool(
            args.drop_overlap
        ),
        "encoder": {
            "architecture": (
                "RR Conv1D + HRV MLP + 64-d fusion embedding"
            ),
            "lr": args.encoder_lr,
            "weight_decay": args.encoder_weight_decay,
            "max_epochs": args.encoder_max_epochs,
            "patience": args.encoder_patience,
            "inner_val_fraction": args.inner_val_fraction,
        },
        "neural_process": {
            "type": "contrastive gated residual temporal attentive neural process",
            "hidden_dim": args.np_hidden_dim,
            "latent_dim": args.np_latent_dim,
            "attention_dim": args.np_attention_dim,
            "context_aux_weight": args.np_context_aux_weight,
            "posterior_predictive_weight": args.np_posterior_weight,
            "alpha_max": args.np_alpha_max,
            "residual_max_logodds": args.np_residual_max,
            "max_applied_logodds_shift": (
                args.np_alpha_max * args.np_residual_max
                if get_ablation_spec(args.ablation_id)[
                    "hard_bound_enabled"
                ]
                else None
            ),
            "gate_init_fraction": args.np_gate_init_fraction,
            "gate_reg_weight": args.np_gate_reg_weight,
            "residual_reg_weight": args.np_residual_reg_weight,
            "contrast_scale": args.np_contrast_scale,
            "residual_init_fraction": args.np_residual_init_fraction,
            "gate_receives_context_count": False,
            "residual_direction": (
                "query-specific contrast constrained"
                if get_ablation_spec(args.ablation_id)[
                    "direction_constraint_enabled"
                ]
                else "free learned signed residual"
            ),
            "lr": args.np_lr,
            "weight_decay": args.np_weight_decay,
            "max_epochs": args.np_max_epochs,
            "patience": args.np_patience,
            "episodes_per_epoch": args.episodes_per_epoch,
            "meta_query_per_class": args.meta_query_per_class,
            "beta_kl": args.np_beta_kl,
            "mc_samples_eval": args.mc_samples_eval,
            "target_gradient_update": False,
            "embedding_normalization_in_personalization_branch": True,
            "global_anchor": (
                "frozen source Global Fusion log-probabilities"
                if get_ablation_spec(args.ablation_id)[
                    "uses_global_anchor"
                ]
                else "none; direct NP log-odds"
            ),
            "residual_parameterization": get_ablation_spec(
                args.ablation_id
            )["label"],
            "source_episode_protocol": "chronological past-context -> future-query",
            "primary_target_prediction": (
                "Global Fusion anchor + gated query-contrastive temporal-attentive "
                "NP residual using deterministic prior mean"
            ),
            "source_model_selection_protocol": (
                "NP early stopping uses inner-train-only encoder/scaler; "
                "inner-val subjects are representation-unseen"
            ),
            "adaptation_strength_calibration": {
                "selection_scope": (
                    "SOURCE-ONLY inner-validation chronological episodes"
                ),
                "candidates": parse_float_list(
                    args.adaptation_gamma_candidates
                ),
                "primary": args.adaptation_calibration_primary,
                "aggregation": (
                    "equal-weight hierarchy: episode -> subject within "
                    "(context,class) -> context/class strata"
                ),
                "tie_break": (
                    "secondary proper score, then smaller gamma"
                ),
                "minimum_balanced_nll_improvement_vs_gamma0": (
                    args.adaptation_min_nll_improvement
                ),
                "activation_rule": (
                    "activate raw source-selected gamma only when its "
                    "balanced NLL improves on gamma=0 by at least the "
                    "pre-specified minimum; otherwise use gamma=0"
                ),
                "gamma0_contract": (
                    "exact frozen Global Fusion probabilities without "
                    "log/exp recalculation"
                ),
                "hard_bound_after_gamma": get_ablation_spec(
                    args.ablation_id
                )["hard_bound_enabled"],
                "outer_target_used_for_selection": False,
            },
        },
        "prototype_temperature": args.prototype_temperature,
        "fine_tuning": {
            "target_gradient_update": True,
            "head_only": True,
            "steps": args.finetune_steps,
            "lr": args.finetune_lr,
            "anchor_lambda": args.finetune_anchor_lambda,
        },
        "query_protocol": (
            "same future query at all context sizes; "
            "most recent balanced nested target history; "
            "context strictly earlier than query"
        ),
    }

    (
        out_dir / "run_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result_parts = []
    audit_parts = []
    invariance_parts = []
    training_rows = []
    gamma_calibration_parts = []

    total_runs = (
        len(seeds)
        * len(targets)
    )
    run_no = 0

    t0 = time.time()

    for seed in seeds:
        for target in targets:
            run_no += 1

            print(
                "\n"
                + "=" * 72
            )
            print(
                f"[{run_no}/{total_runs}] "
                f"seed={seed} "
                f"target={target}"
            )

            fold_dir = (
                checkpoint_dir
                / f"seed_{seed}"
            )

            fold = run_fold_seed(
                data,
                target,
                k_values,
                args,
                device,
                seed,
                fold_dir,
            )

            if fold is None:
                print(
                    "  No matched queries; skipped."
                )
                continue

            result_parts.append(
                fold["results"]
            )
            audit_parts.append(
                fold["context_audit"]
            )
            invariance_parts.append(
                fold[
                    "global_invariance"
                ]
            )
            training_rows.append(
                fold[
                    "training_summary"
                ]
            )
            gamma_calibration_parts.append(
                fold[
                    "source_gamma_calibration"
                ]
            )

            # Incremental safety save after every fold.
            pd.concat(
                result_parts,
                ignore_index=True,
            ).to_csv(
                out_dir
                / "matched_query_results.partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame(
                training_rows
            ).to_csv(
                out_dir
                / "fold_training_summary.partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(
                "  matched queries:",
                fold[
                    "training_summary"
                ][
                    "matched_query_n"
                ],
                "| nonstress:",
                fold[
                    "training_summary"
                ][
                    "query_nonstress_n"
                ],
                "| stress:",
                fold[
                    "training_summary"
                ][
                    "query_stress_n"
                ],
            )
            if not fold["training_summary"]["query_has_both_classes"]:
                print(
                    "  [WARN] This target's matched future-query set contains "
                    "only one true class. Subject-level BA/Macro-F1/AUROC "
                    "will be recorded as NaN; Brier/NLL remain valid."
                )
            print(
                "  encoder best epoch:",
                fold[
                    "training_summary"
                ][
                    "encoder_best_epoch"
                ],
                "val Macro-F1:",
                f"{fold['training_summary']['encoder_best_val_macro_f1']:.4f}",
            )
            print(
                "  NP best epoch:",
                fold[
                    "training_summary"
                ][
                    "np_best_epoch"
                ],
                "val NLL:",
                f"{fold['training_summary']['np_best_val_nll']:.4f}",
            )
            print(
                "  source-selected adaptation gamma:",
                fold[
                    "training_summary"
                ][
                    "source_selected_adaptation_gamma"
                ],
                "| balanced source-val NLL:",
                f"{fold['training_summary']['source_gamma_calibration_selected_balanced_nll']:.4f}",
            )

    if not result_parts:
        raise SystemExit(
            "No folds produced results."
        )

    results = pd.concat(
        result_parts,
        ignore_index=True,
    )

    context_audit = pd.concat(
        audit_parts,
        ignore_index=True,
    )

    invariance = pd.concat(
        invariance_parts,
        ignore_index=True,
    )

    training_summary = pd.DataFrame(
        training_rows
    )

    source_gamma_calibration = (
        pd.concat(
            gamma_calibration_parts,
            ignore_index=True,
        )
        if gamma_calibration_parts
        else pd.DataFrame()
    )

    # Final invariant check.
    if (
        invariance["max_abs_range"]
        .max()
        > 1e-12
    ):
        raise AssertionError(
            "Global prediction invariance failed."
        )

    subject_metrics = make_subject_metrics(
        results
    )

    pooled_metrics = make_pooled_metrics(
        results
    )

    min_context_total = 2 * min(
        k_values
    )
    max_context_total = 2 * max(
        k_values
    )

    gains = make_progressive_gain_subject(
        subject_metrics,
        min_context_total,
        max_context_total,
    )

    stats_df = make_paired_stats(
        subject_metrics,
        min_context_total,
        max_context_total,
    )

    context_sensitivity = (
        make_context_sensitivity_audit(
            results
        )
    )

    np_safety = make_np_safety_audit(
        results
    )

    if (
        len(np_safety)
        and get_ablation_spec(args.ablation_id)[
            "hard_bound_enabled"
        ]
    ):
        max_allowed = (
            args.np_alpha_max
            * args.np_residual_max
        )
        observed_max = float(
            np_safety[
                "np_applied_logodds_shift"
            ]
            .abs()
            .max()
        )
        if observed_max > max_allowed + 1e-5:
            raise AssertionError(
                "Final NP safety audit found an applied "
                "log-odds shift outside the configured bound."
            )

    # Write final outputs.
    results.to_csv(
        out_dir
        / "matched_query_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    context_audit.to_csv(
        out_dir
        / "context_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subject_metrics.to_csv(
        out_dir
        / "subject_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_metrics.to_csv(
        out_dir
        / "pooled_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gains.to_csv(
        out_dir
        / "progressive_gain_subject.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stats_df.to_csv(
        out_dir
        / "paired_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    training_summary.to_csv(
        out_dir
        / "fold_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_gamma_calibration.to_csv(
        out_dir
        / "source_gamma_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    invariance.to_csv(
        out_dir
        / "global_invariance_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    context_sensitivity.to_csv(
        out_dir
        / "context_sensitivity_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np_safety.to_csv(
        out_dir
        / "np_safety_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Remove partial files after successful completion.
    for p in [
        out_dir
        / "matched_query_results.partial.csv",
        out_dir
        / "fold_training_summary.partial.csv",
    ]:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print(
        "Fold-seed runs:",
        len(training_summary),
    )
    print(
        "Target subjects evaluated:",
        training_summary[
            "target_subject"
        ].nunique(),
    )
    print(
        "Matched query rows (unique seed/query):",
        results[
            [
                "seed",
                "target_subject",
                "query_original_sample_index",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )
    print(
        "Result rows:",
        len(results),
    )
    print(
        "Global max probability range across contexts:",
        f"{invariance['max_abs_range'].max():.3e}",
    )

    np_sens = context_sensitivity[
        context_sensitivity["method"]
        == "NeuralProcess"
    ]
    if len(np_sens):
        print(
            "NP median |p_stress(Cmax)-p_stress(Cmin)|:",
            f"{np_sens['abs_p_change'].median():.4f}",
        )
        print(
            "NP mean   |p_stress(Cmax)-p_stress(Cmin)|:",
            f"{np_sens['abs_p_change'].mean():.4f}",
        )

    if len(training_summary):
        print(
            "Source-selected gamma by fold:"
        )
        print(
            training_summary[
                [
                    "seed",
                    "target_subject",
                    "source_selected_adaptation_gamma",
                    "source_gamma_calibration_selected_balanced_nll",
                    "source_gamma_calibration_selected_balanced_brier",
                ]
            ].to_string(
                index=False
            )
        )

    if len(np_safety):
        print(
            "NP median gate:",
            f"{np_safety['np_gate'].median():.4f}",
            "| max gate:",
            f"{np_safety['np_gate'].max():.4f}",
            "| configured alpha_max:",
            f"{args.np_alpha_max:.4f}",
        )
        print(
            "NP max |applied log-odds shift|:",
            f"{np_safety['np_applied_logodds_shift'].abs().max():.4f}",
            "| hard limit:",
            f"{args.np_alpha_max * args.np_residual_max:.4f}",
        )
        print(
            "NP median |P(stress)-Global P(stress)|:",
            f"{np_safety['np_p_stress_minus_global'].abs().median():.4f}",
        )
        if "np_residual_magnitude_fraction" in np_safety.columns:
            print(
                "NP residual magnitude fraction median/max:",
                f"{np_safety['np_residual_magnitude_fraction'].median():.4f}",
                "/",
                f"{np_safety['np_residual_magnitude_fraction'].max():.4f}",
            )
        if "np_signed_contrast" in np_safety.columns:
            sign_ok = (
                np.sign(
                    np_safety["np_signed_contrast"].to_numpy(dtype=float)
                )
                * np.sign(
                    np_safety["np_bounded_residual_logodds"].to_numpy(dtype=float)
                )
                >= 0
            )
            print(
                "NP contrast/residual direction agreement:",
                f"{100.0 * float(np.mean(sign_ok)):.1f}%",
            )
    print(
        "Elapsed minutes:",
        f"{elapsed / 60.0:.1f}",
    )
    print("Output:", out_dir)

    print("\nPooled metrics:")
    print(
        pooled_metrics[
            [
                "seed",
                "method",
                "context_total",
                "n",
                "balanced_accuracy",
                "macro_f1",
                "auroc",
                "brier",
                "nll",
            ]
        ].to_string(index=False)
    )

    print(
        "\nFirst inspect pooled_metrics.csv, subject_metrics.csv, "
        "progressive_gain_subject.csv and global_invariance_audit.csv, "
        "then source_gamma_calibration.csv and np_safety_audit.csv "
        "before interpreting NP gains."
    )


if __name__ == "__main__":
    main()
