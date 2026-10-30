from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "frozen" / "v1_7_safe_source_calibrated.py"
spec = importlib.util.spec_from_file_location("v1_7_safe", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_gamma_zero_is_exact_global() -> None:
    base = np.array([0.432123456789, 0.567876543211], dtype=np.float64)
    out = module._calibrated_binary_probability_from_shift(
        base_prob=base,
        uncalibrated_shift=0.918,
        gamma=0.0,
        hard_bound=1.0,
    )
    assert np.array_equal(out, base)
    assert out is not base


def run_selection(best_nll: float):
    original_catalog = module.build_chronological_episode_catalog
    original_collect = module.collect_source_calibration_records
    original_score = module.score_source_calibration_gamma
    try:
        module.build_chronological_episode_catalog = lambda *args, **kwargs: {1: {0: {}}}
        module.collect_source_calibration_records = lambda *args, **kwargs: pd.DataFrame(
            [{"subject": 1}]
        )

        def fake_score(records, gamma, hard_bound):
            nll = 0.5000 if gamma == 0.0 else best_nll
            return {
                "gamma": float(gamma),
                "balanced_nll": nll,
                "balanced_brier": nll / 2,
                "episode_nll": nll,
                "episode_brier": nll / 2,
                "n_episode": 10,
                "n_subject": 3,
                "n_k_class_strata": 6,
            }

        module.score_source_calibration_gamma = fake_score
        args = SimpleNamespace(
            np_alpha_max=0.5,
            np_residual_max=2.0,
            adaptation_calibration_primary="nll",
            adaptation_min_nll_improvement=0.001,
        )
        return module.select_source_adaptation_gamma(
            model=None,
            embeddings=None,
            global_probs=None,
            y=None,
            subject_ids=None,
            anchor_ns=None,
            calibration_subjects=[1, 2, 3],
            k_values=[1, 2, 3],
            gamma_candidates=[0.0, 0.25],
            args=args,
            device="cpu",
        )
    finally:
        module.build_chronological_episode_catalog = original_catalog
        module.collect_source_calibration_records = original_collect
        module.score_source_calibration_gamma = original_score


def test_small_improvement_falls_back() -> None:
    result = run_selection(best_nll=0.4995)
    assert result["raw_best_gamma"] == 0.25
    assert result["selected_gamma"] == 0.0
    assert result["fallback_to_global"] is True
    assert result["fallback_reason"] == "nll_improvement_below_minimum"


def test_sufficient_improvement_activates() -> None:
    result = run_selection(best_nll=0.4980)
    assert result["raw_best_gamma"] == 0.25
    assert result["selected_gamma"] == 0.25
    assert result["fallback_to_global"] is False
    assert result["fallback_reason"] == "source_nll_improvement_met"


if __name__ == "__main__":
    test_gamma_zero_is_exact_global()
    test_small_improvement_falls_back()
    test_sufficient_improvement_activates()
    print("3 safe-gamma tests passed")
