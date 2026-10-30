"""Fast structural checks for the matched A1--A6 ablation runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "ablations" / "controlled_ablation_v17.py"
SPEC = importlib.util.spec_from_file_location("controlled_ablation_v17", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def make_model(ablation_id: str):
    return module.ContrastiveGatedResidualTemporalAttentiveNP(
        embedding_dim=8,
        hidden_dim=12,
        latent_dim=4,
        attention_dim=6,
        alpha_max=0.5,
        residual_max=2.0,
        ablation_id=ablation_id,
    ).eval()


def inputs():
    torch.manual_seed(7)
    context_x = torch.randn(4, 8)
    context_y = torch.tensor([0, 1, 0, 1])
    context_age = torch.tensor([4.0, 3.0, 2.0, 1.0])
    query_x = torch.randn(3, 8)
    base_a = torch.log(torch.tensor([[0.8, 0.2]]).repeat(3, 1))
    base_b = torch.log(torch.tensor([[0.2, 0.8]]).repeat(3, 1))
    return context_x, context_y, context_age, query_x, base_a, base_b


def run_checks():
    cx, cy, ca, qx, base_a, base_b = inputs()

    # A1 is a direct NP: changing the supplied Global probability cannot
    # change its output.
    a1 = make_model("A1")
    p1 = a1.predict_mean(cx, cy, ca, qx, base_a)["prob"]
    p2 = a1.predict_mean(cx, cy, ca, qx, base_b)["prob"]
    assert torch.equal(p1, p2)

    # A2 retains the contrast direction but is deliberately not hard-clipped.
    a2 = make_model("A2")
    with torch.no_grad():
        a2.residual_out.bias.fill_(20.0)
        a2.gate_out.bias.fill_(20.0)
    out2 = a2.predict_mean(cx, cy, ca, qx, base_a)
    assert torch.max(torch.abs(out2["applied_logodds_shift"])).item() > 1.0
    direction2 = out2["contrast_direction"]
    residual2 = out2["bounded_residual_logodds"]
    assert torch.all(direction2 * residual2 >= -1e-8)

    # A3 permits either sign but keeps the same configured hard shift bound.
    a3 = make_model("A3")
    with torch.no_grad():
        a3.residual_out.bias.fill_(20.0)
        a3.gate_out.bias.fill_(20.0)
    out3 = a3.predict_mean(cx, cy, ca, qx, base_a)
    assert torch.max(torch.abs(out3["applied_logodds_shift"])).item() <= 1.0 + 1e-6

    # Complete structural variants remain bounded and direction-constrained.
    for variant in ("A4", "A5", "A6"):
        model = make_model(variant)
        out = model.predict_mean(cx, cy, ca, qx, base_a)
        assert torch.max(torch.abs(out["applied_logodds_shift"])).item() <= 1.0 + 1e-6
        assert torch.all(
            out["contrast_direction"]
            * out["bounded_residual_logodds"]
            >= -1e-8
        )

    # The numpy safety helper must still implement exact gamma=0 fallback.
    base = np.asarray([0.123456789, 0.876543211], dtype=np.float64)
    returned = module._calibrated_binary_probability_from_shift(
        base, 99.0, 0.0, 1.0
    )
    assert np.array_equal(returned, base)

    print("PASS: matched A1--A6 structural checks")


if __name__ == "__main__":
    run_checks()
