"""Negative tests for the RELAX three-timestamp history contract.

These tests exercise the actual release Target and Source builders.  They do
not initialize, train, or evaluate a neural network.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "release"
    / "safe_np_relax_release_v18.py"
)
SPEC = importlib.util.spec_from_file_location("safe_np_release_v18", RELEASE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def utc(hour: int) -> pd.Timestamp:
    return pd.Timestamp(f"2024-01-01T{hour:02d}:00:00Z")


def make_manifest(invalid_kind: str) -> pd.DataFrame:
    """Create one final query with a newer invalid and older valid substitute."""
    anchors = [utc(0), utc(1), utc(2), utc(3), utc(10)]
    labels = [0, 1, 0, 1, 0]
    available = [utc(0), utc(1), utc(2), utc(3), utc(10)]
    ifb_submitted = list(available)
    rr_ends = [utc(0), utc(1), utc(2), utc(3), utc(10)]

    if invalid_kind == "delayed_label":
        available[2] = utc(11)
    elif invalid_kind == "label_equal_query":
        available[2] = utc(10)
    elif invalid_kind == "missing_timestamp":
        available[3] = pd.NaT
    elif invalid_kind == "missing_rr_end":
        rr_ends[3] = pd.NaT
    elif invalid_kind == "explicit_alias_differs_from_ifb":
        # Proves that builders consume the explicit canonical field rather
        # than falling back to a still-valid IFB timestamp.
        available[2] = utc(11)
        ifb_submitted[2] = utc(2)
    elif invalid_kind == "rr_overflow":
        rr_ends[3] = utc(11)
    elif invalid_kind == "rr_equal_query":
        rr_ends[3] = utc(10)
    else:
        raise ValueError(invalid_kind)

    return pd.DataFrame(
        {
            "sample_index": np.arange(5, dtype=int),
            "original_sample_index": np.arange(5, dtype=int),
            "model_row": np.arange(5, dtype=int),
            "user": np.ones(5, dtype=int),
            "label": labels,
            "anchor_utc": anchors,
            "ifb_submission_utc": ifb_submitted,
            "label_available_utc": available,
            "rr_end_utc": rr_ends,
        }
    )


def final_target_context(manifest: pd.DataFrame) -> list[int]:
    queries, _ = MODULE.build_matched_queries(manifest, 1, [1])
    final = next(q for q in queries if q["query_model_row"] == 4)
    return [int(row["model_row"]) for row in final["contexts"][1]]


def final_source_context(manifest: pd.DataFrame) -> list[int]:
    anchor_ns = manifest["anchor_utc"].astype("int64").to_numpy(dtype=np.int64)
    available_ns = (
        manifest["label_available_utc"].astype("int64").to_numpy(dtype=np.int64)
    )
    rr_end_ns = manifest["rr_end_utc"].astype("int64").to_numpy(dtype=np.int64)
    catalog = MODULE.build_chronological_episode_catalog(
        manifest["user"].to_numpy(dtype=np.int64),
        manifest["label"].to_numpy(dtype=np.int64),
        anchor_ns,
        [1],
        [1],
        label_available_ns=available_ns,
        rr_end_ns=rr_end_ns,
    )
    final = next(
        episode
        for episode in catalog[1][0][1]
        if int(episode["query_idx"][0]) == 4
    )
    return [int(i) for i in final["context_idx"]]


class ThreeTimestampContractTests(unittest.TestCase):
    def assert_both_builders(self, invalid_kind: str, expected: list[int]) -> None:
        manifest = make_manifest(invalid_kind)
        self.assertEqual(final_target_context(manifest), expected)
        self.assertEqual(final_source_context(manifest), expected)

    def test_delayed_label_is_excluded_and_older_class_peer_is_used(self) -> None:
        self.assert_both_builders("delayed_label", [0, 3])

    def test_missing_timestamp_fails_closed(self) -> None:
        self.assert_both_builders("missing_timestamp", [1, 2])

    def test_missing_rr_end_fails_closed(self) -> None:
        self.assert_both_builders("missing_rr_end", [1, 2])

    def test_explicit_label_available_field_overrides_valid_ifb_time(self) -> None:
        self.assert_both_builders("explicit_alias_differs_from_ifb", [0, 3])

    def test_rr_window_ending_after_query_is_excluded(self) -> None:
        self.assert_both_builders("rr_overflow", [1, 2])

    def test_equal_label_availability_is_not_strictly_before(self) -> None:
        self.assert_both_builders("label_equal_query", [0, 3])

    def test_equal_rr_window_end_is_not_strictly_before(self) -> None:
        self.assert_both_builders("rr_equal_query", [1, 2])

    def test_source_timestamp_array_length_mismatch_is_rejected(self) -> None:
        manifest = make_manifest("delayed_label")
        anchor_ns = manifest["anchor_utc"].astype("int64").to_numpy()
        available_ns = manifest["label_available_utc"].astype("int64").to_numpy()
        rr_end_ns = manifest["rr_end_utc"].astype("int64").to_numpy()
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            MODULE.build_chronological_episode_catalog(
                manifest["user"].to_numpy(),
                manifest["label"].to_numpy(),
                anchor_ns,
                [1],
                [1],
                label_available_ns=available_ns[:-1],
                rr_end_ns=rr_end_ns,
            )

    def test_loader_materializes_explicit_label_available_field(self) -> None:
        n = 2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "tiny.npz"
            manifest_path = root / "tiny.csv"
            anchors = pd.Series([utc(0), utc(1)])
            anchor_ns = anchors.astype("int64").to_numpy(dtype=np.int64)
            np.savez(
                npz_path,
                X_rr=np.ones((n, 64), dtype=np.float32),
                X_hrv=np.ones((n, 32), dtype=np.float32),
                y=np.asarray([0, 1], dtype=np.int64),
                subject_ids=np.asarray([1, 1], dtype=np.int64),
                anchor_unix_ns=anchor_ns,
            )
            pd.DataFrame(
                {
                    "sample_index": [0, 1],
                    "user": [1, 1],
                    "label": [0, 1],
                    "anchor_utc": [str(utc(0)), str(utc(1))],
                    "ifb_submission_utc": [str(utc(0)), str(utc(1))],
                    "rr_end_utc": [str(utc(0)), str(utc(1))],
                }
            ).to_csv(manifest_path, index=False)

            data = MODULE.load_prepared_data(npz_path, manifest_path, False)
            self.assertIn("label_available_utc", data["manifest"].columns)
            self.assertEqual(data["label_available_source"], "ifb_submission_utc")
            np.testing.assert_array_equal(data["label_available_ns"], anchor_ns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
