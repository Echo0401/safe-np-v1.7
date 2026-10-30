"""Rebuild RELAX Target/Source index maps with the actual v18 release code.

This audit performs no model fitting and never reads model predictions.
Exit status 0 means both release maps are row-for-row equal to the archived
availability-aware reference maps; status 2 means at least one map differs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[1]
RELEASE = PACKAGE / "src" / "release" / "safe_np_relax_release_v18.py"
DATA_DIR = PACKAGE / "data" / "restricted" / "relax" / "prepared"
REFERENCE_DIR = PACKAGE / "data" / "restricted" / "relax" / "index_audit"
OUT = PACKAGE / "runs" / "relax_index_audit"

SPEC = importlib.util.spec_from_file_location("safe_np_release_v18", RELEASE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def iso_utc(value) -> str:
    return pd.Timestamp(value).isoformat()


def canonicalize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    if "query_anchor_utc" in out:
        out["query_anchor_utc"] = pd.to_datetime(
            out["query_anchor_utc"], utc=True, format="mixed"
        )
    for column in columns:
        if column == "context_original_sample_indices":
            out[column] = out[column].fillna("").astype(str)
        elif column != "query_anchor_utc":
            out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    return out.sort_values(columns, kind="stable").reset_index(drop=True)


def compare(
    label: str,
    actual: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
) -> dict:
    left = canonicalize(actual, columns)
    right = canonicalize(reference, columns)
    exact = left.equals(right)
    summary = {
        "actual_rows": int(len(left)),
        "reference_rows": int(len(right)),
        "exact_row_for_row": bool(exact),
    }
    if not exact:
        merged = left.merge(right, on=columns, how="outer", indicator=True)
        diff = merged[merged["_merge"] != "both"].copy()
        diff.to_csv(OUT / f"{label}_index_differences.csv", index=False)
        summary["actual_only"] = int((diff["_merge"] == "left_only").sum())
        summary["reference_only"] = int((diff["_merge"] == "right_only").sum())
    else:
        summary["actual_only"] = 0
        summary["reference_only"] = 0
    return summary


def main() -> int:
    data = MODULE.load_prepared_data(
        DATA_DIR / "relax_longitudinal_v2.npz",
        DATA_DIR / "relax_longitudinal_v2_manifest.csv",
        drop_overlap=True,
    )
    manifest = data["manifest"]
    k_values = [1, 2, 3]
    reference_target = pd.read_csv(
        REFERENCE_DIR / "target_map_label_available_rebuilt.csv"
    )
    reference_source = pd.read_csv(
        REFERENCE_DIR / "source_map_label_available_rebuilt.csv"
    )
    targets = sorted(reference_target["target_subject"].astype(int).unique())

    target_rows = []
    for subject in targets:
        queries, _ = MODULE.build_matched_queries(manifest, int(subject), k_values)
        for query in queries:
            for k in k_values:
                context = query["contexts"][k]
                target_rows.append(
                    {
                        "target_subject": int(subject),
                        "query_model_row": int(query["query_model_row"]),
                        "query_original_sample_index": int(
                            query["query_original_sample_index"]
                        ),
                        "query_anchor_utc": iso_utc(query["query_anchor_utc"]),
                        "y_true": int(query["query_label"]),
                        "k_per_class": int(k),
                        "context_total": int(2 * k),
                        "context_original_sample_indices": ";".join(
                            str(int(item["original_sample_index"]))
                            for item in context
                        ),
                    }
                )
    actual_target = pd.DataFrame(target_rows)
    actual_target.to_csv(OUT / "release_target_map.csv", index=False)

    subjects = sorted(manifest["user"].astype(int).unique())
    catalog = MODULE.build_chronological_episode_catalog(
        data["subject_ids"],
        data["y"],
        data["anchor_ns"],
        subjects,
        k_values,
        label_available_ns=data["label_available_ns"],
        rr_end_ns=data["rr_end_ns"],
    )
    source_rows = []
    for k in k_values:
        for query_class in (0, 1):
            for subject, episodes in catalog[k][query_class].items():
                for episode in episodes:
                    query_model_row = int(episode["query_idx"][0])
                    context_rows = [int(i) for i in episode["context_idx"]]
                    source_rows.append(
                        {
                            "source_subject": int(subject),
                            "query_original_sample_index": int(
                                manifest.iloc[query_model_row]["original_sample_index"]
                            ),
                            "query_anchor_utc": iso_utc(
                                manifest.iloc[query_model_row]["anchor_utc"]
                            ),
                            "query_label": int(query_class),
                            "k_per_class": int(k),
                            "context_total": int(2 * k),
                            "context_original_sample_indices": ";".join(
                                str(
                                    int(
                                        manifest.iloc[row][
                                            "original_sample_index"
                                        ]
                                    )
                                )
                                for row in context_rows
                            ),
                        }
                    )
    actual_source = pd.DataFrame(source_rows)
    actual_source.to_csv(OUT / "release_source_map.csv", index=False)

    three_time_audit = MODULE.audit_label_availability_contract(
        manifest, k_values
    )
    three_time_audit.to_csv(
        OUT / "release_three_time_context_audit.csv", index=False
    )

    target_columns = list(reference_target.columns)
    source_columns = list(reference_source.columns)
    summary = {
        "scope": "actual release builders; index reconstruction only; no training",
        "release_script": str(RELEASE),
        "label_available_source": data["label_available_source"],
        "manifest_rows_after_overlap_filter": int(len(manifest)),
        "target_unique_queries": int(
            actual_target[
                ["target_subject", "query_original_sample_index"]
            ].drop_duplicates().shape[0]
        ),
        "target_query_context_rows": int(len(actual_target)),
        "source_unique_queries": int(
            actual_source[
                ["source_subject", "query_original_sample_index"]
            ].drop_duplicates().shape[0]
        ),
        "source_query_context_rows": int(len(actual_source)),
        "source_history_links": int(actual_source["context_total"].sum()),
        "three_time_context_links_audited": int(len(three_time_audit)),
        "three_time_context_violations": int(
            (
                ~three_time_audit[
                    [
                        "anchor_before_query",
                        "label_available_before_query",
                        "window_complete_before_query",
                    ]
                ].all(axis=1)
            ).sum()
        ),
        "target_comparison": compare(
            "target", actual_target, reference_target, target_columns
        ),
        "source_comparison": compare(
            "source", actual_source, reference_source, source_columns
        ),
    }
    summary["all_indices_exact"] = bool(
        summary["target_comparison"]["exact_row_for_row"]
        and summary["source_comparison"]["exact_row_for_row"]
    )
    (OUT / "release_index_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_indices_exact"] else 2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    if cli.out.exists():
        raise SystemExit("Choose a new output directory; existing audits are preserved.")
    DATA_DIR = cli.data_dir
    REFERENCE_DIR = cli.reference_dir
    OUT = cli.out
    OUT.mkdir(parents=True)
    raise SystemExit(main())
