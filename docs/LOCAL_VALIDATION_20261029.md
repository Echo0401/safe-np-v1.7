# Local validation record — 29 October 2026

**Evidence source:** Console output supplied by the researcher from their own Windows machine. The release preparer reviewed the shared output; this record does not claim direct remote execution, nor preserve confidential source data or local absolute paths.

## Environment and independent setup

- Windows; Python 3.10.0; CPU-only PyTorch 2.12.1+cpu.
- NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2.
- A newly created independent virtual environment was used, separate from the original development `.venv`.
- `pip check`: `No broken requirements found.`
- `requirements.txt` installation and test execution were completed in the independent environment, per the researcher.

## Offline verification observed in the console output

| Command | Reported status | Scope |
|---|---|---|
| `scripts/verify_integrity.py` | passed; 79 checks on the **original** 80-file candidate | Source inventory and SHA-256 before this documentation revision |
| `scripts/check_reference.py` | passed; 7 variants; 2,565 records/variant | Saved aggregate design evidence |
| `scripts/check_wesad_reference.py` | passed; 5 runs; 15 participants/run | Saved WESAD group means, no retraining |
| `tests/run_smoke.py` | passed | Gamma 3 tests; time-contract 9 tests; ablation structure; 0 training steps |
| `tests/check_wesad_cli.py` | passed; 5 profiles | Configuration parsing, 0 training steps |
| `scripts/scan_release.py` | passed; 80 files; no findings on the **original** candidate | Static scan, not ownership or privacy certification |

The updated documentation candidate has a **new** inventory manifest. Re-run the commands above after applying the patch; the previous count is not a claim about the updated file set.

## Real-input minimal RELAX experiment

- Authorized prepared RELAX NPZ and manifest were found in two original locations each; all four file comparisons matched the expected SHA-256 values in `provenance/restricted_input_hashes.json`.
- The two inputs were copied into ignored `data/restricted/relax/prepared/`; no data was added to the release archive.
- The archived A6 executable was invoked with 1 target subject, 1 seed, only 1 encoder epoch and 1 NP epoch, 2 episodes/epoch, 2 validation episodes, 1 fine-tuning step, and 2 MC evaluation samples.
- Reported result: `EXPERIMENT COMPLETE`; 1 fold-seed run; 2 eligible future queries; 24 output rows; exit code `0`; results written under ignored `runs/`.
- Both eligible future queries had a single true class (Non-stress), so BA, macro-F1 and AUROC were `NaN` by design. NLL and Brier remained defined.
- Source calibration selected `gamma=0`, causing exact fallback to the Global probabilities for this run. No nonzero personalization effect was demonstrated by this smoke test.

## Limits and publishing safeguards

1. **No** formal 13-target × 5-seed full retraining, GPU/other-OS check, exact numerical reproduction of all historical model weights, or end-to-end raw-data preprocessing validation was performed.
2. This run establishes a limited environment and train/predict chain check, not a new scientific result. Do not substitute smoke metrics for paper values.
3. This archive excludes the local NPZ, manifest with participant timestamps, individual predictions, exact query indices, and experimental outputs. Keep those files out of the public repository.
4. Third-party code attribution, code-holders' consent, data-license conditions, publication of group-level tables and the final software `LICENSE` remain open author decisions.
5. The original source A6, ablation training definitions and frozen group result tables were not modified as part of this documentation update.
