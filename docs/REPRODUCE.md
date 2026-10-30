# Reproduction steps and tested scope

Run commands from the repository root. Python 3.10 and the pinned core dependencies were used for this audit. The historical training environment was not recovered completely. An author-run clean virtual-environment dependency installation and a minimal RELAX train-and-predict smoke test passed on Windows Python 3.10 on 29 October 2026. No GPU experiment, complete 65-fold retraining, or paper-wide retraining replication was performed. See [the dated validation record](LOCAL_VALIDATION_20261029.md).

## 1 Offline checks

```bash
python -B scripts/verify_integrity.py
python -B scripts/check_reference.py
python -B scripts/check_wesad_reference.py
python -B tests/run_smoke.py
python -B tests/check_wesad_cli.py
python -B scripts/scan_release.py
```

The smoke suite exercises the actual gamma helper, actual v18 Target/Source builders and actual ablation classes using synthetic inputs. It checks five generated commands against the archived parsers without invoking their main training loops. This proves neither predictive performance nor equivalence of v18 on arbitrary inputs.

## 1a Windows minimal training smoke test (not formal experiment replication)

After authorized RELAX prepared inputs are copied into the ignored `data/restricted/relax/prepared/` directory, compare their SHA-256 values with `provenance/restricted_input_hashes.json`. **Never commit these files or the resulting `runs/` directory.** Use a newly empty output directory:

```powershell
$py = "<independent-python310-venv>\Scripts\python.exe"
& $py -B src/frozen/v1_7_safe_source_calibrated.py `
  --data-npz data/restricted/relax/prepared/relax_longitudinal_v2.npz `
  --manifest data/restricted/relax/prepared/relax_longitudinal_v2_manifest.csv `
  --output-dir runs/relax_smoke_p13_seed2026 `
  --target-subjects 13 --seeds 2026 --context-per-class 1,2,3 `
  --device cpu --encoder-max-epochs 1 --np-max-epochs 1 `
  --episodes-per-epoch 2 --np-val-episodes 2 `
  --finetune-steps 1 --mc-samples-eval 2 --no-save-checkpoints
```

This fast sanity run uses reduced training and evaluation settings, so **do not compare its metrics with paper tables**. The 29 October run completed (`exit code 0`); its two eligible future queries contained one true class, so BA, macro-F1 and AUROC were correctly undefined (`NaN`). Source-only gamma was 0, exercising exact Global fallback, **not** demonstrating benefit from activated personalization. The result is a training-chain compatibility check only.

## 2 Rebuild saved-output main statistics

Supply authorized frozen run folders locally. Each source needs the files read by the archived analysis, including predictions, safety and training summaries. Recomputed outputs must use a fresh directory.

```bash
python -B scripts/reproduce_statistics.py --dataset relax --source data/restricted/relax/raw_runs/RELAX_A6_formal --out runs/relax_statistics_new
python -B scripts/reproduce_statistics.py --dataset polar --source data/restricted/polar/raw_runs/POLAR_V17_external_pilot_5seeds --out runs/polar_statistics_new
```

These commands do not train a network. Their required individual-level CSVs are deliberately absent from the public candidate. Group summaries already provided in `results/reference/` allow inspection without receiving those inputs.

## 3 Rebuild formal A0–A6 aggregation

Supply frozen A1/A2/A3/A6 run folders. Use formal mode; do not reduce the bootstrap count and call the output the paper result.

```bash
python -B src/ablations/aggregate_controlled_ablation.py --a1-dir data/restricted/relax/raw_runs/RELAX_A1_5seeds --a2-dir data/restricted/relax/raw_runs/RELAX_A2_5seeds --a3-dir data/restricted/relax/raw_runs/RELAX_A3_5seeds --a6-dir data/restricted/relax/raw_runs/RELAX_A6_formal --output-dir runs/ablation_statistics_new --analysis-mode formal --bootstrap-reps 10000 --bootstrap-seed 20261013
```

Formal mode validates 13 participants×5 seeds, query matching and reconstructed probabilities. The aggregator uses A6's recorded residual/gamma evidence for A0/A4/A5 and checks the saved A6 reconstruction. It does not generate new neural predictions.

## 4 Inspect frozen experiment commands

```bash
python -B scripts/run_experiment.py --profile relax_a6 --out runs/relax_a6_new
python -B scripts/run_experiment.py --profile polar_a6 --out runs/polar_a6_new
python -B scripts/run_experiment.py --profile relax_a1 --out runs/relax_a1_new
python -B scripts/run_experiment.py --profile relax_a2 --out runs/relax_a2_new
python -B scripts/run_experiment.py --profile relax_a3 --out runs/relax_a3_new
```

The default action only prints a structured argv list. Adding `--execute` would train into the new output location, after checking required inputs exist and match the frozen input SHA-256 values. This option was not used in the audit.

Profiles explicitly select C2/C4/C6, all five seeds, the frozen target set, CPU, encoder maximum epochs 30/patience 8 and NP maximum epochs 40/patience 10, and all recorded scientific hyperparameters. Batch size and validation episode count of 64 are inherited from archived code defaults where the old configuration omitted them; checkpoint saving is an explicit runtime choice. Changing device or library build can affect numerical reproducibility. The archived code does not enforce all deterministic PyTorch algorithms, so identical seeds are not a bitwise guarantee.

## 5 Raw-data and other analyses

RELAX preprocessing requires the provider data, archived alignment products and matching preparation configuration. Polar preparation requires controlled local raw data. WESAD has its own reference entry and cannot be inferred from the Safe NP runner. See the data and WESAD notes.

Temperature-calibration refitting would fit source models and was not rerun. Historical P15 and coverage-risk scripts have additional working-directory and restricted-input dependencies; they are inventoried for review rather than advertised as portable commands. No claim is made that omitted exploratory scripts or manuscript figure generation are one-command reproducible here.

## What a successful local run does not establish

A command may execute with newly prepared data and still fail to reproduce the frozen paper inputs. Compare versioned input hashes, target/source indices, seed sets and table definitions before interpreting a numerical difference. Passing code contracts does not resolve publication rights, ethics, retrospective selection or validation reuse.

## 6 Rebuild time-eligible indices without model training

Supply the frozen prepared input and the two authorized historical reference maps, named `target_map_label_available_rebuilt.csv` and `source_map_label_available_rebuilt.csv`, in the reference directory. These maps contain individual/time indices and are not bundled.

```bash
python -B scripts/verify_release_indices.py --data-dir data/restricted/relax/prepared --reference-dir data/restricted/relax/index_audit --out runs/relax_index_audit_new
```

The adapter changes only path setup and adds this explicit CLI. The original index-building and comparison functions are structurally identical. Outputs contain individual/time indices, remain local under ignored `runs/`, and must not be published automatically.
