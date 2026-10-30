# Safe NP v1.7 reproducibility candidate

Local release candidate for review, prepared on 28 October 2026. No public repository has been created as part of this preparation. A code license has not yet been granted; see [license choices](docs/LICENSE_OPTIONS.md). Author names, repository URL and manuscript DOI must be supplied by the authors before release.

Safe NP applies a bounded, direction-constrained log-odds correction to a frozen source Global classifier. A source-selected fold-level gamma controls activation; gamma=0 returns the stored Global probability exactly. The continuous neural soft gate is a different quantity. The current results do not establish stable average superiority or a clinical safety guarantee.

[中文说明](README.zh-CN.md) · [Validation record](docs/LOCAL_VALIDATION_20261029.md) · [Run instructions](docs/REPRODUCE.md) · [Data access](docs/DATA_ACCESS.md) · [Method and evidence](docs/METHOD_AND_EVIDENCE.md) · [Version selection](docs/VERSIONS.md) · [Release checklist](docs/RELEASE_CHECKLIST.md)

## Repository layout

```text
src/frozen/         Exact archived script that generated the frozen v1.7 A6 run
src/ablations/      Matched A1–A3 training and A0–A6 aggregation
src/release/        Later v18 time-contract audit derivative, not the A6 training script
src/preprocessing/ RELAX V2 and Polar preparation
src/analysis/      Participant statistics with source/output path adapters
configs/frozen/    Archived experiment settings with relative input paths
scripts/           Command printing, statistics, integrity and publication checks
tests/             Data-free contracts exercising the actual archived implementations
results/reference/ Group-level saved tables; no individual predictions or signals
provenance/        Source hashes, adaptations and validation evidence
data/              Input specification only; restricted inputs are excluded
runs/              Generated locally, ignored by Git
```

## Quick checks without study data

Use Python 3.10. The dependency versions in `requirements.txt` match the environment tested for this candidate; they are not a recovered lockfile of the original training environment. An author-run independent Windows Python 3.10 environment completed a clean dependency install, offline checks, and one **minimal** RELAX model-training smoke test on 29 October 2026. This is not a reproduction of all 65 fold-seed runs; see [local validation notes](docs/LOCAL_VALIDATION_20261029.md). Other operating systems and GPU configurations have not been tested.

```bash
python -m venv .venv
# Activate the environment for your operating system, then:
python -m pip install -r requirements.txt
python -B scripts/verify_integrity.py
python -B scripts/check_reference.py
python -B tests/run_smoke.py
python -B scripts/scan_release.py
```

The first two checks need only the Python standard library. Contract tests require the research dependencies and do not optimize a model. The author-run independent Windows test subsequently installed `requirements.txt` and passed all six offline checks listed in the validation notes. Run these checks yourself after cloning; successful tests do not certify data rights or exact numerical retraining reproducibility.

## Reproduction levels

| Level | What is supported | Required inputs |
|---|---|---|
| Offline checks | Integrity, aggregate design checks, gamma fallback, time eligibility and ablation structure | Repository and dependencies |
| Saved-output statistics | RELAX/Polar participant statistics and A0–A6 formal aggregation | Authorized frozen prediction/audit CSVs, supplied locally |
| Model experiments | One author-run minimal RELAX train-and-predict smoke test passed on Windows Python 3.10. **No complete 65-fold retraining or exact results reproduction** was performed in the release preparation | Authorized prepared NPZ/manifests, suitable compute |
| WESAD reference | See the separate WESAD evidence note; do not equate a historical reference model with Safe NP v1.7 | Its own archived code/configuration and prepared data |

To print a complete frozen command without starting training:

```bash
python -B scripts/run_experiment.py --profile relax_a6 --out runs/relax_a6_new
```

`--execute` would start model training and is deliberately absent above. Read [all reproduction steps and limitations](docs/REPRODUCE.md) before using it.

## Interpretation boundaries

C2/C4/C6 mean 1/2/3 historical examples per class for the same future query. The archived runner defaults also include C8, so use the provided profiles. Participants are the inference units after averaging seeds within each participant.

The frozen RELAX data use retrospective ±30-minute Non-stress screening; source validation episodes were reused for NP early stopping and gamma selection. Fixed queries do not isolate a causal sample-count effect. Polar is an exploratory external pilot. The later v18 audit strengthens timestamp eligibility and has evidence of index equality on the frozen RELAX input; it must not silently replace the exact original result script. See [method details](docs/METHOD_AND_EVIDENCE.md).

## Data and privacy

No raw physiological data, questionnaires, participant-level predictions, exact event timestamps, model weights, consent records, private correspondence or recovered chat history are included. Get data from their original providers and place authorized files under `data/restricted/`, which is ignored. Group-level tables are review candidates, not a declaration that all publication rights are settled.
