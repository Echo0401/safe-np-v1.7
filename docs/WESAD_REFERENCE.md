# Recovered WESAD controlled reference experiments

Five saved CNN/Prototype reference runs were recovered with seeds 2026, 2036, 2046, 2056 and 2066. Each summary contains 15 subjects. Selecting method `Global FUSION Encoder` and `k_shot_per_class=0`, the means across five run means are BA 0.8343057988295015, Macro-F1 0.808809931704028 and AUROC 0.9175362925422688, matching the manuscript's 0.834/0.809/0.918. Prototype rows use 1/3/5 shots per class and are a separate static reference setting, not RELAX C2/C4/C6.

The selected `src/reference/cnn_prototype_loso_v2.py` has SHA-256 `b182bc304ce8c1d1c6a1d9d8a6998d176c5dcddf59339434de36f7d4564a63d7` and matches a pre-existing source archive byte for byte. The five saved configurations and results support the connection to the paper. The historical runs did not record a signed/per-run source-code hash, so this is a supported candidate source, not proof of a cryptographically recorded original executable.

These are CNN/Prototype reference results. Neither the old one-seed NP Shrinkage experiment nor RF baseline folders are Safe NP v1.7. Do not relabel any of them as A6 or as five runs of Safe NP on WESAD.

## Run-independent result check

```bash
python -B scripts/check_wesad_reference.py
python -B tests/check_wesad_cli.py
python -B scripts/run_wesad_reference.py --seed 2026 --out runs/wesad_reference_seed2026_new
```

The first recalculates saved group means with the standard library. The second parses five commands without entering training. The third prints a complete command. Repeat command printing for seeds 2036/2046/2056/2066. Adding `--execute` would train; it was not used in this audit. Place authorized matching `wesad_windows.npz` and `wesad_manifest.csv` under `data/restricted/wesad/prepared/` first.

## Raw preprocessing limitations

The existing preparation summary records 1284 windows, 64 RR values, 32 HRV features and 15 subjects. The historical configuration selected R-peak method `auto`, but all 45 processed subject/state segments actually used the NeuroKit backend. An exact NeuroKit2 version was not recovered. Do not allow `auto` to fall back to SciPy and then claim the same frozen inputs were reconstructed.

`src/preprocessing/prepare_wesad_and_loso.py` is included for inspection, but a full raw reconstruction command is not certified until the detector version and preprocessing configuration are fixed and output hashes/indices are compared. Its `--help` works with the core dependencies; raw NeuroKit processing additionally needs `neurokit2`. This missing historical version is recorded rather than replaced with an invented pin. No raw WESAD data, subject tables, individual predictions or checkpoint files are bundled.

## Leakage and protocol boundaries checked in the reference source

The outer target is excluded from source encoder/scaler fitting; inner validation uses source participants, and the final scaler is refit on all source participants. Within each target's separate state run, support and query pools use a split with a 64-RR gap. Support pools from different state runs are then merged. This does not establish that every query only uses supports occurring earlier in the complete experiment timeline.

The 1/3/5-shot comparisons share the test-sample indices, but support sets are sampled separately using a seed that includes k. They are not nested C2/C4/C6 histories. These properties support a static controlled reference, not the RELAX longitudinal protocol. The recovered configuration points to the WESAD prepared input, and no old pseudo-participant mixed-emotion input is selected by this candidate; a complete raw-source provenance audit was not performed.
