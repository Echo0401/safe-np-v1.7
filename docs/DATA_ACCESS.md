# Data access and local input contract

The repository contains no raw or individual-level study data. Data access and code licensing are separate decisions. Do not infer permission to redistribute participants' records from the availability of a dataset download or from a statistical `publishable=true` field.

## RELAX

Obtain the versioned dataset through the [RELAX Zenodo DOI](https://doi.org/10.5281/zenodo.18693288). The original authors' [validation-code repository](https://github.com/halmichchristina/longitudinal-stress-dataset) documents the dataset structure and download route. Check the actual record's current access conditions and license. The record license could not be read directly during this packaging audit; no assumed license or mirror is bundled.

The prepared frozen input names are `relax_longitudinal_v2.npz` and `relax_longitudinal_v2_manifest.csv`. Place authorized matching copies in `data/restricted/relax/prepared/`. Retain the original sample order and preprocessing settings; changing them invalidates the connection to frozen predictions.

The V2 preparation script also requires prior EMA/IBI alignment products (`current_ema_ibi_alignment.csv` and `stress_event_ibi_alignment.csv`). Those products are not obtained simply by downloading a prepared NPZ. A historical alignment builder has been inventoried but is not yet promoted to a fully tested portable raw-to-table chain. The supplied preparation entry is:

```bash
python -B src/preprocessing/prepare_relax_longitudinal_v2.py --root data/restricted/relax/raw/RELAXDataset --alignment-dir data/restricted/relax/alignment --out runs/relax_prepared_new
```

This command is conditional and was not run. The archived settings are included in `configs/frozen/relax_preparation_v2.json`; compare every preparation option before running, rather than relying on unchecked defaults. It preserves the retrospective event-screening definition rather than silently replacing it with a different online rule.

## Polar H10

Polar is a privately collected longitudinal pilot. No public download URL or transferable raw-data permission was established. Do not upload raw RR, questionnaires, session timestamps, participant maps, signed consent or data derived from identifiable sessions without the appropriate authorization.

The main prepared inputs are `polar_longitudinal_strict.npz` and `polar_longitudinal_strict_manifest.csv` in `data/restricted/polar/prepared/`. The frozen run contains 13 participants and five seeds. Other files named `Polar_H10_prior_session_only...` must not be assumed identical to this frozen input. Main and collector-flag-free sensitivity cohorts must remain distinct.

```bash
python -B src/preprocessing/polar_qc_prepare.py --root data/restricted/polar/raw --out runs/polar_prepared_new
```

This is a conditional preparation entry; its output requires comparison to the frozen input hash and manifest before any claim of full replication. Structure-only or signal-quality checks are not ethics/consent verification.

## WESAD

The [UCI WESAD entry](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) points to the original provider and states that licensing information is on the linked dataset site. Follow that site's terms and acknowledgement rules; this candidate does not assert a generic UCI license or redistribute WESAD.

WESAD reference experiments and the v1.7 RELAX/Polar method have different roles. Read `WESAD_REFERENCE.md` for the recovered experiment provenance and remaining portability conditions.

## Prepared NPZ and manifest schema

For the v1.7 runner, NPZ fields include `X_rr` (N×64), `X_hrv` (N×32), `y`, `subject_ids` and `anchor_unix_ns`. The manifest carries sample indices, participant codes, labels and anchor times; v18 additionally validates label-availability and RR-end times. These are schema descriptions, not permission to publish those columns.

Keep original input SHA-256 locally and compare them before reusing predictions. The repository's public hashes of code and aggregates are not a substitute for the restricted input manifest.
