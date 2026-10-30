# Version selection and frozen provenance

## Selected result version

`src/frozen/v1_7_safe_source_calibrated.py` is byte-identical to the script named by the frozen RELAX A6 configuration and the 28 October manuscript audit. SHA-256:

`54e1a4ef5b75c64206c6e80f58df25a922affb93c9df47413bca1a6ba8f9deae`

Files named `NP_v1_7_safe_source_calibrated.py` in several older packages have this same hash. File names and modification dates alone were not used to choose the version. The earlier `NP_RELAX_最终确认包` includes v1.6 results/configuration as well as inputs reused by v1.7; its result directory must not be substituted for formal v1.7 A6.

## Separate release audit derivative

`src/release/safe_np_relax_release_v18.py` is retained separately for the explicit three-timestamp eligibility contract. It differs in multiple core definitions from the original result script, not just file paths. Its inherited filename v18 does not mean a new v1.8 paper model. A prior index audit established equality for the particular frozen RELAX input. This candidate does not claim to have regenerated the frozen predictions with v18.

## Included and excluded code

The formal A1–A3 runner and A0–A6 aggregator are selected from the archived controlled-ablation package. Old v1.3–v1.6 models, exploratory v2 models, pilot runs and manuscript/figure-building scratch scripts are inventoried privately but are not installed as alternative primary algorithms.

The source manifest records original and candidate hashes. Exact algorithm files remain byte-identical. Adaptations are limited to input/output paths, generic help-path examples and test loading locations. The participant-statistics adapter was compared structurally with its source after normalizing only SOURCE/OUT assignments; no statistical function changed. Test-loader and preprocessing adaptations also have local review diffs. A future algorithm change must be a distinct version, not a patch hidden in this packaging work.

Group summaries are copied from the frozen v1.7 analysis. Full inventories with local file locations and private configurations are kept outside this repository in the local review directory.

The archived A1 configuration retains inherited narrative strings such as a Global-anchor prediction description. The explicit ablation_spec and controlled runner define A1 as direct/unanchored NP. Numeric archived settings are preserved; inherited prose is not used to override the implementation.

Git text normalization is disabled in `.gitattributes` so a future clone preserves the reviewed file bytes and hash checks. File encoding remains UTF-8 and Python accepts the retained line endings.
