# Safe NP definitions and evidence boundaries

## Forward definition

The archived code uses a frozen fusion representation and query-dependent, class-wise time-attention prototypes. For the two class prototypes, define

$$e(q)=\cos(h_q,A_1(q))-\cos(h_q,A_0(q)),\qquad d(q)=\tanh(2e(q)).$$

The learned magnitude is $m(q)\in(0,1)$ and the continuous soft gate is $\alpha(q)=0.5\,\mathrm{sigmoid}(\cdot)$. The residual and deployed shift are

$$r(q)=2d(q)m(q),\quad u(q)=\alpha(q)r(q),\quad c(q)=\mathrm{clip}(\gamma u(q),-1,1).$$

For nonzero gamma, the stress probability corresponds to a logit shift by $c$ after the implementation's probability clipping/normalization. The code uses stable two-class log-probability arithmetic. Gamma=0 takes a separate explicit path and copies the stored Global probabilities exactly; the continuous alpha is not a binary fallback switch.

The source gamma grid is 0, 0.25, 0.5, 1, 2, 4, 8. Positive adaptation requires a source balanced-NLL improvement of at least 0.001 over gamma=0. These choices were examined after earlier target results; freezing the files does not turn them into preregistered confirmatory evidence. Of 65 frozen RELAX runs, 15 activated and 50 fell back. The output bound does not guarantee BA/NLL improvement or absence of individual harm.

## Context and leakage checks

C2/C4/C6 are total balanced history sizes 2/4/6, with the most recent 1/2/3 examples per class and identical future queries. History size also changes time span and composition.

The exact original result script explicitly filters by anchor time. The later v18 release requires all three historical times to precede the query anchor: physiological anchor, label availability and RR-window end. Missing/equal/future timestamps are excluded. RELAX V2 label availability maps to IFB submission time where needed. A saved audit reports equality of 513 target query-context rows and 835 source definitions for the frozen input, with 171 unique target queries and no audited three-time violations. This saved audit is not a proof for arbitrary new inputs.

The original V2 Non-stress candidate construction excludes events in both past and future 30-minute neighborhoods. Passing the history checks does not remove this upstream retrospective information. Do not label the whole pipeline strictly online causal.

The outer target is excluded from source fitting and selection. However, NP early stopping and gamma selection reuse the same source inner-validation episode catalog (65/65 folds). This is not independent nested calibration. FineTuning is a comparator with target gradient updates; Safe NP target prediction uses no gradient update.

## Ablation definitions

| ID | Definition | Evidence type |
|---|---|---|
| A0 | Frozen Global probability | Reconstructed from frozen A6 run |
| A1 | Direct attentive NP without Global anchor | Independently matched retraining; broad architecture comparison |
| A2 | Anchored directional residual, no hard bound, gamma=1 | Independently matched retraining |
| A3 | Anchored bounded residual, unconstrained learned direction, gamma=1 | Independently matched retraining |
| A4 | Anchored bounded directional residual, gamma=1 | Frozen A6 counterfactual reconstruction |
| A5 | A4 with raw source-selected gamma, before minimum-improvement activation | Frozen A6 counterfactual reconstruction |
| A6 | Full method with minimum improvement and exact gamma=0 fallback | Frozen formal v1.7 run |

A1 is not a one-factor ablation. Do not independently retrain A4/A5 and describe the outputs as the archived reconstruction experiment.

## Statistical definitions

Average seeds within each participant before inference. RELAX discrimination metrics use 12 eligible participants and probability scores use 13; seeds/windows/query records are not independent participants. Each ablation contains 2565 query×seed×context rows: 855 per context, 13 participants and five seeds.

Main method comparisons use 20,000 participant bootstrap resamples and separate RELAX/Polar 25-comparison Holm families. A0–A6 comparisons use 10,000 resamples and six candidate comparisons within each baseline×context×metric family. BA benefit is candidate minus baseline; NLL/Brier benefit is baseline minus candidate. Some raw tables use the opposite probability-score sign: read the column definition.

Saved RELAX C6 Global/Safe NP BA is 0.511464/0.527188 and NLL is 0.692851/0.695555. Polar C6 BA is 0.553846/0.551923 and NLL is 0.709038/0.713462. No stable average superiority is established. Polar contains 13 participants and is an exploratory external pilot, not an independent confirmatory external test.
