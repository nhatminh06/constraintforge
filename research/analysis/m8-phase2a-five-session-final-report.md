# ConstraintForge M8 Phase 2A: Prospective Reference-Panel Reliability on Apple M5 Metal

> Manually authored scientific interpretation, 2026-11-15. Numerical results
> come from the qualified deterministic analyzer outputs in
> `m8-phase2a-five-session-final`; this file is not part of that output's
> `SHA256SUMS` manifest.

## Abstract

This prospective descriptive pilot tested whether two independently scheduled
reference panels within each physical session produced the same empirical-best
MatMul candidate and similar complete rankings. Five physical Apple M5/Metal
sessions measured two workloads, ten frozen candidates, two paired panels, and
three blocks per candidate. All 6,000 retained GPU timestamp samples and all
600 correctness admissions passed the frozen validation rules.

For the small `7x19x11` workload, panels agreed on the empirical-best candidate
in 2/5 sessions; rank agreement remained positive but imperfect, and the three
disagreement sessions had positive worst-direction cross-panel regret between
13.3027% and 14.4691%. For `256x64x384`, panels agreed in 5/5 sessions on the
same empirical-best candidate; directional regret was zero throughout and rank
agreement was nearly exact. These observations describe five sessions on one
device and do not establish a true optimum or generalize to other workloads or
hardware. Independent archive restoration and analyzer rebuilding reproduced
the complete deterministic output tree byte-for-byte.

## Research question and design

Reference-panel reliability here means agreement between paired repeated
reference measurements in empirical-best membership, full ranking, top-three
membership, and the observed decision regret incurred when one panel's winner
is evaluated by the other. Neither panel is treated as oracle truth.

The frozen protocol identity is
`d8e80f9a37359dadb86aa798ba5e798f22237e4a7cf0a3ee625eba239005939e`.
Session indices 0--4 used seeds 211, 223, 227, 229, and 233. Each physical
session contained 2 workloads x 2 panels x 10 candidates = 40 candidate-panel
records. Each record aggregated 30 finite positive GPU timestamp samples using
their median, for 120 blocks and 1,200 retained samples per session. Panels,
candidates, blocks, and samples are paired or nested; the independent unit is
the physical session, so the effective independent sample size is five.

All measurements identify Apple M5 Metal, supported timestamp queries, and the
frozen measurement revision
`7b6a74f27d8acd87e7ed3dc10a94629d69b73c3c`, source tree
`89d0025d258c8932fd2ea3c8c9e8ee5a44f11d25a4794309e4b08c9a4a2c5e3e`,
Cargo.lock `2155ddae...78a13`, toolchain `a0746d27...f711`, and implementation
`m8-phase2a-v1`.

## Prespecified methodology

The analyzer used the median of all 30 samples per candidate-panel. Exact equal
medians share best-set membership and average rank; no ties occurred in the
reported best sets. Primary metrics were exact best-set agreement, best-set
Jaccard, Spearman correlation, Kendall tau-b, and top-3 Jaccard. Directional
regret applies panel A's empirical-best set to B and vice versa, relative to the
receiving panel's observed minimum. Secondary diagnostics were relative A/B
latency difference, relative best-to-runner-up separation, maximum within-block
coefficient of variation (CV), CV of block medians, rank changes, and retained
sample extrema. No samples were trimmed, winsorized, imputed, or discarded.

The frozen protocol specifies no inferential interval, p-value, bootstrap,
outlier policy, or significance threshold. Accordingly, uncertainty is shown
as the complete five-session distribution; no post hoc inferential estimate is
introduced.

Candidate aliases used below omit the common `tiled-` prefix and
`-wg...-shader1` suffix: `8x8x8`, `8x16x8`, `16x8x8`, `16x16x8`,
`16x16x16`, `16x16x32`, `32x8x8`, `8x32x8`, `32x8x16`, and `8x32x16`.

## Per-session results

Regret columns are percentages; all other agreement metrics are fractions.

| Session | Workload | Best A | Best B | Exact/Jaccard | Spearman | Kendall | Top-3 Jaccard | A->B regret | B->A regret |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 001 | small | 8x8x8 | 8x8x8 | yes / 1 | 0.684848 | 0.511111 | 0.5 | 0% | 0% |
| 002 | small | 8x8x8 | 8x8x8 | yes / 1 | 0.830303 | 0.688889 | 0.5 | 0% | 0% |
| 003 | small | 8x16x8 | 16x16x8 | no / 0 | 0.781818 | 0.600000 | 0.5 | 14.4691% | 0.5459% |
| 004 | small | 8x16x8 | 8x8x8 | no / 0 | 0.769697 | 0.555556 | 0.5 | 13.6624% | 13.6894% |
| 005 | small | 16x16x8 | 8x16x8 | no / 0 | 0.660606 | 0.511111 | 0.2 | 9.7790% | 13.3027% |
| 001 | large | 16x16x32 | 16x16x32 | yes / 1 | 1.000000 | 1.000000 | 1 | 0% | 0% |
| 002 | large | 16x16x32 | 16x16x32 | yes / 1 | 1.000000 | 1.000000 | 1 | 0% | 0% |
| 003 | large | 16x16x32 | 16x16x32 | yes / 1 | 0.987879 | 0.955556 | 1 | 0% | 0% |
| 004 | large | 16x16x32 | 16x16x32 | yes / 1 | 1.000000 | 1.000000 | 1 | 0% | 0% |
| 005 | large | 16x16x32 | 16x16x32 | yes / 1 | 1.000000 | 1.000000 | 1 | 0% | 0% |

Across small-workload panels, `8x8x8` was empirical best four times,
`8x16x8` three times, and `16x16x8` three times. The large workload selected
`16x16x32` in all ten panels. No empirical-best set contained a tie.

## Best-to-runner-up separation

Values are the frozen relative separation `(runner-up - best) / best`.

| Session | Small A / B | Large A / B |
|---:|---:|---:|
| 001 | 2.1341% / 2.0468% | 2.8832% / 2.6507% |
| 002 | 2.0280% / 2.3294% | 3.2586% / 3.1557% |
| 003 | 0.5459% / 4.1993% | 3.3765% / 2.5112% |
| 004 | 6.4246% / 5.8404% | 3.1967% / 3.2071% |
| 005 | 0.2643% / 8.6174% | 3.0528% / 2.9877% |

For the large workload the runner-up was always `16x16x16`. Small-workload
runner-ups varied. Particularly close observed leaders occurred in Session 003
panel A (0.5459%) and Session 005 panel A (0.2643%); these are descriptive
separations, not tests of statistical distinguishability.

## Timing variability and retained extremes

`Max |A/B diff|` is the largest absolute candidate median relative difference
within that session/workload. CV values are maxima across candidates and both
panels. Every listed extreme remains in the analysis.

| Session | Workload | Max absolute A/B median difference | Max within-block CV | Max block-median CV | Retained sample range (ns) |
|---:|---|---:|---:|---:|---:|
| 001 | small | 18.0272% | 2.4713 | 0.1156 | 19,750--1,206,833 |
| 002 | small | 14.5249% | 2.4366 | 0.1008 | 19,875--999,709 |
| 003 | small | 10.3577% | 0.8613 | 0.0938 | 19,875--118,291 |
| 004 | small | 15.0806% | 2.1416 | 0.2913 | 12,583--761,417 |
| 005 | small | 24.4906% | 1.9770 | 0.2392 | 20,125--2,189,083 |
| 001 | large | 0.5539% | 1.0654 | 0.4828 | 102,708--3,490,875 |
| 002 | large | 0.5937% | 0.6516 | 0.0372 | 142,583--1,724,167 |
| 003 | large | 0.4289% | 0.4620 | 0.0151 | 112,333--645,625 |
| 004 | large | 0.4818% | 0.5826 | 0.0894 | 155,541--1,636,334 |
| 005 | large | 0.4088% | 0.6457 | 0.0037 | 120,250--1,788,542 |

Large individual timestamps and high block CVs occurred in both workloads, but
candidate medians were much closer between panels for the large workload. This
is why retained timing outliers alone do not imply empirical-best disagreement.
No causal mechanism for the contrast was tested.

## Interpretation

The small workload showed incomplete reference-panel reliability: exact winner
agreement occurred in only two sessions, ranking diagnostics varied across all
five, and disagreement sometimes carried material directional regret. The
largest observed directional regret was Session 003 A->B at 0.144691, or
14.4691%. Session 003 also illustrates asymmetry: the reverse regret was only
0.005459, or 0.5459%.

The large workload was substantially more stable in these observations. All
five sessions agreed on the empirical best, all top-three sets matched, and
only Session 003 had any rank-order difference; even there Spearman was
0.987879 and Kendall tau-b 0.955556. Cross-panel regret was zero in every
direction. This does not prove a universal winner or explain why workload
behavior differed.

## Threats to validity

The study has only five independent sessions, one Apple M5 device, two matrix
sizes, and a fixed ten-candidate space. Within-session panels and the 6,000
samples are not independent replications. Power conditions varied: Session 001
reported AC power while the battery reported discharging; Sessions 002--005
reported battery/discharging at 67%, 58%, 48%, and 39%. Thermal telemetry was
missing or limited, quantitative GPU-load telemetry was unavailable, host load
was uncontrolled, and large timing excursions were retained. The selected
estimators summarize this design only; no independently observed true optimum,
population inference, causal explanation, or cross-hardware claim is available.

## Reproducibility and evidence integrity

Authoritative archives and SHA-256 values:

- Session 001 canonical v2: `cb143ebfc0bcbb830897d8e1226fb43cfcaa05a74c36b49fee4f20b469316571`
- Session 002: `4486b4f0f64aa9d4283cca285409f295326b6929f92805ada461b057851a1146`
- Session 003: `87e5606d935bf6d34e72db506587d07a7147a4aa0064d41fff50dbd14b93134a`
- Session 004: `efc85a3cc692f9019b5e64556ea009d257da76e763e25c1beeb9501c903395a1`
- Session 005: `97c53a51235c253b4a3f2442e0d3126e8c03848cf36eb5dff6766642a0a15479`

The preserved defective Session 001 archive remains historical provenance only
(`b232d918...ae9235`) and was not extracted for final analysis. Raw-inventory
hashes for Sessions 001--005 are respectively `1bbf6238...f565`,
`7ce6c839...7663`, `c30fb65b...7a1`, `ec292688...a9b2`, and
`9ff5e6fc...12b`; all 42 entries in each inventory verified.

The hardened c8 validator accepted 5 sessions, 200 candidate-panel records,
600 blocks, and 6,000 samples with zero errors. Additional frozen accounting
was 1,800 warmups, 600 accepted correctness admissions, 8,400 runner-accounted
candidate invocations, 600 completed durable attempts, and zero unresolved
attempts.

The qualified analyzer revision was
`c8cd5cf773b40b48a7a5bf1547eadbb09136d613`. Primary output is
`research/analysis/m8-phase2a-five-session-final`; independent output is
`research/analysis/m8-phase2a-five-session-final-repro`. Both manifests hash to
`1ae7e9808400946b888f482552c0c71d5ea285422d65a32f4472d584df01472d`,
and recursive comparison found no byte difference. The reproduction restored
all inputs from the five permanent archives, cloned c8 from the archived Git
bundle, rebuilt it with `cargo build --locked --release -p
constraintforge-cli` in offline mode, independently revalidated the campaign,
and reran the same complete-campaign analysis. The qualified analyzer produces
tables and machine-readable summaries but no figures; none were invented.

## Conclusion and future work

Within this frozen study, small-workload empirical-best identity and rankings
were less stable between reference panels than the large workload, while the
large workload showed complete winner agreement and zero observed directional
regret. These are bounded observations, not universal reliability claims.
Prospective follow-up could increase independent sessions, add workloads and
hardware, control power and thermal conditions, record quantitative GPU load,
and evaluate improved reference procedures. Those are proposals, not completed
experiments.
