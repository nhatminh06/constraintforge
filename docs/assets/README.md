# Presentation assets

These SVGs are descriptive views of the published v0.1.0 source and qualified M8 analysis. They are a subsequent documentation update. They do not alter the historical release tag, scientific report, raw records, or qualified analyzer outputs.

## Architecture

`architecture.svg` follows the responsibilities in [architecture.md](../architecture.md), the seven published crate manifests, and the CLI commands in `crates/constraintforge-cli/src/main.rs`. It shows responsibility and execution boundaries, not every Cargo dependency edge. In particular, `constraintforge-reference` is the CPU crate; production AUTO and offline tuning remain separate.

## M8 figures

From the repository root, regenerate the four SVGs using only the Python standard library:

```sh
python3 docs/assets/generate_figures.py
```

The script reads only the qualified output directory `research/analysis/m8-phase2a-five-session-final/`:

| Figure | Source CSV | Display |
| --- | --- | --- |
| `m8-best-agreement.svg` | `best-set-agreement.csv` | Exact panel agreement, one paired dot per physical session and workload. |
| `m8-directional-regret.svg` | `cross-panel-regret.csv` | Designated A→B and B→A observed regret, expressed as percentages. |
| `m8-rank-repeatability.svg` | `rank-agreement.csv` | Spearman and Kendall tau-b for each paired full ranking. |
| `m8-timing-medians.svg` | `panel-candidates.csv` | Candidate-panel median GPU timestamp in microseconds for fixed `8×8×8` and `16×16×32` candidates. |

The timing figure shows 30-sample candidate-panel medians, paired by physical session. The displayed candidates were chosen after viewing the analysis as illustrative common winners, not as a prespecified comparison or a performance claim. It does not display a pooled 6,000-sample distribution or an inferential interval: samples and blocks are nested inside panels, and only five physical sessions are independent. No raw samples are excluded or recomputed by the figure script. See the [scientific report](../../research/analysis/m8-phase2a-five-session-final-report.md) for the prespecified metrics, retained extremes, and study limits.
