# Frozen candidate specification: M7 retrospective robustness analysis

## Status

This is a human-readable Phase 0 specification, not an executable protocol. It becomes frozen only when the owner approves a dated revision before result-producing M7 implementation. M6 outcomes were already observed, so every analysis here is retrospective and exploratory.

## Data identity

- Input root: `results/m6-v2-metal`.
- Include only sessions `m6-v2-session-001` through `m6-v2-session-005` that pass the existing validator.
- Require M6-v2 protocol SHA-256 `81296ac7da1448b98ea75e70a6e4404021775d5bbb31ed2011c084104ce3dd22` and source-tree SHA-256 `e7ca6493a5a83129bfdb763c29a7f7e701e294ee233d0c8ac6a440fdeb1a5ef2`.
- Exclude all M6-v1, pilot, resume-qualification, and future records.
- Never edit inputs or existing analysis directories.

## Questions

1. How variable are recorded search batches and selected-candidate evaluations within trials?
2. How stable are exhaustive search ranks and top sets across the five sessions for the same workload and budget?
3. How often do policies select the same candidate across sessions under the same condition?
4. How optimistic is a measured policy's selected search score relative to fresh evaluation of that same candidate?
5. How sensitive are the existing resource-aware/random selected-policy ratios to declared robust evaluation summaries and omission of one session?
6. How do recorded tuning cost and selection agreement vary descriptively with distinct-candidate budget?

No question asks for true unselected-candidate latency, true regret, a causal noise effect, or counterfactual policy replay.

## Inclusion and failure rules

- Retain all finite raw observations, including outliers.
- A session or trial failing the existing validator remains visible and makes the corresponding analysis incomplete; do not silently drop it.
- Non-finite or non-positive timestamps are reported as validation failures and excluded only from a metric whose mathematical definition cannot accept them. Report numerator and denominator before and after exclusion.
- Missing matched conditions produce explicit missing cells; do not impute.
- Failed search batches remain in history and are counted. A metric requiring a valid median reports unavailable for that batch.
- Ties use average ranks for Spearman and Kendall tau-b for rank agreement. Top-k ties crossing the boundary include every tied candidate and report the enlarged set.
- Fixed and heuristic have no search-score or search-margin metric; report not applicable, not zero.

## Frozen metric definitions

For positive samples `x`:

- `median(x)`: ordinary midpoint median after numeric sort.
- `MAD(x) = median(|x_i - median(x)|)`.
- `Q_p(x)`: sort ascending, set `h = (n - 1) * p`, and linearly interpolate between zero-based elements `floor(h)` and `ceil(h)`; `IQR(x) = Q_0.75(x) - Q_0.25(x)`.
- `CV(x) = sample_standard_deviation(x) / mean(x)`; report only when at least two values and positive mean exist.
- `range_ratio(x) = max(x) / min(x)`.
- Candidate search estimate: median of all valid raw search samples allocated to that candidate in that trial, matching the M6 selection estimator.
- Exhaustive rank: ascending rank of the ten per-candidate one-batch medians within one exhaustive trial.
- Top-k set: candidates occupying rank positions through k after the tie rule, for `k in {1, 3}`.
- Winner agreement: modal selected-candidate count divided by five sessions within one workload/budget/policy condition; report the full five IDs.
- Observed selection margin: `(second_lowest_estimate - lowest_estimate) / lowest_estimate` among candidates measured in that trial; unavailable when fewer than two valid candidates exist.
- Selected-score optimism ratio: `selected_search_score / evaluation_median` for measured policies. Ratios below one mean the search estimate was lower than fresh evaluation; this is descriptive, not bias-corrected inference.
- Primary sensitivity ratio: resource-aware summary divided by matched random summary using the same session/workload/budget. Summaries are median (main); 20% symmetric trimmed mean, which removes the two smallest and two largest of ten values (sensitivity); and the one-sample Hodges-Lehmann estimator, the median of the 55 averages `(x_i + x_j) / 2` for `i <= j` (sensitivity).
- Cost: recorded `search_wall_time_ns`, `tuning_wall_time_ns`, and `search_decision_overhead_ns`, never substituted for GPU latency.

## Analyses

### A. Within-trial variation

Report batch CV, MAD, IQR, and range ratio by session/workload/budget/policy, preserving the batch nesting. Separately report the same metrics over each selected candidate's ten evaluation samples. Do not pool individual timestamps for inferential tests.

### B. Exhaustive rank stability

For each workload/budget, report all five exhaustive ranks, every pairwise Spearman rho, Kendall tau-b, top-1 agreement, and top-3 Jaccard overlap. Summarize pairwise metrics by median and full range. Budget conditions are separate trial executions, not independent replications of a budget effect.

### C. Policy winner stability and margin

For each workload/budget/policy, report all five selected IDs, modal agreement, and observed margins where applicable. Compare policies descriptively only; candidate subsets and allocation differ.

### D. Selection optimism

For random, resource-ranked, resource-aware, and exhaustive trials, report all selected-score optimism ratios by condition and session. Stratify by policy and budget. Do not interpret fixed/heuristic as zero optimism and do not claim causal winner's-curse correction.

### E. Existing effect sensitivity

For each of the 20 resource-aware/random cells, report the five paired ratios for all three evaluation summaries. Report geometric mean and full range descriptively. Repeat after omitting each session, yielding five four-session geometric means. Do not add a global confirmatory test or count dispatch samples as replications.

### F. Cost and budget description

Report session-level distributions of the three recorded cost fields by policy and budget alongside winner agreement. Do not infer that budget caused stability or that lower cost compensates for latency without a prespecified utility function.

## Dependence and uncertainty

The independent unit is the session. Workloads, budgets, policies, trials, batches, candidates, and timestamps within a session are dependent to varying degrees. Primary output is the complete set of five session values plus descriptive median and range. Phase 0 specifies no new confidence interval, bootstrap, p-value, or stochastic procedure; therefore no analysis seed applies. Adding one requires an approved specification revision. No candidate-, trial-, batch-, or dispatch-level resampling may be presented as session-level uncertainty.

## Multiple comparisons

No p-values or declarations of statistical significance are planned. Every defined cell and metric is reported; no result-dependent filtering is allowed. If future work adds formal tests, it requires a new approved specification with an explicit family and correction before implementation.

## Reproducibility

- Read-only input access and a new output directory that must not exist.
- Deterministic traversal and stable sorting by session, workload, budget, policy, batch, and candidate ID.
- Machine-readable output containing definitions, counts, missingness, and provenance alongside Markdown/CSV views.
- No random seed because every specified analysis is deterministic.
- Independent rerun must be byte-identical.
- SHA-256 manifest over outputs.
- Tests must cover tied ranks, missing candidates, failed batches, zero/non-finite samples, adaptive repeats, and session-level grouping without pseudoreplication. This resolves the pre-implementation wording conflict with the frozen prohibition on stochastic procedures; no resampling is performed.

## Interpretation boundary

The analysis can establish what is present and stable in the recorded M6 paths. It cannot estimate the latency of unselected candidates, reconstruct unobserved adaptive paths, establish true regret, identify physical causes of variation, validate other GPUs, or convert exploratory sensitivity into a confirmatory M6 finding.
