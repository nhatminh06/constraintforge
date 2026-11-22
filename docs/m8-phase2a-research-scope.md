# M8 Phase 2A prospective reference reliability

> **PROSPECTIVE INFRASTRUCTURE IMPLEMENTED. REAL HARDWARE EVIDENCE NOT COLLECTED.**

M8 Phase 2A asks how reproducible empirical-best identities and complete candidate rankings are between two independently scheduled reference panels within one physical GPU session. It is a five-session descriptive pilot on one Apple M5/Metal system, not a confirmatory test, a search-policy experiment, or evidence of a true-best kernel.

## Design and estimands

The independent unit is a separately scheduled physical session. Each session contains the existing small `7x19x11` and large `256x64x384` workloads. For each workload, panels A and B independently measure the same ten legal M5 tiled candidates. Panels, workloads, candidates, blocks, and timestamps are paired or nested; the 20 planned workload/panel combinations are not 20 independent sessions.

Each candidate/panel has three separately prepared blocks, three warmups and ten retained GPU timestamp samples per block. The reused isolated reference-batch path performs a CPU-reference correctness admission during every block preparation. This makes the frozen accounting per session 40 candidate-panel records, 120 blocks, 1,200 samples, 360 warmups, 120 correctness admissions, and 1,680 candidate invocations. Five sessions contain 200 records, 600 blocks, 6,000 samples, 1,800 warmups, 600 admissions, and 8,400 invocations. This deliberately replaces the proposal's provisional 40-admission count with the behavior the qualified runner actually provides.

Candidate latency is the median of all 30 finite positive GPU compute-pass samples, not the median of block medians. Exact-equal medians share average rank and empirical-best membership. Primary outcomes are exact best-set agreement, best-set Jaccard, Spearman correlation, Kendall tau-b, and top-3 Jaccard. Secondary descriptions include A/B latency ratios and relative differences, ranking change, best-to-runner-up separation, within- and between-panel variability, retained extremes, and directional cross-panel regret. Undefined correlations remain undefined.

For every candidate in A's tied best set, A-to-B regret is `(B latency - B minimum) / B minimum`; the reverse is analogous. Outputs retain the minimum and maximum tied-set regret. The lexicographically first tied identity is the frozen operational representative, never a post-hoc favorable choice. Neither panel is oracle truth.

## Scheduling and isolation

Seeds are fixed at `211, 223, 227, 229, 233`. Sessions 1, 3, and 5 are A-first; sessions 2 and 4 are B-first, documenting the unavoidable 3/2 imbalance. Workload order alternates on the same schedule. Panel labels remain identities rather than chronological positions. SHA-256-keyed permutations are generated from session seed, workload, panel, and block before measurement. The two panels have distinct streams, records, warmups, prepared result buffers, and durable attempts; observed latency never changes an order.

The frozen protocol is [`research/protocols/m8-phase2a-v1.toml`](../research/protocols/m8-phase2a-v1.toml). It requires Apple M5, Metal, supported GPU timestamps with a positive finite timestamp period, a committed clean source tree, exact source/protocol identity, and explicitly recorded environment values when available. Missing telemetry stays unavailable. A 45-minute per-session wall cap is a stop, not permission to discard or impute data.

## Persistence, validation, and recovery

Source metadata is captured before the output root is created. The runner validates the resolved plan, creates and write-tests every record directory, then durably writes a `started_or_possibly_executed` attempt before each GPU call. It atomically persists a valid block before marking that attempt `completed_and_durably_recorded`. A committed block can be reconciled after interruption; an intent without a durable block is ambiguous and cannot be remeasured. Existing output, duplicates, missing data, changed source/protocol/panel identity, invalid samples, correctness failure, unsupported hardware, and ordering disagreement stop the run.

Validation and analysis are read-only. Qualification analysis accepts exactly one complete session and labels it `SINGLE-SESSION QUALIFICATION / DESCRIPTIVE ONLY`; final analysis accepts exactly five. Stable outputs are `validation.json`, `resolved-protocol.json`, six CSV tables, `session-summary.json`, `summary.md`, and `SHA256SUMS`. Reproduction means rerunning analysis into a fresh directory and comparing every file byte-for-byte; it is not hardware replication.

## Authorization and stopping boundary

Plan, validation, and analysis do not open a GPU. Every `reference-panel-run` requires separate owner authorization for one named session and a fresh destination. Stop on any preflight, identity, storage, adapter, timestamp, correctness, accounting, ambiguity, wall-cap, or environment concern. Do not amend the frozen protocol after measurements begin; create a new version instead. M6, M7, and M8 Phase 1 evidence remains separate and immutable.

Five sessions on one GPU can assess feasibility and descriptive stability. They cannot establish true-best identity, sample sufficiency, general reliability, panel independence, physical causes of outliers, cross-device validity, or algorithmic superiority. No p-values or population claims are prespecified.
