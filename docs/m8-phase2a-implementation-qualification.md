# M8 Phase 2A implementation qualification

> **PROSPECTIVE INFRASTRUCTURE IMPLEMENTED. REAL HARDWARE EVIDENCE NOT COLLECTED.**

This document records the local, non-GPU engineering qualification of implementation `m8-phase2a-v1`. The prospective protocol is frozen for owner review. Synthetic execution is not physical evidence and supports no scientific inference.

## Acceptance criteria

Readiness requires a valid deterministic five-session plan; exact accounting; production-runner synthetic coverage; destination preflight before measurement; durable pre-invocation intent; ambiguous-remeasurement refusal; committed-block reconciliation; strict source, protocol, plan, adapter, timestamp, sample, and duplicate validation; one-session and five-session analysis gates; byte-identical independent analysis; historical evidence verification; and a clean committed checkpoint before any real run.

The four CLI operations are:

```text
cargo run --release -p constraintforge-cli -- research reference-panel-plan --protocol research/protocols/m8-phase2a-v1.toml --session-index 0
WGPU_BACKEND=metal cargo run --release -p constraintforge-cli -- research reference-panel-run --protocol research/protocols/m8-phase2a-v1.toml --session m8-phase2a-session-001 --session-index 0 --output results/m8-phase2a-metal/session-001
cargo run --release -p constraintforge-cli -- research reference-panel-validate --input <session-or-campaign>
cargo run --release -p constraintforge-cli -- research reference-panel-analyze --input <session-or-campaign> --output <fresh-output> [--qualification-only]
```

The GPU command above is proposed only. It was not executed and requires separate explicit owner authorization after a committed clean checkpoint and immediate hardware/environment preflight.

## Qualified behavior

The exact protocol, methods, budget, seeds, 3/2 A/B ordering balance, alternating workload order, independent SHA-256-keyed schedules, estimator, tie handling, regret definition, schema identities, output layout, environmental fields, and scientific limits are documented in [the research scope](m8-phase2a-research-scope.md). Session, candidate-panel, attempt, plan, protocol, validation, and analysis records are versioned. All scientific tables are deterministically ordered, manifests exclude themselves, and analysis refuses an existing destination.

Synthetic tests exercise the actual session runner through a mock measurement closure; no adapter is initialized. They cover the five plans, counts and candidate coverage, panel stream isolation, ties and undefined correlations, one- and five-session analysis, four-session refusal, byte equality, invalid/missing/duplicate samples, invalid timestamp period, source/protocol mismatch, storage failure with zero calls, overwrite refusal, pre-call intent, post-intent failure, ambiguous refusal, and committed reconciliation. The mock intentionally makes panel B distinct so analysis covers changed measurements without fabricating GPU observations.

Multi-session validation uses Session index 0 as the explicit authoritative measurement source. Every accepted session must have the same Git revision, canonical source-tree SHA-256, complete per-file source manifest, implementation version, Cargo.lock SHA-256, and rust-toolchain SHA-256. Each per-file manifest must reproduce its recorded tree identity. Missing or conflicting measurement provenance rejects both validation and final analysis; the analyzer's own revision remains a separate archival identity.

The final command results, hashes, source identity, historical verification, and READY/NOT READY decision are filled from actual qualification evidence below; no value in this section may be treated as a physical measurement.

## Qualification evidence

- Canonical protocol SHA-256 recorded by plans and sessions: `d8e80f9a37359dadb86aa798ba5e798f22237e4a7cf0a3ee625eba239005939e`; protocol file SHA-256: `7e17031786dc6f3743cb425b52cbe266507662d9bae0a316c161580098f3a723`.
- Experiment-bearing source-tree SHA-256: `89d0025d258c8932fd2ea3c8c9e8ee5a44f11d25a4794309e4b08c9a4a2c5e3e`. This is the qualified Phase 2A content identity before and after checkpointing, not an M7/M8 Phase 1 identity.
- Pre-checkpoint Git base: `d2f4ff6036ceafb285afdb990badd6399dc5164e`. The resulting checkpoint commit identity is recorded by the post-commit qualification report because a commit cannot embed its own SHA.
- Cargo.lock SHA-256: `2155ddae7f843c894b42141d52c5b140e2193f8bf69c2c297758077d4e678a13`.
- rust-toolchain SHA-256: `a0746d273e75bc5f39c640b540058ddb6b0fe6c0627dd3babf91ba3d3224f711`; `rustc 1.87.0`, `cargo 1.87.0`.
- Synthetic tests and workspace regression: PASS. `cargo fmt --all -- --check`, Clippy for all workspace targets with warnings denied, `cargo test --workspace`, `cargo build --workspace`, and `git diff --check` passed. Tests: 81 passed, 0 failed, 4 real-GPU tests ignored. The eight Phase 2A tests passed without initializing an adapter.
- Synthetic reproduction manifest SHA-256: single-session `5550afa358265759c79aa63db25d5ef27cba6b6a552036354306b9e384d961d6`; five-session `68a8010ef804bb401d056c6ce1ead40469fd40250ff135f04cd6eef85151c46f`. Independent output trees were byte-identical.
- M6/M7/M8 Phase 1 integrity: PASS. The 916-entry M6 inventory, 150-entry M7 accepted-input inventory, and both M8 Phase 1 `SHA256SUMS` verified. M6 validation remained 5 sessions/900 complete/0 incomplete; M7 remained 5 sessions/40 search trials/100 reference records. Both M8 Phase 1 manifests remain `e42bfe261006a116f5bcb51eef795f3e176ab439265a5f6d99b3696dd7e47f5a` and byte-identical.

## Preliminary status

M8 Phase 2A implementation: **PASS**. Synthetic end-to-end qualification: **PASS**. Experiment protocol: **FROZEN**. Historical M6/M7/M8 Phase 1: **UNCHANGED**. Real hardware experiments: **NOT RUN**. Scientific inference: **NOT YET AVAILABLE**.

At initial qualification the first GPU session was **NOT READY** because the real runner correctly requires a committed clean source checkpoint and the authorized implementation was uncommitted. The owner subsequently authorized one local checkpoint commit, followed by the same offline gates and immediate Apple M5/Metal/timestamp/environment preflight. The resulting commit identity and final READY/NOT READY decision belong in the post-commit qualification report because a commit cannot truthfully embed its own SHA. No protocol or experimental implementation change is otherwise required.
