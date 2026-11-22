# M8 Phase 2A public evidence and CPU-only reproduction

This directory contains the original raw payload bytes from the five qualified private session archives, arranged as `session-001` through `session-005`. It contains no private Git bundle. The frozen protocol is [`../../protocols/m8-phase2a-v1.toml`](../../protocols/m8-phase2a-v1.toml). `EVIDENCE-SHA256SUMS` covers every file in this public evidence directory except itself and this guide.

Source identities are distinct:

- Public release and qualified analysis source: Git revision `c8cd5cf773b40b48a7a5bf1547eadbb09136d613`, selected from its committed tree. The release candidate README and licenses are new public-preparation files.
- Historical measurement source: Git revision `7b6a74f27d8acd87e7ed3dc10a94629d69b73c3c`, source-tree SHA-256 `89d0025d258c8932fd2ea3c8c9e8ee5a44f11d25a4794309e4b08c9a4a2c5e3e` as recorded by the sessions.
- Frozen protocol SHA-256: `d8e80f9a37359dadb86aa798ba5e798f22237e4a7cf0a3ee625eba239005939e` as recorded by the sessions.

The original final scientific report hashes to `d3b76bd70d08b2f582fec9efdb7df389c776e9e40bad6561997300edb37c937f`. Both original primary and independent reproduction `SHA256SUMS` files hash to `1ae7e9808400946b888f482552c0c71d5ea285422d65a32f4472d584df01472d`.

From the repository root, with Rust 1.87 or newer and dependencies available locally, run:

```sh
cd research/evidence/m8-phase2a
shasum -a 256 -c EVIDENCE-SHA256SUMS
cd ../../..
cargo build --locked --offline -p constraintforge-cli
cargo run --locked --offline -p constraintforge-cli -- research reference-panel-validate --input research/evidence/m8-phase2a
cargo run --locked --offline -p constraintforge-cli -- research reference-panel-analyze --input research/evidence/m8-phase2a --output /tmp/constraintforge-m8-reproduction
cmp research/analysis/m8-phase2a-five-session-final/SHA256SUMS /tmp/constraintforge-m8-reproduction/SHA256SUMS
```

The output directory must not already exist. The validator should report 5 valid sessions, 200 candidate records, 600 blocks, and 6,000 timed samples. The analyzer should report 5 sessions and 10 workload pairs; its complete 11-file output tree should match `research/analysis/m8-phase2a-five-session-final` byte-for-byte. The comparison command checks the manifest; compare the entire tree before claiming independent reproduction.

Each `session-XXX-artifacts.sha256` is an unchanged original inventory with historical paths beginning `results/m8-phase2a-metal/`. `EVIDENCE-SHA256SUMS` checks the rearranged public paths. `PRIVATE_ARCHIVE_REFERENCES.json` records original archive SHA-256 values for provenance; the private archives are not part of this public selection. Session 001 uses the canonical-v2 archive. Do not extract the duplicate-member historical Session 001 archive.

This is a descriptive five-session, one-Apple-M5/Metal study of two workloads and ten candidates. Panels and timestamps are nested within sessions. Thermal and quantitative GPU-load telemetry are incomplete. The frozen protocol specified no inferential uncertainty method, and no true optimum or causal effect was measured. The 600 correctness admissions are records from the historical M8 runner. They do not use the later unfinished M9 numerical-comparison evidence contract. No cross-hardware or general performance conclusion follows.
