# v0.1 reviewed release selection

This is a proposed public source snapshot. It contains no private Git history or M9 development. Publication remains pending explicit owner approval. The project owner has confirmed release rights and MIT OR Apache-2.0 terms for project-owned material.

The Rust workspace (`Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, and `crates/`), selected methodology and architecture documents, all eight `research/protocols/` files, and the ten-file M6 pilot fixture come byte-for-byte from committed revision `c8cd5cf773b40b48a7a5bf1547eadbb09136d613`. The M6 fixture and historical protocols are included because existing CPU tests read them. They are not presented as new M8 evidence.

The M8 final report, primary analysis, and independent reproduced analysis come byte-for-byte from the R1-preserved main worktree. Their required SHA-256 values are in the reproduction guide. M8 session raw payloads come byte-for-byte from the canonical-v2 Session 001 archive and Sessions 002–005; the public layout changes paths only. Their original inventories are preserved, and `research/evidence/m8-phase2a/EVIDENCE-SHA256SUMS` checks the public paths. No private source bundle or session archive is included.

This candidate adds a rewritten README, finalized license texts, a research-artifact license scope, retained third-party notices, a complete locked-dependency license inventory, this selection record, and CPU-only reproduction instructions. `RELEASE-SHA256SUMS` is the complete candidate-file inventory excluding itself. All file paths are relative to the candidate root.

Excluded: the uncommitted GPU/CLI edits, all unfinished M9 files, private preservation archives and Git bundles, the original repository's Git history and worktree metadata, build caches, local configuration, and unrelated M6/M7 historical results outside the ten-file test fixture. Historical M8 measurement source `7b6a74f27d8acd87e7ed3dc10a94629d69b73c3c` and qualified analyzer source `c8cd5cf773b40b48a7a5bf1547eadbb09136d613` remain distinct identities. The candidate's new documentation and license texts do not change either historical source identity.
