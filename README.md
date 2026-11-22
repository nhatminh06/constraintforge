# ConstraintForge

ConstraintForge is a Rust-first GPU execution and scheduling research prototype. It represents small F32 neural-network graphs as a typed DAG, checks them with a deterministic CPU reference, and runs them through `wgpu` compute shaders. Its benchmark and offline tuner record raw measurements and enforce explicit tuning budgets. It is a research artifact, not a production inference service or a general-purpose compiler.

## Architecture and scope

![ConstraintForge architecture diagram](<docs/ConstraintForge Architecture Diagram.png>)

The workspace separates IR (`constraintforge-ir`), CPU reference (`constraintforge-reference`), GPU execution and WGSL (`constraintforge-gpu`), prepared benchmarking (`constraintforge-bench`), bounded offline tuning (`constraintforge-tuner`), scientific validation and analysis (`constraintforge-research`), and the CLI. Supported graph operations are rank-2 MatMul, BiasAdd, and ReLU on static row-major F32 tensors. GPU execution keeps intermediate tensors on the device. The naive MatMul kernel remains an independent baseline; tiled candidates and deterministic production AUTO selection are separate from offline tuning.

## Build and CPU-only quick start

Install Rust 1.87 or newer with `rustfmt` and `clippy`. The workspace uses Rust 2024 and a pinned `Cargo.lock`. From the repository root:

```sh
cargo build --locked --workspace
cargo run --locked -p constraintforge-cli -- demo --backend cpu
cargo run --locked -p constraintforge-cli -- demo --backend cpu --example mlp
cargo run --locked -p constraintforge-cli -- tune candidates --m 128 --k 128 --n 128
```

The CPU demo and candidate listing do not request a GPU. Benchmarking and measured tuning require a suitable GPU; `cargo run --locked -p constraintforge-cli -- --help` lists commands and options. See [benchmark methodology](docs/benchmark-methodology.md) and [tuning methodology](docs/tuning-methodology.md) before comparing timings.

For GPU execution, use a Metal-capable Mac or a Linux system with a supported Vulkan adapter and driver. An adapter must meet the selected kernel's limits; timestamp-query support is needed for GPU-timed benchmarks. GPU requests never silently fall back to CPU. This release was qualified with CPU-only checks; the historical M8 physical study used one Apple M5/Metal system. It is not cross-hardware validation.

## Tests

```sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

Hardware-dependent tests are marked ignored and require a separately chosen adapter. Passing the commands above is not evidence of GPU execution.

## Completed M8 study

The prospective M8 Phase 2A pilot measured five physical Apple M5/Metal sessions, two MatMul workloads, ten fixed candidates, and two independently scheduled reference panels per workload and session. The preserved records contain 600 measurement blocks, 6,000 timed samples, and 600 correctness admissions recorded by the M8 runner. The final analysis was independently reproduced byte-for-byte.

For `7×19×11`, the panels agreed on the empirical-best candidate in 2/5 sessions; the largest observed directional cross-panel regret was 14.4691%. For `256×64×384`, they agreed in 5/5 sessions and observed directional regret was zero. These are descriptive results from one device and five independent sessions. Workload diversity is limited; thermal and GPU-load telemetry is incomplete; there was no prespecified inferential uncertainty procedure, independent true-optimum measurement, causal test, or cross-hardware comparison. Historical M8 correctness admissions are runner records, not evidence that the later, unfinished M9 numerical-comparison contract was used.

Read the [qualified report](research/analysis/m8-phase2a-five-session-final-report.md), [frozen protocol](research/protocols/m8-phase2a-v1.toml), and [CPU-only reproduction guide](research/evidence/m8-phase2a/REPRODUCE.md). The report and original final analysis remain byte-for-byte unchanged from private preservation. M9 is unfinished exploratory development and is excluded from this release candidate.

## Limits and status

The graph supports only the operations and tensor type listed above. There is no training, import pipeline, fusion, distributed execution, production AUTO search, or claim of superiority over established GPU libraries. Current release source, M8 measurement source, and qualified analysis source have separate revision identities in the reproduction guide. This is a v0.1 research prototype; publication is pending owner approval and release qualification.

## License

Project-owned code, WGSL, documentation, and reviewed research artifacts are licensed under **MIT OR Apache-2.0**, at your choice. See [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE), and the [research artifact licensing scope](research/RESEARCH_LICENSE.md). Dependencies retain their own licenses and notices; see the [third-party inventory](THIRD_PARTY_LICENSES.md). This candidate distributes source and research data, not compiled binaries.
