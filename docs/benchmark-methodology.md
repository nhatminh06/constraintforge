# Benchmark methodology

M4 measures prepared F32 MatMul without changing kernel selection. The default protocol uses 10 warmups and 30 measured invocations per kernel; both values are bounded CLI inputs. Warmups are synchronized but excluded from raw samples and statistics. Comparison order alternates naive and tiled, with the seed determining which starts.

## Measurement scopes

- **Preparation wall time** includes adapter/device setup, supported-feature negotiation, pipeline compilation, input upload, buffer allocation, and bind-group/query creation.
- **Prepared invocation end-to-end wall time** uses monotonic host time around command recording, submission, GPU execution, synchronization, query resolution/readback, and output readback.
- **GPU compute-pass duration** exists only when `TIMESTAMP_QUERY` is supported and executed. Beginning/end compute-pass ticks are resolved after the measured pass and multiplied by `Queue::get_timestamp_period()` to obtain nanoseconds. It includes GPU-side pass overhead and is not pure shader arithmetic time.

CPU submission overhead is not inferred by subtracting unlike clocks. Samples execute one synchronized invocation at a time. Pipelines and workload buffers are reused.

## Workloads and correctness

Inputs come from a fixed-seed deterministic 64-bit linear congruential generator and require no datasets. The smoke suite contains M/K/N shapes `2/2/2`, `15/17/13`, `31/33/29`, and `64/64/64`; explicit CLI shapes are never changed silently. Each tensor is capped at 1 GiB before GPU allocation.

Before warmups, every requested kernel is compared with the CPU reference using `|actual-reference| <= 1e-5 + 1e-4*|reference|`. Non-finite values fail. Performance samples are rejected if correctness fails.

## Statistics and results

JSON schema `constraintforge-benchmark-v1` preserves raw host nanoseconds and optional raw GPU compute-pass nanoseconds. Summaries include count, minimum, maximum, mean, median, nearest-rank p95 (`ceil(0.95*n)`, unavailable for one sample), and Bessel-corrected sample standard deviation (unavailable for one sample). No outliers are removed.

Results include exact shape, seed, kernel resources, dispatch, protocol, adapter/limits, timestamp capability/period, Rust/Cargo/wgpu/platform/build metadata, and source provenance. Without Git, provenance says `unavailable/unversioned` and includes an FNV-1a fingerprint over the root `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, and every `.rs`, `.toml`, and `.wgsl` file under `crates/`, sorted by relative path. This fingerprint is not a Git commit or cryptographic identity.

Output uses a same-directory temporary file and a no-overwrite hard link. Existing result paths are rejected. Metal and Vulkan artifacts must remain separately identified; cross-device differences cannot be attributed to kernels alone.
