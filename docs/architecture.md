# Architecture

## Responsibilities and dependency direction

`constraintforge-ir` owns tensor and graph meaning. `constraintforge-reference` is the correctness oracle. `constraintforge-gpu` executes graphs and compiles prepared candidate kernels. `constraintforge-bench` owns measurement, statistics, provenance, and benchmark serialization. `constraintforge-tuner` owns policies, budgets, histories, and independent evaluation. `constraintforge-research` resolves protocols, records one session at a time, validates evidence, and analyzes persisted trials. `constraintforge-cli` presents commands. Dependencies point inward and the tuner reuses M4 timing rather than duplicating it; production AUTO is unchanged.

## Graph and tensors

The graph is an insertion-ordered DAG. A node may reference only an earlier node, making insertion order a deterministic topological order. Nodes contain an inferred `TensorSpec` and one of Input, MatMul, BiasAdd, or ReLU. Graph outputs are explicit.

Tensors are static, non-empty, row-major F32 arrays. Zero dimensions are rejected in M1. Element counts use checked multiplication. MatMul follows `A[M,K] x B[K,N] -> C[M,N]`; BiasAdd broadcasts a rank-1 `[N]` tensor over each row.

## Execution paths

The CPU executor validates the graph, validates every input length, and executes nodes in insertion order with safe indexed vectors. It is deterministic and contains no GPU behavior.

The GPU backend converts the validated IR directly into a minimal execution plan containing node order, operations, specifications, byte requirements, dispatch sizes, and outputs. It validates inputs and adapter limits before allocation. Each graph node owns a distinct storage buffer for the entire execution, avoiding aliasing and premature reuse. Input buffers are initialized from host values; intermediate buffers are never mapped or read by the CPU. Only declared output buffers receive `COPY_SRC` usage and corresponding readback buffers.

Device limits cover both graph tensors and the fixed 16-byte uniform parameter block used by every operator. This matters for graphs whose tensors are smaller than the parameter block: requesting a 4-byte `max_buffer_size` for a 1×1 MatMul makes the later 16-byte parameter buffer invalid even when the adapter supports it. Parameter creation is therefore preflighted against `Device::limits()` before `create_buffer_init`, preserving an actionable typed error rather than allowing an invalid mapped-at-creation buffer to cause a secondary mapped-range panic.

Naive MatMul uses an 8x8 two-dimensional dispatch. BiasAdd and ReLU use 64-element one-dimensional dispatches. Each kernel bounds-checks non-aligned dimensions. One command encoder records operators in insertion order, which establishes dependencies between writes and subsequent reads; one queue submission executes the graph. Mapping begins after submission, device polling has a timeout, and validation scopes surface pipeline or command errors. The M1 standalone MatMul API builds a three-node graph and delegates to this path.

### MatMul kernels and selection

The naive baseline assigns one output element to each invocation in an 8×8 workgroup and reads operands directly from storage buffers. The tiled candidate assigns a 16×16 output tile to a 16×16 workgroup. Each invocation loads one element from A and one from B into two 256-element workgroup arrays. Out-of-range loads become zero. A uniform loop covers `ceil(K/16)` reduction tiles; one barrier publishes each load and a second prevents overwrite before all reads finish. No invocation exits before a barrier.

`MatMulKernelConfig` records the variant, tile dimensions, workgroup dimensions, and exact estimated shared-memory requirement. Validation checks nonzero dimensions, checked invocation and shared-memory arithmetic, per-axis workgroup limits, total invocation count, workgroup storage, dispatch dimensions, tensor buffer size, and storage-binding limits.

Explicit `naive` or `tiled` requests either dispatch that GPU kernel or return a typed error. `auto` deterministically chooses tiled only for shapes with M/K/N ≥16 and M×N ≥4,096 when legal; otherwise it chooses naive. AUTO does not benchmark, use history, search, or execute CPU code. Every MatMul selection and reason is retained in execution metadata.

## Errors and capabilities

Each library exposes typed errors for graph/input validation, resource and dispatch limits, buffer creation, device creation, pipeline validation, execution, and mapping. Buffer-creation diagnostics include the operator, parameter dimensions, buffer label, requested bytes and usage, and relevant requested device limits. Adapter discovery reports the runtime backend, device type, feature set, and compute/buffer limits. Absence of an adapter is a normal explicit error. Hardware tests remain separate and ignored unless intentionally requested; compilation does not count as hardware verification.

## Optimization boundary

M2 established complete unoptimized graph execution and a two-layer MLP correctness fixture. Its naive path remains the M3 baseline.

M3 adds one fixed tiled path but no optimization search. The preliminary comparison command retains its original fresh-invocation host end-to-end meaning.

M4 adds a separate prepared MatMul path. One device owns shared inputs, precompiled naive/tiled pipelines, stable bind groups, per-kernel outputs, and reusable output/query readbacks. Timed invocations never recreate pipelines or workload buffers. Comparisons deterministically alternate variants on the same device. Each invocation is synchronized before the next, preventing accidental overlap and ambiguous attribution.

When supported, only `TIMESTAMP_QUERY` is requested. Compute-pass descriptors write query slots at pass beginning and end; resolution and CPU mapping occur after the pass. Raw tick differences are rejected when reversed and multiplied by the queue timestamp period to produce nanoseconds. No inside-encoder or inside-pass timestamp capability is required. Host prepared end-to-end duration remains separately labeled and includes command recording, submission, synchronization, and readback. Neither metric is called pure arithmetic time.

M4 records all non-warmup samples, nearest-rank p95, sample standard deviation, numerical errors, hardware/software metadata, execution order, and unversioned source fingerprints. It does not alter AUTO selection or implement M5 search.

## M5 configurable kernels and tuning

M5's finite candidate identity contains the variant, BM/BN/BK tile, workgroup dimensions, and shader-generation version. For tiled candidates the workgroup is `[BN, BM, 1]`; every invocation computes one output while linearized invocations cooperatively load all `BM×BK` and `BK×BN` staged values using strided loops. Both barriers remain uniform. Generated static arrays exactly match `4×(BM×BK + BK×BN)` bytes.

Candidate legality is checked against the actual adapter before compilation. The recorded features include padded output/reduction ratios, a data-reuse proxy, shared bytes, and dispatch count. These are transparent engineering proxies, not exact occupancy or performance predictions.

The tuner supports fixed default, deterministic static heuristic, seeded random without duplicate candidates, and resource-aware measurement-guided search. Resource-aware search ranks by fixed static features, measures distinct candidates within the candidate budget, then spends remaining batch budget repeating the current measured best. Every batch is charged, including failures and repeats. Preparation, correctness admission, search timing, decision overhead, and independent final evaluation are separately recorded. Production AUTO is untouched.

