# Tuning methodology

M5 tunes only the explicit ten-candidate F32 MatMul space documented in the README. Each identity fixes tile dimensions, workgroup dimensions, and shader-generation version. Candidates exceeding actual workgroup, dispatch, or shared-memory limits are rejected before preparation.

All policies see the same legal pool and deterministic inputs. Fixed and heuristic policies consume zero selection measurement batches. Seeded random measures distinct candidates in deterministic shuffled order. Resource-aware search sorts by a fixed score:

```text
padded_output_ratio × padded_reduction_ratio
+ shared_bytes / 65536
+ 1 / reuse_proxy
```

where `reuse_proxy = BM×BN/(BM+BN)`. It measures a bounded distinct prefix, then repeats the currently best measured median while batch and time budgets remain. The score is a transparent prioritization proxy, not an occupancy model.

Candidate correctness is checked against the CPU reference before timing is accepted. GPU compute-pass timestamps are the default metric. Host prepared end-to-end samples are permitted only through an explicit CLI measurement mode and are never mixed with GPU timestamps.

Budgets separately cap distinct candidates, timed batches, and tuning wall time. A repeated batch consumes another unit. Raw samples, failures, preparation time, correctness time, decision overhead, measured dispatch count, and termination reason are preserved under schema `constraintforge-tuning-v1`.

After selection, the chosen candidate receives fresh warmups and samples. Those independent samples cannot change the selection. M5 does not execute held-out experiments or claim that resource-aware search outperforms random search; that belongs to a preregistered M6 campaign on separately identified Metal and Vulkan devices.
