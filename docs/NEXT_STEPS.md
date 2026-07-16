# M5 work queue — Multithreading and performance engineering

M4's implementation, strict builds, and many-seed numerical exit gate have passed locally. Before
calling M4 complete, replace the source-finalization note in `docs/M4_DELTA_VALIDATION.md` with the
versioned implementation commit. The next implementation milestone is **M5: multithreading and
performance engineering**.

## Task 1 — Freeze a scalar benchmark protocol

- Define raw path evolution, effective sample, throughput, latency, and target-confidence-interval
  timing consistently for plain, antithetic, and control-variate estimators.
- Add a release-only benchmark harness with warm-up, repeated trials, steady-clock timing, and
  machine-readable output.
- Record compiler, flags, hardware, seed policy, path counts, observation counts, and timing method.

## Task 2 — Profile before optimizing

- Profile representative European and arithmetic-Asian scalar workloads.
- Record the commands and dominant costs without claiming that profiler percentages are universal.
- Use the profile to select memory reuse, loop, RNG, or other scalar changes; preserve the exact
  numerical problem and validate before/after results.

## Task 3 — Add deterministic multithreading

- Define master-seed-to-worker stream construction and document reproducibility scope.
- Partition effective samples without sharing RNG or path buffers between workers.
- Use per-thread statistics and a deterministic reduction order; avoid false sharing.
- Cover non-divisible path counts and thread counts larger than useful work.

## Task 4 — Validate and measure scaling

- Check price and Delta invariance across configured thread counts within documented floating-point
  tolerance.
- Measure million raw path evolutions per second, speedup over the one-thread scalar baseline,
  portfolio-style latency, and time to a stated target confidence interval.
- Report scaling limits and profile evidence. Add SIMD only if the measured bottleneck supports it.

## Scope boundary

M5 does not add Python, label generation, PyTorch, ONNX Runtime, neural guardrails, or resume numbers
that have not been measured. The dependency-free scalar path remains the correctness reference.
