# Benchmarks and numerical experiments

Reproducible benchmark programs and tracked summary reports will live here. Local raw output belongs
in `benchmarks/results/` and is ignored by Git.

M2 introduces `m2_convergence.cpp`, a numerical validation experiment rather than a performance
benchmark. Run it with `make convergence`; it is intentionally excluded from `make check` so unit
tests remain short and deterministic.

M3 adds `m3_variance_reduction.cpp`, a matched-draw comparison of the plain arithmetic-Asian,
antithetic, and independent-pilot control-variate estimators. Run it with `make variance`; it is
also excluded from `make check`.

M4 adds `m4_delta_validation.cpp`, a many-seed comparison of pathwise Delta with analytical Delta
and common-random-number centered finite differences. Run it with `make delta-validation`; it is
also excluded from `make check`.

M5 adds `m5_performance.cpp`, a release-only JSON Lines harness. It performs one untimed warm-up and
seven timed repetitions for fixed-work throughput/scaling rows, reports the median and the empirical
p99 (the maximum with seven observations), and separately searches effective-path counts
`1,000 * 4^k` through 1,024,000 for the first run whose full 95% CI width is at most 0.10. Throughput
always means **million raw path evolutions per second**: raw equals effective for plain, is twice
effective for antithetic, and includes pilot plus pricing paths for the control variate. The harness
emits raw machine-readable output to stdout; redirect local runs to ignored `benchmarks/results/`.

The M5 workloads are one European plain call and one 12-observation arithmetic-Asian call using
plain, antithetic, and geometric-control-variate estimators. These are isolated contract workloads,
not portfolio scenarios. `--profile-european` and `--profile-arithmetic` provide long scalar runs for
sampling profilers. See `docs/M5_PERFORMANCE.md` for exact commands and measured evidence.

M7 adds `m7-polynomial-ridge-v1.json`, a tracked machine-readable held-out summary rather than a C++
benchmark program. It records the fixed normalized/absolute error metrics, Delta RMSE, style/type and
domain-boundary slices, and warmed Python batch timings. Recreate it with `make baseline-evaluate`
only after regenerating the checksum-bound M7 dataset and deterministic model artifact.

A benchmark report must include commit, compiler and flags, build type, CPU/hardware, thread count,
seed policy, path/contract/scenario counts, warm-up, repetitions, timing method, and definitions of
reported metrics.
