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

A benchmark report must include commit, compiler and flags, build type, CPU/hardware, thread count,
seed policy, path/contract/scenario counts, warm-up, repetitions, timing method, and definitions of
reported metrics.
