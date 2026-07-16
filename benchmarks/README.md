# Benchmarks and numerical experiments

Reproducible benchmark programs and tracked summary reports will live here. Local raw output belongs
in `benchmarks/results/` and is ignored by Git.

M2 introduces `m2_convergence.cpp`, a numerical validation experiment rather than a performance
benchmark. Run it with `make convergence`; it is intentionally excluded from `make check` so unit
tests remain short and deterministic.

A benchmark report must include commit, compiler and flags, build type, CPU/hardware, thread count,
seed policy, path/contract/scenario counts, warm-up, repetitions, timing method, and definitions of
reported metrics.
