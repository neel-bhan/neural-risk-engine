# M5 multithreading and performance report

This report records profile-guided scalar work, multithread scaling, and time to a declared Monte
Carlo confidence-interval target. Results apply only to the stated machine, build, and workloads;
they are not a low-latency or universal scaling claim.

Source implementation commit: `95bece6a2e26d3373ab76e91ab0bde88cc61c5d6`.

## Protocol and metric definitions

- Date: 2026-07-16.
- Machine: Apple M4 MacBook Air, arm64, 10 reported logical/physical CPUs.
- OS: macOS 26.5.1 (25F80), Darwin 25.5.0.
- Compiler: Apple Clang 17.0.0 (`clang-1700.0.13.5`).
- Flags: C++20, `-O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
- Timer: `std::chrono::steady_clock` around one complete backend-neutral pricing call.
- Fixed-work timing: one untimed warm-up followed by seven repetitions with the same master seed.
  Median is the fourth sorted latency. With only seven observations, empirical p99 is the maximum;
  it is useful for run-to-run visibility but not a production tail-latency estimate.
- Master seed: `2026071601`; control pilot seed: `2026071602`; control pilot: 20,000 paths.
- Thread counts: 1, 2, 4, 8, and 10. One is the explicit scalar reference.
- European fixed workload: 1,000,000 effective/raw paths and one maturity observation.
- Arithmetic fixed workload: 250,000 effective samples and 12 post-initial observations. Raw paths
  equal 250,000 for plain, 500,000 for antithetic, and 270,000 for control variate including pilot.
- Throughput: raw path evolutions divided by median latency, reported in millions/second. It does not
  relabel antithetic pairs as one path.
- Speedup: scalar median latency divided by the median latency for the same numerical workload.
- Time to target: one timed call at each grid count `1,000, 4,000, ..., 1,024,000`; the first full
  normal-approximation 95% CI width no greater than 0.10 is reported. It is a grid result, not an
  interpolated optimum. Control-variate time includes its fixed 20,000-path pilot.

Reproduce machine-readable JSONL from a clean release build:

```bash
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' performance \
  > benchmarks/results/m5-performance-apple-m4.jsonl
```

## Threading and reduction design

`thread_count = 1` preserves the M2–M4 scalar master-seed draw order. For multiworker execution,
SplitMix64 deterministically maps `(master seed, worker index)` to distinct `mt19937_64` seed values.
Each worker owns a Box-Muller generator, reusable draw vector, price statistics, and Delta
statistics. Quotient/remainder partitioning gives earlier workers one extra effective sample when
necessary; active workers are capped at effective samples.

Worker records are cache-line aligned and no path-loop allocation or shared accumulator exists.
After joining, worker accumulators are merged in ascending logical-worker order using parallel
Chan/Welford formulas. Thus a fixed thread count is reproducible. A different thread count uses a
different finite draw set by design, so stochastic equality is assessed with documented statistical
tolerances; deterministic zero-volatility cases agree to `1e-14` in automated tests.

## Profile evidence and supported scalar change

The unoptimized scalar M5 implementation was sampled for three seconds while the harness repeated
30 calls. `/usr/bin/sample` top-of-stack counts showed transcendental math and normal generation as
the dominant costs:

| Workload | Samples in pricing call | `exp` | Box-Muller `sincos` | `log` |
|---|---:|---:|---:|---:|
| European plain | 2,182 | 944 | 439 | 289 |
| Arithmetic plain | 2,526 | 731 | 678 | 443 |

These are sampling counts from this run, not universal CPU percentages. The profile also exposed
repeat calculation of the constant discount, drift, and diffusion scale. M5 therefore hoists the
European discount/step constants outside the path loop and Asian step constants outside the
observation loop without changing the GBM model, draws, observation schedule, or estimator.

The same release profile loops measured before/after wall times:

| Profile workload | Work per run | Before | After | Reduction |
|---|---:|---:|---:|---:|
| European plain | 30 × 5,000,000 paths | 6,175.10 ms | 5,081.57 ms | 17.7% |
| Arithmetic plain | 30 × 500,000 paths | 4,956.33 ms | 4,786.16 ms | 3.4% |

Commands were `./build/make/m5_performance --profile-european` and
`--profile-arithmetic`; `/usr/bin/sample <pid> 3` produced local untracked profile files. Automated
fixtures and convergence tests remained unchanged. SIMD is deferred: the measured hot path is led
by scalar library transcendental functions and Box-Muller generation, and M5 has no verified SIMD
implementation that preserves the reference sequence and numerical definitions.

## Measured fixed-work results

| Workload | Threads | Median ms | Empirical p99 ms | M raw paths/s | Speedup |
|---|---:|---:|---:|---:|---:|
| European plain | 1 | 28.881 | 30.519 | 34.625 | 1.00× |
| European plain | 4 | 9.278 | 9.386 | 107.783 | 3.11× |
| European plain | 8 | 6.136 | 6.220 | 162.963 | 4.71× |
| European plain | 10 | 5.202 | 6.550 | 192.218 | 5.55× |
| Arithmetic plain | 1 | 86.642 | 91.176 | 2.885 | 1.00× |
| Arithmetic plain | 4 | 22.687 | 25.224 | 11.020 | 3.82× |
| Arithmetic plain | 8 | 15.441 | 16.682 | 16.191 | 5.61× |
| Arithmetic plain | 10 | 14.572 | 16.318 | 17.156 | 5.95× |
| Arithmetic antithetic | 1 | 109.700 | 128.349 | 4.558 | 1.00× |
| Arithmetic antithetic | 4 | 31.296 | 34.557 | 15.976 | 3.51× |
| Arithmetic antithetic | 8 | 20.204 | 22.791 | 24.748 | 5.43× |
| Arithmetic antithetic | 10 | 18.264 | 20.274 | 27.376 | 6.01× |
| Arithmetic control | 1 | 98.535 | 115.682 | 2.740 | 1.00× |
| Arithmetic control | 4 | 29.180 | 31.423 | 9.253 | 3.38× |
| Arithmetic control | 8 | 19.942 | 20.728 | 13.540 | 4.94× |
| Arithmetic control | 10 | 17.223 | 18.839 | 15.676 | 5.72× |

The harness also emitted two-thread rows; the table keeps the scaling endpoints concise. Scaling is
sublinear and generally flattens from eight to ten threads on this machine. Thread creation is paid
inside each pricing call, so these measurements include orchestration overhead.

## Measured grid time to CI target

| Workload | Threads | First effective paths | Achieved full width | Time (ms) |
|---|---:|---:|---:|---:|
| European plain | 1 | 1,024,000 | 0.05360 | 31.150 |
| European plain | 10 | 1,024,000 | 0.05353 | 7.553 |
| Arithmetic plain | 1 | 256,000 | 0.06234 | 89.617 |
| Arithmetic plain | 10 | 256,000 | 0.06256 | 14.808 |
| Arithmetic antithetic | 1 | 64,000 | 0.06490 | 27.359 |
| Arithmetic antithetic | 10 | 64,000 | 0.06418 | 6.203 |
| Arithmetic control | 1 | 1,000 | 0.02910 | 7.660 |
| Arithmetic control | 10 | 1,000 | 0.02649 | 2.283 |

The geometric control reaches this deliberately coarse target at the first grid point; its runtime
is dominated by the included pilot. Different thread counts use different deterministic streams,
so achieved widths vary slightly. These rows establish the M5 gate for the declared inputs and
hardware only. Portfolio repricing, ML inference, and matched-error backend comparisons remain later
milestones.
