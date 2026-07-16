# M10 portfolio repricing and final project report

This report closes the staged project with a measured many-contract risk workload. The system is a
**C++20 pricing and risk engine with an optional ML acceleration backend**: analytical formulas are
validation references, multithreaded Monte Carlo is trusted for all supported products, and ONNX
Runtime accelerates only candidates accepted by C++ engineering guardrails.

Source implementation commit: `9e803a0a687a94afea5f0de08b23d85971e13277`.

## Frozen synthetic workload

The benchmark constructs 18 positions: European, geometric-Asian, and arithmetic-Asian calls and
puts, with three strikes (80/100/120), maturities (0.50/1.00/1.75 years), volatilities
(0.18/0.25/0.32), and Asian observation counts (12/24/52). These are synthetic positions, not
client data or calibrated market inputs.

Each position is repriced under nine ordered spot/volatility scenarios, producing 162 results in
position-major, scenario-minor order. Seven scenarios stay inside the frozen neural domain. The
other two deliberately set spot to 55% of base or add 0.45 absolute volatility; all 36 such
repricings are valid for Monte Carlo but outside the neural deployment domain. Protocol FNV-1a-64
is `fb71f80210738f0c`. The seed rule starts from `202607160010` and deterministically separates
every pricing and control-pilot stream.

## Reference and matched-error rule

The same reference vector is used for every backend. The 108 European/geometric results use their
analytical price and Delta. The 54 arithmetic Asians use a 131,072-effective-path geometric
control-variate run with a separate 32,768-path pilot. Across those arithmetic references, maximum
price and Delta standard errors were 0.02203 currency units and 0.000612 respectively. This is a
finite Monte Carlo reference, not an exact answer.

Normalized price error is `abs(prediction-reference)/max(reference, 1 currency unit)`, fixed before
measurement. The all-Monte-Carlo comparison searches the predeclared effective-path grid
128/512/2,048/8,192/32,768/131,072. The selected point is the first whose median and p99 normalized
price errors and Delta RMSE are each no worse than the full guarded route on the identical 162
references. Arithmetic candidates use a pilot of `max(64, paths/4)`; every request uses 10 threads.

The selected all-Monte-Carlo point was 512 effective paths with a 128-path arithmetic pilot:

| Full workload | Median normalized price error | p99 normalized price error | Delta RMSE |
|---|---:|---:|---:|
| Guarded ONNX + fallback | 0.02691 | 1.06084 | 0.02671 |
| Matched all-Monte-Carlo | 0.01056 | 0.20576 | 0.01709 |

The Monte Carlo row is better than, rather than merely equal to, all three thresholds. The large
guarded p99 is retained: accepted near-zero/low-price options remain a difficult neural slice.

## Routing and error evidence

The C++ router accepted 95/162 candidates (58.64%) and sent 67/162 (41.36%) to unchanged Monte
Carlo fallback. Rejections were 36 input-domain, 22 price-bound, 2 sampled spot-monotonicity, and 7
sampled volatility-monotonicity. No real run produced a non-finite or runtime failure; deterministic
tests safely inject both and verify fallback, stable ordering, and exact counter accounting.

Accepted-neural median/p99 normalized price error was 0.04683/1.08210 with Delta RMSE 0.03278.
Full routing improved those values to 0.02691/1.06084 and 0.02671. Twenty-nine reference prices were
below the fixed one-unit floor. The five worst rows, plus all style/type and scenario slices, are
stored in `benchmarks/m10-portfolio-v1.json`; the worst normalized error was 1.36330 on an accepted
low-price arithmetic-Asian call. Guardrails catch useful failure classes but do not make the model
uniformly accurate.

## Measured timing and component profile

Measurements used Apple Clang 17, C++20, `-O3 -DNDEBUG` plus strict warnings, ONNX Runtime 1.27.1,
and an Apple M4 MacBook Air reporting 10 hardware threads. End-to-end scopes use two warm-ups and
seven repetitions; the raw neural batch uses 20 warm-ups and 100 repetitions. Timing is
`std::chrono::steady_clock`. With seven samples, empirical p99 is near the observed maximum and is
not a production-tail estimate.

| Scope for all 162 repricings | Median ms | Empirical p99 ms | Repricings/s at median |
|---|---:|---:|---:|
| Raw ONNX price + centered-bump Delta | 0.164 | 0.170 | 987,053 |
| Guardrails + 67 Monte Carlo fallbacks | 15.216 | 15.382 | 10,647 |
| Matched-tolerance all Monte Carlo | 31.557 | 32.897 | 5,134 |

The exact rejected subset alone measured 13.364 ms median, identifying fallback work and per-call
thread creation as the dominant routed cost; raw ONNX inference is about 1% of end-to-end routed
time. The guarded route measured **2.07x** faster than all Monte Carlo at the stated matched-error
rule. This result is specific to this portfolio, path grid, guard policy, and machine—not a
universal speedup or low-latency claim.

Historical M5 engine evidence remains unchanged: on the same machine, the arithmetic control-
variate workload sustained 15.676 million raw path evolutions/s at 10 threads (5.72x its scalar
latency), while arithmetic antithetic reached 27.376 million raw paths/s (6.01x). Those are isolated
fixed-work path benchmarks, not the M10 portfolio workload.

## Reproduce

```bash
# Dependency-free trusted engine and offline ML tests
make clean
make CXX=g++-16 CXXFLAGS='-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' check
make clean
make CXX=clang++ CXXFLAGS='-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' check

# Optional Homebrew ONNX Runtime backend and measured release benchmark
brew install onnxruntime
make clean
make CXX=clang++ CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' \
  onnx-check portfolio-benchmark

# Default and optional CMake/CTest builds
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release -DNRE_WARNINGS_AS_ERRORS=ON
cmake --build build/cmake && ctest --test-dir build/cmake --output-on-failure
cmake -S . -B build/cmake-onnx -DCMAKE_BUILD_TYPE=Release -DNRE_WARNINGS_AS_ERRORS=ON \
  -DNRE_ENABLE_ONNX=ON -DCMAKE_PREFIX_PATH="$(brew --prefix onnxruntime)"
cmake --build build/cmake-onnx && ctest --test-dir build/cmake-onnx --output-on-failure
```

The tracked JSON contains every candidate measurement, style/type and scenario slices, routing
counters, worst examples, result checksums, compiler/runtime metadata, exact timing scopes, and
limitations. Generated datasets remain ignored; the benchmark consumes only the checksum-bound
tracked ONNX artifact.

## Verification matrix

The final worktree passed all of these gates on 2026-07-16:

- GNU GCC 16: warnings-as-errors `make check` and C++ ONNX parity (`onnx-check`).
- Apple Clang 17: the same warnings-as-errors Make and ONNX gates.
- Apple Clang CMake/CTest: 10/10 default tests and 11/11 ONNX-enabled tests in Release with
  `NRE_WARNINGS_AS_ERRORS=ON`.
- Python 3.14 environment: 23/23 loader, baseline, neural/autograd, ONNX, and M10 report-schema
  tests.
- A fresh CMake Release benchmark reproduced protocol, references, candidate selection, metrics,
  routing/slice counters, worst examples, and limitations exactly. Timing was deliberately excluded
  from byte/equality comparison because repeated wall-clock samples are not deterministic.

## Boundaries

No result establishes production readiness, formal no-arbitrage, calibrated confidence, reliable
out-of-distribution accuracy, or a universal latency advantage. The engine assumes Black-Scholes
GBM inputs rather than market calibration; American exercise and stochastic volatility remain out
of scope. Sampled guardrails reject observed violations and route to a trusted numerical backend,
but accepted candidates can still have material model error.
