# Neural Risk Engine

A learning-first, performance-oriented **C++20 derivatives pricing and risk engine** with an
optional PyTorch/ONNX acceleration backend.

The long-term system prices European and Asian options with a trusted Monte Carlo engine, validates
against analytical references, and uses a guarded neural surrogate for eligible batched workloads.
The neural model is an accelerator: invalid, unsafe, or out-of-domain results fall back to Monte
Carlo.

> Status: M3 is complete. The single-thread scalar engine prices European, geometric-Asian, and
> arithmetic-Asian calls and puts, with antithetic sampling and an independent-pilot
> geometric-Asian control variate for arithmetic Asians. Delta, threading, ML, ONNX, and general
> performance claims are not implemented yet.

## Why this project is ordered this way

The project begins with financial conventions and deterministic reference prices, then adds a
correct scalar Monte Carlo implementation before concurrency or ML. That order gives every later
optimization and surrogate a trusted result to compare against.

```text
domain + conventions
        |
analytical references
        |
scalar Monte Carlo -> variance reduction -> multithreaded engine
        |                                      |
        +-> generated labels -> baselines -> neural model -> ONNX backend
                                                       |
                                   guardrails + Monte Carlo fallback
                                                       |
                                        portfolio risk benchmark
```

## Build and test

Only an Apple Clang or GCC-compatible C++20 compiler and `make` are needed right now.

```bash
make check
make run
make convergence
make variance
```

`make convergence` runs the external M2 many-seed validation experiment; it is intentionally not
part of the short unit-test target. Its measured report is in
[`docs/M2_CONVERGENCE.md`](docs/M2_CONVERGENCE.md).

`make variance` runs the external M3 matched-draw variance-reduction experiment. Its measured
report is in [`docs/M3_VARIANCE_REDUCTION.md`](docs/M3_VARIANCE_REDUCTION.md).

The equivalent CMake workflow is available once CMake 3.24+ is installed:

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

## Repository map

```text
include/nre/          Public C++ interfaces and domain types
src/                  C++ implementations and CLI entry points
tests/                Deterministic tests
docs/                 Architecture, conventions, roadmap, and next tasks
python/               Reserved for the later ML toolchain
benchmarks/           Reserved for reproducible benchmark programs
```

Start with [the quant primer](docs/QUANT_PRIMER.md), then read
[the conventions](docs/CONVENTIONS.md). The complete staged plan is in
[the roadmap](docs/ROADMAP.md), and the next implementable tasks are in
[the work queue](docs/NEXT_STEPS.md).

## Intended scope

- European, discrete geometric Asian, and discrete arithmetic Asian calls and puts.
- Black-Scholes and geometric-Asian analytical validation.
- Antithetic sampling and a geometric-Asian control variate.
- Profile-guided C++ optimization, multithreading, and memory reuse.
- Monte Carlo-generated price and Delta labels.
- A derivative-supervised PyTorch MLP and a simpler surrogate baseline.
- Batched ONNX Runtime inference in C++ with explicit guardrails and fallback.
- Portfolio repricing across spot and volatility shocks.

American exercise, stochastic volatility, market calibration, a trading system, and formal
arbitrage guarantees are out of scope for the first complete version.

## Metrics (record only after measuring)

- Million Monte Carlo paths per second.
- Speedup over the single-thread scalar baseline.
- Time to reach a target confidence interval.
- Median and p99 normalized pricing error.
- Delta RMSE.
- Neural speedup versus Monte Carlo at a stated matched error tolerance.
- Portfolio repricing latency.
- Neural acceptance and Monte Carlo fallback rates.

Resume bullets remain templates until the benchmark protocol produces real values. See
[the roadmap](docs/ROADMAP.md#resume-ready-finish-line) for the evidence required before filling
them in.
