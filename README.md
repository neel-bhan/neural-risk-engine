# Neural Risk Engine

A learning-first, performance-oriented **C++20 derivatives pricing and risk engine** with an
optional PyTorch/ONNX acceleration backend.

The long-term system prices European and Asian options with a trusted Monte Carlo engine, validates
against analytical references, and uses a guarded neural surrogate for eligible batched workloads.
The neural model is an accelerator: invalid, unsafe, or out-of-domain results fall back to Monte
Carlo.

> Status: M8 is complete. The dependency-free C++ engine remains the trusted label backend. A
> scalar-price PyTorch MLP now supplies Delta only through automatic differentiation and is
> compared fairly with the M7 polynomial-ridge baseline. Evidence and limitations are in
> [`docs/M8_NEURAL_MODEL.md`](docs/M8_NEURAL_MODEL.md). ONNX/C++ neural inference and guarded
> fallback remain M9 work.

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
make delta-validation
make dataset-small
make dataset-verify
make python-test
make dataset-m7
make baseline-reproduce
make baseline-evaluate
make neural-train
make neural-reproduce
make neural-evaluate
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' performance
```

`make convergence` runs the external M2 many-seed validation experiment; it is intentionally not
part of the short unit-test target. Its measured report is in
[`docs/M2_CONVERGENCE.md`](docs/M2_CONVERGENCE.md).

`make variance` runs the external M3 matched-draw variance-reduction experiment. Its measured
report is in [`docs/M3_VARIANCE_REDUCTION.md`](docs/M3_VARIANCE_REDUCTION.md).

`make delta-validation` runs the external M4 many-seed pathwise/CRN Delta experiment. Its measured
report is in [`docs/M4_DELTA_VALIDATION.md`](docs/M4_DELTA_VALIDATION.md).

`make performance` runs the release M5 JSONL benchmark. It is intentionally separate from unit
tests; the protocol and a versioned result summary are in
[`docs/M5_PERFORMANCE.md`](docs/M5_PERFORMANCE.md).

`make dataset-small` generates the 60-row M6 evidence dataset under ignored `data/generated/` and
validates it while writing. `make dataset-verify` independently checks its row counts and checksum;
`make dataset-reproduce` requires two fresh runs to be byte-identical. The schema, configurable
larger command, and measured small run are documented in [`data/README.md`](data/README.md).

M7 uses an isolated `.venv` and one pinned numerical dependency. `make python-test` creates the
environment when needed. `make dataset-m7` generates the ignored 1,200-row label set,
`make baseline-reproduce` proves byte-identical deterministic training, and
`make baseline-evaluate` writes the versioned held-out summary. See
[`python/README.md`](python/README.md) and [`docs/M7_BASELINE.md`](docs/M7_BASELINE.md).

M8 layers pinned PyTorch onto the same isolated environment. `make neural-train` performs the
validation-only capacity/regularization search for both ablations, `make neural-reproduce`
verifies exact selected tensor checksums, and `make neural-evaluate` reproduces the frozen
held-out/slice/timing report. See [`docs/M8_NEURAL_MODEL.md`](docs/M8_NEURAL_MODEL.md).

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
python/               Validated dataset loader and offline surrogate toolchain
benchmarks/           Reserved for reproducible benchmark programs
data/                 Versioned dataset schema/config; generated labels are ignored
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
