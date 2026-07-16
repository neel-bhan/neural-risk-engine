# Neural Risk Engine

A performance-oriented **C++20 derivatives pricing and risk engine** with an optional PyTorch/ONNX
acceleration backend. The standard-library Monte Carlo path is the trusted implementation; the
neural model is a guarded batch accelerator, never the sole pricing path.

The system prices European and Asian options with a trusted Monte Carlo engine, validates
against analytical references, and uses a guarded neural surrogate for eligible batched workloads.
The neural model is an accelerator: invalid, unsafe, or out-of-domain results fall back to Monte
Carlo.

> Status: M0–M10 are complete. The final 162-repricing synthetic portfolio measured 9.51 ms median
> for guarded ONNX plus fallback versus 23.47 ms for all Monte Carlo at the documented matched-error
> rule on Apple M4. Neural acceptance was 58.6%; every rejection used the trusted engine. See the
> [final report](docs/M10_FINAL_REPORT.md) for scope, p99, error, setup, and limitations.

## Evidence at a glance

| Layer | Measured evidence |
|---|---|
| C++ Monte Carlo | Arithmetic control variate: 15.676M raw paths/s at 10 threads, 5.72x scalar latency; [M5 protocol](docs/M5_PERFORMANCE.md) |
| Neural model | 840 accepted train points; held-out p99 normalized error 1.7388 and Delta RMSE 0.0495 versus ridge 8.6795/0.1209; [M8 report](docs/M8_NEURAL_MODEL.md) |
| ONNX deployment | Float64 dynamic batch; C++ price/Delta parity, reasoned guardrails and fallback; [M9 report](docs/M9_ONNX_DEPLOYMENT.md) |
| Portfolio routing | 18 contracts × 9 shocks; guarded median/p99 9.508/9.944 ms, 2.47x matched-MC median speedup; [M10 report](docs/M10_FINAL_REPORT.md) |

Normalized price error uses a fixed one-currency-unit floor. These are machine/workload-specific
measurements, not production latency, universal speedup, formal no-arbitrage, or OOD guarantees.

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

The trusted engine needs a C++20 compiler and `make`; `make test` stays dependency-free. The full
`make check` also needs Python 3 and creates `.venv` with pinned NumPy/PyTorch/ONNX packages. CMake
3.24+ and ONNX Runtime 1.27.x are optional deployment dependencies.

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
make onnx-export
make onnx-evaluate
make onnx-check
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' onnx-evaluate-cpp
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' portfolio-benchmark
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

M9 adds pinned ONNX export/runtime dependencies in `python/requirements-m9.txt`. The default
`make check` remains independent of the C++ runtime. On Apple Silicon, install the optional C++
runtime with `brew install onnxruntime`; then `make onnx-check` runs cross-language goldens and
`make onnx-evaluate-cpp` reproduces guarded held-out evidence. The latter requires the ignored M7
dataset from `make dataset-m7`.

M10's `make portfolio-benchmark` needs the same Homebrew ONNX Runtime but no generated dataset. It
recreates the tracked benchmark from the frozen portfolio, reference policy, ONNX artifact, seed
rule, path grid, warm-ups, and repetitions. It records raw inference, complete guarded routing,
the exact fallback subset, and matched all-Monte-Carlo as separate timing scopes.

The equivalent CMake workflow is available once CMake 3.24+ is installed:

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

Optional CMake ONNX build on Homebrew:

```bash
cmake -S . -B build/cmake-onnx -DCMAKE_BUILD_TYPE=Release -DNRE_WARNINGS_AS_ERRORS=ON \
  -DNRE_ENABLE_ONNX=ON -DCMAKE_PREFIX_PATH="$(brew --prefix onnxruntime)"
cmake --build build/cmake-onnx
ctest --test-dir build/cmake-onnx --output-on-failure
```

## Repository map

```text
include/nre/          Public C++ interfaces and domain types
src/                  C++ pricing, risk, dataset, and CLI implementations
tests/                Deterministic tests
docs/                 Architecture, conventions, roadmap, and next tasks
python/               Validated dataset loader and offline surrogate toolchain
benchmarks/           Reproducible numerical/performance programs and tracked JSON evidence
data/                 Versioned dataset schema/config; generated labels are ignored
```

Start with [the quant primer](docs/QUANT_PRIMER.md), then read
[the conventions](docs/CONVENTIONS.md). The complete staged plan is in
[the roadmap](docs/ROADMAP.md). Optional follow-up ideas are separated in
[the post-project backlog](docs/NEXT_STEPS.md).

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

## Resume-ready summary

The [roadmap finish line](docs/ROADMAP.md#resume-ready-finish-line) contains three bullets populated
only from versioned M5, M8, and M10 measurements. Keep their Apple M4, synthetic-workload,
normalization, and matched-error scope when using them.
