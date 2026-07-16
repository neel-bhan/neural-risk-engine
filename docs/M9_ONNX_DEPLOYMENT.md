# M9 ONNX C++ deployment and guarded fallback report

This report records deployment of the frozen M8 derivative-supervised scalar-price model. The C++
Monte Carlo engine remains the trusted backend. Source implementation commit: `PENDING`.

## Frozen export contract

- Python dependencies: PyTorch 2.13.0, ONNX 1.22.0, ONNX Runtime 1.27.0, NumPy 2.3.5.
- Measured C++ runtime: Homebrew ONNX Runtime 1.27.1 on Apple Silicon.
- Artifact: `nre.onnx.scalar_price.v1`, opset 18, dynamic batch, float64 input/output.
- Input is the exact nine-feature M8 order and train-only means/scales stored in
  `models/m9/scalar-price-v1.json`. The ONNX checksum is `1f60d27245e9e262`.
- Output is one normalized scalar price. Physical price is `strike * output`; there is no Delta
  head. C++ Delta is a centered bump of the same price graph with
  `max(1e-4 * spot, 1e-6)`.

Python parity used all 180 frozen accepted held-out rows and 12 fixed boundary probes covering all
style/type combinations and both domain endpoints. Maximum physical-price difference versus
PyTorch was `5.24e-14`; maximum centered-bump Delta difference versus PyTorch autograd was
`1.75e-08`. Dynamic batches 1, 2, 7, 32, and 192 passed the predeclared tolerances. Exact evidence
is in `benchmarks/m9-onnx-python-v1.json`.

## Guarded C++ routing

The caller supplies a valid Monte Carlo fallback configuration before inference. The router checks
the frozen deployment domain, finite outputs, option-specific price bounds, and sampled spot and
volatility monotonicity. Every rejection keeps request order, calls the unchanged trusted fallback,
and records one reason. These checks are engineering controls, not formal no-arbitrage guarantees,
calibrated confidence, or general OOD detection.

Measured on the unchanged 180-row held-out split:

| Result set | Count | Median normalized price error | p99 normalized price error | Delta RMSE |
|---|---:|---:|---:|---:|
| Accepted neural | 133 | 0.04351 | 0.68518 | 0.03901 |
| Full routed | 180 | 0.02326 | 0.66518 | 0.03353 |

Neural acceptance was 73.89%; Monte Carlo fallback was 26.11% (47 rows): 25 price-bound, 10 spot-
monotonicity, and 12 volatility-monotonicity rejections. There were no held-out domain, non-finite,
or runtime failures. Tests inject and verify every output/runtime rejection reason; invalid
financial inputs are rejected before inference because no meaningful fallback price exists.

The C++ ONNX price-plus-centered-Delta batch of 180 rows measured 138.96 microseconds median and
158.08 microseconds empirical p99 over 300 repetitions after 20 warm-ups. This excludes guardrail
probes and Monte Carlo fallback and is descriptive for the measured Apple M4 machine; it is not a
matched-error neural speedup, portfolio latency, or production-tail claim. A single full guarded
evaluation including 47 fallbacks took 1.091 seconds.

## Reproduction

```bash
make dataset-m7
make python-test
make onnx-export
make onnx-evaluate
brew install onnxruntime
make CXXFLAGS='-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' onnx-check
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' onnx-evaluate-cpp
cmake -S . -B build/cmake-onnx -DCMAKE_BUILD_TYPE=Release -DNRE_WARNINGS_AS_ERRORS=ON \
  -DNRE_ENABLE_ONNX=ON -DCMAKE_PREFIX_PATH="$(brew --prefix onnxruntime)"
cmake --build build/cmake-onnx
ctest --test-dir build/cmake-onnx --output-on-failure
```

Machine-readable reports contain compiler/runtime versions, hardware concurrency, warm-up,
repetitions, batch sizes, checksums, exact metrics, fallback counts, and limitations.
