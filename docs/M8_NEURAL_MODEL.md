# M8 derivative-supervised PyTorch model report

This report records the frozen M8 price-only and derivative-supervised neural experiments. Both
models are offline comparison artifacts; the C++ Monte Carlo engine remains the trusted backend.
M8 does not include ONNX, C++ neural inference, deployment guardrails, or fallback.

Source implementation commit: `PENDING` (finalized mechanically by the committing agent).

## Frozen protocol

- Dataset and held-out protocol: the exact M7 `nre.dataset.v1` labels and accepted splits, with
  checksums `ae034bf46ccb5285` (config), `b2b609903da61cf0` (labels), and `f8468ba180f2eac2`
  (manifest). Accepted counts are 840 train, 178 validation, and 180 test.
- Preprocessing is fitted on train only. Nine inputs are the M7 base features: log spot/strike,
  maturity, volatility, rate, dividend yield, log observations, two Asian-style indicators, and a
  put indicator. Means and scales are stored in each artifact.
- Each MLP has one scalar output representing price divided by strike. There is no Delta head.
  Physical price is `strike * network(scaled_features)`. Delta is PyTorch automatic
  differentiation of that physical price with respect to the unscaled spot tensor. The graph
  therefore includes `log(spot/strike)`, training-only feature scaling, and output rescaling.
- Price-only and derivative-supervised variants use the same four candidates: hidden widths
  `[32,32]` or `[64,64]`, each with weight decay zero or `1e-6`. Both use smooth `tanh`
  activations, float64, Adam at 0.003, batches of 256, StepLR by 0.5 every 100 epochs, at most 400
  epochs, and validation-only early stopping after at least 100 epochs with patience 60.
- Candidate seeds are `2026071608` through `2026071611`. CPU deterministic algorithms and one
  PyTorch training thread are requested. Validation median normalized price error is primary;
  validation p99 error, Delta RMSE, then parameter count break exact ties.
- Price loss is squared error in `price/strike`, divided by the squared training-only standard
  deviation of that target (`0.2096201574`). Derivative loss is squared Delta error divided by the
  squared training-only Delta standard deviation (`0.5930767076`). The derivative-supervised
  objective gives these normalized terms weights one and one; price-only gives Delta weight zero.
- Training and selection functions accept only train and validation rows. Test rows were evaluated
  once after both final checkpoints were frozen; no hyperparameter was changed afterward.

The full predeclared settings are versioned in `python/config/m8-neural-v1.json`, whose FNV-1a-64
checksum is `e148b19107434b52`.

## Selected models and reproducibility

| Variant | Selected candidate | Parameters | Best epoch | Training seconds | Canonical tensor SHA-256 |
|---|---|---:|---:|---:|---|
| Price-only | `[64,64]`, weight decay `1e-6`, seed `2026071611` | 4,865 | 385 | 6.98 | `4eeee5bbce00387b570763644d89fac9081ea5cbedae6aa590797a144605fb5e` |
| Derivative-supervised | `[32,32]`, weight decay `0`, seed `2026071608` | 1,409 | 337 | 7.24 | `3fce199452fb0b1cfc82a915819863e7709146216335245faff52f143e2f089c` |

Measured on 2026-07-16 on an Apple M4 arm64 MacBook Air running macOS 26.5.1. The environment used
Python 3.14.6, NumPy 2.3.5, and PyTorch 2.13.0 on CPU. Apple MPS was not used or evaluated.
`make neural-reproduce` performed a fresh full search and reproduced both selected candidate
records and canonical tensor SHA-256 checksums exactly. Binary `.pt` container checksums are also
recorded but the canonical tensor checksum is the cross-run identity.

## Held-out comparison

Normalized price error is unchanged from M7:
`abs(predicted-reference)/max(reference, 1 currency unit)`. All three models use the same 180
accepted held-out rows and fixed slice definitions.

| Model | Median normalized | p95 normalized | p99 normalized | Maximum normalized | Median absolute | p99 absolute | Delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| M7 polynomial ridge | 0.08905 | 3.40269 | 8.67955 | 12.33523 | 0.92245 | 11.02249 | 0.12094 |
| M8 price-only MLP | 0.05116 | 1.06167 | 1.92373 | 2.49501 | 0.54114 | 3.08700 | 0.09771 |
| M8 derivative-supervised MLP | 0.04851 | 1.13189 | 1.73883 | 2.04656 | 0.46066 | 2.76302 | 0.04951 |

The derivative-supervised model beats the M7 baseline on the predeclared primary held-out metric
and improves Delta RMSE by about 59%. It also has the best p99 normalized and absolute errors. It
does not dominate the price-only model on every metric: its p95 normalized error is 1.13189 versus
1.06167. The useful conclusion is that derivative supervision materially improved derivative
consistency and most summary errors under this protocol, not that it guarantees better behavior
everywhere.

Thirty-eight held-out reference prices are below the fixed one-unit floor. On that slice the
derivative-supervised model has median/p99 normalized error 0.51700/1.96019 and Delta RMSE 0.07089.
Its call slice remains substantially harder than puts (p99 normalized error 1.83880 versus
0.63941), geometric Asians have the worst style-slice p99 at 1.75764, and the rate boundary has the
worst populated boundary p99 at 1.96486. The predeclared maturity boundary has no accepted test
points, so no maturity-boundary claim is possible.

The five stored worst examples show the main failure mode: near-zero calls can receive prices of
roughly one to two currency units, including negative prices and incorrectly signed Delta. These
failures are explicitly retained in `benchmarks/m8-neural-v1.json`; they motivate M9 price-bound,
finite-value, domain, and monotonicity rejection with Monte Carlo fallback. They are not hidden or
interpreted as no-arbitrage/OOD evidence.

## Warmed Python CPU timing

Timing includes Python row tensorization, the PyTorch scalar-price forward pass, and autograd spot
Delta. Each batch has 20 warm-ups and 300 measured repetitions using `time.perf_counter_ns`.
Evaluation observed four PyTorch CPU threads. Empirical p99 is a local repeated-run summary, not a
production tail-latency guarantee.

| Variant | Batch | Median microseconds | Empirical p99 microseconds |
|---|---:|---:|---:|
| Price-only | 1 | 192.85 | 397.64 |
| Price-only | 32 | 260.67 | 536.08 |
| Price-only | 128 | 478.08 | 753.67 |
| Price-only | 180 | 566.73 | 925.80 |
| Derivative-supervised | 1 | 168.90 | 303.73 |
| Derivative-supervised | 32 | 233.69 | 487.48 |
| Derivative-supervised | 128 | 438.92 | 677.03 |
| Derivative-supervised | 180 | 503.23 | 800.57 |

Architecture size differs because validation selected capacity separately under the same search
budget. These Python timings are descriptive only and are not a neural-versus-Monte-Carlo speedup
claim. M9 will measure the exported model under the C++ runtime.

## Verification and artifacts

```bash
make python-test
make neural-train
make neural-reproduce
make neural-evaluate
make check
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release -DNRE_WARNINGS_AS_ERRORS=ON
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

- Versioned state and metadata: `models/m8/price-only-v1.{pt,json}` and
  `models/m8/derivative-supervised-v1.{pt,json}`.
- Machine-readable held-out metrics, exact slices, failures, timing, and checksums:
  `benchmarks/m8-neural-v1.json`.
- Tests prove the scalar output has no Delta head, spot/preprocessing/output chain-rule units are
  correct, preprocessing is train-only, test rows cannot enter selection, seeded training
  reproduces exact tensor state, outputs are finite, and checkpoint reload preserves price and
  Delta exactly.

No ONNX artifact is created in M8. The state dictionaries plus versioned architecture,
preprocessing, feature order, dtype, and output representation are the input to M9 export.
