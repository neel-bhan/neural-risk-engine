# M7 simple surrogate baseline report

This report records a deterministic polynomial-ridge comparison model for the later neural
surrogate. It is an offline Python baseline, not a new trusted pricing backend, and its measured
errors expose material failure regions.

Source implementation commit: `efeafd90af7814912567c59688e1e8dc624fdd1f`.

## Predeclared protocol

- Dataset schema: `nre.dataset.v1`; config: `data/config/m7-baseline.cfg`.
- Design: 200 points for each of six style/type combinations, 1,200 rows before quality gates.
- Trusted labels: C++20 Monte Carlo through `nre::price`; 50,000 effective paths for training and
  200,000 for validation/test. Arithmetic Asians also use 10,000/40,000 independent pilot paths.
- Split ownership remains the M6 deterministic 70/15/15 parameter-point rule.
- The loader verifies the config and label checksums recorded by the manifest, schema/version,
  exact fields, finite values, unique ids, disjoint splits, manifest counts, and quality flags.
- Accepted training rows alone fit preprocessing and model coefficients. Validation alone selects
  degree and ridge penalty. The test split remained sealed until the model was fixed, and no model
  or hyperparameter changed after test access.
- Base features are log spot/strike, maturity, volatility, rate, dividend yield, log observations,
  two style indicators, and a put indicator. Training means/scales are versioned, followed by a
  complete polynomial basis.
- Candidate polynomial degrees are 1, 2, and 3; ridge penalties are `1e-8`, `1e-5`, `1e-2`, and
  `1`. The primary selection metric is validation median normalized price error, then p99 error and
  lower degree as deterministic tie breakers.
- Price is fitted as price divided by strike. Delta is a second, separately fitted target and is
  not the derivative of the price output.
- Normalized price error is `abs(predicted-reference)/max(reference, 1.0)` with the one-currency-unit
  floor fixed before evaluation. Absolute errors and the below-floor slice are reported alongside.

## Environment and dataset generation

Measured on 2026-07-16 on an Apple M4 arm64 MacBook Air running macOS 26.5.1. Python was 3.14.6 and
the sole runtime dependency was pinned NumPy 2.3.5. The C++ generator used Apple Clang 17.0.0,
C++20, `-O3 -DNDEBUG`, strict warnings, and 10 requested threads.

The cleanly rebuilt release generator completed label generation in 14.89 seconds wall time
(114.08 seconds aggregate user CPU). It produced 1,200 rows: 840 train, 180 validation, and 180
test. Two validation European-call rows exceeded the predeclared price-SE limit and were retained
but excluded; accepted model counts were 840/178/180. All 800 analytical cross-checks passed.

| Provenance item | FNV-1a-64 |
|---|---|
| Config | `ae034bf46ccb5285` |
| Labels | `b2b609903da61cf0` |
| Manifest | `f8468ba180f2eac2` |
| Model artifact | `121b6fe0ebf71cc1` |

Two fresh training commands selected degree 3 and ridge penalty 1.0 and produced byte-identical
51,750-byte JSON artifacts. The artifact versions preprocessing, polynomial order, coefficients,
the full validation candidate table, and dataset provenance.

## Held-out results

The final test set contains 180 accepted rows, 60 per style and 90 per option type. Thirty-eight
reference prices are below the fixed one-unit denominator floor.

| Metric | Median | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Normalized price error | 0.08905 | 3.40269 | 8.67955 | 12.33523 |
| Absolute price error | 0.92245 | 6.90940 | 11.02249 | 12.33523 |

Delta RMSE is 0.12094. This is the error of a separately fitted Delta output; it is not evidence of
price/Delta derivative consistency.

The error distribution is poor in the tail. The 38 below-floor options have median normalized and
absolute error 1.99773, p99 12.18978, and maximum 12.33523. Calls are the worst type slice by p99
normalized error (11.98536 versus 1.35506 for puts). Geometric Asians are the worst style slice by
p99 normalized error (9.66669), followed by arithmetic Asians (9.26484) and Europeans (6.55560).
This baseline is therefore a real but limited benchmark, not an accuracy target or deployment
candidate.

Outer-10%-of-domain boundary slices, exact slice counts, and their price/Delta metrics are stored in
`benchmarks/m7-polynomial-ridge-v1.json`. The cutoffs come from the predeclared M6 domain rather than
from observed model errors. The rate boundary is the worst populated boundary by p99 normalized
error at 12.19764; the spot/strike boundary follows at 12.05612. The deterministic test split has no
accepted point in the predeclared maturity outer-10% slice, so no maturity-boundary accuracy claim
is possible from this dataset.

## Warmed Python inference timing

Timing uses `time.perf_counter_ns` around feature transformation plus NumPy matrix multiplication,
after 20 warm-ups and across 300 repetitions per batch. The empirical p99 is a run summary, not a
production tail-latency estimate. On the measured machine, the initial run observed:

| Batch | Median microseconds | Empirical p99 microseconds |
|---:|---:|---:|
| 1 | 1,001.29 | 1,323.34 |
| 32 | 869.35 | 1,027.61 |
| 128 | 1,085.42 | 1,370.30 |
| 180 | 1,323.77 | 1,671.06 |

These Python results include an unoptimized column-wise polynomial expansion. They are retained for
fair later protocol comparison and do not support a low-latency or C++ acceleration claim.

## Reproduction

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements-m7.txt
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' dataset-m7
make python-test
make baseline-reproduce
make baseline-train
make baseline-evaluate
```

Generated labels and temporary reproduction artifacts stay ignored. The versioned model is
`models/m7/polynomial-ridge-v1.json`; the comparison-ready metrics are
`benchmarks/m7-polynomial-ridge-v1.json`.
