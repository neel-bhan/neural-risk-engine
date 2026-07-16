# M6 reproducible dataset generation report

This report records the small M6 evidence run. It establishes deterministic generation, split
integrity, provenance, and label-quality behavior; it is not a claim about the eventual training
dataset size or surrogate accuracy.

Source implementation commit: `PENDING`.

## Declared design

- Schema: `nre.dataset.v1`, documented in `data/DATASET_SCHEMA_V1.md`.
- Six contract combinations: European, geometric Asian, and arithmetic Asian, each as call and put.
- Ten points per style/type combination, 60 total.
- Continuous domain: spot and strike 60–140; maturity 0.25–2 years; volatility 0.05–0.60; rate
  -0.02–0.10; dividend yield -0.01–0.08.
- Asian observations: 2–52 post-initial observations on `t_i=iT/m`; `t=0` is excluded.
- Design: both endpoints plus deterministic radical-inverse interior values.
- Split: unique parameter-point index modulo 20 gives 70% train, 15% validation, and 15% test.
- Estimators: plain Monte Carlo for European/geometric Asian; independent-pilot geometric control
  variate for arithmetic Asian. Price and pathwise Delta come from the same backend-neutral call.
- Seed: master `2026071606`; pricing uses SplitMix64 of master plus twice the point index, and the
  pilot uses the adjacent input. Requested threads are four.
- Bulk training labels use 4,096 effective paths and 1,024 pilot paths. Validation/test labels use
  the tighter reference tier with 16,384 effective paths and 4,096 pilot paths.

The full predeclared ranges, tolerances, and build label are versioned in
`data/config/m6-small.cfg`. The larger, configurable preset is `data/config/m6-large.cfg`; it was
not run for this report.

## Quality policy

Every row must have finite price/Delta labels, non-negative standard errors, consistent 95%
confidence intervals, and complete execution metadata. Price and Delta are checked against
style/type bounds with a four-standard-error allowance for finite Monte Carlo noise. Call Delta
must be non-negative and put Delta non-positive within that allowance.

European and geometric-Asian Monte Carlo labels are checked against their analytical price and
Delta. A cross-check passes when absolute error is no greater than the declared absolute floor plus
four reported Monte Carlo standard errors. Training price/Delta SE limits are 1.50/0.10; held-out
limits are tighter at 0.40/0.03. A failing row is retained for audit with a reason and
`included_for_training=false`.

## Reproduction

```bash
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' dataset-small
make dataset-verify
make dataset-reproduce
```

Measured on 2026-07-16 with Apple Clang 17.0.0 on an Apple M4 arm64 MacBook Air (10 reported
hardware threads). The source field remains `PENDING` until the implementation commit exists.

## Measured small-run results

| Measure | Result |
|---|---:|
| Total rows | 60 |
| Train / validation / test | 42 / 9 / 9 |
| Accepted / rejected | 59 / 1 |
| Analytical rows checked | 40 |
| Analytical cross-check failures | 0 |
| Maximum price SE | 1.5185640508 |
| Maximum Delta SE | 0.0150444560 |
| Maximum analytical price absolute error | 2.4266035884 |
| Maximum analytical Delta absolute error | 0.0221412081 |
| Config FNV-1a-64 | `d3ca20eac5b33e67` |
| `labels.csv` FNV-1a-64 | `47b1998f7bc14381` |

The one rejected row is the upper-boundary European call training point. Its price SE was
1.5185640508, above the predeclared 1.50 bulk threshold; it had no analytical, bounds, finite-value,
or Delta failure. It remains in the CSV and is excluded from training. Absolute analytical errors
must be interpreted with their reported sampling errors: all 40 rows passed the predeclared
absolute-floor-plus-four-SE gate.

The independent reproduction command produced byte-identical CSV and manifest artifacts. Generated
artifacts remain ignored and are not committed.
