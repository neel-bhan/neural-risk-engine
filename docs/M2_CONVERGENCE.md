# M2 convergence and confidence-interval coverage

This report records a measured numerical-validation experiment for the scalar M2 Monte Carlo
pricers. It is not a latency or throughput benchmark.

## Method

- Source: `codex/m2-streaming-statistics`, uncommitted M2 working tree based on `c247ef0`.
- Date: 2026-07-16.
- Compiler: Apple Clang 17.0.0 (`clang-1700.0.13.5`).
- Flags: C++20, `-O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
- Hardware: Apple M4, arm64 macOS Darwin 25.5.0.
- Threads: one.
- Contracts: at-the-money one-year European call and 12-observation geometric-Asian call.
- Market: spot 100, volatility 0.20, continuously compounded rate 0.05, dividend yield 0.02.
- Analytical references: the M1 Black-Scholes and discrete geometric-Asian implementations.
- Path counts: 1,000, 4,000, and 16,000 effective paths.
- Seed policy: 200 runs using seeds 1,000,000 through 1,000,199. Each contract and path count
  restarts the same seed set.
- Metrics: RMSE and mean absolute error versus the analytical price, mean sample standard error
  reported by the pricer, and the fraction of normal-approximation 95% intervals containing the
  analytical price.

Reproduce the experiment from a clean build with:

```bash
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' convergence
```

## Measured results

| Contract | Paths | Runs | Analytical price | RMSE | Mean absolute error | Mean reported SE | 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| European call | 1,000 | 200 | 9.227005508 | 0.444789096 | 0.346854062 | 0.438446687 | 94.5% |
| European call | 4,000 | 200 | 9.227005508 | 0.245256638 | 0.197382729 | 0.219168468 | 92.0% |
| European call | 16,000 | 200 | 9.227005508 | 0.103072050 | 0.084249429 | 0.109409210 | 97.0% |
| Geometric-Asian call | 1,000 | 200 | 5.327705787 | 0.240043087 | 0.191165929 | 0.248073060 | 95.5% |
| Geometric-Asian call | 4,000 | 200 | 5.327705787 | 0.124234960 | 0.099257933 | 0.123962243 | 95.0% |
| Geometric-Asian call | 16,000 | 200 | 5.327705787 | 0.063976481 | 0.051141188 | 0.061968848 | 94.0% |

## Interpretation

Quadrupling the path count should halve standard error under independent sampling. The measured
mean reported standard errors do so within 0.1% for both contracts. RMSE also falls at
approximately the expected `1/sqrt(N)` rate, with ordinary finite-repetition variation.

Observed interval coverage ranges from 92% to 97% across 200 runs per row. This is consistent with
an approximate 95% diagnostic at this experiment size; it is not evidence of calibrated model
uncertainty. The analytical comparisons cover both M2 pricers and use many seeds rather than a
single favorable run.
