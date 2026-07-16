# M3 arithmetic-Asian variance-reduction experiment

This report records a measured matched-draw comparison of the plain, antithetic, and
geometric-control-variate arithmetic-Asian estimators. It is a focused numerical experiment, not a
general latency or throughput benchmark.

## Estimator definitions

- **Plain:** one effective sample is one discounted arithmetic-Asian payoff from one raw path.
- **Antithetic:** one effective sample is the mean of discounted payoffs from `z` and `-z`, using
  two raw paths and the same observation schedule.
- **Control variate:** a separate pilot stream estimates
  `beta = covariance(arithmetic payoff, geometric payoff) / variance(geometric payoff)`. The pricing
  stream then accumulates `X - beta * (Y - E[Y])`, where `E[Y]` is the M1 discrete
  geometric-Asian analytical price. The pilot and pricing seeds are required to differ, so the
  fitted coefficient is independent of the reported pricing sample.

If the pilot control variance is non-finite or no larger than
`64 * epsilon * max(1, pilot_control_mean^2)`, or if the fitted coefficient is non-finite, the
implementation uses `beta = 0` and reports that the control was not applied. This scale-aware
threshold avoids dividing by variance that is numerically negligible in `double`; zero-volatility
tests exercise the fallback.

## Method and reproducibility

- Source: commit `1364d217a187a1a4d8c030bdaba4de0ff42fc2e6`.
- Date: 2026-07-16.
- Compiler: Apple Clang 17.0.0 (`clang-1700.0.13.5`).
- Flags: C++20, `-O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
- Hardware: Apple M4, arm64 MacBook Air.
- OS: macOS 26.5.1, Darwin 25.5.0.
- Threads: one.
- Effective path counts: 1,000, 4,000, and 16,000.
- Repetitions: 40 per case, estimator, and path count.
- Pricing seeds: 3,000,000 through 3,000,039. Every estimator and path count restarts the same seed
  set. Thus the plain and control pricing samples use identical draws, and each antithetic sample
  uses the same base draws plus their negatives.
- Control pilot: 1,000 paths using separate seeds 4,000,000 through 4,000,039. Pilot work is included
  in raw-path counts and elapsed time.
- Timing: `std::chrono::steady_clock` around each complete pricing call after one untimed warm-up per
  case and estimator.
- Target: full normal-approximation 95% confidence-interval width no greater than 0.50 currency
  units. Grid time-to-target is the measured mean runtime at the first tested effective-path count
  whose mean width meets the target; it is not an interpolated optimum.

The cases were:

| Case | Type | Spot | Strike | Maturity | Observations | Volatility | Rate | Dividend yield |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ATM one-year call | Call | 100 | 100 | 1.0 | 12 | 0.20 | 0.05 | 0.02 |
| OTM two-year put | Put | 110 | 100 | 2.0 | 24 | 0.30 | 0.01 | 0.015 |

Reproduce from a clean build with:

```bash
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' variance
```

## Measured results

Sample variance is recovered from each pricer result as
`standard_error^2 * effective_paths`, then averaged over repetitions. Raw paths for the control
include its 1,000 pilot paths.

### ATM one-year call

| Estimator | Effective paths | Mean raw paths | Mean estimate | Mean sample variance | Mean SE | Mean 95% CI width | Mean runtime (ms) | Target hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain | 1,000 | 1,000 | 5.567764457 | 65.14486487 | 0.2551063611 | 1.000016936 | 0.344684525 | 0% |
| Plain | 4,000 | 4,000 | 5.540556264 | 65.61086602 | 0.1280493605 | 0.5019534932 | 1.357921875 | 45% |
| Plain | 16,000 | 16,000 | 5.527189145 | 65.14312699 | 0.06380529735 | 0.2501167656 | 5.693019850 | 100% |
| Antithetic | 1,000 | 2,000 | 5.533968540 | 17.27157145 | 0.1313523805 | 0.5149013314 | 0.444661575 | 15% |
| Antithetic | 4,000 | 8,000 | 5.533108867 | 17.35834153 | 0.06586810196 | 0.2582029597 | 1.829530200 | 100% |
| Antithetic | 16,000 | 32,000 | 5.516042271 | 17.29461839 | 0.03287606255 | 0.1288741652 | 7.145027150 | 100% |
| Control variate | 1,000 | 2,000 | 5.521234601 | 0.04961386278 | 0.007009625191 | 0.02747773075 | 0.801918775 | 100% |
| Control variate | 4,000 | 5,000 | 5.520212432 | 0.04976934538 | 0.003523738599 | 0.01381305531 | 1.961563500 | 100% |
| Control variate | 16,000 | 17,000 | 5.519922609 | 0.04839137412 | 0.001738771704 | 0.006815985081 | 6.678385475 | 100% |

The control was applied in 100% of runs and its mean pilot coefficient was 1.030860972.

### OTM two-year put

| Estimator | Effective paths | Mean raw paths | Mean estimate | Mean sample variance | Mean SE | Mean 95% CI width | Mean runtime (ms) | Target hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain | 1,000 | 1,000 | 6.276383101 | 100.3915616 | 0.3166741445 | 1.241362647 | 0.666363575 | 0% |
| Plain | 4,000 | 4,000 | 6.238856904 | 99.14517244 | 0.1574220263 | 0.6170943431 | 2.670349950 | 0% |
| Plain | 16,000 | 16,000 | 6.228586502 | 99.37340047 | 0.07880653962 | 0.3089216353 | 10.58782075 | 100% |
| Antithetic | 1,000 | 2,000 | 6.304383162 | 30.53218189 | 0.1746838573 | 0.6847607208 | 0.868204225 | 0% |
| Antithetic | 4,000 | 8,000 | 6.242590886 | 30.08317380 | 0.08671824066 | 0.3399355034 | 3.475741775 | 100% |
| Antithetic | 16,000 | 32,000 | 6.241376652 | 30.32133585 | 0.04353179470 | 0.1706446352 | 13.90141042 | 100% |
| Control variate | 1,000 | 2,000 | 6.234244373 | 0.4666195663 | 0.02156801783 | 0.08454662988 | 1.506002150 | 100% |
| Control variate | 4,000 | 5,000 | 6.237480638 | 0.4605593293 | 0.01072486139 | 0.04204145665 | 3.691646925 | 100% |
| Control variate | 16,000 | 17,000 | 6.236173043 | 0.4626347005 | 0.005376575046 | 0.02107617418 | 12.30095630 | 100% |

The control was applied in 100% of runs and its mean pilot coefficient was 0.9299347892.

## Grid time to target and interpretation

| Case | Plain | Antithetic | Control variate |
|---|---:|---:|---:|
| ATM one-year call | 5.6930 ms at 16,000 | 1.8295 ms at 4,000 | 0.8019 ms at 1,000 |
| OTM two-year put | 10.5878 ms at 16,000 | 3.4757 ms at 4,000 | 1.5060 ms at 1,000 |

At 16,000 effective samples, the measured mean sample-variance ratios of plain to antithetic were
3.77 for the call and 3.28 for the put. The corresponding plain-to-control ratios were 1,346.17 and
214.80. On the tested grid, plain time divided by variance-reduced time at the first target-meeting
row was 3.11 and 3.05 for antithetic sampling, and 7.10 and 7.03 for the control variate.

These measurements support the M3 exit gate for the two stated cases and configuration. They do not
establish universal variance-reduction factors or general low-latency performance. In particular,
the grid time-to-target ratios depend on the selected target, tested path counts, fixed pilot size,
compiler, and hardware.
