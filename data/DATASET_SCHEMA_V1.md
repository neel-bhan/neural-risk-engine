# Dataset schema `nre.dataset.v1`

The artifact is a UTF-8, comma-separated `labels.csv` with one header and no quoted fields. Text
values are constrained to comma-free enums/flags. Floating-point values use enough decimal digits
to round-trip a C++ `double`. Missing estimator-specific diagnostics are empty fields, never zero or
`NaN` sentinels.

One row is one unique contract/market parameter point and one Monte Carlo label observation. The
split belongs to that parameter point. M6 never generates stochastic duplicates of a point and
therefore cannot place repeated observations of one point in different splits.

## Field groups

| Group | Columns | Meaning |
|---|---|---|
| Identity and quality | `schema_version`, `parameter_id`, `split`, `included_for_training`, `quality_status`, `quality_flags` | Schema compatibility, point-level split, and auditable gate decision. Rejected rows remain present but consumers must exclude them. |
| Contract | `option_style`, `option_type`, `strike`, `maturity_years`, `observations` | Fixed-strike European, geometric-Asian, or arithmetic-Asian call/put. European observations are one. |
| Market | `spot`, `volatility`, `risk_free_rate`, `dividend_yield` | Currency/model inputs using `docs/CONVENTIONS.md` units. |
| Label policy | `backend`, `estimator`, `label_tier` | Backend is `monte_carlo`; analytical formulas are cross-checks only. Train uses `bulk_training`; validation/test use `heldout_reference`. |
| Price label | `price`, `price_standard_error`, `price_ci_95_lower`, `price_ci_95_upper` | Time-zero price and Monte Carlo diagnostics. |
| Delta label | `delta`, `delta_standard_error`, `delta_ci_95_lower`, `delta_ci_95_upper` | Pathwise spot Delta and its separately accumulated diagnostics. |
| Execution | `effective_paths`, `raw_paths`, `pricing_seed`, `requested_threads`, `active_threads` | M5 execution and sample-count provenance. |
| Control variate | `pilot_paths`, `pilot_seed`, `pilot_active_threads`, `price_control_coefficient`, `price_control_expectation`, `price_control_applied`, `delta_control_coefficient`, `delta_control_expectation`, `delta_control_applied` | Present only for arithmetic-Asian geometric-control labels. Pilot and pricing streams are distinct. |
| Analytical audit | `analytical_price`, `analytical_delta`, `analytical_price_absolute_error`, `analytical_delta_absolute_error` | Present for European and geometric-Asian rows; empty for arithmetic Asians. |

Option styles are `european`, `geometric_asian`, and `arithmetic_asian`; types are `call` and `put`;
splits are `train`, `validation`, and `test`. Quality flags are semicolon-separated. `none` means all
gates passed.

## Manifest

`manifest.json` is deterministic and records:

- schema and source-finalization commit;
- compiler, language mode, build mode, platform, and hardware concurrency;
- config source and its FNV-1a-64 checksum;
- generation command, declared domain, monitoring convention, and point/split design;
- master seed and per-point seed derivation;
- path, pilot-path, thread, and estimator policies;
- predeclared quality thresholds and failure handling;
- split/acceptance/cross-check counts and measured maxima; and
- the `labels.csv` FNV-1a-64 checksum.

FNV-1a-64 is used only to detect regeneration drift. It is not a cryptographic integrity or
security claim.

## Versioning rule

Any removal, renaming, unit change, semantic change, split change, or monitoring change requires a
new schema version and corresponding loader update. Adding a backward-compatible optional field
still requires documenting the addition. Model artifacts must record the exact schema version and
dataset manifest checksum they consume.
