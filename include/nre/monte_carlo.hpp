#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "nre/domain.hpp"
#include "nre/statistics.hpp"

namespace nre {

struct MonteCarloConfig {
  std::uint64_t seed;
  std::size_t path_count;
  // One retains the scalar reference draw order. Zero is invalid.
  std::size_t thread_count{1};
};

struct EstimateDiagnostics {
  double estimate;
  double sample_standard_error;
  ConfidenceInterval confidence_interval_95;
};

struct MonteCarloResult {
  // These three fields are the price diagnostics retained for compatibility with the M2/M3 API.
  double estimate;
  double sample_standard_error;
  ConfidenceInterval confidence_interval_95;
  EstimateDiagnostics delta;
  std::size_t effective_paths;
  std::size_t raw_paths;
  std::uint64_t seed;
  std::size_t requested_threads;
  std::size_t active_threads;
};

struct ControlVariateConfig {
  MonteCarloConfig pricing;
  std::uint64_t pilot_seed;
  std::size_t pilot_path_count;
};

struct ControlVariateResult {
  MonteCarloResult monte_carlo;
  double coefficient;
  double control_expectation;
  bool control_applied;
  double delta_coefficient;
  double delta_control_expectation;
  bool delta_control_applied;
  std::size_t pilot_paths;
  std::uint64_t pilot_seed;
  std::size_t pilot_active_threads;
};

struct PathwiseSample {
  double discounted_payoff;
  double discounted_delta;
};

struct SpotBumpRule {
  // h = max(relative_size * spot, minimum_absolute_size). The centered estimator requires h < spot.
  double relative_size{1.0e-4};
  double minimum_absolute_size{1.0e-6};
};

struct BumpAndRevalueResult {
  EstimateDiagnostics delta;
  std::size_t effective_paths;
  std::size_t raw_paths;
  std::uint64_t seed;
  double spot_bump;
  std::size_t requested_threads;
  std::size_t active_threads;
};

struct ControlVariateBumpAndRevalueResult {
  BumpAndRevalueResult bump_and_revalue;
  double coefficient;
  double control_expectation;
  bool control_applied;
  std::size_t pilot_paths;
  std::uint64_t pilot_seed;
  std::size_t pilot_active_threads;
};

// Inputs must be finite, with positive spot and non-negative time and volatility.
[[nodiscard]] double exact_gbm_step(double spot, double time_step_years, const MarketState& market,
                                    double normal_draw);

// Evolves one path over equally spaced observations and excludes the initial spot from the mean.
// Throws std::invalid_argument when normal_draws is empty.
[[nodiscard]] double geometric_average_from_gbm_path(const MarketState& market,
                                                     double maturity_years,
                                                     std::span<const double> normal_draws);

// Evolves one path over the same equally spaced observations and excludes the initial spot from
// the mean. Throws std::invalid_argument when normal_draws is empty.
[[nodiscard]] double arithmetic_average_from_gbm_path(const MarketState& market,
                                                      double maturity_years,
                                                      std::span<const double> normal_draws);

[[nodiscard]] double option_payoff(OptionType type, double underlying, double strike) noexcept;

[[nodiscard]] double control_variate_adjusted_sample(double target_sample, double control_sample,
                                                     double control_expectation,
                                                     double coefficient) noexcept;

// Computes one discounted payoff and its pathwise spot derivative from the same supplied draws.
// European contracts require one draw; Asian contracts require exactly contract.observations.
// At an exact payoff kink, the derivative uses half of the left/right jump.
[[nodiscard]] PathwiseSample discounted_pathwise_sample(const OptionContract& contract,
                                                        const MarketState& market,
                                                        std::span<const double> normal_draws);

// Returns the scale-aware centered spot bump and rejects a non-finite/non-positive rule or h >= S.
[[nodiscard]] double centered_spot_bump(double spot, const SpotBumpRule& rule);

// Contracts and markets must have passed validate before pricing. Each function rejects a contract
// of the wrong style, and all reject configurations with fewer than two paths or zero threads.
// Requested workers are capped at the effective sample count. Fixed-seed reproducibility includes
// thread_count; changing it intentionally selects a different deterministic set of worker streams.
[[nodiscard]] MonteCarloResult price_european_monte_carlo(const OptionContract& contract,
                                                          const MarketState& market,
                                                          const MonteCarloConfig& config);
[[nodiscard]] MonteCarloResult price_geometric_asian_monte_carlo(const OptionContract& contract,
                                                                 const MarketState& market,
                                                                 const MonteCarloConfig& config);
[[nodiscard]] MonteCarloResult price_arithmetic_asian_monte_carlo(const OptionContract& contract,
                                                                  const MarketState& market,
                                                                  const MonteCarloConfig& config);
// One effective sample is the mean of payoffs from z and -z, so raw_paths is twice
// effective_paths. The reusable draw buffer is negated in place between paired paths.
[[nodiscard]] MonteCarloResult price_arithmetic_asian_antithetic_monte_carlo(
    const OptionContract& contract, const MarketState& market, const MonteCarloConfig& config);

// Fits beta on an independent pilot stream, then prices with target - beta * (control - E[control])
// on the pricing stream. The pilot and pricing seeds must differ and both samples need at least two
// paths. Degenerate control variance produces the plain estimator with control_applied=false.
[[nodiscard]] ControlVariateResult price_arithmetic_asian_control_variate_monte_carlo(
    const OptionContract& contract, const MarketState& market, const ControlVariateConfig& config);

// Independent centered spot bump-and-revalue Delta estimators. Up/down valuations reuse each
// effective sample's normal draws (common random numbers), so their uncertainty is accumulated
// from paired finite-difference samples rather than inferred from two price standard errors.
[[nodiscard]] BumpAndRevalueResult delta_bump_and_revalue_monte_carlo(
    const OptionContract& contract, const MarketState& market, const MonteCarloConfig& config,
    const SpotBumpRule& bump_rule = {});
[[nodiscard]] BumpAndRevalueResult delta_bump_and_revalue_arithmetic_antithetic_monte_carlo(
    const OptionContract& contract, const MarketState& market, const MonteCarloConfig& config,
    const SpotBumpRule& bump_rule = {});
[[nodiscard]] ControlVariateBumpAndRevalueResult
delta_bump_and_revalue_arithmetic_control_variate_monte_carlo(const OptionContract& contract,
                                                              const MarketState& market,
                                                              const ControlVariateConfig& config,
                                                              const SpotBumpRule& bump_rule = {});

}  // namespace nre
