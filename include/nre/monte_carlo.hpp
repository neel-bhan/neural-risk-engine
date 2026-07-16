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
};

struct MonteCarloResult {
  double estimate;
  double sample_standard_error;
  ConfidenceInterval confidence_interval_95;
  std::size_t effective_paths;
  std::size_t raw_paths;
  std::uint64_t seed;
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
  std::size_t pilot_paths;
  std::uint64_t pilot_seed;
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

// Contracts and markets must have passed validate before pricing. Each function rejects a contract
// of the wrong style, and all reject configurations with fewer than two paths.
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
    const OptionContract& contract, const MarketState& market,
    const ControlVariateConfig& config);

}  // namespace nre
