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
  std::uint64_t seed;
};

// Inputs must be finite, with positive spot and non-negative time and volatility.
[[nodiscard]] double exact_gbm_step(double spot, double time_step_years, const MarketState& market,
                                    double normal_draw);

// Evolves one path over equally spaced observations and excludes the initial spot from the mean.
// Throws std::invalid_argument when normal_draws is empty.
[[nodiscard]] double geometric_average_from_gbm_path(const MarketState& market,
                                                     double maturity_years,
                                                     std::span<const double> normal_draws);

[[nodiscard]] double option_payoff(OptionType type, double underlying, double strike) noexcept;

// Contracts and markets must have passed validate before pricing. Each function rejects a contract
// of the wrong style, and both reject configurations with fewer than two paths.
[[nodiscard]] MonteCarloResult price_european_monte_carlo(const OptionContract& contract,
                                                          const MarketState& market,
                                                          const MonteCarloConfig& config);
[[nodiscard]] MonteCarloResult price_geometric_asian_monte_carlo(const OptionContract& contract,
                                                                 const MarketState& market,
                                                                 const MonteCarloConfig& config);

}  // namespace nre
