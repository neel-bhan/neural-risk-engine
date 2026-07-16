#include "nre/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "nre/random.hpp"

namespace nre {
namespace {

void validate_config(const MonteCarloConfig& config) {
  if (config.path_count < 2) {
    throw std::invalid_argument("Monte Carlo pricing requires at least two paths");
  }
}

MonteCarloResult make_result(const StreamingStatistics& statistics,
                             const MonteCarloConfig& config) {
  const auto summary = statistics.summary();
  return {
      .estimate = summary.estimate,
      .sample_standard_error = summary.sample_standard_error,
      .confidence_interval_95 = summary.confidence_interval_95,
      .effective_paths = summary.sample_count,
      .seed = config.seed,
  };
}

}  // namespace

double exact_gbm_step(double spot, double time_step_years, const MarketState& market,
                      double normal_draw) {
  const double variance = market.volatility * market.volatility;
  const double drift =
      (market.risk_free_rate - market.dividend_yield - 0.5 * variance) * time_step_years;
  const double diffusion = market.volatility * std::sqrt(time_step_years) * normal_draw;
  return spot * std::exp(drift + diffusion);
}

double geometric_average_from_gbm_path(const MarketState& market, double maturity_years,
                                       std::span<const double> normal_draws) {
  if (normal_draws.empty()) {
    throw std::invalid_argument("a geometric-average path requires at least one observation");
  }

  const double observation_count = static_cast<double>(normal_draws.size());
  const double time_step_years = maturity_years / observation_count;
  double spot = market.spot;
  double sum_log_spots = 0.0;
  for (const double normal_draw : normal_draws) {
    spot = exact_gbm_step(spot, time_step_years, market, normal_draw);
    sum_log_spots += std::log(spot);
  }
  return std::exp(sum_log_spots / observation_count);
}

double option_payoff(OptionType type, double underlying, double strike) noexcept {
  if (type == OptionType::call) {
    return std::max(underlying - strike, 0.0);
  }
  return std::max(strike - underlying, 0.0);
}

MonteCarloResult price_european_monte_carlo(const OptionContract& contract,
                                            const MarketState& market,
                                            const MonteCarloConfig& config) {
  if (contract.style != OptionStyle::european) {
    throw std::invalid_argument("European Monte Carlo pricing requires a European option");
  }
  validate_config(config);

  NormalGenerator normal_generator(config.seed);
  StreamingStatistics statistics;
  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);

  for (std::size_t path = 0; path < config.path_count; ++path) {
    const double terminal_spot =
        exact_gbm_step(market.spot, contract.maturity_years, market, normal_generator.next());
    statistics.add(discount_factor * option_payoff(contract.type, terminal_spot, contract.strike));
  }

  return make_result(statistics, config);
}

MonteCarloResult price_geometric_asian_monte_carlo(const OptionContract& contract,
                                                   const MarketState& market,
                                                   const MonteCarloConfig& config) {
  if (contract.style != OptionStyle::geometric_asian) {
    throw std::invalid_argument(
        "geometric-Asian Monte Carlo pricing requires a geometric Asian option");
  }
  validate_config(config);

  NormalGenerator normal_generator(config.seed);
  StreamingStatistics statistics;
  std::vector<double> normal_draws(contract.observations);
  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);

  for (std::size_t path = 0; path < config.path_count; ++path) {
    for (double& normal_draw : normal_draws) {
      normal_draw = normal_generator.next();
    }
    const double geometric_average =
        geometric_average_from_gbm_path(market, contract.maturity_years, normal_draws);
    statistics.add(discount_factor *
                   option_payoff(contract.type, geometric_average, contract.strike));
  }

  return make_result(statistics, config);
}

}  // namespace nre
