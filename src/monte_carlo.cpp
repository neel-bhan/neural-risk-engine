#include "nre/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "nre/analytics.hpp"
#include "nre/random.hpp"

namespace nre {
namespace {

using PathAverageFunction = double (*)(const MarketState&, double, std::span<const double>);

// A variance below this scale cannot support a stable control coefficient in double precision.
constexpr double kControlVarianceRelativeTolerance =
    64.0 * std::numeric_limits<double>::epsilon();

struct AsianAverages {
  double arithmetic;
  double geometric;
};

struct DiscountedAsianPayoffs {
  double arithmetic;
  double geometric;
};

void validate_config(const MonteCarloConfig& config) {
  if (config.path_count < 2) {
    throw std::invalid_argument("Monte Carlo pricing requires at least two paths");
  }
}

MonteCarloResult make_result(const StreamingStatistics& statistics, const MonteCarloConfig& config,
                             std::size_t raw_paths) {
  const auto summary = statistics.summary();
  return {
      .estimate = summary.estimate,
      .sample_standard_error = summary.sample_standard_error,
      .confidence_interval_95 = summary.confidence_interval_95,
      .effective_paths = summary.sample_count,
      .raw_paths = raw_paths,
      .seed = config.seed,
  };
}

std::size_t checked_path_sum(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first) {
    throw std::invalid_argument("Monte Carlo raw path count overflows size_t");
  }
  return first + second;
}

void fill_normal_draws(NormalGenerator& normal_generator, std::span<double> normal_draws) {
  for (double& normal_draw : normal_draws) {
    normal_draw = normal_generator.next();
  }
}

template <typename Observer>
void for_each_gbm_observation(const MarketState& market, double maturity_years,
                              std::span<const double> normal_draws, Observer&& observer) {
  if (normal_draws.empty()) {
    throw std::invalid_argument("an Asian path requires at least one observation");
  }

  const double observation_count = static_cast<double>(normal_draws.size());
  const double time_step_years = maturity_years / observation_count;
  double spot = market.spot;
  for (const double normal_draw : normal_draws) {
    spot = exact_gbm_step(spot, time_step_years, market, normal_draw);
    observer(spot);
  }
}

AsianAverages asian_averages_from_gbm_path(const MarketState& market, double maturity_years,
                                           std::span<const double> normal_draws) {
  double sum_spots = 0.0;
  double sum_log_spots = 0.0;
  for_each_gbm_observation(market, maturity_years, normal_draws, [&](double spot) {
    sum_spots += spot;
    sum_log_spots += std::log(spot);
  });
  const double observation_count = static_cast<double>(normal_draws.size());
  return {
      .arithmetic = sum_spots / observation_count,
      .geometric = std::exp(sum_log_spots / observation_count),
  };
}

DiscountedAsianPayoffs discounted_asian_payoffs(const OptionContract& contract,
                                                const MarketState& market,
                                                double discount_factor,
                                                std::span<const double> normal_draws) {
  const auto averages =
      asian_averages_from_gbm_path(market, contract.maturity_years, normal_draws);
  return {
      .arithmetic = discount_factor *
                    option_payoff(contract.type, averages.arithmetic, contract.strike),
      .geometric = discount_factor *
                   option_payoff(contract.type, averages.geometric, contract.strike),
  };
}

MonteCarloResult price_asian_monte_carlo(const OptionContract& contract,
                                         const MarketState& market,
                                         const MonteCarloConfig& config,
                                         OptionStyle required_style,
                                         const char* style_error,
                                         PathAverageFunction path_average) {
  if (contract.style != required_style) {
    throw std::invalid_argument(style_error);
  }
  validate_config(config);

  NormalGenerator normal_generator(config.seed);
  StreamingStatistics statistics;
  std::vector<double> normal_draws(contract.observations);
  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);

  for (std::size_t path = 0; path < config.path_count; ++path) {
    fill_normal_draws(normal_generator, normal_draws);
    const double average = path_average(market, contract.maturity_years, normal_draws);
    statistics.add(discount_factor * option_payoff(contract.type, average, contract.strike));
  }

  return make_result(statistics, config, config.path_count);
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
  const double observation_count = static_cast<double>(normal_draws.size());
  double sum_log_spots = 0.0;
  for_each_gbm_observation(market, maturity_years, normal_draws, [&](double spot) {
    sum_log_spots += std::log(spot);
  });
  return std::exp(sum_log_spots / observation_count);
}

double arithmetic_average_from_gbm_path(const MarketState& market, double maturity_years,
                                        std::span<const double> normal_draws) {
  const double observation_count = static_cast<double>(normal_draws.size());
  double sum_spots = 0.0;
  for_each_gbm_observation(market, maturity_years, normal_draws, [&](double spot) {
    sum_spots += spot;
  });
  return sum_spots / observation_count;
}

double option_payoff(OptionType type, double underlying, double strike) noexcept {
  if (type == OptionType::call) {
    return std::max(underlying - strike, 0.0);
  }
  return std::max(strike - underlying, 0.0);
}

double control_variate_adjusted_sample(double target_sample, double control_sample,
                                       double control_expectation, double coefficient) noexcept {
  return target_sample - coefficient * (control_sample - control_expectation);
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

  return make_result(statistics, config, config.path_count);
}

MonteCarloResult price_geometric_asian_monte_carlo(const OptionContract& contract,
                                                   const MarketState& market,
                                                   const MonteCarloConfig& config) {
  return price_asian_monte_carlo(
      contract, market, config, OptionStyle::geometric_asian,
      "geometric-Asian Monte Carlo pricing requires a geometric Asian option",
      geometric_average_from_gbm_path);
}

MonteCarloResult price_arithmetic_asian_monte_carlo(const OptionContract& contract,
                                                    const MarketState& market,
                                                    const MonteCarloConfig& config) {
  return price_asian_monte_carlo(
      contract, market, config, OptionStyle::arithmetic_asian,
      "arithmetic-Asian Monte Carlo pricing requires an arithmetic Asian option",
      arithmetic_average_from_gbm_path);
}

MonteCarloResult price_arithmetic_asian_antithetic_monte_carlo(
    const OptionContract& contract, const MarketState& market, const MonteCarloConfig& config) {
  if (contract.style != OptionStyle::arithmetic_asian) {
    throw std::invalid_argument(
        "arithmetic-Asian antithetic pricing requires an arithmetic Asian option");
  }
  validate_config(config);
  const std::size_t raw_paths = checked_path_sum(config.path_count, config.path_count);

  NormalGenerator normal_generator(config.seed);
  StreamingStatistics statistics;
  std::vector<double> normal_draws(contract.observations);
  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);

  for (std::size_t sample = 0; sample < config.path_count; ++sample) {
    fill_normal_draws(normal_generator, normal_draws);
    const double positive_average =
        arithmetic_average_from_gbm_path(market, contract.maturity_years, normal_draws);
    for (double& normal_draw : normal_draws) {
      normal_draw = -normal_draw;
    }
    const double negative_average =
        arithmetic_average_from_gbm_path(market, contract.maturity_years, normal_draws);
    const double paired_payoff =
        0.5 * discount_factor *
        (option_payoff(contract.type, positive_average, contract.strike) +
         option_payoff(contract.type, negative_average, contract.strike));
    statistics.add(paired_payoff);
  }

  return make_result(statistics, config, raw_paths);
}

ControlVariateResult price_arithmetic_asian_control_variate_monte_carlo(
    const OptionContract& contract, const MarketState& market,
    const ControlVariateConfig& config) {
  if (contract.style != OptionStyle::arithmetic_asian) {
    throw std::invalid_argument(
        "arithmetic-Asian control-variate pricing requires an arithmetic Asian option");
  }
  validate_config(config.pricing);
  if (config.pilot_path_count < 2) {
    throw std::invalid_argument("control-variate pilot requires at least two paths");
  }
  if (config.pilot_seed == config.pricing.seed) {
    throw std::invalid_argument("control-variate pilot and pricing seeds must differ");
  }
  const std::size_t raw_paths =
      checked_path_sum(config.pricing.path_count, config.pilot_path_count);

  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);
  std::vector<double> normal_draws(contract.observations);
  NormalGenerator pilot_generator(config.pilot_seed);
  StreamingBivariateStatistics pilot_statistics;
  for (std::size_t path = 0; path < config.pilot_path_count; ++path) {
    fill_normal_draws(pilot_generator, normal_draws);
    const auto payoffs = discounted_asian_payoffs(contract, market, discount_factor, normal_draws);
    pilot_statistics.add(payoffs.arithmetic, payoffs.geometric);
  }

  const auto pilot_summary = pilot_statistics.summary();
  const double control_scale =
      std::max(1.0, pilot_summary.second_mean * pilot_summary.second_mean);
  const double variance_threshold = kControlVarianceRelativeTolerance * control_scale;
  double coefficient = 0.0;
  if (std::isfinite(pilot_summary.sample_covariance) &&
      std::isfinite(pilot_summary.second_sample_variance) &&
      pilot_summary.second_sample_variance > variance_threshold) {
    const double candidate =
        pilot_summary.sample_covariance / pilot_summary.second_sample_variance;
    if (std::isfinite(candidate)) {
      coefficient = candidate;
    }
  }
  const bool control_applied = coefficient != 0.0;

  auto geometric_contract = contract;
  geometric_contract.style = OptionStyle::geometric_asian;
  const double control_expectation =
      geometric_asian_analytical(geometric_contract, market).price;

  NormalGenerator pricing_generator(config.pricing.seed);
  StreamingStatistics pricing_statistics;
  for (std::size_t path = 0; path < config.pricing.path_count; ++path) {
    fill_normal_draws(pricing_generator, normal_draws);
    const auto payoffs = discounted_asian_payoffs(contract, market, discount_factor, normal_draws);
    pricing_statistics.add(control_variate_adjusted_sample(
        payoffs.arithmetic, payoffs.geometric, control_expectation, coefficient));
  }

  return {
      .monte_carlo = make_result(pricing_statistics, config.pricing, raw_paths),
      .coefficient = coefficient,
      .control_expectation = control_expectation,
      .control_applied = control_applied,
      .pilot_paths = config.pilot_path_count,
      .pilot_seed = config.pilot_seed,
  };
}

}  // namespace nre
