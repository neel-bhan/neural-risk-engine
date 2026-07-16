#include "nre/monte_carlo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "nre/analytics.hpp"
#include "nre/random.hpp"

namespace nre {
namespace {

using PathAverageFunction = double (*)(const MarketState&, double, std::span<const double>);

// A variance below this scale cannot support a stable control coefficient in double precision.
constexpr double kControlVarianceRelativeTolerance = 64.0 * std::numeric_limits<double>::epsilon();

struct AsianAverages {
  double arithmetic;
  double geometric;
};

struct DiscountedAsianPathwiseSamples {
  PathwiseSample arithmetic;
  PathwiseSample geometric;
};

struct ControlFit {
  double coefficient;
  bool applied;
};

struct StatisticsPair {
  StreamingStatistics price;
  StreamingStatistics delta;
};

struct BivariateStatisticsPair {
  StreamingBivariateStatistics price;
  StreamingBivariateStatistics delta;
};

struct ControlSamples {
  double target_price;
  double control_price;
  double target_delta;
  double control_delta;
};

struct alignas(64) PathWorkerState {
  StatisticsPair statistics;
  std::vector<double> normal_draws;
};

struct alignas(64) ControlWorkerState {
  BivariateStatisticsPair statistics;
  std::vector<double> normal_draws;
};

constexpr std::uint64_t kWorkerSeedIncrement = 0x9E3779B97F4A7C15ULL;

std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += kWorkerSeedIncrement;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

std::uint64_t worker_seed(std::uint64_t master_seed, std::size_t worker_index,
                          std::size_t active_threads) noexcept {
  if (active_threads == 1U) {
    return master_seed;
  }
  return splitmix64(master_seed +
                    kWorkerSeedIncrement * (static_cast<std::uint64_t>(worker_index) + 1U));
}

std::size_t active_thread_count(std::size_t requested_threads, std::size_t sample_count) noexcept {
  return std::min(requested_threads, sample_count);
}

std::size_t worker_sample_count(std::size_t total, std::size_t active_threads,
                                std::size_t worker_index) noexcept {
  const std::size_t base = total / active_threads;
  return base + (worker_index < total % active_threads ? 1U : 0U);
}

void validate_config(const MonteCarloConfig& config) {
  if (config.path_count < 2) {
    throw std::invalid_argument("Monte Carlo pricing requires at least two paths");
  }
  if (config.thread_count == 0U) {
    throw std::invalid_argument("Monte Carlo thread count must be positive");
  }
}

EstimateDiagnostics make_estimate(const StreamingStatistics& statistics) {
  const auto summary = statistics.summary();
  return {
      .estimate = summary.estimate,
      .sample_standard_error = summary.sample_standard_error,
      .confidence_interval_95 = summary.confidence_interval_95,
  };
}

MonteCarloResult make_result(const StreamingStatistics& price_statistics,
                             const StreamingStatistics& delta_statistics,
                             const MonteCarloConfig& config, std::size_t raw_paths) {
  const auto price = make_estimate(price_statistics);
  return {
      .estimate = price.estimate,
      .sample_standard_error = price.sample_standard_error,
      .confidence_interval_95 = price.confidence_interval_95,
      .delta = make_estimate(delta_statistics),
      .effective_paths = price_statistics.count(),
      .raw_paths = raw_paths,
      .seed = config.seed,
      .requested_threads = config.thread_count,
      .active_threads = active_thread_count(config.thread_count, config.path_count),
  };
}

std::size_t checked_path_sum(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first) {
    throw std::invalid_argument("Monte Carlo raw path count overflows size_t");
  }
  return first + second;
}

std::size_t checked_path_product(std::size_t count, std::size_t multiplier) {
  if (count > std::numeric_limits<std::size_t>::max() / multiplier) {
    throw std::invalid_argument("Monte Carlo raw path count overflows size_t");
  }
  return count * multiplier;
}

void fill_normal_draws(NormalGenerator& normal_generator, std::span<double> normal_draws) {
  for (double& normal_draw : normal_draws) {
    normal_draw = normal_generator.next();
  }
}

template <typename SampleFunction>
StatisticsPair run_path_samples(const MonteCarloConfig& config, std::size_t draw_count,
                                SampleFunction&& sample_function) {
  const std::size_t active_threads = active_thread_count(config.thread_count, config.path_count);
  std::vector<PathWorkerState> workers(active_threads);
  for (auto& worker : workers) {
    worker.normal_draws.resize(draw_count);
  }

  auto run_worker = [&](std::size_t worker_index) {
    auto& worker = workers[worker_index];
    NormalGenerator generator(worker_seed(config.seed, worker_index, active_threads));
    const std::size_t samples =
        worker_sample_count(config.path_count, active_threads, worker_index);
    for (std::size_t sample_index = 0; sample_index < samples; ++sample_index) {
      fill_normal_draws(generator, worker.normal_draws);
      const auto sample = sample_function(generator, worker.normal_draws);
      worker.statistics.price.add(sample.discounted_payoff);
      worker.statistics.delta.add(sample.discounted_delta);
    }
  };

  if (active_threads == 1U) {
    run_worker(0U);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(active_threads);
    try {
      for (std::size_t worker_index = 0; worker_index < active_threads; ++worker_index) {
        threads.emplace_back(run_worker, worker_index);
      }
    } catch (...) {
      for (auto& thread : threads) {
        thread.join();
      }
      throw;
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  StatisticsPair combined;
  for (const auto& worker : workers) {
    combined.price.merge(worker.statistics.price);
    combined.delta.merge(worker.statistics.delta);
  }
  return combined;
}

template <typename SampleFunction>
BivariateStatisticsPair run_control_samples(std::uint64_t master_seed, std::size_t sample_count,
                                            std::size_t thread_count, std::size_t draw_count,
                                            SampleFunction&& sample_function) {
  const std::size_t active_threads = active_thread_count(thread_count, sample_count);
  std::vector<ControlWorkerState> workers(active_threads);
  for (auto& worker : workers) {
    worker.normal_draws.resize(draw_count);
  }

  auto run_worker = [&](std::size_t worker_index) {
    auto& worker = workers[worker_index];
    NormalGenerator generator(worker_seed(master_seed, worker_index, active_threads));
    const std::size_t samples = worker_sample_count(sample_count, active_threads, worker_index);
    for (std::size_t sample_index = 0; sample_index < samples; ++sample_index) {
      fill_normal_draws(generator, worker.normal_draws);
      const auto sample = sample_function(worker.normal_draws);
      worker.statistics.price.add(sample.target_price, sample.control_price);
      worker.statistics.delta.add(sample.target_delta, sample.control_delta);
    }
  };

  if (active_threads == 1U) {
    run_worker(0U);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(active_threads);
    try {
      for (std::size_t worker_index = 0; worker_index < active_threads; ++worker_index) {
        threads.emplace_back(run_worker, worker_index);
      }
    } catch (...) {
      for (auto& thread : threads) {
        thread.join();
      }
      throw;
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  BivariateStatisticsPair combined;
  for (const auto& worker : workers) {
    combined.price.merge(worker.statistics.price);
    combined.delta.merge(worker.statistics.delta);
  }
  return combined;
}

template <typename Observer>
void for_each_gbm_observation(const MarketState& market, double maturity_years,
                              std::span<const double> normal_draws, Observer&& observer) {
  if (normal_draws.empty()) {
    throw std::invalid_argument("an Asian path requires at least one observation");
  }

  const double observation_count = static_cast<double>(normal_draws.size());
  const double time_step_years = maturity_years / observation_count;
  const double variance = market.volatility * market.volatility;
  const double drift =
      (market.risk_free_rate - market.dividend_yield - 0.5 * variance) * time_step_years;
  const double diffusion_scale = market.volatility * std::sqrt(time_step_years);
  double spot = market.spot;
  for (const double normal_draw : normal_draws) {
    spot *= std::exp(drift + diffusion_scale * normal_draw);
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

double payoff_spot_derivative(OptionType type, double underlying, double strike,
                              double underlying_spot_derivative) noexcept {
  if (type == OptionType::call) {
    if (underlying > strike) {
      return underlying_spot_derivative;
    }
    if (underlying < strike) {
      return 0.0;
    }
    return 0.5 * underlying_spot_derivative;
  }
  if (underlying < strike) {
    return -underlying_spot_derivative;
  }
  if (underlying > strike) {
    return 0.0;
  }
  return -0.5 * underlying_spot_derivative;
}

PathwiseSample discounted_sample_from_underlying(const OptionContract& contract,
                                                 const MarketState& market, double discount_factor,
                                                 double underlying) {
  const double underlying_spot_derivative = underlying / market.spot;
  return {
      .discounted_payoff =
          discount_factor * option_payoff(contract.type, underlying, contract.strike),
      .discounted_delta =
          discount_factor * payoff_spot_derivative(contract.type, underlying, contract.strike,
                                                   underlying_spot_derivative),
  };
}

DiscountedAsianPathwiseSamples discounted_asian_pathwise_samples(
    const OptionContract& contract, const MarketState& market, double discount_factor,
    std::span<const double> normal_draws) {
  const auto averages = asian_averages_from_gbm_path(market, contract.maturity_years, normal_draws);
  return {
      .arithmetic =
          discounted_sample_from_underlying(contract, market, discount_factor, averages.arithmetic),
      .geometric =
          discounted_sample_from_underlying(contract, market, discount_factor, averages.geometric),
  };
}

ControlFit fit_control(const BivariateStatisticsSummary& summary) {
  const double control_scale = std::max(1.0, summary.second_mean * summary.second_mean);
  const double variance_threshold = kControlVarianceRelativeTolerance * control_scale;
  double coefficient = 0.0;
  if (std::isfinite(summary.sample_covariance) && std::isfinite(summary.second_sample_variance) &&
      summary.second_sample_variance > variance_threshold) {
    const double candidate = summary.sample_covariance / summary.second_sample_variance;
    if (std::isfinite(candidate)) {
      coefficient = candidate;
    }
  }
  return {.coefficient = coefficient, .applied = coefficient != 0.0};
}

MonteCarloResult price_asian_monte_carlo(const OptionContract& contract, const MarketState& market,
                                         const MonteCarloConfig& config, OptionStyle required_style,
                                         const char* style_error,
                                         PathAverageFunction path_average) {
  if (contract.style != required_style) {
    throw std::invalid_argument(style_error);
  }
  validate_config(config);

  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);
  const auto statistics = run_path_samples(
      config, contract.observations,
      [&](NormalGenerator&, std::span<double> normal_draws) {
        const double average = path_average(market, contract.maturity_years, normal_draws);
        return discounted_sample_from_underlying(contract, market, discount_factor, average);
      });

  return make_result(statistics.price, statistics.delta, config, config.path_count);
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
  for_each_gbm_observation(market, maturity_years, normal_draws,
                           [&](double spot) { sum_log_spots += std::log(spot); });
  return std::exp(sum_log_spots / observation_count);
}

double arithmetic_average_from_gbm_path(const MarketState& market, double maturity_years,
                                        std::span<const double> normal_draws) {
  const double observation_count = static_cast<double>(normal_draws.size());
  double sum_spots = 0.0;
  for_each_gbm_observation(market, maturity_years, normal_draws,
                           [&](double spot) { sum_spots += spot; });
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

PathwiseSample discounted_pathwise_sample(const OptionContract& contract, const MarketState& market,
                                          std::span<const double> normal_draws) {
  const std::size_t required_draws =
      contract.style == OptionStyle::european ? 1U : contract.observations;
  if (normal_draws.size() != required_draws) {
    throw std::invalid_argument("pathwise sample draw count does not match the contract schedule");
  }

  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);
  if (contract.style == OptionStyle::european) {
    const double terminal_spot =
        exact_gbm_step(market.spot, contract.maturity_years, market, normal_draws.front());
    return discounted_sample_from_underlying(contract, market, discount_factor, terminal_spot);
  }

  if (contract.style == OptionStyle::geometric_asian) {
    const double average =
        geometric_average_from_gbm_path(market, contract.maturity_years, normal_draws);
    return discounted_sample_from_underlying(contract, market, discount_factor, average);
  }
  if (contract.style == OptionStyle::arithmetic_asian) {
    const double average =
        arithmetic_average_from_gbm_path(market, contract.maturity_years, normal_draws);
    return discounted_sample_from_underlying(contract, market, discount_factor, average);
  }
  throw std::invalid_argument("unsupported option style for a pathwise sample");
}

double centered_spot_bump(double spot, const SpotBumpRule& rule) {
  if (!std::isfinite(spot) || spot <= 0.0) {
    throw std::invalid_argument("centered spot bump requires finite positive spot");
  }
  if (!std::isfinite(rule.relative_size) || rule.relative_size <= 0.0 ||
      !std::isfinite(rule.minimum_absolute_size) || rule.minimum_absolute_size <= 0.0) {
    throw std::invalid_argument("spot bump sizes must be finite and positive");
  }
  const double bump = std::max(rule.relative_size * spot, rule.minimum_absolute_size);
  if (!std::isfinite(bump) || bump >= spot) {
    throw std::invalid_argument("centered spot bump must leave a finite positive down spot");
  }
  return bump;
}

MonteCarloResult price_european_monte_carlo(const OptionContract& contract,
                                            const MarketState& market,
                                            const MonteCarloConfig& config) {
  if (contract.style != OptionStyle::european) {
    throw std::invalid_argument("European Monte Carlo pricing requires a European option");
  }
  validate_config(config);
  const double variance = market.volatility * market.volatility;
  const double drift = (market.risk_free_rate - market.dividend_yield - 0.5 * variance) *
                       contract.maturity_years;
  const double diffusion_scale = market.volatility * std::sqrt(contract.maturity_years);
  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);
  const auto statistics = run_path_samples(
      config, 1U, [&](NormalGenerator&, std::span<double> normal_draws) {
        const double terminal_spot =
            market.spot * std::exp(drift + diffusion_scale * normal_draws.front());
        return discounted_sample_from_underlying(contract, market, discount_factor, terminal_spot);
      });

  return make_result(statistics.price, statistics.delta, config, config.path_count);
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

MonteCarloResult price_arithmetic_asian_antithetic_monte_carlo(const OptionContract& contract,
                                                               const MarketState& market,
                                                               const MonteCarloConfig& config) {
  if (contract.style != OptionStyle::arithmetic_asian) {
    throw std::invalid_argument(
        "arithmetic-Asian antithetic pricing requires an arithmetic Asian option");
  }
  validate_config(config);
  const std::size_t raw_paths = checked_path_sum(config.path_count, config.path_count);

  const auto statistics = run_path_samples(
      config, contract.observations,
      [&](NormalGenerator&, std::span<double> normal_draws) {
        const auto positive = discounted_pathwise_sample(contract, market, normal_draws);
        for (double& normal_draw : normal_draws) {
          normal_draw = -normal_draw;
        }
        const auto negative = discounted_pathwise_sample(contract, market, normal_draws);
        return PathwiseSample{
            .discounted_payoff =
                0.5 * (positive.discounted_payoff + negative.discounted_payoff),
            .discounted_delta =
                0.5 * (positive.discounted_delta + negative.discounted_delta),
        };
      });

  return make_result(statistics.price, statistics.delta, config, raw_paths);
}

ControlVariateResult price_arithmetic_asian_control_variate_monte_carlo(
    const OptionContract& contract, const MarketState& market, const ControlVariateConfig& config) {
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
  const auto pilot_statistics = run_control_samples(
      config.pilot_seed, config.pilot_path_count, config.pricing.thread_count,
      contract.observations, [&](std::span<double> normal_draws) {
        const auto samples =
            discounted_asian_pathwise_samples(contract, market, discount_factor, normal_draws);
        return ControlSamples{
            .target_price = samples.arithmetic.discounted_payoff,
            .control_price = samples.geometric.discounted_payoff,
            .target_delta = samples.arithmetic.discounted_delta,
            .control_delta = samples.geometric.discounted_delta,
        };
      });

  const auto price_fit = fit_control(pilot_statistics.price.summary());
  const auto delta_fit = fit_control(pilot_statistics.delta.summary());

  auto geometric_contract = contract;
  geometric_contract.style = OptionStyle::geometric_asian;
  const auto analytical_control = geometric_asian_analytical(geometric_contract, market);

  const auto pricing_statistics = run_path_samples(
      config.pricing, contract.observations,
      [&](NormalGenerator&, std::span<double> normal_draws) {
        const auto samples =
            discounted_asian_pathwise_samples(contract, market, discount_factor, normal_draws);
        return PathwiseSample{
            .discounted_payoff = control_variate_adjusted_sample(
                samples.arithmetic.discounted_payoff, samples.geometric.discounted_payoff,
                analytical_control.price, price_fit.coefficient),
            .discounted_delta = control_variate_adjusted_sample(
                samples.arithmetic.discounted_delta, samples.geometric.discounted_delta,
                analytical_control.delta, delta_fit.coefficient),
        };
      });

  return {
      .monte_carlo =
          make_result(pricing_statistics.price, pricing_statistics.delta, config.pricing, raw_paths),
      .coefficient = price_fit.coefficient,
      .control_expectation = analytical_control.price,
      .control_applied = price_fit.applied,
      .delta_coefficient = delta_fit.coefficient,
      .delta_control_expectation = analytical_control.delta,
      .delta_control_applied = delta_fit.applied,
      .pilot_paths = config.pilot_path_count,
      .pilot_seed = config.pilot_seed,
      .pilot_active_threads =
          active_thread_count(config.pricing.thread_count, config.pilot_path_count),
  };
}

BumpAndRevalueResult delta_bump_and_revalue_monte_carlo(const OptionContract& contract,
                                                        const MarketState& market,
                                                        const MonteCarloConfig& config,
                                                        const SpotBumpRule& bump_rule) {
  validate_config(config);
  const double bump = centered_spot_bump(market.spot, bump_rule);
  auto up_market = market;
  auto down_market = market;
  up_market.spot += bump;
  down_market.spot -= bump;

  const std::size_t draw_count =
      contract.style == OptionStyle::european ? 1U : contract.observations;
  const auto statistics = run_path_samples(
      config, draw_count, [&](NormalGenerator&, std::span<double> normal_draws) {
        const double up_payoff =
            discounted_pathwise_sample(contract, up_market, normal_draws).discounted_payoff;
        const double down_payoff =
            discounted_pathwise_sample(contract, down_market, normal_draws).discounted_payoff;
        return PathwiseSample{
            .discounted_payoff = 0.0,
            .discounted_delta = (up_payoff - down_payoff) / (2.0 * bump),
        };
      });

  return {
      .delta = make_estimate(statistics.delta),
      .effective_paths = config.path_count,
      .raw_paths = checked_path_product(config.path_count, 2U),
      .seed = config.seed,
      .spot_bump = bump,
      .requested_threads = config.thread_count,
      .active_threads = active_thread_count(config.thread_count, config.path_count),
  };
}

BumpAndRevalueResult delta_bump_and_revalue_arithmetic_antithetic_monte_carlo(
    const OptionContract& contract, const MarketState& market, const MonteCarloConfig& config,
    const SpotBumpRule& bump_rule) {
  if (contract.style != OptionStyle::arithmetic_asian) {
    throw std::invalid_argument(
        "arithmetic-Asian antithetic bump-and-revalue requires an arithmetic Asian option");
  }
  validate_config(config);
  const double bump = centered_spot_bump(market.spot, bump_rule);
  auto up_market = market;
  auto down_market = market;
  up_market.spot += bump;
  down_market.spot -= bump;

  const auto statistics = run_path_samples(
      config, contract.observations,
      [&](NormalGenerator&, std::span<double> normal_draws) {
        const double positive_up =
            discounted_pathwise_sample(contract, up_market, normal_draws).discounted_payoff;
        const double positive_down =
            discounted_pathwise_sample(contract, down_market, normal_draws).discounted_payoff;
        for (double& normal_draw : normal_draws) {
          normal_draw = -normal_draw;
        }
        const double negative_up =
            discounted_pathwise_sample(contract, up_market, normal_draws).discounted_payoff;
        const double negative_down =
            discounted_pathwise_sample(contract, down_market, normal_draws).discounted_payoff;
        const double up_pair = 0.5 * (positive_up + negative_up);
        const double down_pair = 0.5 * (positive_down + negative_down);
        return PathwiseSample{
            .discounted_payoff = 0.0,
            .discounted_delta = (up_pair - down_pair) / (2.0 * bump),
        };
      });

  return {
      .delta = make_estimate(statistics.delta),
      .effective_paths = config.path_count,
      .raw_paths = checked_path_product(config.path_count, 4U),
      .seed = config.seed,
      .spot_bump = bump,
      .requested_threads = config.thread_count,
      .active_threads = active_thread_count(config.thread_count, config.path_count),
  };
}

ControlVariateBumpAndRevalueResult delta_bump_and_revalue_arithmetic_control_variate_monte_carlo(
    const OptionContract& contract, const MarketState& market, const ControlVariateConfig& config,
    const SpotBumpRule& bump_rule) {
  if (contract.style != OptionStyle::arithmetic_asian) {
    throw std::invalid_argument(
        "arithmetic-Asian control-variate bump-and-revalue requires an arithmetic Asian option");
  }
  validate_config(config.pricing);
  if (config.pilot_path_count < 2) {
    throw std::invalid_argument("control-variate pilot requires at least two paths");
  }
  if (config.pilot_seed == config.pricing.seed) {
    throw std::invalid_argument("control-variate pilot and pricing seeds must differ");
  }

  const double bump = centered_spot_bump(market.spot, bump_rule);
  auto up_market = market;
  auto down_market = market;
  up_market.spot += bump;
  down_market.spot -= bump;
  const double discount_factor = std::exp(-market.risk_free_rate * contract.maturity_years);
  const double denominator = 2.0 * bump;

  const auto pilot_statistics = run_control_samples(
      config.pilot_seed, config.pilot_path_count, config.pricing.thread_count,
      contract.observations, [&](std::span<double> normal_draws) {
        const auto up =
            discounted_asian_pathwise_samples(contract, up_market, discount_factor, normal_draws);
        const auto down =
            discounted_asian_pathwise_samples(contract, down_market, discount_factor, normal_draws);
        const double target =
            (up.arithmetic.discounted_payoff - down.arithmetic.discounted_payoff) / denominator;
        const double control =
            (up.geometric.discounted_payoff - down.geometric.discounted_payoff) / denominator;
        return ControlSamples{
            .target_price = target,
            .control_price = control,
            .target_delta = target,
            .control_delta = control,
        };
      });
  const auto fit = fit_control(pilot_statistics.price.summary());

  auto geometric_contract = contract;
  geometric_contract.style = OptionStyle::geometric_asian;
  const double up_control = geometric_asian_analytical(geometric_contract, up_market).price;
  const double down_control = geometric_asian_analytical(geometric_contract, down_market).price;
  const double control_expectation = (up_control - down_control) / denominator;

  const auto statistics = run_path_samples(
      config.pricing, contract.observations,
      [&](NormalGenerator&, std::span<double> normal_draws) {
        const auto up =
            discounted_asian_pathwise_samples(contract, up_market, discount_factor, normal_draws);
        const auto down =
            discounted_asian_pathwise_samples(contract, down_market, discount_factor, normal_draws);
        const double target_sample =
            (up.arithmetic.discounted_payoff - down.arithmetic.discounted_payoff) / denominator;
        const double control_sample =
            (up.geometric.discounted_payoff - down.geometric.discounted_payoff) / denominator;
        return PathwiseSample{
            .discounted_payoff = 0.0,
            .discounted_delta = control_variate_adjusted_sample(
                target_sample, control_sample, control_expectation, fit.coefficient),
        };
      });

  const std::size_t all_effective_paths =
      checked_path_sum(config.pricing.path_count, config.pilot_path_count);
  return {
      .bump_and_revalue =
          {
              .delta = make_estimate(statistics.delta),
              .effective_paths = config.pricing.path_count,
              .raw_paths = checked_path_product(all_effective_paths, 2U),
              .seed = config.pricing.seed,
              .spot_bump = bump,
              .requested_threads = config.pricing.thread_count,
              .active_threads =
                  active_thread_count(config.pricing.thread_count, config.pricing.path_count),
          },
      .coefficient = fit.coefficient,
      .control_expectation = control_expectation,
      .control_applied = fit.applied,
      .pilot_paths = config.pilot_path_count,
      .pilot_seed = config.pilot_seed,
      .pilot_active_threads =
          active_thread_count(config.pricing.thread_count, config.pilot_path_count),
  };
}

}  // namespace nre
