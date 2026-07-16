#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "nre/monte_carlo.hpp"

namespace {

constexpr std::size_t kRepetitions = 40;
constexpr std::size_t kPilotPaths = 1000;
constexpr std::uint64_t kFirstPricingSeed = 3000000ULL;
constexpr std::uint64_t kFirstPilotSeed = 4000000ULL;
constexpr double kTargetConfidenceIntervalWidth = 0.50;
constexpr std::array<std::size_t, 3> kEffectivePathCounts{1000, 4000, 16000};

enum class Estimator { plain, antithetic, control_variate };

struct ExperimentCase {
  std::string_view name;
  nre::OptionContract contract;
  nre::MarketState market;
};

struct TimedResult {
  nre::MonteCarloResult monte_carlo;
  double elapsed_milliseconds;
  double control_coefficient;
  bool control_applied;
};

struct ExperimentSummary {
  double mean_raw_paths;
  double mean_estimate;
  double mean_sample_variance;
  double mean_standard_error;
  double mean_confidence_interval_width;
  double mean_runtime_milliseconds;
  double target_width_hit_percent;
  double control_applied_percent;
  double mean_control_coefficient;
};

std::string_view estimator_name(Estimator estimator) {
  switch (estimator) {
    case Estimator::plain:
      return "plain";
    case Estimator::antithetic:
      return "antithetic";
    case Estimator::control_variate:
      return "control_variate";
  }
  return "unknown";
}

TimedResult run_once(const ExperimentCase& experiment_case, Estimator estimator,
                     std::size_t path_count, std::uint64_t pricing_seed,
                     std::uint64_t pilot_seed) {
  const nre::MonteCarloConfig config{.seed = pricing_seed, .path_count = path_count};
  const auto start = std::chrono::steady_clock::now();

  if (estimator == Estimator::plain) {
    const auto result = nre::price_arithmetic_asian_monte_carlo(
        experiment_case.contract, experiment_case.market, config);
    const auto end = std::chrono::steady_clock::now();
    return {
        .monte_carlo = result,
        .elapsed_milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count(),
        .control_coefficient = 0.0,
        .control_applied = false,
    };
  }

  if (estimator == Estimator::antithetic) {
    const auto result = nre::price_arithmetic_asian_antithetic_monte_carlo(
        experiment_case.contract, experiment_case.market, config);
    const auto end = std::chrono::steady_clock::now();
    return {
        .monte_carlo = result,
        .elapsed_milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count(),
        .control_coefficient = 0.0,
        .control_applied = false,
    };
  }

  if (estimator == Estimator::control_variate) {
    const nre::ControlVariateConfig control_config{
        .pricing = config,
        .pilot_seed = pilot_seed,
        .pilot_path_count = kPilotPaths,
    };
    const auto result = nre::price_arithmetic_asian_control_variate_monte_carlo(
        experiment_case.contract, experiment_case.market, control_config);
    const auto end = std::chrono::steady_clock::now();
    return {
        .monte_carlo = result.monte_carlo,
        .elapsed_milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count(),
        .control_coefficient = result.coefficient,
        .control_applied = result.control_applied,
    };
  }

  throw std::logic_error("unknown M3 estimator");
}

ExperimentSummary run_experiment(const ExperimentCase& experiment_case, Estimator estimator,
                                 std::size_t path_count) {
  double sum_raw_paths = 0.0;
  double sum_estimates = 0.0;
  double sum_sample_variances = 0.0;
  double sum_standard_errors = 0.0;
  double sum_confidence_interval_widths = 0.0;
  double sum_runtime_milliseconds = 0.0;
  double sum_control_coefficients = 0.0;
  std::size_t target_width_hits = 0;
  std::size_t control_applied_count = 0;

  for (std::size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    const auto pricing_seed = kFirstPricingSeed + static_cast<std::uint64_t>(repetition);
    const auto pilot_seed = kFirstPilotSeed + static_cast<std::uint64_t>(repetition);
    const auto timed =
        run_once(experiment_case, estimator, path_count, pricing_seed, pilot_seed);
    const double effective_paths = static_cast<double>(timed.monte_carlo.effective_paths);
    const double sample_variance = timed.monte_carlo.sample_standard_error *
                                   timed.monte_carlo.sample_standard_error * effective_paths;
    const double confidence_interval_width =
        timed.monte_carlo.confidence_interval_95.upper -
        timed.monte_carlo.confidence_interval_95.lower;

    sum_raw_paths += static_cast<double>(timed.monte_carlo.raw_paths);
    sum_estimates += timed.monte_carlo.estimate;
    sum_sample_variances += sample_variance;
    sum_standard_errors += timed.monte_carlo.sample_standard_error;
    sum_confidence_interval_widths += confidence_interval_width;
    sum_runtime_milliseconds += timed.elapsed_milliseconds;
    sum_control_coefficients += timed.control_coefficient;
    if (confidence_interval_width <= kTargetConfidenceIntervalWidth) {
      ++target_width_hits;
    }
    if (timed.control_applied) {
      ++control_applied_count;
    }
  }

  const double repetitions = static_cast<double>(kRepetitions);
  return {
      .mean_raw_paths = sum_raw_paths / repetitions,
      .mean_estimate = sum_estimates / repetitions,
      .mean_sample_variance = sum_sample_variances / repetitions,
      .mean_standard_error = sum_standard_errors / repetitions,
      .mean_confidence_interval_width = sum_confidence_interval_widths / repetitions,
      .mean_runtime_milliseconds = sum_runtime_milliseconds / repetitions,
      .target_width_hit_percent =
          100.0 * static_cast<double>(target_width_hits) / repetitions,
      .control_applied_percent =
          100.0 * static_cast<double>(control_applied_count) / repetitions,
      .mean_control_coefficient = sum_control_coefficients / repetitions,
  };
}

void warm_up(const ExperimentCase& experiment_case) {
  for (const auto estimator :
       {Estimator::plain, Estimator::antithetic, Estimator::control_variate}) {
    static_cast<void>(
        run_once(experiment_case, estimator, kEffectivePathCounts.front(), 2999999ULL, 3999999ULL));
  }
}

void print_case(const ExperimentCase& experiment_case) {
  warm_up(experiment_case);
  for (const auto estimator :
       {Estimator::plain, Estimator::antithetic, Estimator::control_variate}) {
    for (const std::size_t path_count : kEffectivePathCounts) {
      const auto summary = run_experiment(experiment_case, estimator, path_count);
      std::cout << experiment_case.name << ',' << estimator_name(estimator) << ',' << path_count
                << ',' << summary.mean_raw_paths << ',' << kRepetitions << ','
                << summary.mean_estimate << ',' << summary.mean_sample_variance << ','
                << summary.mean_standard_error << ',' << summary.mean_confidence_interval_width
                << ',' << summary.mean_runtime_milliseconds << ','
                << summary.target_width_hit_percent << ',' << summary.control_applied_percent << ','
                << summary.mean_control_coefficient << '\n';
    }
  }
}

}  // namespace

int main() {
  const std::array cases{
      ExperimentCase{
          .name = "atm_one_year_call",
          .contract =
              {
                  .type = nre::OptionType::call,
                  .style = nre::OptionStyle::arithmetic_asian,
                  .strike = 100.0,
                  .maturity_years = 1.0,
                  .observations = 12,
              },
          .market =
              {
                  .spot = 100.0,
                  .volatility = 0.20,
                  .risk_free_rate = 0.05,
                  .dividend_yield = 0.02,
              },
      },
      ExperimentCase{
          .name = "otm_two_year_put",
          .contract =
              {
                  .type = nre::OptionType::put,
                  .style = nre::OptionStyle::arithmetic_asian,
                  .strike = 100.0,
                  .maturity_years = 2.0,
                  .observations = 24,
              },
          .market =
              {
                  .spot = 110.0,
                  .volatility = 0.30,
                  .risk_free_rate = 0.01,
                  .dividend_yield = 0.015,
              },
      },
  };

  std::cout << std::setprecision(10);
  std::cout << "# compiler=" << __VERSION__ << '\n';
  std::cout << "# timing=std::chrono::steady_clock around each complete pricing call; one untimed "
               "warm-up per estimator and case\n";
  std::cout << "# matched_draws=pricing seeds " << kFirstPricingSeed << " through "
            << kFirstPricingSeed + static_cast<std::uint64_t>(kRepetitions) - 1ULL
            << "; every estimator and path count restarts the same seed set\n";
  std::cout << "# pilot=independent seeds " << kFirstPilotSeed << " through "
            << kFirstPilotSeed + static_cast<std::uint64_t>(kRepetitions) - 1ULL << "; "
            << kPilotPaths << " pilot paths per control-variate run\n";
  std::cout << "# target_95_ci_width=" << kTargetConfidenceIntervalWidth << " currency units\n";
  std::cout << "case,estimator,effective_paths,mean_raw_paths,repetitions,mean_estimate,"
               "mean_sample_variance,mean_standard_error,mean_ci_width,mean_runtime_ms,"
               "target_width_hit_percent,control_applied_percent,mean_control_coefficient\n";

  for (const auto& experiment_case : cases) {
    print_case(experiment_case);
  }
}
