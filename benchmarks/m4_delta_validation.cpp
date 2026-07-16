#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "nre/analytics.hpp"
#include "nre/monte_carlo.hpp"
#include "nre/statistics.hpp"

namespace {

constexpr std::size_t kRuns = 100;
constexpr std::uint64_t kFirstPricingSeed = 5'000'000ULL;
constexpr std::uint64_t kFirstPilotSeed = 6'000'000ULL;
constexpr std::size_t kPilotPaths = 1'000;
constexpr nre::SpotBumpRule kBumpRule{.relative_size = 1.0e-4, .minimum_absolute_size = 1.0e-6};

struct ValidationSummary {
  nre::StreamingStatistics errors;
  double sum_squared_error{0.0};
  double sum_absolute_error{0.0};
  double sum_reported_standard_error{0.0};
  double elapsed_milliseconds{0.0};
  std::size_t coverage_count{0};

  void add(double estimate, double reference, double reported_standard_error,
           const nre::ConfidenceInterval& interval, double milliseconds) {
    const double error = estimate - reference;
    errors.add(error);
    sum_squared_error += error * error;
    sum_absolute_error += std::abs(error);
    sum_reported_standard_error += reported_standard_error;
    elapsed_milliseconds += milliseconds;
    if (interval.lower <= reference && reference <= interval.upper) {
      ++coverage_count;
    }
  }
};

struct ComparisonSummary {
  nre::StreamingStatistics differences;
  double sum_squared_difference{0.0};
  double sum_absolute_difference{0.0};
  double sum_pathwise_standard_error{0.0};
  double sum_bump_standard_error{0.0};
  double pathwise_milliseconds{0.0};
  double bump_milliseconds{0.0};

  void add(const nre::EstimateDiagnostics& pathwise, const nre::EstimateDiagnostics& bumped,
           double pathwise_time, double bump_time) {
    const double difference = pathwise.estimate - bumped.estimate;
    differences.add(difference);
    sum_squared_difference += difference * difference;
    sum_absolute_difference += std::abs(difference);
    sum_pathwise_standard_error += pathwise.sample_standard_error;
    sum_bump_standard_error += bumped.sample_standard_error;
    pathwise_milliseconds += pathwise_time;
    bump_milliseconds += bump_time;
  }
};

template <typename Function>
auto timed(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  auto result = function();
  const auto stop = std::chrono::steady_clock::now();
  const double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
  return std::pair{std::move(result), milliseconds};
}

void print_validation_row(const std::string& contract_name, std::size_t paths, double reference,
                          const ValidationSummary& summary) {
  const double runs = static_cast<double>(kRuns);
  const auto error_summary = summary.errors.summary();
  std::cout << contract_name << ',' << paths << ',' << reference << ','
            << std::sqrt(summary.sum_squared_error / runs) << ','
            << summary.sum_absolute_error / runs << ',' << error_summary.estimate << ','
            << error_summary.sample_standard_error << ','
            << summary.sum_reported_standard_error / runs << ','
            << 100.0 * static_cast<double>(summary.coverage_count) / runs << ','
            << summary.elapsed_milliseconds / runs << '\n';
}

void print_comparison_row(const std::string& estimator, std::size_t paths,
                          const ComparisonSummary& summary) {
  const double runs = static_cast<double>(kRuns);
  const auto difference_summary = summary.differences.summary();
  std::cout << estimator << ',' << paths << ',' << difference_summary.estimate << ','
            << difference_summary.sample_standard_error << ','
            << std::sqrt(summary.sum_squared_difference / runs) << ','
            << summary.sum_absolute_difference / runs << ','
            << summary.sum_pathwise_standard_error / runs << ','
            << summary.sum_bump_standard_error / runs << ',' << summary.pathwise_milliseconds / runs
            << ',' << summary.bump_milliseconds / runs << '\n';
}

void run_analytical_validation() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.20,
      .risk_free_rate = 0.05,
      .dividend_yield = 0.02,
  };
  const nre::OptionContract european{
      .type = nre::OptionType::call,
      .style = nre::OptionStyle::european,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = 1,
  };
  const nre::OptionContract geometric{
      .type = nre::OptionType::put,
      .style = nre::OptionStyle::geometric_asian,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = 12,
  };
  const double european_reference = nre::black_scholes_european(european, market).delta;
  const double geometric_reference = nre::geometric_asian_analytical(geometric, market).delta;

  std::cout << "analytical_validation\n";
  std::cout << "contract,paths,analytical_delta,rmse,mean_absolute_error,mean_error,"
               "error_se,mean_reported_se,coverage_percent,mean_runtime_ms\n";
  for (const std::size_t paths : {1'000U, 4'000U, 16'000U}) {
    ValidationSummary european_summary;
    ValidationSummary geometric_summary;
    for (std::size_t run = 0; run < kRuns; ++run) {
      const nre::MonteCarloConfig config{
          .seed = kFirstPricingSeed + static_cast<std::uint64_t>(run),
          .path_count = paths,
      };
      auto [european_result, european_time] =
          timed([&] { return nre::price_european_monte_carlo(european, market, config); });
      european_summary.add(european_result.delta.estimate, european_reference,
                           european_result.delta.sample_standard_error,
                           european_result.delta.confidence_interval_95, european_time);

      auto [geometric_result, geometric_time] =
          timed([&] { return nre::price_geometric_asian_monte_carlo(geometric, market, config); });
      geometric_summary.add(geometric_result.delta.estimate, geometric_reference,
                            geometric_result.delta.sample_standard_error,
                            geometric_result.delta.confidence_interval_95, geometric_time);
    }
    print_validation_row("European call", paths, european_reference, european_summary);
    print_validation_row("Geometric-Asian put", paths, geometric_reference, geometric_summary);
  }
}

void run_european_and_geometric_crn_comparison() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.20,
      .risk_free_rate = 0.05,
      .dividend_yield = 0.02,
  };
  const nre::OptionContract european{
      .type = nre::OptionType::call,
      .style = nre::OptionStyle::european,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = 1,
  };
  const nre::OptionContract geometric{
      .type = nre::OptionType::put,
      .style = nre::OptionStyle::geometric_asian,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = 12,
  };

  std::cout << "analytical_contract_crn_comparison\n";
  std::cout << "estimator,paths,mean_difference,difference_se,rmse_difference,"
               "mean_absolute_difference,mean_pathwise_se,mean_bump_se,"
               "mean_pathwise_runtime_ms,mean_bump_runtime_ms\n";
  for (const std::size_t paths : {1'000U, 4'000U, 16'000U}) {
    ComparisonSummary european_summary;
    ComparisonSummary geometric_summary;
    for (std::size_t run = 0; run < kRuns; ++run) {
      const nre::MonteCarloConfig config{
          .seed = kFirstPricingSeed + static_cast<std::uint64_t>(run),
          .path_count = paths,
      };
      auto [european_pathwise, european_pathwise_time] =
          timed([&] { return nre::price_european_monte_carlo(european, market, config); });
      auto [european_bump, european_bump_time] = timed([&] {
        return nre::delta_bump_and_revalue_monte_carlo(european, market, config, kBumpRule);
      });
      european_summary.add(european_pathwise.delta, european_bump.delta, european_pathwise_time,
                           european_bump_time);

      auto [geometric_pathwise, geometric_pathwise_time] =
          timed([&] { return nre::price_geometric_asian_monte_carlo(geometric, market, config); });
      auto [geometric_bump, geometric_bump_time] = timed([&] {
        return nre::delta_bump_and_revalue_monte_carlo(geometric, market, config, kBumpRule);
      });
      geometric_summary.add(geometric_pathwise.delta, geometric_bump.delta, geometric_pathwise_time,
                            geometric_bump_time);
    }
    print_comparison_row("European call", paths, european_summary);
    print_comparison_row("Geometric-Asian put", paths, geometric_summary);
  }
}

void run_arithmetic_comparison() {
  const nre::MarketState market{
      .spot = 105.0,
      .volatility = 0.30,
      .risk_free_rate = 0.02,
      .dividend_yield = 0.01,
  };
  const nre::OptionContract contract{
      .type = nre::OptionType::call,
      .style = nre::OptionStyle::arithmetic_asian,
      .strike = 100.0,
      .maturity_years = 1.5,
      .observations = 18,
  };

  std::cout << "arithmetic_crn_comparison\n";
  std::cout << "estimator,paths,mean_difference,difference_se,rmse_difference,"
               "mean_absolute_difference,mean_pathwise_se,mean_bump_se,"
               "mean_pathwise_runtime_ms,mean_bump_runtime_ms\n";
  for (const std::size_t paths : {1'000U, 4'000U, 16'000U}) {
    ComparisonSummary plain_summary;
    ComparisonSummary antithetic_summary;
    ComparisonSummary control_summary;
    for (std::size_t run = 0; run < kRuns; ++run) {
      const std::uint64_t run_offset = static_cast<std::uint64_t>(run);
      const nre::MonteCarloConfig config{
          .seed = kFirstPricingSeed + run_offset,
          .path_count = paths,
      };
      const nre::ControlVariateConfig control_config{
          .pricing = config,
          .pilot_seed = kFirstPilotSeed + run_offset,
          .pilot_path_count = kPilotPaths,
      };

      auto [plain, plain_time] =
          timed([&] { return nre::price_arithmetic_asian_monte_carlo(contract, market, config); });
      auto [plain_bump, plain_bump_time] = timed([&] {
        return nre::delta_bump_and_revalue_monte_carlo(contract, market, config, kBumpRule);
      });
      plain_summary.add(plain.delta, plain_bump.delta, plain_time, plain_bump_time);

      auto [antithetic, antithetic_time] = timed([&] {
        return nre::price_arithmetic_asian_antithetic_monte_carlo(contract, market, config);
      });
      auto [antithetic_bump, antithetic_bump_time] = timed([&] {
        return nre::delta_bump_and_revalue_arithmetic_antithetic_monte_carlo(contract, market,
                                                                             config, kBumpRule);
      });
      antithetic_summary.add(antithetic.delta, antithetic_bump.delta, antithetic_time,
                             antithetic_bump_time);

      auto [control, control_time] = timed([&] {
        return nre::price_arithmetic_asian_control_variate_monte_carlo(contract, market,
                                                                       control_config);
      });
      auto [control_bump, control_bump_time] = timed([&] {
        return nre::delta_bump_and_revalue_arithmetic_control_variate_monte_carlo(
            contract, market, control_config, kBumpRule);
      });
      control_summary.add(control.monte_carlo.delta, control_bump.bump_and_revalue.delta,
                          control_time, control_bump_time);
    }
    print_comparison_row("Plain", paths, plain_summary);
    print_comparison_row("Antithetic", paths, antithetic_summary);
    print_comparison_row("Control variate", paths, control_summary);
  }
}

}  // namespace

int main() {
  std::cout << std::setprecision(10);
  run_analytical_validation();
  run_european_and_geometric_crn_comparison();
  run_arithmetic_comparison();
  return 0;
}
