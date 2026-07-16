#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>

#include "nre/analytics.hpp"
#include "nre/monte_carlo.hpp"

namespace {

constexpr std::size_t kRepetitions = 200;
constexpr std::uint64_t kFirstSeed = 1000000ULL;
constexpr std::array<std::size_t, 3> kPathCounts{1000, 4000, 16000};

struct ExperimentRow {
  double root_mean_squared_error;
  double mean_absolute_error;
  double mean_reported_standard_error;
  double coverage_percent;
};

template <typename Pricer>
ExperimentRow run_experiment(const nre::OptionContract& contract, const nre::MarketState& market,
                             std::size_t path_count, double reference_price, Pricer pricer) {
  double sum_squared_error = 0.0;
  double sum_absolute_error = 0.0;
  double sum_standard_error = 0.0;
  std::size_t intervals_containing_reference = 0;

  for (std::size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    const auto seed = kFirstSeed + static_cast<std::uint64_t>(repetition);
    const nre::MonteCarloConfig config{.seed = seed, .path_count = path_count};
    const auto result = pricer(contract, market, config);
    const double error = result.estimate - reference_price;
    sum_squared_error += error * error;
    sum_absolute_error += std::abs(error);
    sum_standard_error += result.sample_standard_error;
    if (result.confidence_interval_95.lower <= reference_price &&
        reference_price <= result.confidence_interval_95.upper) {
      ++intervals_containing_reference;
    }
  }

  const double repetitions = static_cast<double>(kRepetitions);
  return {
      .root_mean_squared_error = std::sqrt(sum_squared_error / repetitions),
      .mean_absolute_error = sum_absolute_error / repetitions,
      .mean_reported_standard_error = sum_standard_error / repetitions,
      .coverage_percent = 100.0 * static_cast<double>(intervals_containing_reference) / repetitions,
  };
}

template <typename Pricer>
void print_case(std::string_view case_name, const nre::OptionContract& contract,
                const nre::MarketState& market, double reference_price, Pricer pricer) {
  for (const std::size_t path_count : kPathCounts) {
    const auto row = run_experiment(contract, market, path_count, reference_price, pricer);
    std::cout << case_name << ',' << path_count << ',' << kRepetitions << ',' << reference_price
              << ',' << row.root_mean_squared_error << ',' << row.mean_absolute_error << ','
              << row.mean_reported_standard_error << ',' << row.coverage_percent << '\n';
  }
}

}  // namespace

int main() {
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
  const nre::OptionContract geometric_asian{
      .type = nre::OptionType::call,
      .style = nre::OptionStyle::geometric_asian,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = 12,
  };

  std::cout << std::setprecision(10);
  std::cout << "# compiler=" << __VERSION__ << '\n';
  std::cout << "# seed_policy=seeds " << kFirstSeed << " through "
            << kFirstSeed + static_cast<std::uint64_t>(kRepetitions) - 1ULL
            << "; each path count and contract restarts the same seed set\n";
  std::cout << "case,paths,repetitions,reference_price,rmse,mean_absolute_error,"
               "mean_reported_standard_error,coverage_percent\n";

  print_case("european_call", european, market, nre::black_scholes_european(european, market).price,
             nre::price_european_monte_carlo);
  print_case("geometric_asian_call", geometric_asian, market,
             nre::geometric_asian_analytical(geometric_asian, market).price,
             nre::price_geometric_asian_monte_carlo);
}
