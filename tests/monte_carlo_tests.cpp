#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nre/analytics.hpp"
#include "nre/monte_carlo.hpp"

namespace {

// These fixtures use hand-calculated deterministic paths. The tolerance covers last-place
// differences in exp, log, and sqrt without acting as a statistical Monte Carlo tolerance.
constexpr double kDeterministicTolerance = 2.0e-12;
constexpr double kIntervalTolerance = 2.0e-12;

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected
              << ", tolerance=" << tolerance << ")\n";
    ++failures;
  }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  } catch (...) {
    std::cerr << "FAIL: " << message << " (unexpected exception type)\n";
    ++failures;
    return;
  }

  std::cerr << "FAIL: " << message << " (no exception)\n";
  ++failures;
}

nre::OptionContract european(nre::OptionType type, double strike = 100.0) {
  return {
      .type = type,
      .style = nre::OptionStyle::european,
      .strike = strike,
      .maturity_years = 1.0,
      .observations = 1,
  };
}

nre::OptionContract geometric_asian(nre::OptionType type, double strike = 100.0,
                                    std::size_t observations = 4) {
  return {
      .type = type,
      .style = nre::OptionStyle::geometric_asian,
      .strike = strike,
      .maturity_years = 1.0,
      .observations = observations,
  };
}

void expect_interval_consistent(const nre::MonteCarloResult& result, const std::string& message) {
  const double expected_half_width = 1.96 * result.sample_standard_error;
  expect_near(result.confidence_interval_95.lower, result.estimate - expected_half_width,
              kIntervalTolerance, message + " lower bound");
  expect_near(result.confidence_interval_95.upper, result.estimate + expected_half_width,
              kIntervalTolerance, message + " upper bound");
}

void test_exact_gbm_step() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.20,
      .risk_free_rate = 0.05,
      .dividend_yield = 0.02,
  };
  const double evolved = nre::exact_gbm_step(market.spot, 0.25, market, 0.5);
  expect_near(evolved, 105.39025620785374, kDeterministicTolerance, "exact GBM quarter-year step");

  auto deterministic_market = market;
  deterministic_market.volatility = 0.0;
  expect_near(nre::exact_gbm_step(100.0, 2.0, deterministic_market, -100.0), 100.0 * std::exp(0.06),
              kDeterministicTolerance, "zero-volatility GBM should ignore the normal draw");

  deterministic_market.risk_free_rate = -0.01;
  expect_near(nre::exact_gbm_step(100.0, 2.0, deterministic_market, 100.0), 100.0 * std::exp(-0.06),
              kDeterministicTolerance, "zero-volatility GBM should support a negative rate");
}

void test_geometric_path_schedule() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.20,
      .risk_free_rate = 0.05,
      .dividend_yield = 0.02,
  };
  constexpr std::array draws{0.0, 0.0};
  expect_near(nre::geometric_average_from_gbm_path(market, 1.0, draws), 100.75281954445339,
              kDeterministicTolerance,
              "geometric path should use t=T/2 and t=T while excluding t=0");

  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            nre::geometric_average_from_gbm_path(market, 1.0, std::span<const double>{}));
      },
      "an empty observation schedule should be rejected");
}

void test_payoffs() {
  expect_near(nre::option_payoff(nre::OptionType::call, 110.0, 100.0), 10.0,
              kDeterministicTolerance, "in-the-money call payoff");
  expect_near(nre::option_payoff(nre::OptionType::call, 90.0, 100.0), 0.0, kDeterministicTolerance,
              "out-of-the-money call payoff");
  expect_near(nre::option_payoff(nre::OptionType::put, 90.0, 100.0), 10.0, kDeterministicTolerance,
              "in-the-money put payoff");
  expect_near(nre::option_payoff(nre::OptionType::put, 110.0, 100.0), 0.0, kDeterministicTolerance,
              "out-of-the-money put payoff");
  expect_near(nre::option_payoff(nre::OptionType::call, 100.0, 100.0), 0.0, kDeterministicTolerance,
              "at-the-money payoff");
}

void test_configuration_and_style_rejection() {
  const nre::MarketState market{};
  const nre::MonteCarloConfig too_few_paths{.seed = 1ULL, .path_count = 1};
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            price_european_monte_carlo(european(nre::OptionType::call), market, too_few_paths));
      },
      "European pricing should reject fewer than two paths");

  const nre::MonteCarloConfig config{.seed = 1ULL, .path_count = 2};
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            price_european_monte_carlo(geometric_asian(nre::OptionType::call), market, config));
      },
      "European pricing should reject a geometric Asian contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            price_geometric_asian_monte_carlo(european(nre::OptionType::call), market, config));
      },
      "geometric-Asian pricing should reject a European contract");
}

void test_zero_volatility_against_analytics() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.0,
      .risk_free_rate = 0.03,
      .dividend_yield = 0.01,
  };
  const nre::MonteCarloConfig config{.seed = 77ULL, .path_count = 32};

  for (const auto type : {nre::OptionType::call, nre::OptionType::put}) {
    const auto european_contract = european(type, 100.0);
    const auto european_result = nre::price_european_monte_carlo(european_contract, market, config);
    expect_near(european_result.estimate,
                nre::black_scholes_european(european_contract, market).price,
                kDeterministicTolerance, "zero-volatility European analytical agreement");
    expect_near(european_result.sample_standard_error, 0.0, kDeterministicTolerance,
                "zero-volatility European standard error");

    const auto asian_contract = geometric_asian(type, 100.0, 4);
    const auto asian_result =
        nre::price_geometric_asian_monte_carlo(asian_contract, market, config);
    expect_near(asian_result.estimate,
                nre::geometric_asian_analytical(asian_contract, market).price,
                kDeterministicTolerance, "zero-volatility geometric-Asian analytical agreement");
    expect_near(asian_result.sample_standard_error, 0.0, kDeterministicTolerance,
                "zero-volatility geometric-Asian standard error");
  }
}

void test_fixed_seed_reproducibility_and_metadata() {
  const nre::MarketState market{};
  const nre::MonteCarloConfig config{.seed = 20260716ULL, .path_count = 4096};
  const auto contract = european(nre::OptionType::call);
  const auto first = nre::price_european_monte_carlo(contract, market, config);
  const auto second = nre::price_european_monte_carlo(contract, market, config);

  expect(first.estimate == second.estimate, "fixed seed should reproduce the estimate");
  expect(first.sample_standard_error == second.sample_standard_error,
         "fixed seed should reproduce the standard error");
  expect(first.effective_paths == config.path_count, "result should report effective path count");
  expect(first.seed == config.seed, "result should report its seed");
  expect(first.sample_standard_error > 0.0,
         "a stochastic European run should have positive standard error");
  expect_interval_consistent(first, "European Monte Carlo interval");

  const auto asian = nre::price_geometric_asian_monte_carlo(
      geometric_asian(nre::OptionType::put, 100.0, 12), market, config);
  expect(asian.effective_paths == config.path_count,
         "geometric-Asian result should report effective path count");
  expect(asian.seed == config.seed, "geometric-Asian result should report its seed");
  expect(asian.sample_standard_error > 0.0,
         "a stochastic geometric-Asian run should have positive standard error");
  expect_interval_consistent(asian, "geometric-Asian Monte Carlo interval");
}

}  // namespace

int main() {
  test_exact_gbm_step();
  test_geometric_path_schedule();
  test_payoffs();
  test_configuration_and_style_rejection();
  test_zero_volatility_against_analytics();
  test_fixed_seed_reproducibility_and_metadata();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All Monte Carlo tests passed\n";
  return EXIT_SUCCESS;
}
