#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nre/analytics.hpp"
#include "nre/pricing.hpp"
#include "nre/statistics.hpp"

namespace {

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

template <typename Function>
void expect_invalid_argument(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  } catch (...) {
    std::cerr << "FAIL: " << message << " (unexpected exception type)\n";
    ++failures;
    return;
  }
  std::cerr << "FAIL: " << message << " (no exception)\n";
  ++failures;
}

nre::OptionContract option(nre::OptionStyle style) {
  return {
      .type = nre::OptionType::call,
      .style = style,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = style == nre::OptionStyle::european ? 1U : 12U,
  };
}

void test_streaming_merges() {
  nre::StreamingStatistics complete;
  nre::StreamingStatistics first;
  nre::StreamingStatistics second;
  for (int value = 1; value <= 9; ++value) {
    complete.add(static_cast<double>(value));
    (value <= 4 ? first : second).add(static_cast<double>(value));
  }
  first.merge(second);
  const auto expected = complete.summary();
  const auto actual = first.summary();
  expect(actual.sample_count == expected.sample_count, "merged scalar count");
  expect_near(actual.estimate, expected.estimate, 1.0e-15, "merged scalar mean");
  expect_near(actual.sample_standard_error, expected.sample_standard_error, 1.0e-15,
              "merged scalar standard error");

  nre::StreamingBivariateStatistics complete_pair;
  nre::StreamingBivariateStatistics first_pair;
  nre::StreamingBivariateStatistics second_pair;
  for (int value = 1; value <= 9; ++value) {
    const double first_value = static_cast<double>(value);
    const double second_value = 2.0 * first_value + 1.0;
    complete_pair.add(first_value, second_value);
    (value <= 4 ? first_pair : second_pair).add(first_value, second_value);
  }
  first_pair.merge(second_pair);
  const auto expected_pair = complete_pair.summary();
  const auto actual_pair = first_pair.summary();
  expect_near(actual_pair.sample_covariance, expected_pair.sample_covariance, 1.0e-14,
              "merged covariance");
  expect_near(actual_pair.second_sample_variance, expected_pair.second_sample_variance, 1.0e-14,
              "merged second variance");
}

void test_partitioning_reproducibility_and_metadata() {
  const nre::MarketState market{};
  const auto contract = option(nre::OptionStyle::european);
  const nre::MonteCarloConfig config{.seed = 1234567ULL, .path_count = 10003U, .thread_count = 3U};
  const auto first = nre::price_european_monte_carlo(contract, market, config);
  const auto second = nre::price_european_monte_carlo(contract, market, config);
  expect(first.estimate == second.estimate, "fixed threaded configuration reproduces price");
  expect(first.delta.estimate == second.delta.estimate,
         "fixed threaded configuration reproduces Delta");
  expect(first.effective_paths == 10003U, "non-divisible effective count is preserved");
  expect(first.raw_paths == 10003U, "non-divisible raw count is preserved");
  expect(first.requested_threads == 3U, "requested threads are reported");
  expect(first.active_threads == 3U, "active threads are reported");

  const nre::MonteCarloConfig excess_threads{
      .seed = 8ULL, .path_count = 7U, .thread_count = 64U};
  const auto excess = nre::price_european_monte_carlo(contract, market, excess_threads);
  expect(excess.effective_paths == 7U, "excess workers do not lose samples");
  expect(excess.requested_threads == 64U, "excess requested thread count is retained");
  expect(excess.active_threads == 7U, "workers are capped at useful work");

  const nre::PricingRequest request{
      .contract = contract,
      .market = market,
      .backend = nre::PricingBackend::monte_carlo,
      .estimator = nre::PricingEstimator::plain,
      .monte_carlo_config = config,
  };
  const auto unified = nre::price(request);
  expect(unified.metadata.requested_threads == 3U, "router propagates requested threads");
  expect(unified.metadata.active_threads == 3U, "router propagates active threads");

  const nre::MonteCarloConfig invalid{.seed = 1ULL, .path_count = 8U, .thread_count = 0U};
  expect_invalid_argument(
      [&] { static_cast<void>(nre::price_european_monte_carlo(contract, market, invalid)); },
      "zero thread count is rejected");
}

void test_threaded_estimator_agreement() {
  const nre::MarketState market{};
  const nre::MonteCarloConfig scalar{.seed = 20260716ULL, .path_count = 30000U};
  const nre::MonteCarloConfig threaded{
      .seed = 20260716ULL, .path_count = 30000U, .thread_count = 4U};

  for (const auto style : {nre::OptionStyle::european, nre::OptionStyle::geometric_asian}) {
    const auto contract = option(style);
    const auto analytical = style == nre::OptionStyle::european
                                ? nre::black_scholes_european(contract, market)
                                : nre::geometric_asian_analytical(contract, market);
    const auto result = style == nre::OptionStyle::european
                            ? nre::price_european_monte_carlo(contract, market, threaded)
                            : nre::price_geometric_asian_monte_carlo(contract, market, threaded);
    expect_near(result.estimate, analytical.price, 4.0 * result.sample_standard_error,
                "threaded price agrees with analytical reference");
    expect_near(result.delta.estimate, analytical.delta, 4.0 * result.delta.sample_standard_error,
                "threaded Delta agrees with analytical reference");
  }

  const auto arithmetic = option(nre::OptionStyle::arithmetic_asian);
  const auto scalar_plain = nre::price_arithmetic_asian_monte_carlo(arithmetic, market, scalar);
  const auto threaded_plain =
      nre::price_arithmetic_asian_monte_carlo(arithmetic, market, threaded);
  const double plain_tolerance =
      4.0 * std::hypot(scalar_plain.sample_standard_error, threaded_plain.sample_standard_error);
  expect_near(threaded_plain.estimate, scalar_plain.estimate, plain_tolerance,
              "threaded arithmetic price agrees statistically with scalar");

  const auto antithetic =
      nre::price_arithmetic_asian_antithetic_monte_carlo(arithmetic, market, threaded);
  expect(antithetic.raw_paths == 60000U, "threaded antithetic raw paths");
  expect(antithetic.active_threads == 4U, "threaded antithetic active workers");

  const nre::ControlVariateConfig control{
      .pricing = threaded, .pilot_seed = 20260717ULL, .pilot_path_count = 1001U};
  const auto controlled =
      nre::price_arithmetic_asian_control_variate_monte_carlo(arithmetic, market, control);
  expect(controlled.monte_carlo.raw_paths == 31001U, "threaded control raw paths include pilot");
  expect(controlled.monte_carlo.active_threads == 4U, "threaded control active workers");
  expect(controlled.pilot_active_threads == 4U, "threaded control pilot active workers");
  expect(controlled.control_applied, "threaded price control is applied for stochastic case");
  expect(controlled.delta_control_applied,
         "threaded Delta control is applied for stochastic case");

  const nre::PricingRequest control_request{
      .contract = arithmetic,
      .market = market,
      .backend = nre::PricingBackend::monte_carlo,
      .estimator = nre::PricingEstimator::geometric_control_variate,
      .control_variate_config = control,
  };
  const auto unified_control = nre::price(control_request);
  expect(unified_control.metadata.requested_threads == 4U,
         "router propagates control requested workers");
  expect(unified_control.metadata.pilot_active_threads == 4U,
         "router propagates control pilot active workers");
}

void test_zero_volatility_invariance() {
  nre::MarketState market{};
  market.volatility = 0.0;
  const auto arithmetic = option(nre::OptionStyle::arithmetic_asian);
  const nre::MonteCarloConfig scalar{.seed = 12ULL, .path_count = 17U};
  const nre::MonteCarloConfig threaded{.seed = 12ULL, .path_count = 17U, .thread_count = 5U};
  const auto first = nre::price_arithmetic_asian_monte_carlo(arithmetic, market, scalar);
  const auto second = nre::price_arithmetic_asian_monte_carlo(arithmetic, market, threaded);
  expect_near(first.estimate, second.estimate, 1.0e-14,
              "deterministic price is invariant across thread counts");
  expect_near(first.delta.estimate, second.delta.estimate, 1.0e-14,
              "deterministic Delta is invariant across thread counts");
}

}  // namespace

int main() {
  test_streaming_merges();
  test_partitioning_reproducibility_and_metadata();
  test_threaded_estimator_agreement();
  test_zero_volatility_invariance();
  if (failures != 0) {
    std::cerr << failures << " threading test(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "All threading tests passed\n";
  return EXIT_SUCCESS;
}
