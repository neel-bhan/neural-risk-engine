#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "nre/analytics.hpp"
#include "nre/pricing.hpp"

namespace {

constexpr double kTolerance = 2.0e-12;
int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, const std::string& message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > kTolerance) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected << ")\n";
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

nre::OptionContract contract(nre::OptionStyle style, nre::OptionType type = nre::OptionType::call) {
  return {
      .type = type,
      .style = style,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = style == nre::OptionStyle::european ? 1U : 12U,
  };
}

nre::PricingRequest pricing_request(
    const nre::OptionContract& option, const nre::MarketState& market,
    nre::PricingBackend backend = nre::PricingBackend::analytical,
    nre::PricingEstimator estimator = nre::PricingEstimator::analytical,
    std::optional<nre::MonteCarloConfig> monte_carlo_config = std::nullopt,
    std::optional<nre::ControlVariateConfig> control_variate_config = std::nullopt) {
  return {
      .contract = option,
      .market = market,
      .backend = backend,
      .estimator = estimator,
      .monte_carlo_config = monte_carlo_config,
      .control_variate_config = control_variate_config,
  };
}

void expect_stochastic_equal(const nre::UnifiedPricingResult& unified,
                             const nre::MonteCarloResult& direct, const std::string& message) {
  expect_near(unified.price.estimate, direct.estimate, message + " price");
  expect_near(unified.delta.estimate, direct.delta.estimate, message + " Delta");
  expect(unified.price.sample_standard_error.has_value(), message + " price SE is available");
  expect(unified.delta.sample_standard_error.has_value(), message + " Delta SE is available");
  expect(unified.price.confidence_interval_95.has_value(), message + " price CI is available");
  expect(unified.delta.confidence_interval_95.has_value(), message + " Delta CI is available");
  if (unified.price.sample_standard_error.has_value()) {
    expect_near(*unified.price.sample_standard_error, direct.sample_standard_error,
                message + " price SE");
  }
  if (unified.delta.sample_standard_error.has_value()) {
    expect_near(*unified.delta.sample_standard_error, direct.delta.sample_standard_error,
                message + " Delta SE");
  }
  expect(unified.metadata.effective_paths == direct.effective_paths, message + " effective paths");
  expect(unified.metadata.raw_paths == direct.raw_paths, message + " raw paths");
  expect(unified.metadata.seed == direct.seed, message + " seed");
}

void test_analytical_routing() {
  const nre::MarketState market{};
  for (const auto style : {nre::OptionStyle::european, nre::OptionStyle::geometric_asian}) {
    const auto option = contract(style, nre::OptionType::put);
    const auto request = pricing_request(option, market);
    const auto unified = nre::price(request);
    const auto direct = style == nre::OptionStyle::european
                            ? nre::black_scholes_european(option, market)
                            : nre::geometric_asian_analytical(option, market);
    expect_near(unified.price.estimate, direct.price, "unified analytical price");
    expect_near(unified.delta.estimate, direct.delta, "unified analytical Delta");
    expect(!unified.price.sample_standard_error.has_value(), "analytical price has no sampling SE");
    expect(!unified.delta.sample_standard_error.has_value(), "analytical Delta has no sampling SE");
    expect(!unified.metadata.effective_paths.has_value(), "analytical result has no path count");
    expect(unified.backend == nre::PricingBackend::analytical,
           "analytical result records its backend");
    expect(unified.estimator == nre::PricingEstimator::analytical,
           "analytical result records its estimator");
  }
}

void test_monte_carlo_routing_and_metadata() {
  const nre::MarketState market{};
  const nre::MonteCarloConfig config{.seed = 98765ULL, .path_count = 256};

  for (const auto style : {nre::OptionStyle::european, nre::OptionStyle::geometric_asian,
                           nre::OptionStyle::arithmetic_asian}) {
    const auto option = contract(style);
    const auto request = pricing_request(option, market, nre::PricingBackend::monte_carlo,
                                         nre::PricingEstimator::plain, config);
    const auto unified = nre::price(request);
    nre::MonteCarloResult direct{};
    if (style == nre::OptionStyle::european) {
      direct = nre::price_european_monte_carlo(option, market, config);
    } else if (style == nre::OptionStyle::geometric_asian) {
      direct = nre::price_geometric_asian_monte_carlo(option, market, config);
    } else {
      direct = nre::price_arithmetic_asian_monte_carlo(option, market, config);
    }
    expect_stochastic_equal(unified, direct, "unified plain Monte Carlo");
  }

  const auto arithmetic = contract(nre::OptionStyle::arithmetic_asian);
  const auto antithetic_request = pricing_request(arithmetic, market,
                                                  nre::PricingBackend::monte_carlo,
                                                  nre::PricingEstimator::antithetic, config);
  const auto antithetic = nre::price(antithetic_request);
  const auto direct_antithetic =
      nre::price_arithmetic_asian_antithetic_monte_carlo(arithmetic, market, config);
  expect_stochastic_equal(antithetic, direct_antithetic, "unified antithetic Monte Carlo");

  const nre::ControlVariateConfig control_config{
      .pricing = config,
      .pilot_seed = 98766ULL,
      .pilot_path_count = 64,
  };
  const auto control_request = pricing_request(
      arithmetic, market, nre::PricingBackend::monte_carlo,
      nre::PricingEstimator::geometric_control_variate, std::nullopt, control_config);
  const auto control = nre::price(control_request);
  const auto direct_control =
      nre::price_arithmetic_asian_control_variate_monte_carlo(arithmetic, market, control_config);
  expect_stochastic_equal(control, direct_control.monte_carlo,
                          "unified control-variate Monte Carlo");
  expect(control.metadata.pilot_seed == control_config.pilot_seed,
         "unified control result reports pilot seed");
  expect(control.metadata.pilot_paths == control_config.pilot_path_count,
         "unified control result reports pilot paths");
  expect(control.metadata.price_control_coefficient == direct_control.coefficient,
         "unified control result reports price coefficient");
  expect(control.metadata.delta_control_coefficient == direct_control.delta_coefficient,
         "unified control result reports Delta coefficient");
  expect(control.metadata.price_control_applied == direct_control.control_applied,
         "unified control result reports price application status");
  expect(control.metadata.delta_control_applied == direct_control.delta_control_applied,
         "unified control result reports Delta application status");
}

void test_validation_and_unsupported_combinations() {
  const nre::MarketState market{};
  const nre::MonteCarloConfig config{.seed = 1ULL, .path_count = 8};
  const nre::ControlVariateConfig control_config{
      .pricing = config,
      .pilot_seed = 2ULL,
      .pilot_path_count = 8,
  };

  auto invalid_contract = contract(nre::OptionStyle::european);
  invalid_contract.strike = -1.0;
  expect_invalid_argument(
      [&] { static_cast<void>(nre::price(pricing_request(invalid_contract, market))); },
      "router validates contracts");
  auto invalid_market = market;
  invalid_market.spot = std::numeric_limits<double>::quiet_NaN();
  expect_invalid_argument(
      [&] {
        static_cast<void>(
            nre::price(pricing_request(contract(nre::OptionStyle::european), invalid_market)));
      },
      "router validates markets");

  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(
            pricing_request(contract(nre::OptionStyle::arithmetic_asian), market)));
      },
      "router rejects analytical arithmetic-Asian requests");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::european), market, nre::PricingBackend::analytical,
            nre::PricingEstimator::plain)));
      },
      "analytical backend rejects a Monte Carlo estimator");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::european), market, nre::PricingBackend::analytical,
            nre::PricingEstimator::analytical, config)));
      },
      "analytical backend rejects extraneous Monte Carlo configuration");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::european), market, nre::PricingBackend::monte_carlo,
            nre::PricingEstimator::plain)));
      },
      "plain Monte Carlo rejects missing configuration");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::european), market, nre::PricingBackend::monte_carlo,
            nre::PricingEstimator::antithetic, config)));
      },
      "router rejects unsupported antithetic style");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::geometric_asian), market,
            nre::PricingBackend::monte_carlo,
            nre::PricingEstimator::geometric_control_variate, std::nullopt, control_config)));
      },
      "router rejects unsupported control-variate style");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::arithmetic_asian), market,
            nre::PricingBackend::monte_carlo,
            nre::PricingEstimator::geometric_control_variate, config, control_config)));
      },
      "router rejects incompatible duplicate configuration");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::european), market,
            static_cast<nre::PricingBackend>(999), nre::PricingEstimator::analytical)));
      },
      "router rejects unknown backends");
  expect_invalid_argument(
      [&] {
        static_cast<void>(nre::price(pricing_request(
            contract(nre::OptionStyle::european), market, nre::PricingBackend::monte_carlo,
            static_cast<nre::PricingEstimator>(999), config)));
      },
      "router rejects unknown estimators");
}

}  // namespace

int main() {
  test_analytical_routing();
  test_monte_carlo_routing_and_metadata();
  test_validation_and_unsupported_combinations();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "All pricing interface tests passed\n";
  return EXIT_SUCCESS;
}
