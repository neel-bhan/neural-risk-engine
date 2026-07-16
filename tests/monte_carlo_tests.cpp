#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "nre/analytics.hpp"
#include "nre/monte_carlo.hpp"
#include "nre/random.hpp"

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

nre::OptionContract arithmetic_asian(nre::OptionType type, double strike = 100.0,
                                     std::size_t observations = 4) {
  return {
      .type = type,
      .style = nre::OptionStyle::arithmetic_asian,
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

void expect_delta_interval_consistent(const nre::MonteCarloResult& result,
                                      const std::string& message) {
  const double expected_half_width = 1.96 * result.delta.sample_standard_error;
  expect_near(result.delta.confidence_interval_95.lower,
              result.delta.estimate - expected_half_width, kIntervalTolerance,
              message + " lower bound");
  expect_near(result.delta.confidence_interval_95.upper,
              result.delta.estimate + expected_half_width, kIntervalTolerance,
              message + " upper bound");
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

void test_arithmetic_path_schedule_and_average() {
  // With unit time steps, sigma=1, and r=sigma^2/2, the GBM drift term is zero. These draws
  // therefore evolve the supplied spot through the hand-calculated observations 110 and 120.
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 1.0,
      .risk_free_rate = 0.5,
      .dividend_yield = 0.0,
  };
  const std::array draws{std::log(1.1), std::log(12.0 / 11.0)};
  const double arithmetic_average = nre::arithmetic_average_from_gbm_path(market, 2.0, draws);
  const double geometric_average = nre::geometric_average_from_gbm_path(market, 2.0, draws);

  expect_near(arithmetic_average, 115.0, kDeterministicTolerance,
              "arithmetic average of the observed spots 110 and 120");
  expect(std::abs(arithmetic_average - 110.0) > kDeterministicTolerance,
         "arithmetic path average should exclude the initial spot of 100");
  expect(std::abs(arithmetic_average - geometric_average) > kDeterministicTolerance,
         "arithmetic and geometric averages should differ on a non-constant path");

  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            nre::arithmetic_average_from_gbm_path(market, 1.0, std::span<const double>{}));
      },
      "an empty arithmetic observation schedule should be rejected");
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

  expect_near(nre::control_variate_adjusted_sample(10.0, 7.0, 5.0, 1.5), 7.0,
              kDeterministicTolerance, "control-variate adjustment sign and centering");
}

void test_supplied_draw_pathwise_delta_and_kinks() {
  const nre::MarketState kink_market{
      .spot = 100.0,
      .volatility = 0.0,
      .risk_free_rate = 0.0,
      .dividend_yield = 0.0,
  };
  constexpr std::array one_draw{9.0};
  const auto call_kink = nre::discounted_pathwise_sample(european(nre::OptionType::call, 100.0),
                                                         kink_market, one_draw);
  const auto put_kink =
      nre::discounted_pathwise_sample(european(nre::OptionType::put, 100.0), kink_market, one_draw);
  expect_near(call_kink.discounted_payoff, 0.0, kDeterministicTolerance,
              "call payoff at a deterministic kink");
  expect_near(call_kink.discounted_delta, 0.5, kDeterministicTolerance,
              "call pathwise Delta uses half the kink jump");
  expect_near(put_kink.discounted_delta, -0.5, kDeterministicTolerance,
              "put pathwise Delta uses half the kink jump");

  // The two post-initial observations are exactly 110 and 120. This checks the arithmetic
  // monitoring schedule and the homogeneous A/S0 path derivative independently of the RNG.
  const nre::MarketState path_market{
      .spot = 100.0,
      .volatility = 1.0,
      .risk_free_rate = 0.5,
      .dividend_yield = 0.0,
  };
  const std::array draws{std::log(1.1), std::log(12.0 / 11.0)};
  auto contract = arithmetic_asian(nre::OptionType::call, 110.0, draws.size());
  contract.maturity_years = 2.0;
  const auto sample = nre::discounted_pathwise_sample(contract, path_market, draws);
  expect_near(sample.discounted_payoff, 5.0 * std::exp(-1.0), kDeterministicTolerance,
              "supplied draws produce the hand-calculated arithmetic payoff");
  expect_near(sample.discounted_delta, 1.15 * std::exp(-1.0), kDeterministicTolerance,
              "arithmetic pathwise Delta is discounted A/S0 on an active call path");

  constexpr std::array zero_draws{0.0, 0.0};
  auto asian_kink_contract = arithmetic_asian(nre::OptionType::call, 100.0, zero_draws.size());
  const auto asian_kink =
      nre::discounted_pathwise_sample(asian_kink_contract, kink_market, zero_draws);
  expect_near(asian_kink.discounted_delta, 0.5, kDeterministicTolerance,
              "arithmetic-Asian pathwise Delta uses half the kink jump");

  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(nre::discounted_pathwise_sample(contract, path_market, one_draw)); },
      "pathwise sampling should reject a draw count that differs from the monitoring schedule");
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
            price_european_monte_carlo(arithmetic_asian(nre::OptionType::call), market, config));
      },
      "European pricing should reject an arithmetic Asian contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            price_geometric_asian_monte_carlo(european(nre::OptionType::call), market, config));
      },
      "geometric-Asian pricing should reject a European contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_geometric_asian_monte_carlo(arithmetic_asian(nre::OptionType::call),
                                                            market, config));
      },
      "geometric-Asian pricing should reject an arithmetic Asian contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_monte_carlo(
            arithmetic_asian(nre::OptionType::call), market, too_few_paths));
      },
      "arithmetic-Asian pricing should reject fewer than two paths");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            price_arithmetic_asian_monte_carlo(european(nre::OptionType::call), market, config));
      },
      "arithmetic-Asian pricing should reject a European contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_monte_carlo(geometric_asian(nre::OptionType::call),
                                                             market, config));
      },
      "arithmetic-Asian pricing should reject a geometric Asian contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_antithetic_monte_carlo(
            european(nre::OptionType::call), market, config));
      },
      "antithetic pricing should reject a European contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_antithetic_monte_carlo(
            geometric_asian(nre::OptionType::call), market, config));
      },
      "antithetic pricing should reject a geometric Asian contract");

  const nre::ControlVariateConfig control_config{
      .pricing = config,
      .pilot_seed = 2ULL,
      .pilot_path_count = 2,
  };
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_control_variate_monte_carlo(
            european(nre::OptionType::call), market, control_config));
      },
      "control-variate pricing should reject a European contract");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_control_variate_monte_carlo(
            geometric_asian(nre::OptionType::call), market, control_config));
      },
      "control-variate pricing should reject a geometric Asian contract");

  auto invalid_control_config = control_config;
  invalid_control_config.pilot_path_count = 1;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_control_variate_monte_carlo(
            arithmetic_asian(nre::OptionType::call), market, invalid_control_config));
      },
      "control-variate pricing should reject fewer than two pilot paths");
  invalid_control_config = control_config;
  invalid_control_config.pilot_seed = invalid_control_config.pricing.seed;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_control_variate_monte_carlo(
            arithmetic_asian(nre::OptionType::call), market, invalid_control_config));
      },
      "control-variate pricing should require independent pilot and pricing seeds");

  const nre::MonteCarloConfig overflowing_config{
      .seed = 1ULL,
      .path_count = std::numeric_limits<std::size_t>::max(),
  };
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(price_arithmetic_asian_antithetic_monte_carlo(
            arithmetic_asian(nre::OptionType::call), market, overflowing_config));
      },
      "antithetic pricing should reject overflowing raw-path metadata");
}

void test_arithmetic_asian_zero_volatility() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.0,
      .risk_free_rate = 0.03,
      .dividend_yield = 0.01,
  };
  const nre::MonteCarloConfig config{.seed = 77ULL, .path_count = 32};
  constexpr std::size_t kObservations = 4;
  constexpr double kMaturity = 1.0;
  const double time_step = kMaturity / static_cast<double>(kObservations);
  double expected_average = 0.0;
  for (std::size_t observation = 1; observation <= kObservations; ++observation) {
    const double time = static_cast<double>(observation) * time_step;
    expected_average +=
        market.spot * std::exp((market.risk_free_rate - market.dividend_yield) * time);
  }
  expected_average /= static_cast<double>(kObservations);
  const double discount = std::exp(-market.risk_free_rate * kMaturity);

  const auto call = nre::price_arithmetic_asian_monte_carlo(
      arithmetic_asian(nre::OptionType::call, 100.0, kObservations), market, config);
  const auto put = nre::price_arithmetic_asian_monte_carlo(
      arithmetic_asian(nre::OptionType::put, 105.0, kObservations), market, config);

  expect_near(call.estimate, discount * std::max(expected_average - 100.0, 0.0),
              kDeterministicTolerance, "zero-volatility arithmetic-Asian call discounted payoff");
  expect_near(put.estimate, discount * std::max(105.0 - expected_average, 0.0),
              kDeterministicTolerance, "zero-volatility arithmetic-Asian put discounted payoff");
  expect_near(call.sample_standard_error, 0.0, kDeterministicTolerance,
              "zero-volatility arithmetic-Asian call standard error");
  expect_near(put.sample_standard_error, 0.0, kDeterministicTolerance,
              "zero-volatility arithmetic-Asian put standard error");
  expect_near(call.delta.estimate, discount * expected_average / market.spot,
              kDeterministicTolerance, "zero-volatility arithmetic-Asian call pathwise Delta");
  expect_near(put.delta.estimate, -discount * expected_average / market.spot,
              kDeterministicTolerance, "zero-volatility arithmetic-Asian put pathwise Delta");
  expect_near(call.delta.sample_standard_error, 0.0, kDeterministicTolerance,
              "zero-volatility arithmetic-Asian call Delta standard error");

  const auto antithetic_call = nre::price_arithmetic_asian_antithetic_monte_carlo(
      arithmetic_asian(nre::OptionType::call, 100.0, kObservations), market, config);
  const auto antithetic_put = nre::price_arithmetic_asian_antithetic_monte_carlo(
      arithmetic_asian(nre::OptionType::put, 105.0, kObservations), market, config);
  expect_near(antithetic_call.estimate, call.estimate, kDeterministicTolerance,
              "zero-volatility antithetic call should equal the deterministic payoff");
  expect_near(antithetic_put.estimate, put.estimate, kDeterministicTolerance,
              "zero-volatility antithetic put should equal the deterministic payoff");
  expect_near(antithetic_call.sample_standard_error, 0.0, kDeterministicTolerance,
              "zero-volatility antithetic call standard error");
  expect_near(antithetic_put.sample_standard_error, 0.0, kDeterministicTolerance,
              "zero-volatility antithetic put standard error");
  expect_near(antithetic_call.delta.estimate, call.delta.estimate, kDeterministicTolerance,
              "zero-volatility antithetic call Delta");
  expect(antithetic_call.effective_paths == config.path_count,
         "antithetic result should report paired effective samples");
  expect(antithetic_call.raw_paths == 2 * config.path_count,
         "antithetic result should report both raw paths in every pair");

  const nre::ControlVariateConfig control_config{
      .pricing = config,
      .pilot_seed = 78ULL,
      .pilot_path_count = 8,
  };
  const auto control_call = nre::price_arithmetic_asian_control_variate_monte_carlo(
      arithmetic_asian(nre::OptionType::call, 100.0, kObservations), market, control_config);
  expect(!control_call.control_applied,
         "zero control variance should fall back to the plain estimator");
  expect_near(control_call.coefficient, 0.0, kDeterministicTolerance,
              "zero control variance should use a zero coefficient");
  expect_near(control_call.monte_carlo.estimate, call.estimate, kDeterministicTolerance,
              "degenerate control variate should retain the plain deterministic estimate");
  expect(!control_call.delta_control_applied,
         "zero Delta control variance should fall back to the plain Delta estimator");
  expect_near(control_call.monte_carlo.delta.estimate, call.delta.estimate, kDeterministicTolerance,
              "degenerate Delta control should retain the deterministic pathwise Delta");
  expect(control_call.monte_carlo.raw_paths == config.path_count + control_config.pilot_path_count,
         "control-variate raw paths should include pilot and pricing work");
  expect(control_call.pilot_paths == control_config.pilot_path_count,
         "control-variate result should report pilot paths");
  expect(control_call.pilot_seed == control_config.pilot_seed,
         "control-variate result should report the pilot seed");
}

void test_antithetic_pair_uses_positive_and_negative_draws() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.30,
      .risk_free_rate = 0.04,
      .dividend_yield = 0.01,
  };
  const auto contract = arithmetic_asian(nre::OptionType::call, 97.0, 3);
  const nre::MonteCarloConfig config{.seed = 314159ULL, .path_count = 2};
  const auto result = nre::price_arithmetic_asian_antithetic_monte_carlo(contract, market, config);

  nre::NormalGenerator generator(config.seed);
  nre::StreamingStatistics expected_statistics;
  std::array<double, 3> draws{};
  const double discount = std::exp(-market.risk_free_rate * contract.maturity_years);
  for (std::size_t sample = 0; sample < config.path_count; ++sample) {
    for (double& draw : draws) {
      draw = generator.next();
    }
    const double positive_average =
        nre::arithmetic_average_from_gbm_path(market, contract.maturity_years, draws);
    for (double& draw : draws) {
      draw = -draw;
    }
    const double negative_average =
        nre::arithmetic_average_from_gbm_path(market, contract.maturity_years, draws);
    expected_statistics.add(0.5 * discount *
                            (nre::option_payoff(contract.type, positive_average, contract.strike) +
                             nre::option_payoff(contract.type, negative_average, contract.strike)));
  }
  const auto expected = expected_statistics.summary();
  expect_near(result.estimate, expected.estimate, kDeterministicTolerance,
              "antithetic estimator should pair supplied z and -z draws");
  expect_near(result.sample_standard_error, expected.sample_standard_error, kDeterministicTolerance,
              "antithetic paired-sample standard error");
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
    expect_near(european_result.delta.estimate,
                nre::black_scholes_european(european_contract, market).delta,
                kDeterministicTolerance, "zero-volatility European Delta agreement");
    expect_near(european_result.delta.sample_standard_error, 0.0, kDeterministicTolerance,
                "zero-volatility European Delta standard error");

    const auto asian_contract = geometric_asian(type, 100.0, 4);
    const auto asian_result =
        nre::price_geometric_asian_monte_carlo(asian_contract, market, config);
    expect_near(asian_result.estimate,
                nre::geometric_asian_analytical(asian_contract, market).price,
                kDeterministicTolerance, "zero-volatility geometric-Asian analytical agreement");
    expect_near(asian_result.sample_standard_error, 0.0, kDeterministicTolerance,
                "zero-volatility geometric-Asian standard error");
    expect_near(asian_result.delta.estimate,
                nre::geometric_asian_analytical(asian_contract, market).delta,
                kDeterministicTolerance, "zero-volatility geometric-Asian Delta agreement");
  }
}

void test_delta_draw_reuse_and_bump_validation() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.25,
      .risk_free_rate = 0.03,
      .dividend_yield = 0.01,
  };
  const nre::MonteCarloConfig config{.seed = 555123ULL, .path_count = 2048};
  const nre::SpotBumpRule tiny_bump{.relative_size = 1.0e-6, .minimum_absolute_size = 1.0e-8};

  // Strike one keeps every sampled call path active for this fixed fixture. Pathwise and CRN
  // finite-difference samples then use the same homogeneous payoff derivative path by path.
  const auto european_contract = european(nre::OptionType::call, 1.0);
  const auto pathwise = nre::price_european_monte_carlo(european_contract, market, config);
  const auto bumped =
      nre::delta_bump_and_revalue_monte_carlo(european_contract, market, config, tiny_bump);
  expect_near(bumped.delta.estimate, pathwise.delta.estimate, 2.0e-10,
              "European pathwise and CRN Delta reuse identical draws");
  expect_near(bumped.delta.sample_standard_error, pathwise.delta.sample_standard_error, 2.0e-10,
              "European pathwise and CRN Delta standard errors agree on active paths");
  expect(bumped.raw_paths == 2 * config.path_count,
         "plain bump-and-revalue counts both up and down evolutions");
  expect(bumped.seed == config.seed, "bump-and-revalue reports its seed");

  const auto arithmetic_contract = arithmetic_asian(nre::OptionType::call, 1.0, 12);
  const auto antithetic =
      nre::price_arithmetic_asian_antithetic_monte_carlo(arithmetic_contract, market, config);
  const auto bumped_antithetic = nre::delta_bump_and_revalue_arithmetic_antithetic_monte_carlo(
      arithmetic_contract, market, config, tiny_bump);
  expect_near(bumped_antithetic.delta.estimate, antithetic.delta.estimate, 2.0e-10,
              "antithetic pathwise and CRN Delta reuse z and -z draws");
  expect(bumped_antithetic.raw_paths == 4 * config.path_count,
         "antithetic bump-and-revalue counts up/down z and -z evolutions");

  const nre::ControlVariateConfig control_config{
      .pricing = config,
      .pilot_seed = 555124ULL,
      .pilot_path_count = 256,
  };
  const auto control = nre::price_arithmetic_asian_control_variate_monte_carlo(
      arithmetic_contract, market, control_config);
  const auto bumped_control = nre::delta_bump_and_revalue_arithmetic_control_variate_monte_carlo(
      arithmetic_contract, market, control_config, tiny_bump);
  expect_near(bumped_control.bump_and_revalue.delta.estimate, control.monte_carlo.delta.estimate,
              2.0e-8, "control-variate pathwise and CRN Delta use matched pilot and pricing draws");
  expect(bumped_control.bump_and_revalue.raw_paths ==
             2 * (config.path_count + control_config.pilot_path_count),
         "control bump-and-revalue raw paths include up/down pilot and pricing evolutions");
  expect(bumped_control.pilot_seed == control_config.pilot_seed,
         "control bump-and-revalue reports its pilot seed");

  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            nre::centered_spot_bump(1.0, {.relative_size = 2.0, .minimum_absolute_size = 1.0e-6}));
      },
      "a centered bump that makes the down spot non-positive should be rejected");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(nre::centered_spot_bump(
            100.0, {.relative_size = 0.0, .minimum_absolute_size = 1.0e-6}));
      },
      "a non-positive relative bump should be rejected");

  const nre::MarketState kink_market{
      .spot = 100.0,
      .volatility = 0.0,
      .risk_free_rate = 0.0,
      .dividend_yield = 0.0,
  };
  const auto kink_delta = nre::delta_bump_and_revalue_monte_carlo(
      european(nre::OptionType::call, 100.0), kink_market, config);
  expect_near(kink_delta.delta.estimate, 0.5, kDeterministicTolerance,
              "centered bump-and-revalue matches the half-jump kink convention");
  expect_near(kink_delta.delta.sample_standard_error, 0.0, kDeterministicTolerance,
              "deterministic kink bump Delta has zero sampling error");
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
  expect(first.raw_paths == config.path_count,
         "plain result should report one raw path per sample");
  expect(first.seed == config.seed, "result should report its seed");
  expect(first.sample_standard_error > 0.0,
         "a stochastic European run should have positive standard error");
  expect_interval_consistent(first, "European Monte Carlo interval");
  expect(first.delta.estimate > 0.0, "a call pathwise Delta estimate should be positive");
  expect(first.delta.sample_standard_error > 0.0,
         "a stochastic European Delta should have positive standard error");
  expect_delta_interval_consistent(first, "European Monte Carlo Delta interval");

  const auto asian = nre::price_geometric_asian_monte_carlo(
      geometric_asian(nre::OptionType::put, 100.0, 12), market, config);
  expect(asian.effective_paths == config.path_count,
         "geometric-Asian result should report effective path count");
  expect(asian.raw_paths == config.path_count,
         "geometric-Asian result should report one raw path per sample");
  expect(asian.seed == config.seed, "geometric-Asian result should report its seed");
  expect(asian.sample_standard_error > 0.0,
         "a stochastic geometric-Asian run should have positive standard error");
  expect(asian.delta.estimate < 0.0, "a put pathwise Delta estimate should be negative");
  expect_interval_consistent(asian, "geometric-Asian Monte Carlo interval");

  const auto arithmetic_contract = arithmetic_asian(nre::OptionType::call, 100.0, 12);
  const auto arithmetic_first =
      nre::price_arithmetic_asian_monte_carlo(arithmetic_contract, market, config);
  const auto arithmetic_second =
      nre::price_arithmetic_asian_monte_carlo(arithmetic_contract, market, config);
  expect(arithmetic_first.estimate == arithmetic_second.estimate,
         "fixed seed should reproduce the arithmetic-Asian estimate");
  expect(arithmetic_first.sample_standard_error == arithmetic_second.sample_standard_error,
         "fixed seed should reproduce the arithmetic-Asian standard error");
  expect(arithmetic_first.effective_paths == config.path_count,
         "arithmetic-Asian result should report effective path count");
  expect(arithmetic_first.raw_paths == config.path_count,
         "plain arithmetic-Asian result should report one raw path per sample");
  expect(arithmetic_first.seed == config.seed, "arithmetic-Asian result should report its seed");
  expect(arithmetic_first.sample_standard_error > 0.0,
         "a stochastic arithmetic-Asian run should have positive standard error");
  expect_interval_consistent(arithmetic_first, "arithmetic-Asian Monte Carlo interval");

  const auto antithetic_first =
      nre::price_arithmetic_asian_antithetic_monte_carlo(arithmetic_contract, market, config);
  const auto antithetic_second =
      nre::price_arithmetic_asian_antithetic_monte_carlo(arithmetic_contract, market, config);
  expect(antithetic_first.estimate == antithetic_second.estimate,
         "fixed seed should reproduce the antithetic estimate");
  expect(antithetic_first.sample_standard_error == antithetic_second.sample_standard_error,
         "fixed seed should reproduce the antithetic standard error");
  expect(antithetic_first.effective_paths == config.path_count,
         "antithetic result should report paired effective samples");
  expect(antithetic_first.raw_paths == 2 * config.path_count,
         "antithetic result should report two raw paths per effective sample");
  expect(antithetic_first.seed == config.seed, "antithetic result should report its seed");
  expect_interval_consistent(antithetic_first, "antithetic Monte Carlo interval");

  for (const auto type : {nre::OptionType::call, nre::OptionType::put}) {
    const auto control_contract = arithmetic_asian(type, 100.0, 12);
    const nre::ControlVariateConfig control_config{
        .pricing = config,
        .pilot_seed = 20260717ULL,
        .pilot_path_count = 512,
    };
    const auto control_first = nre::price_arithmetic_asian_control_variate_monte_carlo(
        control_contract, market, control_config);
    const auto control_second = nre::price_arithmetic_asian_control_variate_monte_carlo(
        control_contract, market, control_config);
    expect(control_first.monte_carlo.estimate == control_second.monte_carlo.estimate,
           "fixed seeds should reproduce the control-variate estimate");
    expect(control_first.monte_carlo.sample_standard_error ==
               control_second.monte_carlo.sample_standard_error,
           "fixed seeds should reproduce the control-variate standard error");
    expect(control_first.control_applied,
           "a stochastic arithmetic Asian payoff should use its geometric control");
    expect(control_first.delta_control_applied,
           "a stochastic arithmetic Asian Delta should use its geometric Delta control");
    expect(std::isfinite(control_first.coefficient),
           "control-variate coefficient should be finite");
    expect(control_first.monte_carlo.effective_paths == config.path_count,
           "control-variate result should report pricing effective paths");
    expect(
        control_first.monte_carlo.raw_paths == config.path_count + control_config.pilot_path_count,
        "control-variate raw paths should include independent pilot work");
    expect(control_first.monte_carlo.seed == config.seed,
           "control-variate result should report the pricing seed");
    expect(control_first.pilot_seed == control_config.pilot_seed,
           "control-variate result should report the pilot seed");
    expect(control_first.pilot_paths == control_config.pilot_path_count,
           "control-variate result should report the pilot path count");
    expect(control_first.monte_carlo.sample_standard_error > 0.0,
           "stochastic control-variate pricing should report positive standard error");
    expect_interval_consistent(control_first.monte_carlo, "control-variate Monte Carlo interval");
    expect_delta_interval_consistent(control_first.monte_carlo,
                                     "control-variate Monte Carlo Delta interval");
  }
}

}  // namespace

int main() {
  test_exact_gbm_step();
  test_geometric_path_schedule();
  test_arithmetic_path_schedule_and_average();
  test_payoffs();
  test_supplied_draw_pathwise_delta_and_kinks();
  test_configuration_and_style_rejection();
  test_arithmetic_asian_zero_volatility();
  test_antithetic_pair_uses_positive_and_negative_draws();
  test_zero_volatility_against_analytics();
  test_delta_draw_reuse_and_bump_validation();
  test_fixed_seed_reproducibility_and_metadata();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All Monte Carlo tests passed\n";
  return EXIT_SUCCESS;
}
