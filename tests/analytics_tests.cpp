#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nre/analytics.hpp"

namespace {

// Fixture values are rounded from 80-decimal independent calculations. This absolute tolerance
// allows for last-place differences in standard-library transcendental functions.
constexpr double kFixtureTolerance = 2.0e-12;
constexpr double kInvariantTolerance = 2.0e-12;
// The centered difference uses a 1e-4 relative bump; this covers its truncation error while still
// being tight enough to catch discounting and sign mistakes.
constexpr double kDeltaTolerance = 2.0e-8;
constexpr double kRelativeSpotBump = 1.0e-4;

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

nre::OptionContract european(nre::OptionType type, double strike = 100.0, double maturity = 1.0) {
  return {
      .type = type,
      .style = nre::OptionStyle::european,
      .strike = strike,
      .maturity_years = maturity,
      .observations = 1,
  };
}

nre::OptionContract geometric_asian(nre::OptionType type, double strike = 100.0,
                                    double maturity = 1.0, std::size_t observations = 12) {
  return {
      .type = type,
      .style = nre::OptionStyle::geometric_asian,
      .strike = strike,
      .maturity_years = maturity,
      .observations = observations,
  };
}

double finite_difference_delta(const nre::OptionContract& contract,
                               const nre::MarketState& market) {
  const double bump = kRelativeSpotBump * market.spot;
  auto up = market;
  auto down = market;
  up.spot += bump;
  down.spot -= bump;
  const double up_price = nre::black_scholes_european(contract, up).price;
  const double down_price = nre::black_scholes_european(contract, down).price;
  return (up_price - down_price) / (2.0 * bump);
}

double geometric_finite_difference_delta(const nre::OptionContract& contract,
                                         const nre::MarketState& market) {
  const double bump = kRelativeSpotBump * market.spot;
  auto up = market;
  auto down = market;
  up.spot += bump;
  down.spot -= bump;
  const double up_price = nre::geometric_asian_analytical(contract, up).price;
  const double down_price = nre::geometric_asian_analytical(contract, down).price;
  return (up_price - down_price) / (2.0 * bump);
}

void test_style_rejection() {
  nre::OptionContract geometric_asian{};
  geometric_asian.style = nre::OptionStyle::geometric_asian;
  geometric_asian.observations = 12;
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(nre::black_scholes_european(geometric_asian, nre::MarketState{})); },
      "Black-Scholes should reject a geometric Asian option");

  nre::OptionContract arithmetic_asian{};
  arithmetic_asian.style = nre::OptionStyle::arithmetic_asian;
  arithmetic_asian.observations = 12;
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(nre::black_scholes_european(arithmetic_asian, nre::MarketState{})); },
      "Black-Scholes should reject an arithmetic Asian option");
}

void test_european_high_precision_fixtures() {
  const nre::MarketState dividend_market{
      .spot = 100.0,
      .volatility = 0.20,
      .risk_free_rate = 0.05,
      .dividend_yield = 0.02,
  };
  const auto call = nre::black_scholes_european(european(nre::OptionType::call), dividend_market);
  const auto put = nre::black_scholes_european(european(nre::OptionType::put), dividend_market);
  expect_near(call.price, 9.2270055081540475, kFixtureTolerance,
              "European call fixture with dividend yield");
  expect_near(call.delta, 0.5868511461347640, kFixtureTolerance,
              "European call Delta fixture with dividend yield");
  expect_near(put.price, 6.3300806275499182, kFixtureTolerance,
              "European put fixture with dividend yield");
  expect_near(put.delta, -0.3933475271719913, kFixtureTolerance,
              "European put Delta fixture with dividend yield");

  const nre::MarketState negative_rate_market{
      .spot = 95.0,
      .volatility = 0.30,
      .risk_free_rate = -0.01,
      .dividend_yield = 0.015,
  };
  const auto negative_rate_call = nre::black_scholes_european(
      european(nre::OptionType::call, 100.0, 2.0), negative_rate_market);
  const auto negative_rate_put =
      nre::black_scholes_european(european(nre::OptionType::put, 100.0, 2.0), negative_rate_market);
  expect_near(negative_rate_call.price, 11.854917553632455, kFixtureTolerance,
              "European call fixture with negative rate");
  expect_near(negative_rate_call.delta, 0.4749185652425172, kFixtureTolerance,
              "European call Delta fixture with negative rate");
  expect_near(negative_rate_put.price, 21.682725869199759, kFixtureTolerance,
              "European put fixture with negative rate");
  expect_near(negative_rate_put.delta, -0.4955269683059910, kFixtureTolerance,
              "European put Delta fixture with negative rate");
}

void test_european_zero_volatility() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.0,
      .risk_free_rate = 0.05,
      .dividend_yield = 0.02,
  };
  const double terminal_spot = 100.0 * std::exp(0.03);
  const double discount = std::exp(-0.05);

  const auto call = nre::black_scholes_european(european(nre::OptionType::call, 90.0), market);
  expect_near(call.price, discount * (terminal_spot - 90.0), kInvariantTolerance,
              "zero-volatility call price");
  expect_near(call.delta, std::exp(-0.02), kInvariantTolerance, "zero-volatility call Delta");

  const auto put = nre::black_scholes_european(european(nre::OptionType::put, 110.0), market);
  expect_near(put.price, discount * (110.0 - terminal_spot), kInvariantTolerance,
              "zero-volatility put price");
  expect_near(put.delta, -std::exp(-0.02), kInvariantTolerance, "zero-volatility put Delta");

  auto kink_market = nre::MarketState{};
  kink_market.volatility = 0.0;
  kink_market.risk_free_rate = 0.0;
  const auto kink_call = nre::black_scholes_european(european(nre::OptionType::call), kink_market);
  const auto kink_put = nre::black_scholes_european(european(nre::OptionType::put), kink_market);
  expect_near(kink_call.price, 0.0, kInvariantTolerance, "zero-volatility call price at the kink");
  expect_near(kink_call.delta, 0.5, kInvariantTolerance,
              "zero-volatility call uses half-Delta at the kink");
  expect_near(kink_put.delta, -0.5, kInvariantTolerance,
              "zero-volatility put uses half-Delta at the kink");
}

void test_european_near_zero_maturity() {
  const nre::MarketState market{
      .spot = 110.0,
      .volatility = 0.20,
      .risk_free_rate = 0.03,
      .dividend_yield = 0.01,
  };
  const auto result =
      nre::black_scholes_european(european(nre::OptionType::call, 100.0, 1.0e-12), market);
  expect(std::isfinite(result.price) && std::isfinite(result.delta),
         "positive near-zero maturity should produce finite analytics");
  expect_near(result.price, 10.0, 5.0e-10,
              "near-zero maturity call should approach intrinsic value");
  expect_near(result.delta, 1.0, 5.0e-12, "near-zero maturity in-the-money call Delta");
}

void test_european_invariants() {
  const nre::MarketState base_market{
      .spot = 100.0,
      .volatility = 0.27,
      .risk_free_rate = 0.03,
      .dividend_yield = 0.01,
  };
  constexpr std::array<double, 5> kSpots{60.0, 80.0, 100.0, 120.0, 140.0};
  double previous_call = -1.0;
  double previous_put = 1.0e9;
  for (const double spot : kSpots) {
    auto market = base_market;
    market.spot = spot;
    const auto call = nre::black_scholes_european(european(nre::OptionType::call), market);
    const auto put = nre::black_scholes_european(european(nre::OptionType::put), market);
    const double discounted_spot = spot * std::exp(-market.dividend_yield);
    const double discounted_strike = 100.0 * std::exp(-market.risk_free_rate);

    expect_near(call.price - put.price, discounted_spot - discounted_strike, kInvariantTolerance,
                "European put-call parity");
    expect(call.price + kInvariantTolerance >= std::max(discounted_spot - discounted_strike, 0.0),
           "European call should satisfy its lower bound");
    expect(call.price <= discounted_spot + kInvariantTolerance,
           "European call should satisfy its upper bound");
    expect(put.price + kInvariantTolerance >= std::max(discounted_strike - discounted_spot, 0.0),
           "European put should satisfy its lower bound");
    expect(put.price <= discounted_strike + kInvariantTolerance,
           "European put should satisfy its upper bound");
    expect(call.price > previous_call, "European call price should increase with spot");
    expect(put.price < previous_put, "European put price should decrease with spot");
    previous_call = call.price;
    previous_put = put.price;
  }

  for (const auto type : {nre::OptionType::call, nre::OptionType::put}) {
    const auto contract = european(type, 105.0, 1.7);
    const auto result = nre::black_scholes_european(contract, base_market);
    expect_near(result.delta, finite_difference_delta(contract, base_market), kDeltaTolerance,
                "European analytical Delta should match centered difference");
  }
}

void test_geometric_asian_style_rejection() {
  expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(
            nre::geometric_asian_analytical(european(nre::OptionType::call), nre::MarketState{}));
      },
      "geometric Asian formula should reject a European option");

  nre::OptionContract arithmetic{};
  arithmetic.style = nre::OptionStyle::arithmetic_asian;
  arithmetic.observations = 12;
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(nre::geometric_asian_analytical(arithmetic, nre::MarketState{})); },
      "geometric Asian formula should reject an arithmetic Asian option");
}

void test_geometric_asian_high_precision_fixtures() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.25,
      .risk_free_rate = 0.03,
      .dividend_yield = 0.01,
  };
  const auto call =
      nre::geometric_asian_analytical(geometric_asian(nre::OptionType::call, 105.0, 1.5), market);
  const auto put =
      nre::geometric_asian_analytical(geometric_asian(nre::OptionType::put, 105.0, 1.5), market);
  expect_near(call.price, 5.5467739788053797, kFixtureTolerance, "geometric Asian call fixture");
  expect_near(call.delta, 0.4357617080392449, kFixtureTolerance,
              "geometric Asian call Delta fixture");
  expect_near(put.price, 9.5114952740204390, kFixtureTolerance, "geometric Asian put fixture");
  expect_near(put.delta, -0.5283884349333594, kFixtureTolerance,
              "geometric Asian put Delta fixture");
}

void test_geometric_asian_zero_volatility() {
  const nre::MarketState market{
      .spot = 100.0,
      .volatility = 0.0,
      .risk_free_rate = 0.04,
      .dividend_yield = 0.01,
  };
  const auto call_contract = geometric_asian(nre::OptionType::call, 95.0, 2.0, 4);
  const double average_time = 2.0 * 5.0 / 8.0;
  const double average = 100.0 * std::exp(0.03 * average_time);
  const double discount = std::exp(-0.08);
  const auto call = nre::geometric_asian_analytical(call_contract, market);
  expect_near(call.price, discount * (average - 95.0), kInvariantTolerance,
              "zero-volatility geometric Asian call price");
  expect_near(call.delta, discount * average / market.spot, kInvariantTolerance,
              "zero-volatility geometric Asian call Delta");

  auto kink_market = nre::MarketState{};
  kink_market.volatility = 0.0;
  kink_market.risk_free_rate = 0.0;
  const auto kink_call =
      nre::geometric_asian_analytical(geometric_asian(nre::OptionType::call), kink_market);
  const auto kink_put =
      nre::geometric_asian_analytical(geometric_asian(nre::OptionType::put), kink_market);
  expect_near(kink_call.delta, 0.5, kInvariantTolerance,
              "zero-volatility geometric call half-Delta at the kink");
  expect_near(kink_put.delta, -0.5, kInvariantTolerance,
              "zero-volatility geometric put half-Delta at the kink");
}

void test_geometric_asian_invariants() {
  const nre::MarketState market{
      .spot = 92.0,
      .volatility = 0.31,
      .risk_free_rate = -0.015,
      .dividend_yield = 0.02,
  };
  constexpr double kStrike = 100.0;
  constexpr double kMaturity = 1.25;
  constexpr std::size_t kObservations = 8;
  const auto call_contract =
      geometric_asian(nre::OptionType::call, kStrike, kMaturity, kObservations);
  const auto put_contract =
      geometric_asian(nre::OptionType::put, kStrike, kMaturity, kObservations);
  const auto call = nre::geometric_asian_analytical(call_contract, market);
  const auto put = nre::geometric_asian_analytical(put_contract, market);

  const double count = static_cast<double>(kObservations);
  const double average_time = kMaturity * (count + 1.0) / (2.0 * count);
  const double variance = market.volatility * market.volatility * kMaturity * (count + 1.0) *
                          (2.0 * count + 1.0) / (6.0 * count * count);
  const double log_mean = std::log(market.spot) + (market.risk_free_rate - market.dividend_yield -
                                                   0.5 * market.volatility * market.volatility) *
                                                      average_time;
  const double expected_average = std::exp(log_mean + 0.5 * variance);
  const double discount = std::exp(-market.risk_free_rate * kMaturity);

  expect_near(call.price - put.price, discount * (expected_average - kStrike), kInvariantTolerance,
              "geometric Asian put-call parity");
  expect(call.price >= 0.0 && call.price <= discount * expected_average,
         "geometric Asian call should satisfy price bounds");
  expect(put.price >= 0.0 && put.price <= discount * kStrike,
         "geometric Asian put should satisfy price bounds");
  expect_near(call.delta, geometric_finite_difference_delta(call_contract, market), kDeltaTolerance,
              "geometric Asian call Delta should match centered difference");
  expect_near(put.delta, geometric_finite_difference_delta(put_contract, market), kDeltaTolerance,
              "geometric Asian put Delta should match centered difference");
}

}  // namespace

int main() {
  test_style_rejection();
  test_european_high_precision_fixtures();
  test_european_zero_volatility();
  test_european_near_zero_maturity();
  test_european_invariants();
  test_geometric_asian_style_rejection();
  test_geometric_asian_high_precision_fixtures();
  test_geometric_asian_zero_volatility();
  test_geometric_asian_invariants();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All analytics tests passed\n";
  return EXIT_SUCCESS;
}
