#include "nre/analytics.hpp"

#include <cmath>
#include <stdexcept>

namespace nre {
namespace {

double normal_cdf(double value) {
  constexpr double kSqrtTwo = 1.4142135623730950488;
  return 0.5 * std::erfc(-value / kSqrtTwo);
}

PricingResult deterministic_option(OptionType type, double underlying, double strike,
                                   double discount, double underlying_spot_derivative) {
  const double discounted_derivative = discount * underlying_spot_derivative;
  if (type == OptionType::call) {
    if (underlying > strike) {
      return {.price = discount * (underlying - strike), .delta = discounted_derivative};
    }
    if (underlying < strike) {
      return {.price = 0.0, .delta = 0.0};
    }
    return {.price = 0.0, .delta = 0.5 * discounted_derivative};
  }

  if (underlying < strike) {
    return {.price = discount * (strike - underlying), .delta = -discounted_derivative};
  }
  if (underlying > strike) {
    return {.price = 0.0, .delta = 0.0};
  }
  return {.price = 0.0, .delta = -0.5 * discounted_derivative};
}

PricingResult lognormal_option(OptionType type, double spot, double strike, double discount,
                               double log_mean, double log_variance) {
  const double standard_deviation = std::sqrt(log_variance);
  const double d1 = (log_mean - std::log(strike) + log_variance) / standard_deviation;
  const double d2 = d1 - standard_deviation;
  const double expected_underlying = std::exp(log_mean + 0.5 * log_variance);
  const double delta_scale = discount * expected_underlying / spot;

  if (type == OptionType::call) {
    return {
        .price = discount * (expected_underlying * normal_cdf(d1) - strike * normal_cdf(d2)),
        .delta = delta_scale * normal_cdf(d1),
    };
  }

  return {
      .price = discount * (strike * normal_cdf(-d2) - expected_underlying * normal_cdf(-d1)),
      .delta = -delta_scale * normal_cdf(-d1),
  };
}

}  // namespace

PricingResult black_scholes_european(const OptionContract& contract, const MarketState& market) {
  if (contract.style != OptionStyle::european) {
    throw std::invalid_argument("Black-Scholes pricing requires a European option");
  }

  const double maturity = contract.maturity_years;
  const double discount = std::exp(-market.risk_free_rate * maturity);
  if (market.volatility == 0.0) {
    const double terminal_spot =
        market.spot * std::exp((market.risk_free_rate - market.dividend_yield) * maturity);
    const double terminal_spot_derivative = terminal_spot / market.spot;
    return deterministic_option(contract.type, terminal_spot, contract.strike, discount,
                                terminal_spot_derivative);
  }

  const double variance = market.volatility * market.volatility * maturity;
  const double log_mean = std::log(market.spot) + (market.risk_free_rate - market.dividend_yield -
                                                   0.5 * market.volatility * market.volatility) *
                                                      maturity;
  return lognormal_option(contract.type, market.spot, contract.strike, discount, log_mean,
                          variance);
}

PricingResult geometric_asian_analytical(const OptionContract& contract,
                                         const MarketState& market) {
  if (contract.style != OptionStyle::geometric_asian) {
    throw std::invalid_argument("geometric Asian pricing requires a geometric Asian option");
  }

  const double maturity = contract.maturity_years;
  const double observation_count = static_cast<double>(contract.observations);
  const double average_time = maturity * (observation_count + 1.0) / (2.0 * observation_count);
  const double discount = std::exp(-market.risk_free_rate * maturity);
  if (market.volatility == 0.0) {
    const double geometric_average =
        market.spot * std::exp((market.risk_free_rate - market.dividend_yield) * average_time);
    const double average_spot_derivative = geometric_average / market.spot;
    return deterministic_option(contract.type, geometric_average, contract.strike, discount,
                                average_spot_derivative);
  }

  const double variance_time = maturity * (observation_count + 1.0) *
                               (2.0 * observation_count + 1.0) /
                               (6.0 * observation_count * observation_count);
  const double variance = market.volatility * market.volatility * variance_time;
  const double log_mean = std::log(market.spot) + (market.risk_free_rate - market.dividend_yield -
                                                   0.5 * market.volatility * market.volatility) *
                                                      average_time;
  return lognormal_option(contract.type, market.spot, contract.strike, discount, log_mean,
                          variance);
}

}  // namespace nre
