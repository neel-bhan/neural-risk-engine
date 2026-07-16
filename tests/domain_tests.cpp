#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "nre/domain.hpp"

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  expect(nre::validate(nre::OptionContract{}).empty(), "default contract should be valid");
  expect(nre::validate(nre::MarketState{}).empty(), "default market should be valid");

  auto asian = nre::OptionContract{};
  asian.style = nre::OptionStyle::arithmetic_asian;
  asian.observations = 12;
  expect(nre::validate(asian).empty(), "monthly arithmetic Asian should be valid");

  auto invalid_asian = asian;
  invalid_asian.observations = 1;
  expect(!nre::validate(invalid_asian).empty(), "Asian option needs multiple observations");

  auto invalid_european = nre::OptionContract{};
  invalid_european.observations = 12;
  expect(!nre::validate(invalid_european).empty(),
         "European option should only observe maturity");

  auto invalid_market = nre::MarketState{};
  invalid_market.spot = 0.0;
  invalid_market.volatility = std::numeric_limits<double>::quiet_NaN();
  expect(nre::validate(invalid_market).size() == 2,
         "validation should return all market errors");

  expect(nre::to_string(nre::OptionType::put) == "put", "put should have a stable name");
  expect(nre::to_string(nre::OptionStyle::geometric_asian) == "geometric Asian",
         "geometric Asian should have a stable name");

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All domain tests passed\n";
  return EXIT_SUCCESS;
}

