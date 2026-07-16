#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "nre/random.hpp"

namespace {

// The fixtures lock the documented engine and draw order. The tolerance permits small libm
// differences in Box-Muller transcendental functions across supported C++ toolchains.
constexpr double kFixtureTolerance = 2.0e-14;

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > kFixtureTolerance) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected
              << ", tolerance=" << kFixtureTolerance << ")\n";
    ++failures;
  }
}

void test_fixed_seed_draw_order() {
  nre::NormalGenerator generator(123456789ULL);
  constexpr std::array expected{
      -0.15368229104453199, 1.4430813627758599,   1.9631432390449084,
      0.35607245128913229,  -0.31386836108161936, 0.42715868946460389,
  };

  for (std::size_t index = 0; index < expected.size(); ++index) {
    expect_near(generator.next(), expected[index],
                "fixed-seed normal draw " + std::to_string(index));
  }
}

void test_seed_reproducibility() {
  nre::NormalGenerator first(987654321ULL);
  nre::NormalGenerator second(987654321ULL);
  for (int draw = 0; draw < 100; ++draw) {
    expect(first.next() == second.next(), "equal seeds should reproduce every draw");
  }
}

void test_different_seeds_change_stream() {
  nre::NormalGenerator first(1ULL);
  nre::NormalGenerator second(2ULL);
  expect(first.next() != second.next(), "different seeds should change the normal stream");
}

}  // namespace

int main() {
  test_fixed_seed_draw_order();
  test_seed_reproducibility();
  test_different_seeds_change_stream();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All random tests passed\n";
  return EXIT_SUCCESS;
}
