#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nre/statistics.hpp"

namespace {

// Expected values below are hand-calculated from small integer datasets. This tolerance covers
// rounding in square root and subsequent arithmetic while remaining tight at the tested scale.
constexpr double kTolerance = 1.0e-12;

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > kTolerance) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected
              << ", tolerance=" << kTolerance << ")\n";
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

void test_fewer_than_two_samples() {
  nre::StreamingStatistics statistics;
  expect(statistics.count() == 0, "a new accumulator should have zero samples");
  expect_throws<std::logic_error>([&] { static_cast<void>(statistics.summary()); },
                                  "an empty accumulator should not produce a summary");

  statistics.add(42.0);
  expect(statistics.count() == 1, "adding one value should update the sample count");
  expect_throws<std::logic_error>([&] { static_cast<void>(statistics.summary()); },
                                  "one sample should not produce a summary");
}

void test_hand_calculated_sequence() {
  nre::StreamingStatistics statistics;
  for (const double sample : std::array{1.0, 2.0, 3.0, 4.0, 5.0}) {
    statistics.add(sample);
  }

  const auto summary = statistics.summary();
  // mean = 3; sample variance = 10 / 4 = 2.5; standard error = sqrt(2.5 / 5).
  const double expected_standard_error = std::sqrt(0.5);
  const double expected_half_width = 1.96 * expected_standard_error;
  expect(summary.sample_count == 5, "summary should return the number of accumulated samples");
  expect_near(summary.estimate, 3.0, "sequence mean");
  expect_near(summary.sample_standard_error, expected_standard_error,
              "sequence sample standard error");
  expect_near(summary.confidence_interval_95.lower, 3.0 - expected_half_width,
              "sequence 95% confidence lower bound");
  expect_near(summary.confidence_interval_95.upper, 3.0 + expected_half_width,
              "sequence 95% confidence upper bound");
}

void test_two_samples() {
  nre::StreamingStatistics statistics;
  statistics.add(-1.0);
  statistics.add(1.0);

  const auto summary = statistics.summary();
  // mean = 0; sample variance = 2; standard error = sqrt(2 / 2) = 1.
  expect(summary.sample_count == 2, "two-sample summary count");
  expect_near(summary.estimate, 0.0, "two-sample mean");
  expect_near(summary.sample_standard_error, 1.0, "two-sample standard error");
  expect_near(summary.confidence_interval_95.lower, -1.96, "two-sample 95% confidence lower bound");
  expect_near(summary.confidence_interval_95.upper, 1.96, "two-sample 95% confidence upper bound");
}

void test_large_offset_sequence() {
  nre::StreamingStatistics statistics;
  constexpr double kOffset = 1.0e12;
  for (const double sample :
       std::array{kOffset + 1.0, kOffset + 2.0, kOffset + 3.0, kOffset + 4.0, kOffset + 5.0}) {
    statistics.add(sample);
  }

  const auto summary = statistics.summary();
  // Adding a constant changes the mean but not the sample variance or standard error.
  const double expected_standard_error = std::sqrt(0.5);
  const double expected_half_width = 1.96 * expected_standard_error;
  expect_near(summary.estimate, kOffset + 3.0, "large-offset sequence mean");
  expect_near(summary.sample_standard_error, expected_standard_error,
              "large-offset sequence sample standard error");
  expect_near(summary.confidence_interval_95.lower, kOffset + 3.0 - expected_half_width,
              "large-offset sequence 95% confidence lower bound");
  expect_near(summary.confidence_interval_95.upper, kOffset + 3.0 + expected_half_width,
              "large-offset sequence 95% confidence upper bound");
}

void test_constant_samples() {
  nre::StreamingStatistics statistics;
  for (const double sample : std::array{7.0, 7.0, 7.0, 7.0}) {
    statistics.add(sample);
  }

  const auto summary = statistics.summary();
  expect_near(summary.estimate, 7.0, "constant-sample mean");
  expect_near(summary.sample_standard_error, 0.0, "constant-sample standard error");
  expect_near(summary.confidence_interval_95.lower, 7.0,
              "constant-sample 95% confidence lower bound");
  expect_near(summary.confidence_interval_95.upper, 7.0,
              "constant-sample 95% confidence upper bound");
}

void test_bivariate_fewer_than_two_samples() {
  nre::StreamingBivariateStatistics statistics;
  expect(statistics.count() == 0, "a new bivariate accumulator should have zero samples");
  expect_throws<std::logic_error>([&] { static_cast<void>(statistics.summary()); },
                                  "an empty bivariate accumulator should not summarize");
  statistics.add(1.0, 2.0);
  expect(statistics.count() == 1, "one pair should update the bivariate sample count");
  expect_throws<std::logic_error>([&] { static_cast<void>(statistics.summary()); },
                                  "one pair should not produce bivariate sample moments");
}

void test_bivariate_hand_calculated_sequence() {
  nre::StreamingBivariateStatistics statistics;
  constexpr std::array first_samples{1.0, 2.0, 3.0};
  constexpr std::array second_samples{2.0, 1.0, 5.0};
  for (std::size_t index = 0; index < first_samples.size(); ++index) {
    statistics.add(first_samples[index], second_samples[index]);
  }

  const auto summary = statistics.summary();
  expect(summary.sample_count == 3, "bivariate summary sample count");
  expect_near(summary.first_mean, 2.0, "bivariate first mean");
  expect_near(summary.second_mean, 8.0 / 3.0, "bivariate second mean");
  expect_near(summary.first_sample_variance, 1.0, "bivariate first sample variance");
  expect_near(summary.second_sample_variance, 13.0 / 3.0,
              "bivariate second sample variance");
  expect_near(summary.sample_covariance, 1.5, "bivariate sample covariance");
}

void test_bivariate_constant_control() {
  nre::StreamingBivariateStatistics statistics;
  for (const double first_sample : std::array{1.0, 4.0, 9.0}) {
    statistics.add(first_sample, 7.0);
  }

  const auto summary = statistics.summary();
  expect_near(summary.second_sample_variance, 0.0, "constant control sample variance");
  expect_near(summary.sample_covariance, 0.0, "constant control sample covariance");
}

}  // namespace

int main() {
  test_fewer_than_two_samples();
  test_hand_calculated_sequence();
  test_two_samples();
  test_large_offset_sequence();
  test_constant_samples();
  test_bivariate_fewer_than_two_samples();
  test_bivariate_hand_calculated_sequence();
  test_bivariate_constant_control();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All statistics tests passed\n";
  return EXIT_SUCCESS;
}
