#pragma once

#include <cstddef>

namespace nre {

struct ConfidenceInterval {
  double lower;
  double upper;
};

struct StatisticsSummary {
  std::size_t sample_count;
  double estimate;
  double sample_standard_error;
  ConfidenceInterval confidence_interval_95;
};

struct BivariateStatisticsSummary {
  std::size_t sample_count;
  double first_mean;
  double second_mean;
  double first_sample_variance;
  double second_sample_variance;
  double sample_covariance;
};

// Accumulates scalar samples using Welford's online algorithm. Callers must supply finite values.
class StreamingStatistics {
 public:
  void add(double sample) noexcept;

  // Combines an independently accumulated block using Chan's parallel variance formula.
  void merge(const StreamingStatistics& other) noexcept;

  [[nodiscard]] std::size_t count() const noexcept;

  // Sample variance, standard error, and a confidence interval require at least two samples.
  // Throws std::logic_error when fewer samples have been added.
  [[nodiscard]] StatisticsSummary summary() const;

 private:
  std::size_t count_{0};
  double mean_{0.0};
  double sum_squared_deviations_{0.0};
};

// Accumulates paired finite samples with a numerically stable online covariance update.
class StreamingBivariateStatistics {
 public:
  void add(double first_sample, double second_sample) noexcept;

  // Combines an independently accumulated block in a deterministic caller-selected order.
  void merge(const StreamingBivariateStatistics& other) noexcept;

  [[nodiscard]] std::size_t count() const noexcept;

  // Sample variances and covariance require at least two paired samples.
  // Throws std::logic_error when fewer samples have been added.
  [[nodiscard]] BivariateStatisticsSummary summary() const;

 private:
  std::size_t count_{0};
  double first_mean_{0.0};
  double second_mean_{0.0};
  double first_sum_squared_deviations_{0.0};
  double second_sum_squared_deviations_{0.0};
  double sum_cross_deviations_{0.0};
};

}  // namespace nre
