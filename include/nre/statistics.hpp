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

// Accumulates scalar samples using Welford's online algorithm. Callers must supply finite values.
class StreamingStatistics {
 public:
  void add(double sample) noexcept;

  [[nodiscard]] std::size_t count() const noexcept;

  // Sample variance, standard error, and a confidence interval require at least two samples.
  // Throws std::logic_error when fewer samples have been added.
  [[nodiscard]] StatisticsSummary summary() const;

 private:
  std::size_t count_{0};
  double mean_{0.0};
  double sum_squared_deviations_{0.0};
};

}  // namespace nre
