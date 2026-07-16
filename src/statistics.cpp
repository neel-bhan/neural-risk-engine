#include "nre/statistics.hpp"

#include <cmath>
#include <stdexcept>

namespace nre {
namespace {

constexpr double kNormal95CriticalValue = 1.96;

}  // namespace

void StreamingStatistics::add(double sample) noexcept {
  ++count_;
  const double count = static_cast<double>(count_);
  const double delta = sample - mean_;
  mean_ += delta / count;
  const double delta_from_updated_mean = sample - mean_;
  sum_squared_deviations_ += delta * delta_from_updated_mean;
}

std::size_t StreamingStatistics::count() const noexcept { return count_; }

StatisticsSummary StreamingStatistics::summary() const {
  if (count_ < 2) {
    throw std::logic_error("streaming statistics require at least two samples");
  }

  const double count = static_cast<double>(count_);
  const double sample_variance = sum_squared_deviations_ / (count - 1.0);
  const double sample_standard_error = std::sqrt(sample_variance / count);
  const double confidence_half_width = kNormal95CriticalValue * sample_standard_error;

  return {
      .sample_count = count_,
      .estimate = mean_,
      .sample_standard_error = sample_standard_error,
      .confidence_interval_95 =
          {
              .lower = mean_ - confidence_half_width,
              .upper = mean_ + confidence_half_width,
          },
  };
}

}  // namespace nre
