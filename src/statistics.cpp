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

void StreamingStatistics::merge(const StreamingStatistics& other) noexcept {
  if (other.count_ == 0U) {
    return;
  }
  if (count_ == 0U) {
    *this = other;
    return;
  }
  const double first_count = static_cast<double>(count_);
  const double second_count = static_cast<double>(other.count_);
  const double combined_count = first_count + second_count;
  const double delta = other.mean_ - mean_;
  sum_squared_deviations_ += other.sum_squared_deviations_ +
                             delta * delta * first_count * second_count / combined_count;
  mean_ += delta * second_count / combined_count;
  count_ += other.count_;
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

void StreamingBivariateStatistics::add(double first_sample, double second_sample) noexcept {
  ++count_;
  const double count = static_cast<double>(count_);
  const double first_delta = first_sample - first_mean_;
  const double second_delta = second_sample - second_mean_;
  first_mean_ += first_delta / count;
  second_mean_ += second_delta / count;
  first_sum_squared_deviations_ += first_delta * (first_sample - first_mean_);
  second_sum_squared_deviations_ += second_delta * (second_sample - second_mean_);
  sum_cross_deviations_ += first_delta * (second_sample - second_mean_);
}

void StreamingBivariateStatistics::merge(const StreamingBivariateStatistics& other) noexcept {
  if (other.count_ == 0U) {
    return;
  }
  if (count_ == 0U) {
    *this = other;
    return;
  }
  const double first_count = static_cast<double>(count_);
  const double second_count = static_cast<double>(other.count_);
  const double combined_count = first_count + second_count;
  const double first_delta = other.first_mean_ - first_mean_;
  const double second_delta = other.second_mean_ - second_mean_;
  const double weight = first_count * second_count / combined_count;
  first_sum_squared_deviations_ +=
      other.first_sum_squared_deviations_ + first_delta * first_delta * weight;
  second_sum_squared_deviations_ +=
      other.second_sum_squared_deviations_ + second_delta * second_delta * weight;
  sum_cross_deviations_ += other.sum_cross_deviations_ + first_delta * second_delta * weight;
  first_mean_ += first_delta * second_count / combined_count;
  second_mean_ += second_delta * second_count / combined_count;
  count_ += other.count_;
}

std::size_t StreamingBivariateStatistics::count() const noexcept { return count_; }

BivariateStatisticsSummary StreamingBivariateStatistics::summary() const {
  if (count_ < 2) {
    throw std::logic_error("bivariate statistics require at least two paired samples");
  }

  const double degrees_of_freedom = static_cast<double>(count_ - 1);
  return {
      .sample_count = count_,
      .first_mean = first_mean_,
      .second_mean = second_mean_,
      .first_sample_variance = first_sum_squared_deviations_ / degrees_of_freedom,
      .second_sample_variance = second_sum_squared_deviations_ / degrees_of_freedom,
      .sample_covariance = sum_cross_deviations_ / degrees_of_freedom,
  };
}

}  // namespace nre
