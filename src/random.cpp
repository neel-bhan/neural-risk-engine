#include "nre/random.hpp"

#include <cmath>
#include <numbers>

namespace nre {
namespace {

constexpr double kInverseTwoToThe53 = 1.0 / 9007199254740992.0;

double uniform_open_unit_interval(std::mt19937_64& engine) {
  const std::uint64_t top_53_bits = engine() >> 11U;
  if (top_53_bits == 0U) {
    return 0.5 * kInverseTwoToThe53;
  }
  return static_cast<double>(top_53_bits) * kInverseTwoToThe53;
}

}  // namespace

NormalGenerator::NormalGenerator(std::uint64_t seed) : engine_(seed) {}

double NormalGenerator::next() {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }

  const double first_uniform = uniform_open_unit_interval(engine_);
  const double second_uniform = uniform_open_unit_interval(engine_);
  const double radius = std::sqrt(-2.0 * std::log(first_uniform));
  const double angle = 2.0 * std::numbers::pi_v<double> * second_uniform;

  spare_ = radius * std::sin(angle);
  has_spare_ = true;
  return radius * std::cos(angle);
}

}  // namespace nre
