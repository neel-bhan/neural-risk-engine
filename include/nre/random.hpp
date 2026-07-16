#pragma once

#include <cstdint>
#include <random>

namespace nre {

// Produces standard-normal draws from std::mt19937_64 with an explicit Box-Muller transform.
class NormalGenerator {
 public:
  explicit NormalGenerator(std::uint64_t seed);

  [[nodiscard]] double next();

 private:
  std::mt19937_64 engine_;
  bool has_spare_{false};
  double spare_{0.0};
};

}  // namespace nre
