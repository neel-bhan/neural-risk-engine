#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "nre/neural_router.hpp"

namespace nre {

// Optional ONNX Runtime implementation. This interface is linked only by ONNX-enabled targets;
// the dependency-free core does not compile or link its implementation.
class OnnxPricingBackend final : public NeuralBatchBackend {
 public:
  OnnxPricingBackend(const std::filesystem::path& metadata_path,
                     const std::filesystem::path& model_path);
  ~OnnxPricingBackend() override;
  OnnxPricingBackend(OnnxPricingBackend&&) noexcept;
  OnnxPricingBackend& operator=(OnnxPricingBackend&&) noexcept;
  OnnxPricingBackend(const OnnxPricingBackend&) = delete;
  OnnxPricingBackend& operator=(const OnnxPricingBackend&) = delete;

  [[nodiscard]] std::vector<NeuralCandidate> predict(
      const std::vector<NeuralInput>& inputs) override;
  [[nodiscard]] const NeuralGuardrailConfig& guardrail_config() const noexcept;
  [[nodiscard]] const std::string& artifact_version() const noexcept;
  [[nodiscard]] const std::string& model_checksum() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nre
