#pragma once

#include "driftless/model_window.h"

#include <memory>
#include <string>

namespace driftless {

// Result of one model inference. See models/MODEL_CONTRACT.md.
struct VelocityModelOutput {
    double speed_ms = 0.0;   // [0] -- what UkfFusionEngine::updateVelocityModel consumes
    double dpsi_rad = 0.0;   // [1] -- heading change, more accurate, not yet wired into fusion
    double dv_ms = 0.0;      // [2] -- speed change over the interval
};

// ONNX Runtime wrapper for the same weights android/.../fusion/VelocityModel.kt
// loads as TFLite -- one training/ checkpoint, two runtimes.
//
// CONTRACT (models/MODEL_CONTRACT.md):
//   input   "imu_window"  [1,14,80] float32, channels-FIRST
//   output  "speed_dpsi"  [1,3] float32
// Normalisation is baked into the graph: feed raw SI units, read SI units.
class VelocityModel {
public:
    // Throws std::runtime_error if the model file is missing or fails to
    // load.
    explicit VelocityModel(const std::string& onnx_path);
    ~VelocityModel();

    VelocityModel(const VelocityModel&) = delete;
    VelocityModel& operator=(const VelocityModel&) = delete;

    // Run inference on a full window from ModelWindow::push().
    [[nodiscard]] VelocityModelOutput infer(const ModelWindow::Tensor& window) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace driftless