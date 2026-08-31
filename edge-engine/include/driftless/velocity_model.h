#pragma once

namespace driftless {

// ONNX Runtime wrapper for the same weights android/.../fusion/VelocityModel.kt
// loads as TFLite — one training/ checkpoint, two runtimes.
class VelocityModel {
public:
    VelocityModel() = default;
    // TODO: Ort::Session over the .onnx export from training/export/to_tflite.py
};

}  // namespace driftless
