#include "driftless/velocity_model.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <stdexcept>

namespace driftless {

struct VelocityModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "driftless_velocity_model"};
    Ort::SessionOptions options{};
    std::unique_ptr<Ort::Session> session;
    std::string input_name = "imu_window";
    std::string output_name = "speed_dpsi";
};

VelocityModel::VelocityModel(const std::string& onnx_path) : impl_(std::make_unique<Impl>()) {
    impl_->options.SetIntraOpNumThreads(1);
    impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        impl_->session = std::make_unique<Ort::Session>(impl_->env, onnx_path.c_str(), impl_->options);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("driftless: failed to load velocity model at '" + onnx_path +
                                  "': " + e.what());
    }
    // NOTE: this trusts the input/output names and shapes from
    // MODEL_CONTRACT.md rather than introspecting the graph. If the export
    // ever renames the graph's I/O, this throws at the first infer() call
    // (Ort::Exception: unknown input/output name), not here -- add a
    // GetInputNameAllocated/GetOutputNameAllocated + shape check here if
    // that's not fast enough feedback for you.
}

VelocityModel::~VelocityModel() = default;

VelocityModelOutput VelocityModel::infer(const ModelWindow::Tensor& window) const {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> input_shape{1, ModelWindow::kChannels, ModelWindow::kTimesteps};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(window.data()), window.size(), input_shape.data(),
        input_shape.size());

    const char* input_names[] = {impl_->input_name.c_str()};
    const char* output_names[] = {impl_->output_name.c_str()};

    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                        output_names, 1);

    const float* out = outputs[0].GetTensorData<float>();
    return VelocityModelOutput{static_cast<double>(out[0]), static_cast<double>(out[1]),
                               static_cast<double>(out[2])};
}

}  // namespace driftless