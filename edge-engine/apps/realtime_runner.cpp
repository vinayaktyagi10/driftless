#include "driftless/model_window.h"
#include "driftless/ukf_fusion_engine.h"
#include "driftless/velocity_model.h"

#include <iostream>

using namespace driftless;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: driftless_realtime_runner <path-to-velocity_model.onnx>\n";
        return 1;
    }

    VelocityModel model(argv[1]);
    Decimator decimator;
    ModelWindow window;

    NavState initial;
    UkfFusionEngine::ErrorMatrix p0 = UkfFusionEngine::ErrorMatrix::Identity();
    UkfFusionEngine engine(initial, p0, ImuNoiseParams::fogGrade());

    // Called once per raw sample from the ~200Hz FOG driver. Everything
    // above this function is one-time setup; this closure is the hot path.
    auto onRawImuSample = [&](const ImuSample& raw) {
        if (!engine.predict(raw)) {
            std::cerr << "predict() rejected a sample (non-monotonic or over-long dt)\n";
        }
        if (auto decimated = decimator.push(raw)) {
            if (auto tensor = window.push(*decimated)) {
                const VelocityModelOutput out = model.infer(*tensor);
                const auto outcome = engine.updateVelocityModel(out.speed_ms);
                if (outcome == UkfFusionEngine::UpdateOutcome::kRejectedByGate) {
                    std::cerr << "velocity model update rejected by NIS gate\n";
                }
            }
        }
    };

    // TODO: replace with the real sensor driver's callback registration.
    // The pipeline above (predict -> decimate -> window -> infer -> fuse)
    // is complete and ready to be driven by real samples.
    (void)onRawImuSample;

    std::cout << "wired up; connect a real IMU source to drive onRawImuSample()\n";
    return 0;
}