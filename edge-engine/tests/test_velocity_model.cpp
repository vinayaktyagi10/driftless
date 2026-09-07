#include "driftless/velocity_model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>

namespace {

std::string modelPath() { return std::string(DRIFTLESS_MODEL_DIR) + "/velocity_model.onnx"; }

bool modelExists() {
    std::ifstream f(modelPath(), std::ios::binary);
    return f.good();
}

}  // namespace

TEST(VelocityModelTest, LoadsAndRunsOnZeroWindow) {
    if (!modelExists()) {
        GTEST_SKIP() << "velocity_model.onnx not present at " << modelPath()
                     << " -- run `cd training && python export/to_tflite.py "
                        "--copy-to-consumers` first.";
    }

    driftless::VelocityModel model(modelPath());
    driftless::ModelWindow::Tensor zero_window{};
    zero_window.fill(0.0f);

    auto out = model.infer(zero_window);
    EXPECT_TRUE(std::isfinite(out.speed_ms));
    EXPECT_TRUE(std::isfinite(out.dpsi_rad));
    EXPECT_TRUE(std::isfinite(out.dv_ms));
}