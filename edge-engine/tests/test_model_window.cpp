#include "driftless/model_window.h"

#include <gtest/gtest.h>

using namespace driftless;

TEST(DecimatorTest, EmitsAtTargetRate) {
    Decimator dec(0.1);  // 10Hz target
    int emitted = 0;
    for (int i = 0; i < 200; ++i) {  // 1s of 200Hz samples
        ImuSample s;
        s.timestamp_nanos = static_cast<std::int64_t>(i) * 5'000'000;  // 5ms step
        if (dec.push(s)) ++emitted;
    }
    EXPECT_GE(emitted, 9);
    EXPECT_LE(emitted, 12);
}

TEST(ModelWindowTest, ProducesTensorOnlyAfter80Samples) {
    ModelWindow window;
    std::optional<ModelWindow::Tensor> result;
    for (int t = 0; t < 79; ++t) {
        ImuSample s;
        s.timestamp_nanos = static_cast<std::int64_t>(t) * 100'000'000;  // 100ms step
        s.accel = Vec3(0.0, 0.0, -9.80665);
        result = window.push(s);
        EXPECT_FALSE(result.has_value());
    }
    ImuSample last;
    last.timestamp_nanos = static_cast<std::int64_t>(79) * 100'000'000;
    last.accel = Vec3(0.0, 0.0, -9.80665);
    result = window.push(last);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), static_cast<std::size_t>(ModelWindow::kChannels * ModelWindow::kTimesteps));
}

TEST(ModelWindowTest, ChannelsFirstLayoutMatchesInsertionOrder) {
    ModelWindow window;
    std::optional<ModelWindow::Tensor> result;
    for (int t = 0; t < 80; ++t) {
        ImuSample s;
        s.timestamp_nanos = static_cast<std::int64_t>(t) * 100'000'000;
        s.accel = Vec3(static_cast<double>(t), 0.0, -9.80665);  // acc_x encodes timestep
        result = window.push(s);
    }
    ASSERT_TRUE(result.has_value());
    for (int t = 0; t < 80; ++t) {
        EXPECT_FLOAT_EQ((*result)[0 * ModelWindow::kTimesteps + t], static_cast<float>(t));
    }
}