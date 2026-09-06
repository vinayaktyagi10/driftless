#include "driftless/model_window.h"
#include "driftless/ukf_fusion_engine.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>

using namespace driftless;
using Clock = std::chrono::steady_clock;

namespace {

ImuSample makeSample(std::int64_t t_nanos, double phase) {
    ImuSample s;
    s.timestamp_nanos = t_nanos;
    s.accel = Vec3(0.1 * std::sin(phase), 0.1 * std::cos(phase), -9.80665);
    s.gyro = Vec3(0.01 * std::cos(phase), 0.0, 0.02);
    return s;
}

}  // namespace

// Proves predict() alone -- the thing that runs on EVERY raw sample at
// 200Hz -- fits inside the 5ms/sample budget with real margin. This is the
// "fast enough to keep up in real time, not just eventually correct" bullet;
// nothing else in the test suite measures wall-clock time.
TEST(RealtimeBudgetTest, PredictFitsWithinFogSampleBudget) {
    NavState initial;
    UkfFusionEngine::ErrorMatrix p0 = UkfFusionEngine::ErrorMatrix::Identity();
    UkfFusionEngine engine(initial, p0, ImuNoiseParams::fogGrade());

    constexpr int kNumSamples = 20000;       // 100s of simulated 200Hz data
    constexpr std::int64_t kStepNanos = 5'000'000;  // 200Hz
    constexpr double kBudgetMs = 5.0;

    double total_ms = 0.0;
    double worst_ms = 0.0;
    std::int64_t t = 0;

    for (int i = 0; i < kNumSamples; ++i) {
        t += kStepNanos;
        ImuSample s = makeSample(t, static_cast<double>(i) * 0.01);

        const auto start = Clock::now();
        engine.predict(s);
        const auto end = Clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
        worst_ms = std::max(worst_ms, ms);
    }

    const double mean_ms = total_ms / kNumSamples;
    std::cout << "predict(): mean=" << mean_ms << "ms  worst=" << worst_ms
              << "ms  budget=" << kBudgetMs << "ms\n";

    // Mean must sit well under budget -- if the mean is already close to 5ms
    // there is no headroom for the model-inference thread, OS jitter, or a
    // slower deployment target. Worst-case is logged, not asserted, because a
    // single page-fault/scheduler hiccup on a shared CI box is not the same
    // failure as a systemically slow filter.
    EXPECT_LT(mean_ms, kBudgetMs * 0.5)
        << "predict() mean latency is eating over half the 5ms/200Hz budget";
}

// Same idea for the model side: it only needs to run once per DECIMATED
// (~10Hz, 100ms-budget) sample, but you still want to know its cost with
// real margin before you commit to running it on the same core as predict().
TEST(RealtimeBudgetTest, ModelWindowBookkeepingFitsWithinDecimatedBudget) {
    ModelWindow window;
    constexpr int kNumSamples = 1000;   // 100s of decimated (10Hz) data
    constexpr std::int64_t kStepNanos = 100'000'000;  // 10Hz
    constexpr double kBudgetMs = 100.0;

    double total_ms = 0.0;
    std::int64_t t = 0;
    for (int i = 0; i < kNumSamples; ++i) {
        t += kStepNanos;
        ImuSample s = makeSample(t, static_cast<double>(i) * 0.05);

        const auto start = Clock::now();
        window.push(s);
        const auto end = Clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    const double mean_ms = total_ms / kNumSamples;
    std::cout << "ModelWindow::push(): mean=" << mean_ms << "ms  budget=" << kBudgetMs << "ms\n";
    EXPECT_LT(mean_ms, kBudgetMs * 0.1);
}