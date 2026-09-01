// CHARACTERIZATION, NOT VALIDATION.
//
// This measures how far the dead-reckoning solution wanders over 1 km with no
// aiding of any kind. It does NOT test the <10%-over-1km bar from the problem
// statement, and no number printed here should ever be quoted as evidence for
// or against it, because:
//
//   1. There is no GNSS update, no non-holonomic constraint and no
//      map-matching in this slice. Nothing bounds the error -- it grows without
//      limit by construction, and the only reason the figures below look
//      acceptable is that 60 seconds is short.
//   2. The IMU error model is a datasheet-class guess (see imu_noise.h), not an
//      Allan-variance fit to the actual sensor.
//   3. The trajectory is analytic. Real roads have vibration, wheel slip,
//      temperature-dependent bias drift, and mounting flex, none of which are
//      simulated here.
//
// What it IS good for: catching a regression. If a change to the mechanization
// or the covariance propagation makes these numbers move, that is worth
// knowing, so the assertions are loose ceilings rather than targets.

#include "driftless/ukf_fusion_engine.h"

#include "support/trajectory_sim.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

namespace {

using driftless::ImuNoiseParams;
using driftless::NavState;
using driftless::UkfFusionEngine;
using driftless::Vec3;
using driftless::testing::ConstantTurnTrajectory;
using driftless::testing::generateImuStream;
using driftless::testing::ImuStreamOptions;

constexpr double kSixtyKmphInMps = 60.0 / 3.6;
constexpr double kOneKilometreSeconds = 1000.0 / kSixtyKmphInMps;  // 60 s

UkfFusionEngine::ErrorMatrix postAlignmentSqrtCovariance() {
    UkfFusionEngine::ErrorVector diagonal;
    diagonal.segment<3>(UkfFusionEngine::kPositionIndex).setConstant(1.0);
    diagonal.segment<3>(UkfFusionEngine::kVelocityIndex).setConstant(0.1);
    diagonal.segment<3>(UkfFusionEngine::kAttitudeIndex)
        .setConstant(0.1 * M_PI / 180.0);
    diagonal.segment<3>(UkfFusionEngine::kAccelBiasIndex).setConstant(0.005);
    diagonal.segment<3>(UkfFusionEngine::kGyroBiasIndex).setConstant(1e-4);
    return diagonal.asDiagonal();
}

struct DriftResult {
    double horizontal_error_m;
    double distance_travelled_m;
    double percent_of_distance;
};

DriftResult runBlackout(double yaw_rate_rps, const ImuStreamOptions& base) {
    const ConstantTurnTrajectory trajectory(kSixtyKmphInMps, yaw_rate_rps);

    ImuStreamOptions options = base;
    options.duration_s = kOneKilometreSeconds;
    options.rate_hz = 200.0;

    UkfFusionEngine engine(trajectory.truthAt(0.0), postAlignmentSqrtCovariance(),
                           ImuNoiseParams::fogGrade());
    for (const auto& sample : generateImuStream(trajectory, options)) {
        EXPECT_TRUE(engine.predict(sample));
    }

    const NavState truth = trajectory.truthAt(options.duration_s);
    const Vec3 error = engine.state().position - truth.position;
    const double horizontal = std::hypot(error.x(), error.y());
    // Distance ALONG THE PATH, which is what the drift budget is a fraction of
    // -- not straight-line displacement, which on a curve is much shorter.
    const double distance = kSixtyKmphInMps * options.duration_s;
    return {horizontal, distance, 100.0 * horizontal / distance};
}

void report(const char* label, const DriftResult& r) {
    std::cout << "[ DRIFT    ] " << label << ": " << r.horizontal_error_m
              << " m over " << r.distance_travelled_m << " m ("
              << r.percent_of_distance << "% of distance)\n";
}

TEST(DriftCharacterization, IdealSensorOverOneKilometre) {
    // Floor case: no bias, no noise. Whatever is left is pure numerical and
    // discretization error, and it should be negligible. If this one ever
    // grows, the problem is arithmetic, not sensors.
    const auto result = runBlackout(0.02, ImuStreamOptions{});
    report("ideal sensor, gentle curve", result);
    EXPECT_LT(result.horizontal_error_m, 0.5);
}

TEST(DriftCharacterization, NoisyFogSensorOverOneKilometre) {
    ImuStreamOptions options;
    options.accel_noise_density = ImuNoiseParams::fogGrade().accel_noise_density;
    options.gyro_noise_density = ImuNoiseParams::fogGrade().gyro_noise_density;
    options.seed = 20260901;

    const auto result = runBlackout(0.02, options);
    report("FOG white noise only, gentle curve", result);
    EXPECT_LT(result.horizontal_error_m, 5.0);
}

TEST(DriftCharacterization, UnobservedBiasesDominateOverOneKilometre) {
    // The honest result. Turn-on bias is what actually kills dead reckoning:
    // white noise averages out over a minute, a constant bias does not. With no
    // update there is nothing to observe these, so the error is very nearly the
    // closed-form 0.5*b*T^2 and the filter's sophistication buys nothing.
    // Bounding this is exactly the job of the update path that is NOT in this
    // slice, which is why no conclusion about the 10% bar can be drawn here.
    ImuNoiseParams fog = ImuNoiseParams::fogGrade();
    ImuStreamOptions options;
    options.accel_noise_density = fog.accel_noise_density;
    options.gyro_noise_density = fog.gyro_noise_density;
    options.accel_bias = Vec3(2e-3, 1e-3, 0.0);   // ~200 ug, plausible turn-on
    options.gyro_bias = Vec3(0.0, 0.0, 5e-6);     // ~1 deg/hr, FOG-class
    options.seed = 20260902;

    const auto result = runBlackout(0.02, options);
    report("FOG noise + turn-on biases, gentle curve", result);

    const double closed_form_from_accel_bias =
        0.5 * std::hypot(2e-3, 1e-3) * kOneKilometreSeconds * kOneKilometreSeconds;
    std::cout << "[ DRIFT    ] closed-form 0.5*b*T^2 from accel bias alone: "
              << closed_form_from_accel_bias << " m\n";

    // Loose non-regression ceiling. NOT a performance target.
    EXPECT_LT(result.horizontal_error_m, 15.0);
}

TEST(DriftCharacterization, StraightLineVersusCurve) {
    // Whether path curvature materially changes the error. It should not, at
    // this grade -- if a curve is much worse, the attitude integration is the
    // suspect, not the sensors.
    ImuNoiseParams fog = ImuNoiseParams::fogGrade();
    ImuStreamOptions options;
    options.accel_noise_density = fog.accel_noise_density;
    options.gyro_noise_density = fog.gyro_noise_density;
    options.accel_bias = Vec3(2e-3, 1e-3, 0.0);
    options.gyro_bias = Vec3(0.0, 0.0, 5e-6);
    options.seed = 20260903;

    const auto straight = runBlackout(0.0, options);
    const auto curve = runBlackout(0.05, options);
    report("straight line", straight);
    report("333 m radius curve", curve);

    EXPECT_LT(curve.horizontal_error_m, 4.0 * straight.horizontal_error_m + 1.0);
}

}  // namespace
