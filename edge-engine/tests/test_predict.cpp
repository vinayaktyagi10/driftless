// Dynamics and numerics of the predict step, against trajectories whose answer
// is known in closed form. The sign-convention tests matter most: a strapdown
// filter with a flipped axis still runs, still produces plausible-looking
// numbers, and is wrong -- so the conventions are asserted directly rather
// than inferred from an end-to-end error being "small".

#include "driftless/ukf_fusion_engine.h"

#include "support/trajectory_sim.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

using driftless::ImuNoiseParams;
using driftless::ImuSample;
using driftless::NavState;
using driftless::UkfFusionEngine;
using driftless::Vec3;
using driftless::testing::ConstantTurnTrajectory;
using driftless::testing::generateImuStream;
using driftless::testing::ImuStreamOptions;
using ErrorMatrix = UkfFusionEngine::ErrorMatrix;

constexpr double kSixtyKmphInMps = 60.0 / 3.6;  // 16.667 m/s

// Negligible initial uncertainty. With the covariance this small the unscented
// transform collapses to plain integration of the mean, so these runs test the
// mechanization exactly -- no second-order correction to confound the result.
// (Exactly zero would make S singular, which is not a state the filter should
// ever be handed.)
ErrorMatrix negligibleSqrtCovariance() {
    UkfFusionEngine::ErrorVector diagonal;
    diagonal.setConstant(1e-9);
    return diagonal.asDiagonal();
}

// A plausible post-alignment initial uncertainty: metre-level position,
// decimetre-per-second velocity, half a degree of attitude, and biases known
// only to their datasheet turn-on repeatability.
ErrorMatrix initialSqrtCovariance() {
    UkfFusionEngine::ErrorVector diagonal;
    diagonal.segment<3>(UkfFusionEngine::kPositionIndex).setConstant(1.0);
    diagonal.segment<3>(UkfFusionEngine::kVelocityIndex).setConstant(0.1);
    diagonal.segment<3>(UkfFusionEngine::kAttitudeIndex)
        .setConstant(0.5 * M_PI / 180.0);
    diagonal.segment<3>(UkfFusionEngine::kAccelBiasIndex).setConstant(0.01);
    diagonal.segment<3>(UkfFusionEngine::kGyroBiasIndex).setConstant(1e-3);
    return diagonal.asDiagonal();
}

double headingOf(const NavState& s) {
    const Vec3 forward = s.orientation * Vec3::UnitX();
    return std::atan2(forward.y(), forward.x());
}

UkfFusionEngine runStream(const std::vector<ImuSample>& samples,
                          const NavState& initial = NavState{},
                          const ErrorMatrix& sqrt_covariance =
                              negligibleSqrtCovariance()) {
    UkfFusionEngine engine(initial, sqrt_covariance, ImuNoiseParams::fogGrade());
    for (const auto& sample : samples) {
        EXPECT_TRUE(engine.predict(sample));
    }
    return engine;
}

// --- Mechanization, in isolation from the filter ---------------------------

TEST(Mechanize, LevelStationaryAccelerometerCancelsGravityExactly) {
    // The single most load-bearing sign in the whole engine. A level, at-rest
    // accelerometer reads -g on its down axis; mechanize must add gravity back
    // and get exactly zero acceleration. Get this backwards and the filter
    // accelerates downward at 2g while every other test still looks fine.
    NavState s;
    const Vec3 accel(0.0, 0.0, -driftless::kGravity);
    const NavState out = UkfFusionEngine::mechanize(
        s, accel, Vec3::Zero(), 0.005, Vec3(0.0, 0.0, driftless::kGravity));
    EXPECT_LT(out.velocity.norm(), 1e-15);
    EXPECT_LT(out.position.norm(), 1e-15);
}

TEST(Mechanize, PositiveYawRateTurnsFromNorthTowardEast) {
    // Pins the handedness of the body-frame angular rate against the NED frame.
    NavState s;
    const NavState out = UkfFusionEngine::mechanize(
        s, Vec3(0.0, 0.0, -driftless::kGravity), Vec3(0.0, 0.0, 0.1), 1.0,
        Vec3(0.0, 0.0, driftless::kGravity));
    const Vec3 forward = out.orientation * Vec3::UnitX();
    EXPECT_NEAR(std::atan2(forward.y(), forward.x()), 0.1, 1e-9);
    EXPECT_GT(forward.y(), 0.0) << "positive yaw rate must swing toward East";
}

// --- The filter on known trajectories --------------------------------------

TEST(Predict, StationaryVehicleDoesNotDrift) {
    ImuStreamOptions options;
    options.duration_s = 60.0;
    const auto engine = runStream(
        generateImuStream(ConstantTurnTrajectory(0.0, 0.0), options));

    // Not exactly zero: process noise inflates the attitude covariance over the
    // minute, and the second-order term below is the consequence. At this
    // magnitude it is ~1e-5 m over 60s, i.e. nothing.
    EXPECT_LT(engine.state().position.norm(), 1e-4);
    EXPECT_LT(engine.state().velocity.norm(), 1e-5);
    EXPECT_LT(std::abs(headingOf(engine.state())), 1e-12);
}

TEST(Predict, ConstantVelocityMatchesAnalyticTruth) {
    ImuStreamOptions options;
    options.duration_s = 60.0;
    const ConstantTurnTrajectory trajectory(kSixtyKmphInMps, 0.0);
    const auto engine = runStream(generateImuStream(trajectory, options),
                                  trajectory.truthAt(0.0));

    const NavState truth = trajectory.truthAt(options.duration_s);
    ASSERT_NEAR(truth.position.norm(), 1000.0, 1.0) << "sanity: ~1km covered";
    EXPECT_LT((engine.state().position - truth.position).norm(), 1e-4);
    EXPECT_LT((engine.state().velocity - truth.velocity).norm(), 1e-5);
}

TEST(Predict, ConstantTurnMatchesAnalyticTruth) {
    // A 0.05 rad/s turn at 60 km/h is a ~333 m radius curve -- a highway
    // interchange, not a hairpin. Residual error here is the price of the
    // first-order attitude integration in mechanize(), and it is bounded well
    // below anything the drift budget cares about.
    ImuStreamOptions options;
    options.duration_s = 60.0;
    const ConstantTurnTrajectory trajectory(kSixtyKmphInMps, 0.05);

    const auto engine = runStream(generateImuStream(trajectory, options),
                                  trajectory.truthAt(0.0));

    const NavState truth = trajectory.truthAt(options.duration_s);
    const double error = (engine.state().position - truth.position).norm();
    std::cout << "[ INFO     ] constant-turn integration error after "
              << options.duration_s << "s over " << truth.position.norm()
              << " m of arc: " << error << " m\n";
    EXPECT_LT(error, 0.2);
    EXPECT_LT((engine.state().velocity - truth.velocity).norm(), 0.01);
}

TEST(Predict, MechanizationErrorIsFirstOrderInStepSize) {
    // mechanize() holds attitude constant across the interval, so its error is
    // O(dt). Asserting the CONVERGENCE RATE rather than a magnitude is what
    // makes this test meaningful: a sign error or a wrong axis would still
    // produce a small number at 400Hz, but it would not halve when dt halves.
    const ConstantTurnTrajectory trajectory(kSixtyKmphInMps, 0.05);

    std::vector<double> errors;
    for (double rate_hz : {50.0, 100.0, 200.0, 400.0}) {
        ImuStreamOptions options;
        options.duration_s = 60.0;
        options.rate_hz = rate_hz;
        const auto engine = runStream(generateImuStream(trajectory, options),
                                      trajectory.truthAt(0.0));
        errors.push_back(
            (engine.state().position - trajectory.truthAt(60.0).position).norm());
    }

    for (std::size_t i = 1; i < errors.size(); ++i) {
        const double ratio = errors[i - 1] / errors[i];
        std::cout << "[ INFO     ] halving dt reduced error by " << ratio << "x\n";
        EXPECT_NEAR(ratio, 2.0, 0.05) << "expected first-order convergence";
    }
}

TEST(Predict, GyroBiasProducesPredictedHeadingDrift) {
    // The filter starts believing its gyro bias is zero, so an injected bias is
    // integrated straight into heading. With no measurement update there is
    // nothing to observe it, so the heading error must be exactly bias*T --
    // and in the direction the bias points. Both halves are the test.
    constexpr double kBias = 2e-4;  // rad/s about body-down
    constexpr double kDuration = 60.0;

    ImuStreamOptions options;
    options.duration_s = kDuration;
    options.gyro_bias = Vec3(0.0, 0.0, kBias);

    const auto engine = runStream(
        generateImuStream(ConstantTurnTrajectory(0.0, 0.0), options));

    EXPECT_NEAR(headingOf(engine.state()), kBias * kDuration, 1e-9);
    EXPECT_GT(headingOf(engine.state()), 0.0)
        << "a positive down-axis bias must drift heading toward East";
}

TEST(Predict, AccelBiasIntegratesIntoPositionQuadratically) {
    // The other half of the dead-reckoning error budget: an unobserved
    // accelerometer bias b produces exactly 0.5*b*T^2 of position error.
    constexpr double kBias = 0.01;  // m/s^2 forward
    constexpr double kDuration = 60.0;

    ImuStreamOptions options;
    options.duration_s = kDuration;
    options.accel_bias = Vec3(kBias, 0.0, 0.0);

    const auto engine = runStream(
        generateImuStream(ConstantTurnTrajectory(0.0, 0.0), options));

    const double expected = 0.5 * kBias * kDuration * kDuration;
    EXPECT_NEAR(engine.state().position.x(), expected, 1e-3 * expected);
}

TEST(Predict, SecondOrderMeanCorrectionScalesWithAttitudeVariance) {
    // With a REALISTIC covariance the propagated mean is deliberately not the
    // integration of the mean. E[R(dtheta) * f] != R * f when attitude is
    // uncertain, and the unscented transform captures that second-order term --
    // it is the main thing a UKF buys over an EKF here.
    //
    // Quantitatively it must be quadratic in the attitude sigma: doubling the
    // attitude uncertainty must quadruple the induced drift. If it came out
    // linear, the "second-order" claim would be false and the filter would be
    // an EKF wearing 31 sigma points.
    ImuStreamOptions options;
    options.duration_s = 60.0;
    const auto samples = generateImuStream(ConstantTurnTrajectory(0.0, 0.0), options);

    std::vector<double> drifts;
    for (double sigma_deg : {0.25, 0.5, 1.0}) {
        UkfFusionEngine::ErrorVector diagonal;
        diagonal.setConstant(1e-9);
        diagonal.segment<3>(UkfFusionEngine::kAttitudeIndex)
            .setConstant(sigma_deg * M_PI / 180.0);
        const ErrorMatrix sqrt_covariance = diagonal.asDiagonal();
        const auto engine = runStream(samples, NavState{}, sqrt_covariance);
        drifts.push_back(engine.state().position.norm());
    }

    for (std::size_t i = 1; i < drifts.size(); ++i) {
        const double ratio = drifts[i] / drifts[i - 1];
        std::cout << "[ INFO     ] doubling attitude sigma scaled drift by "
                  << ratio << "x\n";
        EXPECT_NEAR(ratio, 4.0, 0.1) << "expected quadratic, i.e. second-order";
    }

    // And it must be a correction, not a catastrophe: at half a degree of
    // attitude uncertainty it is order a metre over a minute, well inside the
    // budget. This is the number that would grow teeth if attitude uncertainty
    // were ever allowed to reach several degrees during a long blackout.
    EXPECT_LT(drifts[1], 5.0);
}

// --- Covariance behaviour --------------------------------------------------

TEST(Predict, CovarianceGrowsMonotonicallyWithoutUpdates) {
    // With no measurement there is nothing to shrink the covariance, so its
    // trace must increase at every single step. A step that fails to grow means
    // process noise is not actually being injected.
    ImuStreamOptions options;
    options.duration_s = 10.0;
    const auto samples = generateImuStream(ConstantTurnTrajectory(kSixtyKmphInMps, 0.02), options);

    UkfFusionEngine engine(NavState{}, initialSqrtCovariance(),
                           ImuNoiseParams::fogGrade());
    double previous_trace = engine.covariance().trace();
    ASSERT_TRUE(engine.predict(samples.front()));  // establishes the time origin

    for (std::size_t i = 1; i < samples.size(); ++i) {
        ASSERT_TRUE(engine.predict(samples[i]));
        const double trace = engine.covariance().trace();
        ASSERT_GT(trace, previous_trace) << "step " << i;
        previous_trace = trace;
    }
}

TEST(Predict, SquareRootFactorStaysWellFormedAcrossATwoMinuteBlackout) {
    // The structural claim the square-root form is bought for. 200Hz for 120s
    // is 24,000 predict steps with nothing re-conditioning the covariance --
    // the regime where a plain UKF's repeated P -> chol(P) -> P round trip
    // accumulates asymmetry until the factorization fails.
    ImuStreamOptions options;
    options.duration_s = 120.0;
    options.rate_hz = 200.0;
    const auto samples =
        generateImuStream(ConstantTurnTrajectory(kSixtyKmphInMps, 0.02), options);
    ASSERT_GT(samples.size(), 24000u);

    UkfFusionEngine engine(NavState{}, initialSqrtCovariance(),
                           ImuNoiseParams::fogGrade());
    for (const auto& sample : samples) {
        ASSERT_TRUE(engine.predict(sample));
        const auto& s = engine.covarianceSqrt();
        for (int i = 0; i < UkfFusionEngine::kStateDim; ++i) {
            ASSERT_GT(s(i, i), 0.0);
            ASSERT_TRUE(std::isfinite(s(i, i)));
            for (int j = i + 1; j < UkfFusionEngine::kStateDim; ++j) {
                ASSERT_EQ(s(i, j), 0.0);
            }
        }
    }
}

// --- Step rejection --------------------------------------------------------

TEST(Predict, RejectsNonMonotonicAndOverlongSteps) {
    UkfFusionEngine engine(NavState{}, initialSqrtCovariance(),
                           ImuNoiseParams::fogGrade());
    ImuSample sample;
    sample.accel = Vec3(0.0, 0.0, -driftless::kGravity);

    sample.timestamp_nanos = 1'000'000'000;
    EXPECT_TRUE(engine.predict(sample));  // first sample: time origin only

    sample.timestamp_nanos = 900'000'000;  // backwards
    EXPECT_FALSE(engine.predict(sample));

    sample.timestamp_nanos = 1'000'000'000;  // zero-length step
    EXPECT_FALSE(engine.predict(sample));

    sample.timestamp_nanos = 5'000'000'000;  // 4s gap, over max_step_seconds
    EXPECT_FALSE(engine.predict(sample));

    // A rejected step must leave the filter untouched, so a valid step from the
    // original origin still works.
    sample.timestamp_nanos = 1'005'000'000;
    EXPECT_TRUE(engine.predict(sample));
}

}  // namespace
