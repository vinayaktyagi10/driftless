// Measurement updates: the linear GNSS path, the unscented non-holonomic path,
// and the outlier gate that stands in front of both.

#include "driftless/ukf_fusion_engine.h"

#include "support/trajectory_sim.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

namespace {

using driftless::GnssFix;
using driftless::ImuNoiseParams;
using driftless::NavState;
using driftless::Position;
using driftless::UkfFusionEngine;
using driftless::Vec3;
using Outcome = UkfFusionEngine::UpdateOutcome;
using ErrorMatrix = UkfFusionEngine::ErrorMatrix;

ErrorMatrix diagonalSqrtCovariance(double position_sigma = 5.0,
                                   double velocity_sigma = 0.5,
                                   double attitude_sigma_deg = 1.0) {
    UkfFusionEngine::ErrorVector d;
    d.segment<3>(UkfFusionEngine::kPositionIndex).setConstant(position_sigma);
    d.segment<3>(UkfFusionEngine::kVelocityIndex).setConstant(velocity_sigma);
    d.segment<3>(UkfFusionEngine::kAttitudeIndex)
        .setConstant(attitude_sigma_deg * M_PI / 180.0);
    d.segment<3>(UkfFusionEngine::kAccelBiasIndex).setConstant(0.01);
    d.segment<3>(UkfFusionEngine::kGyroBiasIndex).setConstant(1e-4);
    return d.asDiagonal();
}

UkfFusionEngine makeEngine(const NavState& state = NavState{},
                           const ErrorMatrix& s = diagonalSqrtCovariance()) {
    return UkfFusionEngine(state, s, ImuNoiseParams::fogGrade());
}

GnssFix fixAt(const Vec3& ned, double accuracy = 3.0, int satellites = 12) {
    GnssFix fix;
    fix.position = Position::fromVector(ned);
    fix.horizontal_accuracy_m = accuracy;
    fix.vertical_accuracy_m = 2.0 * accuracy;
    fix.satellites_used = satellites;
    return fix;
}

// --- GNSS ------------------------------------------------------------------

TEST(GnssUpdate, PullsEstimateTowardTheFixAndShrinksUncertainty) {
    auto engine = makeEngine();
    const Vec3 measured(10.0, -4.0, 0.5);

    const double prior_variance = engine.covariance()(0, 0);
    ASSERT_EQ(engine.updateGnss(fixAt(measured)), Outcome::kApplied);

    // Moved toward the fix, but not all the way -- the prior still counts.
    EXPECT_GT(engine.state().position.x(), 0.0);
    EXPECT_LT(engine.state().position.x(), measured.x());
    EXPECT_LT(engine.state().position.y(), 0.0);
    EXPECT_LT(engine.covariance()(0, 0), prior_variance);
    EXPECT_EQ(engine.diagnostics().gnss_applied, 1);
}

TEST(GnssUpdate, AVeryPreciseFixNearlySnapsToTheMeasurement) {
    // The accuracy floor has to be lowered explicitly, or it -- not the fix --
    // decides the result. See AccuracyFloorIgnoresImplausiblePrecisionClaims.
    UkfFusionEngine::Config config;
    config.gnss.min_horizontal_accuracy_m = 0.01;
    config.gnss.min_vertical_accuracy_m = 0.01;
    UkfFusionEngine engine(NavState{}, diagonalSqrtCovariance(),
                           ImuNoiseParams::fogGrade(), config);

    const Vec3 measured(10.0, -4.0, 0.5);
    ASSERT_EQ(engine.updateGnss(fixAt(measured, 0.02)), Outcome::kApplied);
    EXPECT_LT((engine.state().position - measured).norm(), 0.05);
}

TEST(GnssUpdate, AccuracyFloorIgnoresImplausiblePrecisionClaims) {
    // A receiver claiming millimetre accuracy is not to be believed, and the
    // floor is what stops one bad fix from collapsing the covariance to nothing
    // -- after which the gate would reject every subsequent honest fix and the
    // filter would free-run while reporting high confidence. The floor is a
    // guard against that lock-up, not a tuning knob.
    auto floored = makeEngine();
    ASSERT_EQ(floored.updateGnss(fixAt(Vec3(10.0, 0.0, 0.0), 1e-6)),
              Outcome::kApplied);

    UkfFusionEngine::Config config;
    config.gnss.min_horizontal_accuracy_m = 1e-6;
    config.gnss.min_vertical_accuracy_m = 1e-6;
    UkfFusionEngine unfloored(NavState{}, diagonalSqrtCovariance(),
                              ImuNoiseParams::fogGrade(), config);
    ASSERT_EQ(unfloored.updateGnss(fixAt(Vec3(10.0, 0.0, 0.0), 1e-6)),
              Outcome::kApplied);

    // With the floor the posterior stays sane; without it, it collapses.
    EXPECT_GT(floored.covariance()(0, 0), 0.5);
    EXPECT_LT(unfloored.covariance()(0, 0), 1e-6);
}

TEST(GnssUpdate, RepeatedConsistentFixesConvergeAndDoNotOscillate) {
    // A 25 m offset only makes sense against a prior that admits to being that
    // uncertain -- i.e. a filter that has just dead-reckoned through a tunnel.
    // Against the default 5 m prior this same fix is a 4-sigma event and the
    // gate rejects it, correctly.
    auto engine = makeEngine(NavState{}, diagonalSqrtCovariance(20.0));
    const Vec3 truth(20.0, 15.0, 0.0);
    double previous_error = (engine.state().position - truth).norm();
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(engine.updateGnss(fixAt(truth)), Outcome::kApplied);
        const double error = (engine.state().position - truth).norm();
        EXPECT_LT(error, previous_error) << "iteration " << i;
        previous_error = error;
    }
    EXPECT_LT(previous_error, 0.5);
}

TEST(GnssUpdate, UsesVelocityWhenTheFixReportsIt) {
    NavState state;
    state.velocity = Vec3(15.0, 0.0, 0.0);
    auto engine = makeEngine(state, diagonalSqrtCovariance(5.0, 2.0));

    GnssFix fix = fixAt(Vec3(5.0, 0.0, 0.0));
    fix.has_velocity = true;
    fix.velocity_ned = Vec3(18.0, 0.0, 0.0);
    fix.speed_accuracy_mps = 0.2;

    const double prior_velocity_variance = engine.covariance()(3, 3);
    ASSERT_EQ(engine.updateGnss(fix), Outcome::kApplied);

    EXPECT_GT(engine.state().velocity.x(), 17.0);
    EXPECT_LT(engine.covariance()(3, 3), prior_velocity_variance)
        << "a Doppler velocity must actually inform the velocity state";
}

// --- The outlier gate ------------------------------------------------------

TEST(OutlierGate, RejectsAConfidentlyWrongMultipathFix) {
    // The urban-canyon failure mode: a fix 60 m off that claims 3 m accuracy.
    // Un-gated this would drag the position onto the wrong street AND poison
    // the bias states, which is the part that outlives the bad fix.
    auto engine = makeEngine();
    const NavState before = engine.state();
    const ErrorMatrix covariance_before = engine.covarianceSqrt();

    EXPECT_EQ(engine.updateGnss(fixAt(Vec3(60.0, 0.0, 0.0), 3.0)),
              Outcome::kRejectedByGate);

    // A rejected measurement must change nothing at all.
    EXPECT_EQ(engine.state().position, before.position);
    EXPECT_EQ(engine.covarianceSqrt(), covariance_before);
    EXPECT_EQ(engine.diagnostics().gnss_rejected, 1);
    EXPECT_EQ(engine.diagnostics().gnss_applied, 0);
}

TEST(OutlierGate, AcceptsAFixConsistentWithItsOwnStatedAccuracy) {
    // The gate must not be so tight that ordinary noise trips it, or the filter
    // silently free-runs while looking healthy.
    auto engine = makeEngine();
    EXPECT_EQ(engine.updateGnss(fixAt(Vec3(4.0, -3.0, 1.0), 3.0)), Outcome::kApplied);
}

TEST(OutlierGate, TheSameOffsetPassesWhenTheFixHonestlyReportsPoorAccuracy) {
    // NIS tests consistency, not distance. A 60 m innovation is fine if the fix
    // admits to 40 m accuracy -- what gets rejected is the *contradiction*
    // between a large innovation and a confident claim.
    auto engine = makeEngine();
    EXPECT_EQ(engine.updateGnss(fixAt(Vec3(60.0, 0.0, 0.0), 40.0)), Outcome::kApplied);
}

TEST(AdaptiveNoise, FewerSatellitesProducesASmallerCorrection) {
    // Two fixes identical in every respect except satellite count. The one from
    // a degraded constellation must move the estimate less, because its
    // geometry-derived accuracy claim is less trustworthy.
    const Vec3 measured(8.0, 0.0, 0.0);

    auto open_sky = makeEngine();
    ASSERT_EQ(open_sky.updateGnss(fixAt(measured, 3.0, 12)), Outcome::kApplied);

    auto canyon = makeEngine();
    ASSERT_EQ(canyon.updateGnss(fixAt(measured, 3.0, 4)), Outcome::kApplied);

    EXPECT_GT(open_sky.state().position.x(), canyon.state().position.x());
    EXPECT_GT(canyon.state().position.x(), 0.0) << "still informative, just less so";
    std::cout << "[ INFO     ] correction with 12 sats: "
              << open_sky.state().position.x() << " m, with 4 sats: "
              << canyon.state().position.x() << " m\n";
}

// --- Non-holonomic constraint ----------------------------------------------

TEST(NonHolonomic, IsSkippedBelowTheSpeedGate) {
    // At standstill the constraint is satisfied by every heading, so it carries
    // no information -- applying it anyway would let noise steer the attitude.
    auto engine = makeEngine();
    EXPECT_EQ(engine.updateNonHolonomic(), Outcome::kSkipped);
    EXPECT_EQ(engine.diagnostics().nhc_skipped, 1);
}

TEST(NonHolonomic, DrivesBodyLateralVelocityTowardZero) {
    // The filter believes it is travelling due North while pointing 5 degrees
    // off. That implies the vehicle is crabbing sideways at ~1.5 m/s, which a
    // car cannot do -- so the constraint has real information to contribute.
    NavState state;
    state.velocity = Vec3(16.67, 0.0, 0.0);
    state.orientation = driftless::so3::expMap(Vec3(0.0, 0.0, 5.0 * M_PI / 180.0));
    auto engine = makeEngine(state);

    const double before = std::abs(UkfFusionEngine::bodyVelocity(engine.state()).y());
    ASSERT_GT(before, 1.0) << "sanity: the scenario really does violate NHC";

    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(engine.updateNonHolonomic(), Outcome::kApplied) << "iteration " << i;
    }

    const double after = std::abs(UkfFusionEngine::bodyVelocity(engine.state()).y());
    std::cout << "[ INFO     ] body lateral velocity " << before << " -> " << after
              << " m/s\n";
    EXPECT_LT(after, 0.2 * before);
}

TEST(NonHolonomic, SharpensHeadingKnowledgeWithoutAnyPositionFix) {
    // The payoff during a blackout: the constraint is informative about
    // ATTITUDE, which is what bounds heading drift when GNSS is unavailable.
    NavState state;
    state.velocity = Vec3(16.67, 0.0, 0.0);
    auto engine = makeEngine(state);

    const double yaw_variance_before =
        engine.covariance()(UkfFusionEngine::kAttitudeIndex + 2,
                            UkfFusionEngine::kAttitudeIndex + 2);
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(engine.updateNonHolonomic(), Outcome::kApplied);
    }
    const double yaw_variance_after =
        engine.covariance()(UkfFusionEngine::kAttitudeIndex + 2,
                            UkfFusionEngine::kAttitudeIndex + 2);

    EXPECT_LT(yaw_variance_after, yaw_variance_before);
    std::cout << "[ INFO     ] yaw sigma " << std::sqrt(yaw_variance_before) * 180 / M_PI
              << " deg -> " << std::sqrt(yaw_variance_after) * 180 / M_PI << " deg\n";
}

// --- The two update paths must agree where they overlap --------------------

TEST(UpdatePaths, UnscentedAgreesWithLinearOnALinearMeasurement) {
    // The unscented transform is EXACT for a linear measurement function, so
    // running a position fix through the general unscented path must reproduce
    // the closed-form linear update. This is the test that justifies using the
    // cheap path for GNSS: it demonstrates the expensive one would have given
    // the same answer.
    const Vec3 measured(10.0, -4.0, 0.5);
    const Eigen::MatrixXd sqrt_R = 3.0 * Eigen::MatrixXd::Identity(3, 3);

    auto linear = makeEngine();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, UkfFusionEngine::kStateDim);
    H.block<3, 3>(0, UkfFusionEngine::kPositionIndex) = Eigen::Matrix3d::Identity();
    ASSERT_EQ(linear.updateLinear(H, measured - linear.state().position, sqrt_R, 0.99),
              Outcome::kApplied);

    auto unscented = makeEngine();
    ASSERT_EQ(unscented.updateUnscented(
                  [](const NavState& s) -> Eigen::VectorXd { return s.position; },
                  measured, sqrt_R, 0.99),
              Outcome::kApplied);

    EXPECT_LT((linear.state().position - unscented.state().position).norm(), 1e-8);
    EXPECT_LT((linear.state().velocity - unscented.state().velocity).norm(), 1e-8);
    EXPECT_LT((linear.covariance() - unscented.covariance()).cwiseAbs().maxCoeff(),
              1e-6 * linear.covariance().cwiseAbs().maxCoeff());
}

}  // namespace
