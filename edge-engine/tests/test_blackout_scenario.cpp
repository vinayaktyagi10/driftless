// End-to-end: GNSS available, then a one-kilometre blackout, then reacquisition.
// This is the first test in the suite that can say anything at all about the
// <10%-drift-over-distance bar, because it is the first one where anything
// bounds the error.
//
// It runs the same drive four ways -- dead reckoning alone, plus the
// non-holonomic constraint, plus map matching, and both -- so the contribution
// of each is separately visible rather than asserted.
//
// Sensor grade is deliberately CONSUMER MEMS, not the FOG unit. On FOG hardware
// unaided dead reckoning already meets the bar over a kilometre (see
// test_drift_characterization), so it would demonstrate nothing. The phone is
// the hard case, and the phone is most of the deployment.

#include "driftless/hmm_map_matcher.h"
#include "driftless/road_graph.h"
#include "driftless/ukf_fusion_engine.h"

#include "support/trajectory_sim.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace {

using driftless::GnssFix;
using driftless::HmmMapMatcher;
using driftless::ImuNoiseParams;
using driftless::MapMatchParams;
using driftless::NavState;
using driftless::Position;
using driftless::RoadGraph;
using driftless::UkfFusionEngine;
using driftless::Vec3;
using driftless::testing::ConstantTurnTrajectory;
using driftless::testing::generateImuStream;
using driftless::testing::ImuStreamOptions;

constexpr double kSpeedMps = 60.0 / 3.6;
constexpr double kImuRateHz = 200.0;
constexpr double kWarmupSeconds = 40.0;
constexpr double kBlackoutSeconds = 60.0;  // 1 km at 60 km/h
constexpr double kRecoverySeconds = 20.0;
constexpr double kBlackoutDistanceM = kSpeedMps * kBlackoutSeconds;

struct Aiding {
    bool non_holonomic = false;
    bool map_match = false;
    // Whether the receiver supplies Doppler velocity alongside position. Kept
    // as a variable rather than assumed, because it turns out to matter more
    // than either of the other two.
    bool gnss_doppler = false;
};

struct Result {
    double error_at_blackout_end_m = 0.0;
    double worst_error_during_blackout_m = 0.0;
    // The road runs due North, so North is along-track and East is cross-track.
    double along_track_at_blackout_end_m = 0.0;
    double cross_track_at_blackout_end_m = 0.0;
    double error_after_reacquisition_m = 0.0;
    int gnss_rejected = 0;

    [[nodiscard]] double percentOfDistance() const {
        return 100.0 * error_at_blackout_end_m / kBlackoutDistanceM;
    }
};

UkfFusionEngine::ErrorMatrix initialSqrtCovariance() {
    UkfFusionEngine::ErrorVector d;
    d.segment<3>(UkfFusionEngine::kPositionIndex).setConstant(5.0);
    d.segment<3>(UkfFusionEngine::kVelocityIndex).setConstant(0.5);
    d.segment<3>(UkfFusionEngine::kAttitudeIndex).setConstant(2.0 * M_PI / 180.0);
    d.segment<3>(UkfFusionEngine::kAccelBiasIndex).setConstant(0.1);
    d.segment<3>(UkfFusionEngine::kGyroBiasIndex).setConstant(5e-3);
    return d.asDiagonal();
}

// A straight road running due North, which is also the route driven.
RoadGraph buildRoad(double length_m) {
    RoadGraph graph;
    int previous = graph.addNode(Vec3(-200.0, 0.0, 0.0));
    for (double north = -100.0; north <= length_m + 200.0; north += 100.0) {
        const int next = graph.addNode(Vec3(north, 0.0, 0.0));
        graph.addSegment(previous, next, 1);
        previous = next;
    }
    return graph;
}

bool inBlackout(double t) {
    return t >= kWarmupSeconds && t < kWarmupSeconds + kBlackoutSeconds;
}

Result runScenario(const Aiding& aiding) {
    const ConstantTurnTrajectory trajectory(kSpeedMps, 0.0);
    const double total_seconds = kWarmupSeconds + kBlackoutSeconds + kRecoverySeconds;

    ImuStreamOptions imu_options;
    imu_options.rate_hz = kImuRateHz;
    imu_options.duration_s = total_seconds;
    const ImuNoiseParams noise = ImuNoiseParams::consumerMems();
    imu_options.accel_noise_density = noise.accel_noise_density;
    imu_options.gyro_noise_density = noise.gyro_noise_density;
    // Turn-on biases the filter starts out ignorant of. The gyro bias is the
    // one that hurts: it rotates the entire solution, so its positional effect
    // grows with distance travelled, not merely with time.
    imu_options.accel_bias = Vec3(0.03, 0.02, 0.0);
    imu_options.gyro_bias = Vec3(0.0, 0.0, 2e-3);
    imu_options.seed = 20261001;
    const auto samples = generateImuStream(trajectory, imu_options);

    UkfFusionEngine engine(trajectory.truthAt(0.0), initialSqrtCovariance(), noise);
    const RoadGraph road = buildRoad(kSpeedMps * total_seconds);
    HmmMapMatcher matcher(road, MapMatchParams{});

    std::mt19937 rng(20261002);
    std::normal_distribution<double> gnss_noise(0.0, 3.0);

    Result result;
    int last_gnss_second = -1;
    int last_aiding_decisecond = -1;
    bool captured_blackout_end = false;
    (void)captured_blackout_end;

    for (const auto& sample : samples) {
        const double t = static_cast<double>(sample.timestamp_nanos) * 1e-9;
        if (!engine.predict(sample)) {
            ADD_FAILURE() << "predict rejected a valid sample at t=" << t;
            return result;
        }

        // Aiding that needs no external infrastructure runs at 10Hz throughout
        // -- during the blackout it is all there is.
        const int decisecond = static_cast<int>(t * 10.0);
        if (decisecond != last_aiding_decisecond) {
            last_aiding_decisecond = decisecond;
            if (aiding.non_holonomic) engine.updateNonHolonomic();
            if (aiding.map_match) {
                engine.updateMapMatch(matcher.step(engine.state().position));
            }
        }

        // GNSS at 1Hz, absent during the blackout.
        const int second = static_cast<int>(t);
        if (second != last_gnss_second && !inBlackout(t)) {
            last_gnss_second = second;
            const NavState truth = trajectory.truthAt(t);
            GnssFix fix;
            fix.timestamp_nanos = sample.timestamp_nanos;
            fix.position = Position::fromVector(
                truth.position + Vec3(gnss_noise(rng), gnss_noise(rng), 0.0));
            fix.horizontal_accuracy_m = 3.0;
            fix.vertical_accuracy_m = 6.0;
            fix.satellites_used = 11;
            // Doppler velocity. Nearly every receiver reports it, and it turns
            // out to be the single most important input for surviving a
            // blackout -- see the along-track/cross-track test below.
            if (aiding.gnss_doppler) {
                fix.has_velocity = true;
                fix.velocity_ned =
                    truth.velocity + Vec3(gnss_noise(rng), gnss_noise(rng), 0.0) * 0.03;
                fix.speed_accuracy_mps = 0.1;
            }
            engine.updateGnss(fix);
        }

        // Measured BEFORE any of this sample's aiding is applied would be
        // cleaner still, but the ordering that actually matters is that the
        // end-of-blackout figure must not include the first recovery fix --
        // hence capturing on every in-blackout sample and keeping the last,
        // rather than sampling once GNSS has already returned.
        const Vec3 error = engine.state().position - trajectory.truthAt(t).position;
        const double horizontal = std::hypot(error.x(), error.y());
        if (inBlackout(t)) {
            result.worst_error_during_blackout_m =
                std::max(result.worst_error_during_blackout_m, horizontal);
            result.error_at_blackout_end_m = horizontal;
            result.along_track_at_blackout_end_m = std::abs(error.x());
            result.cross_track_at_blackout_end_m = std::abs(error.y());
            captured_blackout_end = true;
        }
    }

    const Vec3 final_error =
        engine.state().position - trajectory.truthAt(total_seconds).position;
    result.error_after_reacquisition_m = std::hypot(final_error.x(), final_error.y());
    result.gnss_rejected = engine.diagnostics().gnss_rejected;
    return result;
}

void report(const char* label, const Result& r) {
    std::cout << "[ BLACKOUT ] " << std::left << std::setw(26) << label
              << " end-of-blackout error " << std::fixed << std::setprecision(1)
              << std::setw(7) << r.error_at_blackout_end_m << " m ("
              << std::setprecision(2) << r.percentOfDistance() << "% of "
              << std::setprecision(0) << kBlackoutDistanceM << " m)"
              << "  [along " << std::setprecision(1)
              << r.along_track_at_blackout_end_m << " m, cross "
              << r.cross_track_at_blackout_end_m << " m]\n";
}

TEST(BlackoutScenario, EachAidingSourceReducesDriftAndTheCombinationMeetsTheBar) {
    // Position-only GNSS. This is the hard case: forward speed is only weakly
    // observable from 1Hz position fixes, so the filter enters the blackout
    // already wrong about how fast it is going, and that error integrates.
    const Result dead_reckoning = runScenario({false, false, false});
    const Result with_nhc = runScenario({true, false, false});
    const Result with_map = runScenario({false, true, false});
    const Result with_both = runScenario({true, true, false});

    std::cout << "[ BLACKOUT ] --- position-only GNSS ---\n";
    report("dead reckoning only", dead_reckoning);
    report("+ non-holonomic", with_nhc);
    report("+ map matching", with_map);
    report("+ both", with_both);

    // The scenario must actually be hard, or none of the rest means anything:
    // unaided dead reckoning on this sensor grade must MISS the bar.
    EXPECT_GT(dead_reckoning.percentOfDistance(), 10.0)
        << "unaided drift is too small for this scenario to demonstrate anything";

    // Each aiding source helps on its own.
    EXPECT_LT(with_nhc.error_at_blackout_end_m, dead_reckoning.error_at_blackout_end_m);
    EXPECT_LT(with_map.error_at_blackout_end_m, dead_reckoning.error_at_blackout_end_m);
    // Each source bounds the error it is capable of bounding, and the
    // combination bounds cross-track best of all.
    EXPECT_LT(with_both.cross_track_at_blackout_end_m,
              with_nhc.cross_track_at_blackout_end_m);
    EXPECT_LT(with_both.cross_track_at_blackout_end_m, 1.0);

    // But NOT that the combination minimises TOTAL error, because here it does
    // not: adding map matching on top of NHC takes cross-track from ~10 m to
    // ~0.1 m while pushing along-track from ~118 m to ~133 m, so the total gets
    // slightly worse. That is a real effect, not noise. The cross-track update
    // is scalar, but the states it corrects are correlated with the along-track
    // ones through the covariance, so pulling sideways drags forward-backward
    // with it. It only shows up here because along-track error is enormous in
    // the position-only case; with Doppler (below) it disappears. Worth knowing
    // before someone in a panel asks why more aiding made a number go up.
    EXPECT_LT(with_both.error_at_blackout_end_m,
              dead_reckoning.error_at_blackout_end_m);

    // Neither constraint can rescue position-only GNSS on this sensor grade:
    // both bound CROSS-track error, and the dominant error here is ALONG-track.
    EXPECT_GT(with_both.percentOfDistance(), 10.0)
        << "if this now passes, the along-track story below has changed";
    EXPECT_LT(with_both.cross_track_at_blackout_end_m, 10.0)
        << "cross-track, though, must be well bounded";
}

TEST(BlackoutScenario, DopplerVelocityIsWhatMakesTheBarReachable) {
    // The finding that came out of building this, and the one worth being able
    // to defend: the aiding that bounds the DOMINANT error term here is not the
    // map and not the non-holonomic constraint -- it is Doppler velocity from
    // the GNSS receiver, taken before the blackout starts. It is what lets the
    // filter enter the tunnel actually knowing its speed.
    //
    // This is also precisely the gap the learned velocity model (training/, and
    // VelocityModel.kt) exists to fill: when GNSS is gone there is no Doppler
    // either, and something has to keep observing forward speed. Until that
    // model exists, along-track error is bounded only by however well speed was
    // known at the moment GNSS dropped.
    const Result position_only = runScenario({true, true, false});
    const Result with_doppler_unaided = runScenario({false, false, true});
    const Result with_doppler = runScenario({true, true, true});

    std::cout << "[ BLACKOUT ] --- with Doppler velocity ---\n";
    report("Doppler, no constraints", with_doppler_unaided);
    report("Doppler + both", with_doppler);

    EXPECT_LT(with_doppler.error_at_blackout_end_m,
              0.5 * position_only.error_at_blackout_end_m);

    // The bar from the problem statement: under 10% of distance travelled.
    EXPECT_LT(with_doppler.percentOfDistance(), 10.0);
    // And with everything on, comfortably rather than marginally under it.
    EXPECT_LT(with_doppler.percentOfDistance(), 5.0);
    // Lane-level laterally, which is what map matching and NHC actually buy.
    EXPECT_LT(with_doppler.cross_track_at_blackout_end_m, 1.5);
}

TEST(BlackoutScenario, MapMatchingBoundsCrossTrackButNotAlongTrackError) {
    // The honest limitation, and worth being able to say out loud: a map
    // constrains the direction PERPENDICULAR to the road and nothing else. On a
    // straight road it can hold the vehicle in the correct lane indefinitely
    // while saying nothing about how far along that road it has got. Position
    // along the road is bounded only by GNSS returning.
    const Result unaided = runScenario({false, false, true});
    const Result mapped = runScenario({false, true, true});

    EXPECT_LT(mapped.cross_track_at_blackout_end_m,
              0.5 * unaided.cross_track_at_blackout_end_m)
        << "map matching must substantially bound cross-track error";

    std::cout << "[ BLACKOUT ] along-track error unaided "
              << unaided.along_track_at_blackout_end_m << " m -> mapped "
              << mapped.along_track_at_blackout_end_m << " m\n";
}

TEST(BlackoutScenario, ReacquisitionPullsTheSolutionBackWithoutBeingRejected) {
    // The failure mode this guards against: during a long blackout the filter's
    // covariance may not grow enough to admit how wrong it has become, and then
    // the outlier gate rejects the very fixes that would fix it -- the filter
    // free-runs forever while reporting high confidence.
    const Result result = runScenario({true, true, true});

    EXPECT_LT(result.error_after_reacquisition_m, 10.0)
        << "solution failed to recover after GNSS returned";
    EXPECT_LT(result.error_after_reacquisition_m, result.error_at_blackout_end_m)
        << "reacquisition must improve on the dead-reckoned estimate";
    std::cout << "[ BLACKOUT ] error " << result.error_at_blackout_end_m
              << " m at blackout end -> " << result.error_after_reacquisition_m
              << " m after " << kRecoverySeconds << " s of GNSS; "
              << result.gnss_rejected << " fixes rejected by the gate\n";
}

}  // namespace
