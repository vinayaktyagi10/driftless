#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdint>

namespace driftless {

// ---------------------------------------------------------------------------
// Frame conventions. These are the contract; everything else in the engine
// assumes them, and the Kotlin port must mirror them exactly.
//
//   Navigation frame : local-tangent NED (x North, y East, z Down), origin
//                      anchored at the first GNSS fix. Flat-earth -- over the
//                      distances a blackout covers, earth-rate and Coriolis
//                      terms are far below this IMU's noise floor.
//   Body frame       : FRD (x Forward, y Right, z Down).
//   Attitude         : Hamilton unit quaternion, body -> nav.
//   Gravity          : g_nav = [0, 0, +9.80665] (down is positive z), so a
//                      level stationary accelerometer reads [0, 0, -9.80665].
//
// NOTE for the Android port: SensorManager delivers an ENU-ish axis convention
// (x right, y forward, z up). A conversion layer belongs on that side of the
// boundary, not in here -- this engine only ever sees FRD/NED.
// ---------------------------------------------------------------------------

using Vec3 = Eigen::Vector3d;
using Quat = Eigen::Quaterniond;

inline constexpr double kGravity = 9.80665;  // m/s^2

// One raw inertial measurement. Mirrors ImuSample in
// android/app/src/main/java/com/driftless/sensors/ImuSampler.kt.
struct ImuSample {
    std::int64_t timestamp_nanos = 0;
    Vec3 accel = Vec3::Zero();  // specific force, body FRD, m/s^2
    Vec3 gyro = Vec3::Zero();   // angular rate, body FRD, rad/s
};

// A point in the local-tangent navigation frame, in metres from the origin.
struct Position {
    double north = 0.0;
    double east = 0.0;
    double down = 0.0;

    [[nodiscard]] Vec3 vector() const { return Vec3(north, east, down); }
    static Position fromVector(const Vec3& v) { return {v.x(), v.y(), v.z()}; }
};

// A GNSS observation, already projected into the local-tangent frame.
// Accuracies are 1-sigma, not the 68%-radius some receivers report -- convert
// at the sampler boundary.
struct GnssFix {
    std::int64_t timestamp_nanos = 0;
    Position position;
    Vec3 velocity_ned = Vec3::Zero();
    // Doppler-derived velocity is a genuinely independent observation and is
    // far better than differencing positions, but not every receiver reports it
    // and it is meaningless at standstill -- so it is opt-in per fix.
    bool has_velocity = false;
    double speed_accuracy_mps = 0.0;
    double horizontal_accuracy_m = 0.0;
    double vertical_accuracy_m = 0.0;
    // Surfaced so fusion can detect degradation before a full outage, per
    // GnssSampler.kt -- a fix from 4 satellites in an urban canyon deserves a
    // very different R than one from 14 in the open.
    int satellites_used = 0;
};

// The nominal navigation state. This is integrated directly and is NOT what
// the UKF covariance is over -- see ukf_fusion_engine.h for why.
struct NavState {
    Vec3 position = Vec3::Zero();        // m, NED
    Vec3 velocity = Vec3::Zero();        // m/s, NED
    Quat orientation = Quat::Identity();  // body -> nav
    Vec3 accel_bias = Vec3::Zero();      // m/s^2, body
    Vec3 gyro_bias = Vec3::Zero();       // rad/s, body
};

}  // namespace driftless
