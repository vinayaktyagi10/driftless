#pragma once

#include "driftless/types.h"

#include <cstdint>
#include <vector>

namespace driftless::testing {

// A constant-speed, constant-turn-rate trajectory in the horizontal plane, and
// the ideal IMU measurements an ideal sensor riding it would report.
//
// Deliberately derived from the analytic definition of the motion rather than
// by inverting UkfFusionEngine::mechanize. If the simulator reused the
// mechanization, any sign error in the mechanization would appear in the
// generated measurements too and cancel exactly -- the tests would pass on a
// filter that integrates gravity the wrong way round. Everything here comes
// from differentiating the trajectory by hand:
//
//   psi(t)   = psi0 + rate * t
//   v_ned(t) = speed * [cos psi, sin psi, 0]
//   a_ned(t) = speed * rate * [-sin psi, cos psi, 0]
//   gyro_b   = [0, 0, rate]                       (yaw is about body-down)
//   accel_b  = R(psi)^T * (a_ned - g_ned)
//            = [0, speed*rate, -g]                (centripetal is body-right)
//
// The last line is the one worth staring at: with NED/FRD and psi increasing,
// the vehicle turns toward East, i.e. to its right, so the centripetal
// acceleration is +y in body frame and the accelerometer reads -g on its
// down axis while level.
class ConstantTurnTrajectory {
public:
    ConstantTurnTrajectory(double speed_mps, double yaw_rate_rps,
                           double initial_heading_rad = 0.0);

    // Ground truth at time t. Bias fields are always zero -- truth has no bias.
    [[nodiscard]] NavState truthAt(double t) const;

    // Ideal (noise- and bias-free) measurements. Constant for this family.
    [[nodiscard]] Vec3 idealAccel() const { return ideal_accel_; }
    [[nodiscard]] Vec3 idealGyro() const { return ideal_gyro_; }

private:
    double speed_;
    double yaw_rate_;
    double initial_heading_;
    Vec3 ideal_accel_;
    Vec3 ideal_gyro_;
};

struct ImuStreamOptions {
    double rate_hz = 200.0;
    double duration_s = 60.0;
    // Constant biases added to the measurements. The filter starts believing
    // the biases are zero, so these are what it has to either estimate or
    // suffer -- and with no measurement update, it suffers.
    Vec3 accel_bias = Vec3::Zero();
    Vec3 gyro_bias = Vec3::Zero();
    // White noise densities; the per-sample sigma is density * sqrt(rate).
    double accel_noise_density = 0.0;
    double gyro_noise_density = 0.0;
    std::uint32_t seed = 0;
};

[[nodiscard]] std::vector<ImuSample> generateImuStream(
    const ConstantTurnTrajectory& trajectory, const ImuStreamOptions& options);

}  // namespace driftless::testing
