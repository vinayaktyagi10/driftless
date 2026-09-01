#include "trajectory_sim.h"

#include "driftless/so3.h"

#include <cmath>
#include <random>

namespace driftless::testing {
namespace {
constexpr double kStraightLineThreshold = 1e-12;
}  // namespace

ConstantTurnTrajectory::ConstantTurnTrajectory(double speed_mps,
                                               double yaw_rate_rps,
                                               double initial_heading_rad)
    : speed_(speed_mps),
      yaw_rate_(yaw_rate_rps),
      initial_heading_(initial_heading_rad),
      ideal_accel_(0.0, speed_mps * yaw_rate_rps, -kGravity),
      ideal_gyro_(0.0, 0.0, yaw_rate_rps) {}

NavState ConstantTurnTrajectory::truthAt(double t) const {
    const double heading = initial_heading_ + yaw_rate_ * t;

    NavState truth;
    if (std::abs(yaw_rate_) < kStraightLineThreshold) {
        truth.position = Vec3(speed_ * t * std::cos(initial_heading_),
                              speed_ * t * std::sin(initial_heading_), 0.0);
    } else {
        // Integral of speed*[cos psi, sin psi, 0] dt with psi linear in t.
        const double radius = speed_ / yaw_rate_;
        truth.position =
            Vec3(radius * (std::sin(heading) - std::sin(initial_heading_)),
                 radius * (std::cos(initial_heading_) - std::cos(heading)), 0.0);
    }
    truth.velocity =
        Vec3(speed_ * std::cos(heading), speed_ * std::sin(heading), 0.0);
    truth.orientation = so3::expMap(Vec3(0.0, 0.0, heading));
    return truth;
}

std::vector<ImuSample> generateImuStream(const ConstantTurnTrajectory& trajectory,
                                         const ImuStreamOptions& options) {
    const double dt = 1.0 / options.rate_hz;
    const auto sample_count =
        static_cast<std::size_t>(options.duration_s * options.rate_hz) + 1;

    // Discrete white noise from a continuous density: sigma = density*sqrt(fs).
    const double accel_sigma =
        options.accel_noise_density * std::sqrt(options.rate_hz);
    const double gyro_sigma =
        options.gyro_noise_density * std::sqrt(options.rate_hz);

    std::mt19937 rng(options.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<ImuSample> samples;
    samples.reserve(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
        ImuSample sample;
        sample.timestamp_nanos =
            static_cast<std::int64_t>(std::llround(i * dt * 1e9));
        sample.accel = trajectory.idealAccel() + options.accel_bias;
        sample.gyro = trajectory.idealGyro() + options.gyro_bias;
        if (accel_sigma > 0.0) {
            sample.accel += accel_sigma * Vec3(gauss(rng), gauss(rng), gauss(rng));
        }
        if (gyro_sigma > 0.0) {
            sample.gyro += gyro_sigma * Vec3(gauss(rng), gauss(rng), gauss(rng));
        }
        samples.push_back(sample);
    }
    return samples;
}

}  // namespace driftless::testing
