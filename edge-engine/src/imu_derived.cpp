#include "driftless/imu_derived.h"

#include <algorithm>
#include <cmath>

namespace driftless {

Vec3 GravityLowpass::push(const Vec3& accel, double dt_s) {
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
        dt_s = 0.1;  // matches the Python fallback in _causal_gravity
    }
    const double a = std::exp(-dt_s / tau_s_);
    if (!initialized_) {
        state_ = accel;
        initialized_ = true;
    } else {
        state_ = a * state_ + (1.0 - a) * accel;
    }
    return state_;
}

ImuDerivedChannels computeImuDerived(const Vec3& accel, const Vec3& gyro,
                                     const Vec3& gravity_estimate) {
    ImuDerivedChannels out;

    const double g_norm = gravity_estimate.norm();
    const Vec3 g_hat = (g_norm > 1e-3) ? Vec3(gravity_estimate / std::max(g_norm, 1e-9))
                                        : Vec3(0.0, 0.0, 1.0);  // Python fallback

    const double acc_norm = accel.norm();
    const double acc_vert = accel.dot(g_hat);
    out.acc_norm = acc_norm;
    out.acc_vert = acc_vert;
    out.acc_horiz = std::sqrt(std::max(acc_norm * acc_norm - acc_vert * acc_vert, 0.0));

    const double gyro_norm = gyro.norm();
    const double gyro_vert = gyro.dot(g_hat);
    out.gyro_vert = gyro_vert;
    out.gyro_horiz = std::sqrt(std::max(gyro_norm * gyro_norm - gyro_vert * gyro_vert, 0.0));

    return out;
}

}  // namespace driftless