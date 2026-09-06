#pragma once

#include "driftless/types.h"

namespace driftless {

// Attitude-invariant channels derived from raw IMU. C++ port of
// `imu_derived()` / `_causal_gravity()` in
// training/driftless_train/preprocess.py. Must match that implementation
// exactly -- these five values are part of the model's trained input
// distribution, not a convenience feature you can approximate.
//
// IMPORTANT CAVEAT: the Python function was validated against a phone
// dataset (IO-VNBD) with a *discovered, phone-specific* gyro-axis
// mislabeling (see GYRO_XYZ_COLUMNS in preprocess.py -- the column literally
// named "Pitch" is actually the yaw-rate axis on that hardware). This port
// assumes the edge engine's professional IMU reports gyro in its true,
// correctly-labeled body-FRD (x,y,z) axes, so no equivalent relabeling is
// applied here. That assumption has NOT been validated against real FOG
// logs. Before trusting model output on real hardware: capture a real
// window, run it through this code AND through the Python reference on the
// identical raw samples, and diff the five outputs.
struct ImuDerivedChannels {
    double acc_norm = 0.0;
    double acc_vert = 0.0;
    double acc_horiz = 0.0;
    double gyro_vert = 0.0;
    double gyro_horiz = 0.0;
};

// Causal one-pole low-pass tracking the gravity direction from the raw
// accelerometer (which includes gravity). Stateful: call push() once per
// sample, in timestamp order. Do NOT reset between windows -- only between
// logically distinct sessions/power-cycles -- or every window pays the
// filter's ~30s warmup again. tau = 10s, matching GRAVITY_TAU_S in
// preprocess.py.
//
//   y[n] = a*y[n-1] + (1-a)*x[n],  a = exp(-dt/tau),  y[0] = x[0]
class GravityLowpass {
public:
    explicit GravityLowpass(double tau_s = 10.0) : tau_s_(tau_s) {}

    // Feed one accelerometer sample (m/s^2, includes gravity) with the
    // elapsed time since the previous call. Returns the current estimate.
    Vec3 push(const Vec3& accel, double dt_s);

    [[nodiscard]] bool hasEstimate() const { return initialized_; }
    [[nodiscard]] Vec3 current() const { return state_; }

private:
    double tau_s_;
    bool initialized_ = false;
    Vec3 state_ = Vec3::Zero();
};

// Compute the five derived channels for ONE sample, given that sample's raw
// accel/gyro and the CURRENT gravity-lowpass estimate (call
// GravityLowpass::push() first, then this, using the same sample).
[[nodiscard]] ImuDerivedChannels computeImuDerived(const Vec3& accel,
                                                   const Vec3& gyro,
                                                   const Vec3& gravity_estimate);

}  // namespace driftless