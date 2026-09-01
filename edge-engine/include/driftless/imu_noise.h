#pragma once

namespace driftless {

// Continuous-time IMU noise densities, which the filter discretizes per step.
//
// PROVENANCE, honestly: these defaults are datasheet-class order-of-magnitude
// figures, not measurements. They are placeholders until an Allan-variance fit
// is run on real logs from the actual sensor -- the FOG unit for the edge
// engine, and a handset for the Android port. Q is the single biggest lever on
// how fast the covariance inflates during a blackout, so tuning it off a guess
// and then quoting the resulting drift as a result would be dishonest.
// TODO: fit from Allan deviation once IO-VNBD / FOG logs are in hand.
struct ImuNoiseParams {
    // Velocity random walk -- accelerometer white noise. m/s^2 / sqrt(Hz).
    double accel_noise_density = 2.45e-4;
    // Angle random walk -- gyroscope white noise. rad/s / sqrt(Hz).
    double gyro_noise_density = 5.82e-6;
    // Bias random walks: how fast the biases are allowed to wander.
    double accel_bias_random_walk = 1.0e-5;  // m/s^3 / sqrt(Hz)
    double gyro_bias_random_walk = 1.0e-8;   // rad/s^2 / sqrt(Hz)

    // ~200 Hz navigation-grade FOG: 0.02 deg/sqrt(h) ARW, 25 ug/sqrt(Hz) VRW.
    static ImuNoiseParams fogGrade() { return {}; }

    // ~100 Hz handset MEMS, for the Android port. Roughly thirty times noisier
    // in gyro and twelve times in accelerometer, which is the entire reason the
    // phone needs map-matching to stay bounded and the FOG unit needs it less.
    static ImuNoiseParams consumerMems() {
        ImuNoiseParams p;
        p.accel_noise_density = 2.94e-3;
        p.gyro_noise_density = 1.75e-4;
        p.accel_bias_random_walk = 1.0e-4;
        p.gyro_bias_random_walk = 1.0e-6;
        return p;
    }
};

}  // namespace driftless
