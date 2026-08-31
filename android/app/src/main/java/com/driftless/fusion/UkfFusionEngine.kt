package com.driftless.fusion

/**
 * Unscented Kalman Filter over [position, velocity, heading, IMU biases].
 * Predict step runs on every IMU sample (dead reckoning); update step
 * runs on GNSS fixes when available and on map-matched corrections
 * otherwise. This is the seam between "trust the IMU" and "trust GNSS/map".
 */
class UkfFusionEngine {
    // TODO: state vector, sigma points, predict(imuSample), updateGnss(fix),
    // updateMapMatch(snappedPosition)
}
