package com.driftless.fusion

/**
 * Continuous-time IMU noise densities, discretized per filter step.
 */
data class ImuNoiseParams(
    /** Velocity random walk — accelerometer white noise. m/s^2 / sqrt(Hz). */
    val accelNoiseDensity: Double = 2.45e-4,
    /** Angle random walk — gyroscope white noise. rad/s / sqrt(Hz). */
    val gyroNoiseDensity: Double = 5.82e-6,
    /** Bias random walk: accelerometer bias wander rate. m/s^3 / sqrt(Hz). */
    val accelBiasRandomWalk: Double = 1.0e-5,
    /** Bias random walk: gyroscope bias wander rate. rad/s^2 / sqrt(Hz). */
    val gyroBiasRandomWalk: Double = 1.0e-8,
) {
    companion object {
        /** ~200 Hz navigation-grade FOG preset. */
        fun fogGrade(): ImuNoiseParams = ImuNoiseParams()

        /** ~100 Hz handset consumer MEMS preset for Android smartphones. */
        fun consumerMems(): ImuNoiseParams = ImuNoiseParams(
            accelNoiseDensity = 2.94e-3,
            gyroNoiseDensity = 1.75e-4,
            accelBiasRandomWalk = 1.0e-4,
            gyroBiasRandomWalk = 1.0e-6,
        )
    }
}
