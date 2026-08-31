package com.driftless.sensors

/**
 * Raw accelerometer/gyroscope/magnetometer stream off SensorManager,
 * timestamped at hardware rate. No filtering happens here — this is
 * the boundary between the phone's sensors and everything downstream.
 */
class ImuSampler {
    // TODO: register SensorEventListener for TYPE_ACCELEROMETER, TYPE_GYROSCOPE,
    // TYPE_MAGNETIC_FIELD at SENSOR_DELAY_FASTEST; expose as a cold Flow<ImuSample>.
}

data class ImuSample(
    val timestampNanos: Long,
    val accel: FloatArray,
    val gyro: FloatArray,
    val mag: FloatArray?
)
