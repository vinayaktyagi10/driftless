package com.driftless.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import com.driftless.frames.AxisConvention
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import java.util.concurrent.atomic.AtomicLong

/**
 * Raw accelerometer/gyroscope/magnetometer stream off SensorManager,
 * timestamped at hardware rate. No filtering happens here — this is
 * the boundary between the phone's sensors and everything downstream.
 *
 * The one thing this class does do is **convert axes**: everything it emits is
 * body FRD, never SensorManager's device frame. See [AxisConvention] for why
 * that boundary is here and not in the fusion engine.
 */
class ImuSampler(context: Context) {

    private val manager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accelerometer = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer = manager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val gravity = manager.getDefaultSensor(Sensor.TYPE_GRAVITY)

    val hasAccelerometer get() = accelerometer != null
    val hasGyroscope get() = gyroscope != null
    val hasMagnetometer get() = magnetometer != null
    val hasGravity get() = gravity != null

    /**
     * Samples the collector could not keep up with.
     *
     * Deliberately counted rather than silently swallowed: a dropped IMU sample
     * is a hole in a dead-reckoned integral, so this being non-zero is a real
     * defect and not a performance detail. Surfaced in the UI during step 6.
     */
    val droppedSamples = AtomicLong(0)

    /**
     * Cold flow of IMU frames, in body FRD.
     *
     * Emission is anchored on the **accelerometer**, carrying the most recent
     * gyro/mag/gravity by zero-order hold. At `SENSOR_DELAY_FASTEST` the
     * resulting skew is under one sample period — about 5 ms at 200 Hz, or 8 cm
     * of travel at 60 km/h, comfortably under the GNSS noise the filter is
     * correcting against. Interpolating instead would be more correct and is
     * not worth the complexity before the pipeline runs end to end.
     *
     * Nothing is emitted until accelerometer **and** gyroscope have each
     * reported at least once. Emitting earlier would send a zeroed gyro
     * downstream, which does not read as "unknown" to a filter — it reads as
     * "perfectly still", and it would be integrated as fact.
     *
     * @param sensorDelay one of the `SensorManager.SENSOR_DELAY_*` constants.
     */
    fun frames(sensorDelay: Int): Flow<ImuFrame> = callbackFlow {
        // Sensor callbacks land on this thread rather than the main looper, so
        // a slow frame on the UI thread cannot stall sampling.
        val thread = HandlerThread("imu-sampler").apply { start() }
        val handler = Handler(thread.looper)

        val latestAccel = FloatArray(3)
        val latestGyro = FloatArray(3)
        val latestMag = FloatArray(3)
        val latestGravity = FloatArray(3)

        var seenAccel = false
        var seenGyro = false
        var seenMag = false
        var seenGravity = false

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                // event.values is recycled by the framework between callbacks,
                // so every path below copies before keeping anything.
                when (event.sensor.type) {
                    Sensor.TYPE_GYROSCOPE -> {
                        event.values.copyInto(latestGyro, endIndex = 3)
                        AxisConvention.deviceToFrdInPlace(latestGyro)
                        seenGyro = true
                    }

                    Sensor.TYPE_MAGNETIC_FIELD -> {
                        event.values.copyInto(latestMag, endIndex = 3)
                        AxisConvention.deviceToFrdInPlace(latestMag)
                        seenMag = true
                    }

                    Sensor.TYPE_GRAVITY -> {
                        event.values.copyInto(latestGravity, endIndex = 3)
                        AxisConvention.deviceToFrdInPlace(latestGravity)
                        seenGravity = true
                    }

                    Sensor.TYPE_ACCELEROMETER -> {
                        event.values.copyInto(latestAccel, endIndex = 3)
                        AxisConvention.deviceToFrdInPlace(latestAccel)
                        seenAccel = true

                        if (!seenGyro) return

                        val frame = ImuFrame(
                            // Already on the elapsedRealtime monotonic base;
                            // GnssSampler joins it there via
                            // Location.getElapsedRealtimeNanos().
                            timestampNanos = event.timestamp,
                            accel = latestAccel.copyOf(),
                            gyro = latestGyro.copyOf(),
                            mag = if (seenMag) latestMag.copyOf() else null,
                            gravity = if (seenGravity) latestGravity.copyOf() else null,
                        )
                        if (trySend(frame).isFailure) {
                            droppedSamples.incrementAndGet()
                        }
                    }
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }

        accelerometer?.let { manager.registerListener(listener, it, sensorDelay, handler) }
        gyroscope?.let { manager.registerListener(listener, it, sensorDelay, handler) }
        magnetometer?.let { manager.registerListener(listener, it, sensorDelay, handler) }
        // Gravity is a fused virtual sensor and does not need the IMU's rate;
        // it feeds three of the velocity model's fourteen channels, which are
        // consumed at 10 Hz.
        gravity?.let {
            manager.registerListener(listener, it, SensorManager.SENSOR_DELAY_GAME, handler)
        }

        awaitClose {
            manager.unregisterListener(listener)
            thread.quitSafely()
        }
    }.buffer(BUFFER_CAPACITY)

    private companion object {
        /**
         * ~2.5 s of headroom at 200 Hz. Large enough to ride out a GC pause or
         * a slow map frame; small enough that a genuinely stalled collector
         * shows up as [droppedSamples] rather than as unbounded memory growth.
         */
        const val BUFFER_CAPACITY = 512
    }
}

/**
 * Everything the sampler knows about one instant, in body FRD.
 *
 * Deliberately *not* the contract type. [ImuSample] is what Role 02's engine
 * consumes and must stay exactly as `types.h` defines it; the velocity model
 * additionally needs the gravity vector for three of its fourteen input
 * channels. Carrying that here rather than widening [ImuSample] keeps the
 * fusion contract stable while giving step 5 what it needs without re-plumbing
 * the sampler.
 *
 * Not a data class: it holds arrays, and a data class would generate
 * identity-comparing `equals`/`hashCode`. Reference equality is the honest
 * default for a plain class, so the trap simply does not arise. [ImuSample],
 * which cannot avoid being a data class, overrides them instead.
 */
class ImuFrame(
    val timestampNanos: Long,
    val accel: FloatArray,
    val gyro: FloatArray,
    val mag: FloatArray?,
    val gravity: FloatArray?,
) {
    fun toImuSample() = ImuSample(timestampNanos, accel, gyro, mag)
}

/**
 * One raw inertial measurement, in the engine's frames.
 *
 * Mirrors `ImuSample` in `edge-engine/include/driftless/types.h`, which names
 * this file explicitly. [accel] and [gyro] are body **FRD** — SensorManager's
 * device axes are converted before construction, never after.
 *
 * [timestampNanos] is on the `elapsedRealtime` monotonic base
 * (`SensorEvent.timestamp`). GNSS fixes must use
 * `Location.getElapsedRealtimeNanos()` to land on the same clock;
 * `Location.getTime()` is UTC epoch millis and mixing the two hands the filter
 * a dt measured in decades.
 *
 * [mag] is nullable and has no C++ counterpart — the engine does not consume it,
 * but the heading fallbacks on this side do, and not every device has one.
 */
data class ImuSample(
    val timestampNanos: Long,
    val accel: FloatArray,
    val gyro: FloatArray,
    val mag: FloatArray?,
) {
    // A data class holding FloatArray generates equals/hashCode that compare
    // array *identity*, so two samples with identical contents are unequal and
    // the same sample rehashes differently after a copy. Harmless for a
    // consume-only stream, silently wrong the moment a test asserts equality or
    // one of these lands in a Set. Overridden now, before the fusion port
    // starts depending on this type.

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ImuSample) return false
        return timestampNanos == other.timestampNanos &&
            accel.contentEquals(other.accel) &&
            gyro.contentEquals(other.gyro) &&
            mag.contentEquals(other.mag)
    }

    override fun hashCode(): Int {
        var result = timestampNanos.hashCode()
        result = 31 * result + accel.contentHashCode()
        result = 31 * result + gyro.contentHashCode()
        result = 31 * result + (mag?.contentHashCode() ?: 0)
        return result
    }
}
