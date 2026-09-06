package com.driftless.fusion

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt

/**
 * TFLite wrapper for the learned forward speed and heading-change model.
 *
 * Implements the contract defined in `MODEL_CONTRACT.md`:
 *   - Input tensor: [1, 80, 14] float32 (80 samples at 10 Hz = 8s context)
 *   - Output tensor: [1, 3] float32 ([speed_ms, dpsi_rad, dv_ms])
 *
 * Channels (14 total):
 *   0..2:   acc_x, acc_y, acc_z (FRD m/s^2)
 *   3..5:   gyro_yaw, gyro_pitch, gyro_roll (rad/s)
 *   6..8:   grav_x, grav_y, grav_z (m/s^2)
 *   9..13:  acc_norm, acc_vert, acc_horiz, gyro_vert, gyro_horiz (attitude-invariant derived features)
 */
class VelocityModel(context: Context? = null) {

    data class Prediction(
        /** Mean forward speed over the 2s prediction interval, m/s. */
        val speedMps: Float,
        /** Heading change across the 2s prediction interval, radians. */
        val dpsiRad: Float,
        /** Change in forward speed across the interval, m/s. */
        val dvMps: Float,
    )

    private var interpreter: Interpreter? = null

    // Ring buffer: 80 samples x 14 channels
    private val buffer = Array(WINDOW_SIZE) { FloatArray(NUM_CHANNELS) }
    private var bufferCount = 0
    private var writeIndex = 0

    // Causal one-pole gravity low-pass filter: y[n] = a * y[n-1] + (1 - a) * x[n]
    private var gravLp = floatArrayOf(0f, 0f, 9.80665f)
    private var gravityInitialized = false

    init {
        if (context != null) {
            try {
                val modelBuffer = loadModelFile(context, MODEL_PATH)
                if (modelBuffer != null) {
                    val options = Interpreter.Options().apply {
                        setNumThreads(2)
                    }
                    interpreter = Interpreter(modelBuffer, options)
                    Log.i(TAG, "TFLite VelocityModel loaded successfully from $MODEL_PATH")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load TFLite model from $MODEL_PATH; using fallback", e)
            }
        }
    }

    /**
     * Feeds one 10 Hz IMU sample into the context window.
     * Note: IMU should arrive in FRD body frame.
     */
    fun addSample(
        accel: Vec3,
        gyro: Vec3,
        gravity: Vec3? = null,
        dtSeconds: Double = 0.1,
    ) {
        val dt = if (dtSeconds in 0.001..1.0) dtSeconds.toFloat() else 0.1f
        val a = exp(-dt / GRAVITY_TAU_S)

        val ax = accel.x.toFloat()
        val ay = accel.y.toFloat()
        val az = accel.z.toFloat()

        if (!gravityInitialized) {
            gravLp[0] = ax
            gravLp[1] = ay
            gravLp[2] = az
            gravityInitialized = true
        } else {
            gravLp[0] = a * gravLp[0] + (1f - a) * ax
            gravLp[1] = a * gravLp[1] + (1f - a) * ay
            gravLp[2] = a * gravLp[2] + (1f - a) * az
        }

        // Gravity unit vector
        val gNorm = sqrt(gravLp[0] * gravLp[0] + gravLp[1] * gravLp[1] + gravLp[2] * gravLp[2])
        val gxNorm = if (gNorm > 1e-3f) gravLp[0] / gNorm else 0f
        val gyNorm = if (gNorm > 1e-3f) gravLp[1] / gNorm else 0f
        val gzNorm = if (gNorm > 1e-3f) gravLp[2] / gNorm else 1f

        // Derived attitude-invariant features
        val accNorm = sqrt(ax * ax + ay * ay + az * az)
        val accVert = ax * gxNorm + ay * gyNorm + az * gzNorm
        val accHoriz = sqrt(max(accNorm * accNorm - accVert * accVert, 0f))

        val gx = gyro.x.toFloat()
        val gy = gyro.y.toFloat()
        val gz = gyro.z.toFloat()
        val gyroNorm = sqrt(gx * gx + gy * gy + gz * gz)
        // Note on gyro axis projection: gz is yaw, gx is pitch/roll in body FRD
        val gyroVert = gx * gxNorm + gy * gyNorm + gz * gzNorm
        val gyroHoriz = sqrt(max(gyroNorm * gyroNorm - gyroVert * gyroVert, 0f))

        val grv = gravity ?: Vec3(gravLp[0].toDouble(), gravLp[1].toDouble(), gravLp[2].toDouble())

        val frame = buffer[writeIndex]
        frame[0] = ax
        frame[1] = ay
        frame[2] = az
        frame[3] = gz // gyro_yaw
        frame[4] = gx // gyro_pitch
        frame[5] = gy // gyro_roll
        frame[6] = grv.x.toFloat()
        frame[7] = grv.y.toFloat()
        frame[8] = grv.z.toFloat()
        frame[9] = accNorm
        frame[10] = accVert
        frame[11] = accHoriz
        frame[12] = gyroVert
        frame[13] = gyroHoriz

        writeIndex = (writeIndex + 1) % WINDOW_SIZE
        if (bufferCount < WINDOW_SIZE) {
            bufferCount++
        }
    }

    val isReady: Boolean
        get() = bufferCount >= 10

    /**
     * Checks if recent IMU buffer is consistent with a stationary device on a desk/mount.
     */
    fun isStationary(): Boolean {
        if (bufferCount < 5) return false
        val lastIdx = (writeIndex - 1 + WINDOW_SIZE) % WINDOW_SIZE
        val lastFrame = buffer[lastIdx]
        val accNorm = lastFrame[9]
        val gx = lastFrame[4]
        val gy = lastFrame[5]
        val gz = lastFrame[3]
        val gyroNorm = sqrt(gx * gx + gy * gy + gz * gz)
        return kotlin.math.abs(accNorm - 9.80665f) < 1.0f && gyroNorm < 0.1f
    }

    /**
     * Runs model inference over the 80-sample window.
     * Returns null if buffer is not full or interpreter is unavailable.
     */
    fun predict(): Prediction? {
        if (!isReady) return null
        if (isStationary()) {
            return Prediction(
                speedMps = 0.0f,
                dpsiRad = 0.0f,
                dvMps = 0.0f,
            )
        }
        if (bufferCount < WINDOW_SIZE) return null
        val interp = interpreter ?: return null

        try {
            // Prepare input buffer: [1, 80, 14] in chronological order
            val input = Array(1) { Array(WINDOW_SIZE) { FloatArray(NUM_CHANNELS) } }
            val startIdx = writeIndex // oldest sample in circular buffer
            for (i in 0 until WINDOW_SIZE) {
                val srcIdx = (startIdx + i) % WINDOW_SIZE
                System.arraycopy(buffer[srcIdx], 0, input[0][i], 0, NUM_CHANNELS)
            }

            val output = Array(1) { FloatArray(3) }
            interp.run(input, output)

            val speed = output[0][0].coerceAtLeast(0f)
            val dpsi = output[0][1]
            val dv = output[0][2]

            return Prediction(
                speedMps = speed,
                dpsiRad = dpsi,
                dvMps = dv,
            )
        } catch (e: Exception) {
            Log.w(TAG, "TFLite inference failed: ${e.message}", e)
            return null
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    companion object {
        const val WINDOW_SIZE = 80
        const val NUM_CHANNELS = 14
        const val GRAVITY_TAU_S = 10.0f
        const val MODEL_PATH = "models/velocity_model.tflite"
        private const val TAG = "VelocityModel"

        private fun loadModelFile(context: Context, path: String): ByteBuffer? {
            return try {
                val fileDescriptor = context.assets.openFd(path)
                val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).apply {
                    order(ByteOrder.nativeOrder())
                }
            } catch (e: Exception) {
                null
            }
        }
    }
}
