package com.driftless.fusion

import com.driftless.frames.LocalTangentFrame
import com.driftless.math.CholeskyUpdate
import com.driftless.math.Matrix
import com.driftless.math.So3
import com.driftless.math.SqrtKalman
import com.driftless.math.conjugate
import com.driftless.math.dot
import com.driftless.math.minus
import com.driftless.math.norm
import com.driftless.math.normalized
import com.driftless.math.plus
import com.driftless.math.rotate
import com.driftless.math.times
import com.driftless.sensors.ImuSample
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 15-State Error-State Square-Root Unscented Kalman Filter (SR-UKF).
 *
 * Direct Kotlin port of `edge-engine/src/ukf_fusion_engine.cpp`.
 * Implements [PositionFusionEngine].
 *
 * State layout (15 states):
 *   Position error:       indices 0..2  (NED metres)
 *   Velocity error:       indices 3..5  (NED m/s)
 *   Attitude error:       indices 6..8  (body rotation vector, radians)
 *   Accel bias error:     indices 9..11 (body m/s^2)
 *   Gyro bias error:      indices 12..14 (body rad/s)
 *
 * Covariance is maintained in square-root lower-triangular Cholesky factor S (15x15).
 */
class UkfFusionEngine(
    initialState: NavState = NavState(),
    initialCovarianceSqrt: Matrix = defaultInitialCovarianceSqrt(),
    private val noise: ImuNoiseParams = ImuNoiseParams.consumerMems(),
    private val config: Config = Config(),
    private val frameProvider: () -> LocalTangentFrame? = { null },
) : PositionFusionEngine {

    data class GnssParams(
        val minHorizontalAccuracyM: Double = 1.0,
        val minVerticalAccuracyM: Double = 2.0,
        val minSpeedAccuracyMps: Double = 0.1,
        val healthySatelliteCount: Int = 8,
        val satelliteDeficitInflation: Double = 0.25,
        val gateConfidence: Double = 0.99,
    )

    data class NonHolonomicParams(
        val lateralSigmaMps: Double = 0.3,
        val verticalSigmaMps: Double = 0.3,
        val minSpeedMps: Double = 2.0,
        val gateConfidence: Double = 0.99,
    )

    data class VelocityModelParams(
        val sigmaMps: Double = 2.893,
        val biasMps: Double = 0.0,
        val correlationInflation: Double = 2.0,
        val gateConfidence: Double = 0.99,
    )

    data class MapMatchParams(
        val crossTrackSigmaM: Double = 2.5,
        val maxDistanceToRoadM: Double = 30.0,
        val gateConfidence: Double = 0.99,
    )

    data class Diagnostics(
        var lastNis: Double = 0.0,
        var gnssApplied: Int = 0,
        var gnssRejected: Int = 0,
        var nhcApplied: Int = 0,
        var nhcRejected: Int = 0,
        var nhcSkipped: Int = 0,
        var mapMatchApplied: Int = 0,
        var mapMatchRejected: Int = 0,
        var mapMatchSkipped: Int = 0,
        var velocityModelApplied: Int = 0,
        var velocityModelRejected: Int = 0,
    )

    data class Config(
        val alpha: Double = 1e-3,
        val beta: Double = 2.0,
        val kappa: Double = 0.0,
        val gravityNed: Vec3 = Vec3(0.0, 0.0, 9.80665),
        val maxStepSeconds: Double = 0.1,
        val gnss: GnssParams = GnssParams(),
        val nonHolonomic: NonHolonomicParams = NonHolonomicParams(),
        val mapMatch: MapMatchParams = MapMatchParams(),
        val velocityModel: VelocityModelParams = VelocityModelParams(),
    )

    private val lock = Any()

    private var nominal: NavState = initialState.copy(orientation = initialState.orientation.normalized())
    private var sqrtCovariance: Matrix = initialCovarianceSqrt.copy()
    private var lastTimestampNanos: Long? = null

    val diagnostics = Diagnostics()

    private var gamma: Double = 0.0
    private var weightMean0: Double = 0.0
    private var weightCov0: Double = 0.0
    private var weightI: Double = 0.0

    private val positions = MutableSharedFlow<FusedPosition>(
        replay = 1,
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private var lastEmittedNanos = 0L

    private var isOrientationInitialized = false

    init {
        computeWeights()
    }

    private fun computeWeights() {
        val n = STATE_DIM.toDouble()
        val lambda = config.alpha * config.alpha * (n + config.kappa) - n
        val nPlusLambda = n + lambda

        gamma = sqrt(nPlusLambda)
        weightMean0 = lambda / nPlusLambda
        weightCov0 = weightMean0 + 1.0 - config.alpha * config.alpha + config.beta
        weightI = 0.5 / nPlusLambda
    }

    /**
     * Aligns filter attitude so body-measured gravity vector points strictly Down in NED frame.
     */
    fun alignGravity(accel: Vec3) {
        val aNorm = accel.norm()
        if (aNorm in 5.0..15.0) {
            val downBody = Vec3(-accel.x, -accel.y, -accel.z).normalized()
            val qAlign = So3.fromTwoVectors(from = downBody, to = Vec3(0.0, 0.0, 1.0))
            nominal = nominal.copy(orientation = qAlign)
            isOrientationInitialized = true
        }
    }

    override fun predict(sample: ImuSample) {
        val accelVec = Vec3(sample.accel[0].toDouble(), sample.accel[1].toDouble(), sample.accel[2].toDouble())
        val gyroVec = Vec3(sample.gyro[0].toDouble(), sample.gyro[1].toDouble(), sample.gyro[2].toDouble())
        predictStep(sample.timestampNanos, accelVec, gyroVec)
    }

    fun predict(timestampNanos: Long, accel: Vec3, gyro: Vec3): Boolean =
        predictStep(timestampNanos, accel, gyro)

    private fun predictStep(timestampNanos: Long, accel: Vec3, gyro: Vec3): Boolean {
        synchronized(lock) {
            // Auto-align orientation with gravity on startup
            if (!isOrientationInitialized) {
                alignGravity(accel)
            }

            val last = lastTimestampNanos
            if (last == null) {
                lastTimestampNanos = timestampNanos
                return true
            }

            val dt = (timestampNanos - last) / NANOS_PER_SECOND
            if (!(dt > 0.0) || dt > config.maxStepSeconds) {
                return false
            }

            // 1. Generate sigma points in error space
            val chi = sigmaPoints(sqrtCovariance, gamma)
            val propStates = Array(SIGMA_POINTS) { NavState() }
            val propErrors = Array(SIGMA_POINTS) { DoubleArray(STATE_DIM) }

            // 2 & 3. Compose onto nominal and propagate through strapdown mechanization
            for (i in 0 until SIGMA_POINTS) {
                val statePoint = compose(nominal, chi[i])
                propStates[i] = mechanize(statePoint, accel, gyro, dt, config.gravityNed)
            }

            // 4. Propagated center point is the new nominal
            val propNominal = propStates[0]

            // 5. Decompose back to error coordinates about the new nominal
            for (i in 0 until SIGMA_POINTS) {
                propErrors[i] = decompose(propStates[i], propNominal)
            }

            // 6. Weighted mean of the error
            val meanError = DoubleArray(STATE_DIM)
            for (k in 0 until STATE_DIM) {
                var sum = weightMean0 * propErrors[0][k]
                for (i in 1 until SIGMA_POINTS) {
                    sum += weightI * propErrors[i][k]
                }
                meanError[k] = sum
            }

            // 7. Square-root covariance update
            val sqrtDt = sqrt(dt)
            val sqrtQ = DoubleArray(STATE_DIM)
            val posQ = noise.accelNoiseDensity * dt * sqrtDt / sqrt(3.0)
            val velQ = noise.accelNoiseDensity * sqrtDt
            val attQ = noise.gyroNoiseDensity * sqrtDt
            val abQ = noise.accelBiasRandomWalk * sqrtDt
            val gbQ = noise.gyroBiasRandomWalk * sqrtDt

            sqrtQ[0] = posQ; sqrtQ[1] = posQ; sqrtQ[2] = posQ
            sqrtQ[3] = velQ; sqrtQ[4] = velQ; sqrtQ[5] = velQ
            sqrtQ[6] = attQ; sqrtQ[7] = attQ; sqrtQ[8] = attQ
            sqrtQ[9] = abQ;  sqrtQ[10] = abQ; sqrtQ[11] = abQ
            sqrtQ[12] = gbQ; sqrtQ[13] = gbQ; sqrtQ[14] = gbQ

            val stacked = Matrix.zeros(2 * STATE_DIM + STATE_DIM, STATE_DIM)
            val sqrtWeightI = sqrt(weightI)
            for (i in 1 until SIGMA_POINTS) {
                val row = DoubleArray(STATE_DIM)
                for (k in 0 until STATE_DIM) {
                    row[k] = sqrtWeightI * (propErrors[i][k] - meanError[k])
                }
                stacked.setRow(i - 1, row)
            }
            for (k in 0 until STATE_DIM) {
                stacked[2 * STATE_DIM + k, k] = sqrtQ[k]
            }

            val newSqrtCov = CholeskyUpdate.qrToLowerTriangular(stacked)

            val centreDeviation = DoubleArray(STATE_DIM)
            for (k in 0 until STATE_DIM) {
                centreDeviation[k] = propErrors[0][k] - meanError[k]
            }

            if (!CholeskyUpdate.cholUpdate(newSqrtCov, centreDeviation, weightCov0)) {
                return false
            }

            // 8. Inject mean error into nominal state
            nominal = compose(propNominal, meanError).let {
                it.copy(orientation = it.orientation.normalized())
            }
            sqrtCovariance = newSqrtCov
            lastTimestampNanos = timestampNanos

            // Publish fused position at ~10 Hz
            emitPositionIfDue(timestampNanos)
            return true
        }
    }

    override fun updateGnss(fix: GnssFix): UpdateOutcome = synchronized(lock) {
        val useVelocity = fix.hasVelocity
        val m = if (useVelocity) 6 else 3

        val H = Matrix.zeros(m, STATE_DIM)
        H[0, POSITION_INDEX] = 1.0
        H[1, POSITION_INDEX + 1] = 1.0
        H[2, POSITION_INDEX + 2] = 1.0

        if (useVelocity) {
            H[3, VELOCITY_INDEX] = 1.0
            H[4, VELOCITY_INDEX + 1] = 1.0
            H[5, VELOCITY_INDEX + 2] = 1.0
        }

        val innovation = DoubleArray(m)
        innovation[0] = fix.position.north - nominal.position.x
        innovation[1] = fix.position.east - nominal.position.y
        innovation[2] = fix.position.down - nominal.position.z

        if (useVelocity) {
            innovation[3] = fix.velocityNed.x - nominal.velocity.x
            innovation[4] = fix.velocityNed.y - nominal.velocity.y
            innovation[5] = fix.velocityNed.z - nominal.velocity.z
        }

        val sqrtR = gnssSqrtNoise(fix, useVelocity)
        val outcome = updateLinear(H, innovation, sqrtR, config.gnss.gateConfidence)

        if (outcome == UpdateOutcome.Applied) {
            diagnostics.gnssApplied++
        } else if (outcome == UpdateOutcome.RejectedByGate) {
            diagnostics.gnssRejected++
        }

        emitPositionIfDue(fix.timestampNanos, force = true)
        outcome
    }

    fun updateZeroVelocity(sigmaMps: Double = 0.05): UpdateOutcome = synchronized(lock) {
        val H = Matrix.zeros(3, STATE_DIM)
        H[0, VELOCITY_INDEX] = 1.0
        H[1, VELOCITY_INDEX + 1] = 1.0
        H[2, VELOCITY_INDEX + 2] = 1.0

        val currentSpeed = nominal.velocity.norm()
        val effectiveSigma = max(sigmaMps, currentSpeed * 0.25)
        val innovation = doubleArrayOf(
            -nominal.velocity.x,
            -nominal.velocity.y,
            -nominal.velocity.z,
        )
        val sqrtR = Matrix.diagonal(doubleArrayOf(effectiveSigma, effectiveSigma, effectiveSigma))
        val outcome = updateLinear(H, innovation, sqrtR, 0.9999)
        if (outcome == UpdateOutcome.RejectedByGate && currentSpeed > 0.0) {
            nominal = nominal.copy(velocity = nominal.velocity * 0.5)
        }
        outcome
    }

    fun updateNonHolonomic(): UpdateOutcome = synchronized(lock) {
        val p = config.nonHolonomic
        val speed = nominal.velocity.norm()
        if (speed < p.minSpeedMps) {
            val outcome = updateZeroVelocity(0.05)
            if (outcome == UpdateOutcome.Applied) {
                diagnostics.nhcApplied++
            }
            return outcome
        }

        val h = { s: NavState ->
            val bVel = bodyVelocity(s)
            doubleArrayOf(bVel.y, bVel.z) // lateral and vertical body velocity
        }

        val sqrtR = Matrix.diagonal(doubleArrayOf(p.lateralSigmaMps, p.verticalSigmaMps))
        val outcome = updateUnscented(h, DoubleArray(2), sqrtR, p.gateConfidence)

        if (outcome == UpdateOutcome.Applied) {
            diagnostics.nhcApplied++
        } else if (outcome == UpdateOutcome.RejectedByGate) {
            diagnostics.nhcRejected++
        }
        outcome
    }

    fun updateVelocityModel(predictedForwardSpeedMps: Double): UpdateOutcome = synchronized(lock) {
        val p = config.velocityModel
        val isStandstill = predictedForwardSpeedMps <= 0.01

        if (isStandstill) {
            val outcome = updateZeroVelocity(0.05)
            if (outcome == UpdateOutcome.Applied) {
                diagnostics.velocityModelApplied++
            }
            return outcome
        }

        val correctedSpeedMps = max(0.0, predictedForwardSpeedMps - p.biasMps)

        val h = { s: NavState ->
            val bVel = bodyVelocity(s)
            doubleArrayOf(bVel.x) // forward body velocity
        }

        val sqrtR = Matrix(1, 1, doubleArrayOf(p.sigmaMps * p.correlationInflation))
        val outcome = updateUnscented(h, doubleArrayOf(correctedSpeedMps), sqrtR, p.gateConfidence)

        if (outcome == UpdateOutcome.Applied) {
            diagnostics.velocityModelApplied++
        } else if (outcome == UpdateOutcome.RejectedByGate) {
            diagnostics.velocityModelRejected++
        }
        outcome
    }

    override fun updateMapMatch(match: MapMatchResult): UpdateOutcome = synchronized(lock) {
        val p = config.mapMatch
        if (!match.matched || match.distanceToRoadM > p.maxDistanceToRoadM) {
            diagnostics.mapMatchSkipped++
            return UpdateOutcome.Skipped
        }

        val dir = match.segmentDirection
        val horizontal = hypot(dir.x, dir.y)
        if (horizontal < 1e-9) {
            diagnostics.mapMatchSkipped++
            return UpdateOutcome.Skipped
        }
        val normal = Vec3(-dir.y / horizontal, dir.x / horizontal, 0.0)

        val H = Matrix.zeros(1, STATE_DIM)
        H[0, POSITION_INDEX] = normal.x
        H[0, POSITION_INDEX + 1] = normal.y
        H[0, POSITION_INDEX + 2] = normal.z

        val deltaPos = Vec3(
            match.snappedPosition.x - nominal.position.x,
            match.snappedPosition.y - nominal.position.y,
            match.snappedPosition.z - nominal.position.z,
        )
        val innovation = doubleArrayOf(normal.dot(deltaPos))
        val sqrtR = Matrix(1, 1, doubleArrayOf(p.crossTrackSigmaM))

        val outcome = updateLinear(H, innovation, sqrtR, p.gateConfidence)
        if (outcome == UpdateOutcome.Applied) {
            diagnostics.mapMatchApplied++
        } else if (outcome == UpdateOutcome.RejectedByGate) {
            diagnostics.mapMatchRejected++
        }
        outcome
    }

    fun updateLinear(H: Matrix, innovation: DoubleArray, sqrtR: Matrix, gateConfidence: Double): UpdateOutcome {
        val result = SqrtKalman.arrayFormUpdate(sqrtCovariance, H, sqrtR)
        val nis = SqrtKalman.normalizedInnovationSquared(result.sqrtInnovationCovariance, innovation)
        diagnostics.lastNis = nis
        if (!nis.isFinite()) return UpdateOutcome.NumericalFailure

        val threshold = SqrtKalman.chiSquaredThreshold(innovation.size, gateConfidence)
        if (threshold < 0.0) return UpdateOutcome.NumericalFailure
        if (nis > threshold) return UpdateOutcome.RejectedByGate

        val correction = result.gain * innovation
        nominal = compose(nominal, correction).let {
            it.copy(orientation = it.orientation.normalized())
        }
        sqrtCovariance = result.sqrtCovariancePosterior
        return UpdateOutcome.Applied
    }

    fun updateUnscented(h: (NavState) -> DoubleArray, z: DoubleArray, sqrtR: Matrix, gateConfidence: Double): UpdateOutcome {
        val m = z.size
        val chi = sigmaPoints(sqrtCovariance, gamma)

        val predicted = Array(SIGMA_POINTS) { i ->
            h(compose(nominal, chi[i]))
        }

        val predictedMean = DoubleArray(m)
        for (k in 0 until m) {
            var sum = weightMean0 * predicted[0][k]
            for (i in 1 until SIGMA_POINTS) {
                sum += weightI * predicted[i][k]
            }
            predictedMean[k] = sum
        }

        val stacked = Matrix.zeros(2 * STATE_DIM + m, m)
        val sqrtWeightI = sqrt(weightI)
        for (i in 1 until SIGMA_POINTS) {
            val row = DoubleArray(m)
            for (k in 0 until m) {
                row[k] = sqrtWeightI * (predicted[i][k] - predictedMean[k])
            }
            stacked.setRow(i - 1, row)
        }
        stacked.setBlock(2 * STATE_DIM, 0, sqrtR.transpose())

        val sqrtInnovation = CholeskyUpdate.qrToLowerTriangular(stacked)
        val centreDev = DoubleArray(m)
        for (k in 0 until m) {
            centreDev[k] = predicted[0][k] - predictedMean[k]
        }
        if (!CholeskyUpdate.cholUpdate(sqrtInnovation, centreDev, weightCov0)) {
            return UpdateOutcome.NumericalFailure
        }

        val innovation = DoubleArray(m)
        for (k in 0 until m) {
            innovation[k] = z[k] - predictedMean[k]
        }

        val nis = SqrtKalman.normalizedInnovationSquared(sqrtInnovation, innovation)
        diagnostics.lastNis = nis
        if (!nis.isFinite()) return UpdateOutcome.NumericalFailure

        val threshold = SqrtKalman.chiSquaredThreshold(m, gateConfidence)
        if (threshold < 0.0) return UpdateOutcome.NumericalFailure
        if (nis > threshold) return UpdateOutcome.RejectedByGate

        // Cross-covariance P_xz (n x m)
        val crossCovariance = Matrix.zeros(STATE_DIM, m)
        for (i in 1 until SIGMA_POINTS) {
            val chiI = chi[i]
            val predDiff = DoubleArray(m) { k -> predicted[i][k] - predictedMean[k] }
            for (r in 0 until STATE_DIM) {
                for (c in 0 until m) {
                    crossCovariance[r, c] += weightI * chiI[r] * predDiff[c]
                }
            }
        }

        // Kalman gain K = P_xz * (S_nu * S_nu^T)^-1
        val temp = sqrtInnovation.solveLowerTriangular(crossCovariance.transpose())
        val gain = sqrtInnovation.transpose().solveUpperTriangular(temp).transpose()

        // Covariance downdate: S+ = cholUpdate(S, -1.0, col_j)
        val factor = sqrtCovariance.copy()
        val downdateCols = gain * sqrtInnovation
        for (j in 0 until m) {
            val colJ = downdateCols.col(j)
            if (!CholeskyUpdate.cholUpdate(factor, colJ, -1.0)) {
                return UpdateOutcome.NumericalFailure
            }
        }

        val correction = gain * innovation
        nominal = compose(nominal, correction).let {
            it.copy(orientation = it.orientation.normalized())
        }
        sqrtCovariance = factor
        return UpdateOutcome.Applied
    }

    private fun gnssSqrtNoise(fix: GnssFix, useVelocity: Boolean): Matrix {
        val p = config.gnss
        val horizontal = max(fix.horizontalAccuracyM, p.minHorizontalAccuracyM)
        val vertical = max(fix.verticalAccuracyM, p.minVerticalAccuracyM)

        val deficit = max(0, p.healthySatelliteCount - fix.satellitesUsed)
        val inflation = 1.0 + p.satelliteDeficitInflation * deficit.toDouble() * deficit.toDouble()

        val m = if (useVelocity) 6 else 3
        val sigmas = DoubleArray(m)
        sigmas[0] = horizontal * inflation
        sigmas[1] = horizontal * inflation
        sigmas[2] = vertical * inflation

        if (useVelocity) {
            val speed = max(fix.speedAccuracyMps, p.minSpeedAccuracyMps) * inflation
            sigmas[3] = speed
            sigmas[4] = speed
            sigmas[5] = speed
        }

        return Matrix.diagonal(sigmas)
    }

    override fun state(): NavState = synchronized(lock) { nominal }

    fun covarianceSqrt(): Matrix = synchronized(lock) { sqrtCovariance }

    fun covariance(): Matrix = synchronized(lock) { sqrtCovariance * sqrtCovariance.transpose() }

    override fun observePosition(): Flow<FusedPosition> = positions.asSharedFlow()

    private fun emitPositionIfDue(nowNanos: Long, force: Boolean = false) {
        if (!force && nowNanos - lastEmittedNanos < EMIT_INTERVAL_NANOS) return
        val frame = frameProvider() ?: return

        lastEmittedNanos = nowNanos
        val posNed = Position(nominal.position.x, nominal.position.y, nominal.position.z)
        val geodetic = frame.toGeodetic(posNed)
        val vel = nominal.velocity
        val speedMps = hypot(vel.x, vel.y).toFloat()
        val headingDeg = if (speedMps > 0.5f) {
            ((Math.toDegrees(atan2(vel.y, vel.x)) + 360.0) % 360.0).toFloat()
        } else {
            val forwardNed = nominal.orientation.rotate(Vec3(1.0, 0.0, 0.0))
            ((Math.toDegrees(atan2(forwardNed.y, forwardNed.x)) + 360.0) % 360.0).toFloat()
        }

        // Covariance-based confidence metric (0..1)
        val s = sqrtCovariance
        val horizVar = s[0, 0] * s[0, 0] + s[0, 1] * s[0, 1] + s[1, 0] * s[1, 0] + s[1, 1] * s[1, 1]
        val sigmaH = sqrt(max(horizVar, 0.01))
        val confidence = (1.0 / (1.0 + sigmaH / 10.0)).toFloat().coerceIn(0.05f, 1.0f)

        val fused = FusedPosition(
            lat = geodetic.latDeg,
            lon = geodetic.lonDeg,
            headingDegrees = headingDeg,
            speedMetersPerSec = speedMps,
            confidence = confidence,
        )
        positions.tryEmit(fused)
    }

    companion object {
        const val STATE_DIM = 15
        const val SIGMA_POINTS = 2 * STATE_DIM + 1

        const val POSITION_INDEX = 0
        const val VELOCITY_INDEX = 3
        const val ATTITUDE_INDEX = 6
        const val ACCEL_BIAS_INDEX = 9
        const val GYRO_BIAS_INDEX = 12

        private const val NANOS_PER_SECOND = 1_000_000_000.0
        private const val EMIT_INTERVAL_NANOS = 100_000_000L // 10 Hz

        fun defaultInitialCovarianceSqrt(): Matrix {
            val sigmas = DoubleArray(STATE_DIM)
            sigmas[0] = 5.0;  sigmas[1] = 5.0;  sigmas[2] = 5.0    // pos 5 m
            sigmas[3] = 1.0;  sigmas[4] = 1.0;  sigmas[5] = 1.0    // vel 1 m/s
            sigmas[6] = 0.1;  sigmas[7] = 0.1;  sigmas[8] = 0.2    // att ~6-11 deg
            sigmas[9] = 0.1;  sigmas[10] = 0.1; sigmas[11] = 0.1  // accel bias 0.1 m/s^2
            sigmas[12] = 0.01; sigmas[13] = 0.01; sigmas[14] = 0.01 // gyro bias 0.01 rad/s
            return Matrix.diagonal(sigmas)
        }

        fun sigmaPoints(sqrtCovariance: Matrix, gamma: Double): Array<DoubleArray> {
            val points = Array(SIGMA_POINTS) { DoubleArray(STATE_DIM) }
            for (i in 0 until STATE_DIM) {
                val colI = sqrtCovariance.col(i)
                val scaled = DoubleArray(STATE_DIM) { k -> gamma * colI[k] }
                val negScaled = DoubleArray(STATE_DIM) { k -> -gamma * colI[k] }
                points[1 + i] = scaled
                points[1 + STATE_DIM + i] = negScaled
            }
            return points
        }

        fun mechanize(s: NavState, accel: Vec3, gyro: Vec3, dt: Double, gravityNed: Vec3): NavState {
            val omega = gyro - s.gyroBias
            val force = accel - s.accelBias
            val accelNed = s.orientation.rotate(force) + gravityNed

            val outPos = s.position + s.velocity * dt + 0.5 * accelNed * dt * dt
            val outVel = s.velocity + accelNed * dt
            val outOri = So3.boxPlus(s.orientation, omega * dt)
            return s.copy(
                position = outPos,
                velocity = outVel,
                orientation = outOri,
            )
        }

        fun compose(nominal: NavState, error: DoubleArray): NavState {
            val dp = Vec3(error[POSITION_INDEX], error[POSITION_INDEX + 1], error[POSITION_INDEX + 2])
            val dv = Vec3(error[VELOCITY_INDEX], error[VELOCITY_INDEX + 1], error[VELOCITY_INDEX + 2])
            val dtheta = Vec3(error[ATTITUDE_INDEX], error[ATTITUDE_INDEX + 1], error[ATTITUDE_INDEX + 2])
            val dba = Vec3(error[ACCEL_BIAS_INDEX], error[ACCEL_BIAS_INDEX + 1], error[ACCEL_BIAS_INDEX + 2])
            val dbg = Vec3(error[GYRO_BIAS_INDEX], error[GYRO_BIAS_INDEX + 1], error[GYRO_BIAS_INDEX + 2])

            return nominal.copy(
                position = nominal.position + dp,
                velocity = nominal.velocity + dv,
                orientation = So3.boxPlus(nominal.orientation, dtheta),
                accelBias = nominal.accelBias + dba,
                gyroBias = nominal.gyroBias + dbg,
            )
        }

        fun decompose(s: NavState, nominal: NavState): DoubleArray {
            val error = DoubleArray(STATE_DIM)
            val dp = s.position - nominal.position
            val dv = s.velocity - nominal.velocity
            val dtheta = So3.boxMinus(s.orientation, nominal.orientation)
            val dba = s.accelBias - nominal.accelBias
            val dbg = s.gyroBias - nominal.gyroBias

            error[POSITION_INDEX] = dp.x; error[POSITION_INDEX + 1] = dp.y; error[POSITION_INDEX + 2] = dp.z
            error[VELOCITY_INDEX] = dv.x; error[VELOCITY_INDEX + 1] = dv.y; error[VELOCITY_INDEX + 2] = dv.z
            error[ATTITUDE_INDEX] = dtheta.x; error[ATTITUDE_INDEX + 1] = dtheta.y; error[ATTITUDE_INDEX + 2] = dtheta.z
            error[ACCEL_BIAS_INDEX] = dba.x; error[ACCEL_BIAS_INDEX + 1] = dba.y; error[ACCEL_BIAS_INDEX + 2] = dba.z
            error[GYRO_BIAS_INDEX] = dbg.x; error[GYRO_BIAS_INDEX + 1] = dbg.y; error[GYRO_BIAS_INDEX + 2] = dbg.z
            return error
        }

        fun bodyVelocity(s: NavState): Vec3 = s.orientation.conjugate().rotate(s.velocity)
    }
}
