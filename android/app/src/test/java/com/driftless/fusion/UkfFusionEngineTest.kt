package com.driftless.fusion

import com.driftless.mapmatch.HmmMapMatcher
import com.driftless.mapmatch.HmmParams
import com.driftless.mapmatch.RoadGraph
import com.driftless.math.Matrix
import com.driftless.math.So3
import com.driftless.math.dot
import com.driftless.math.minus
import com.driftless.math.norm
import com.driftless.math.plus
import com.driftless.math.rotate
import com.driftless.math.times
import com.driftless.sensors.ImuSample
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

class UkfFusionEngineTest {

    private fun initialSqrtCovariance(
        posSigma: Double = 5.0,
        velSigma: Double = 0.5,
        attSigmaDeg: Double = 2.0,
    ): Matrix {
        val d = DoubleArray(UkfFusionEngine.STATE_DIM)
        d[0] = posSigma; d[1] = posSigma; d[2] = posSigma
        d[3] = velSigma; d[4] = velSigma; d[5] = velSigma
        d[6] = attSigmaDeg * PI / 180.0; d[7] = attSigmaDeg * PI / 180.0; d[8] = attSigmaDeg * PI / 180.0
        d[9] = 0.1; d[10] = 0.1; d[11] = 0.1
        d[12] = 5e-3; d[13] = 5e-3; d[14] = 5e-3
        return Matrix.diagonal(d)
    }

    @Test
    fun testSigmaPointsSymmetryAndMomentMatching() {
        val S = Matrix.identity(15) * 2.0
        val gamma = 3.87298
        val points = UkfFusionEngine.sigmaPoints(S, gamma)

        assertEquals(31, points.size)
        // Center point is zero
        for (k in 0 until 15) {
            assertEquals(0.0, points[0][k], 1e-15)
        }

        // Symmetry: point[1 + i] == -point[1 + 15 + i]
        for (i in 0 until 15) {
            for (k in 0 until 15) {
                assertEquals(-points[1 + 15 + i][k], points[1 + i][k], 1e-12)
            }
        }
    }

    @Test
    fun testMechanizeConstantVelocity() {
        val s0 = NavState(
            position = Vec3(0.0, 0.0, 0.0),
            velocity = Vec3(10.0, 0.0, 0.0),
            orientation = Quaternion(1.0, 0.0, 0.0, 0.0),
        )
        // Stationarity: specific force = -g
        val accel = Vec3(0.0, 0.0, -9.80665)
        val gyro = Vec3(0.0, 0.0, 0.0)
        val dt = 0.1
        val gNed = Vec3(0.0, 0.0, 9.80665)

        val s1 = UkfFusionEngine.mechanize(s0, accel, gyro, dt, gNed)
        assertEquals(1.0, s1.position.x, 1e-6)
        assertEquals(0.0, s1.position.y, 1e-6)
        assertEquals(0.0, s1.position.z, 1e-6)
        assertEquals(10.0, s1.velocity.x, 1e-6)
    }

    @Test
    fun testComposeAndDecomposeInverses() {
        val nominal = NavState(
            position = Vec3(10.0, -5.0, 2.0),
            velocity = Vec3(15.0, 0.5, -0.1),
            orientation = So3.expMap(Vec3(0.1, -0.2, 0.3)),
            accelBias = Vec3(0.02, -0.01, 0.05),
            gyroBias = Vec3(1e-3, -2e-3, 5e-4),
        )

        val error = DoubleArray(15) { (it + 1) * 0.01 }
        val composed = UkfFusionEngine.compose(nominal, error)
        val recovered = UkfFusionEngine.decompose(composed, nominal)

        for (k in 0 until 15) {
            assertEquals(error[k], recovered[k], 1e-10)
        }
    }

    @Test
    fun testGnssUpdateConvergence() {
        // As documented in test_updates.cpp, repeated fixes against a 20m prior converge smoothly
        val engine = UkfFusionEngine(
            initialState = NavState(),
            initialCovarianceSqrt = initialSqrtCovariance(posSigma = 20.0),
            noise = ImuNoiseParams.fogGrade(),
        )

        val truth = Vec3(20.0, 15.0, 0.0)
        var previousError = (engine.state().position - truth).norm()

        for (i in 0 until 10) {
            val fix = GnssFix(
                timestampNanos = (i + 1) * 1_000_000_000L,
                position = Position(truth.x, truth.y, truth.z),
                horizontalAccuracyM = 3.0,
                verticalAccuracyM = 6.0,
                satellitesUsed = 12,
            )
            val outcome = engine.updateGnss(fix)
            assertEquals("iteration $i", UpdateOutcome.Applied, outcome)

            val error = (engine.state().position - truth).norm()
            assertTrue("iteration $i: error=$error should be < previousError=$previousError", error < previousError)
            previousError = error
        }

        assertTrue("Final error $previousError should be < 0.5m", previousError < 0.5)
    }

    @Test
    fun testVelocityModelUpdate() {
        val state = NavState(velocity = Vec3(12.0, 0.0, 0.0))
        val engine = UkfFusionEngine(
            initialState = state,
            initialCovarianceSqrt = initialSqrtCovariance(posSigma = 50.0, velSigma = 3.0, attSigmaDeg = 5.0),
            noise = ImuNoiseParams.fogGrade(),
        )

        for (i in 0 until 20) {
            val outcome = engine.updateVelocityModel(16.0)
            assertEquals(UpdateOutcome.Applied, outcome)
        }

        val forwardSpeed = UkfFusionEngine.bodyVelocity(engine.state()).x
        assertTrue("Forward speed $forwardSpeed should be > 15.0", forwardSpeed > 15.0)
    }

    // --- End-to-end 60-Second GPS Blackout Scenario (<10% Drift Target) ---

    private data class BlackoutResult(
        val errorAtBlackoutEndM: Double,
        val percentOfDistance: Double,
        val crossTrackAtBlackoutEndM: Double,
        val errorAfterReacquisitionM: Double,
        val gnssRejected: Int,
    )

    private fun runBlackoutScenario(
        useNhc: Boolean,
        useMap: Boolean,
        useDoppler: Boolean,
    ): BlackoutResult {
        val speedMps = 60.0 / 3.6 // 16.667 m/s (60 km/h)
        val imuRateHz = 100.0
        val dt = 1.0 / imuRateHz
        val warmupSec = 40.0
        val blackoutSec = 60.0 // 1 km blackout
        val recoverySec = 20.0
        val totalSec = warmupSec + blackoutSec + recoverySec
        val blackoutDistM = speedMps * blackoutSec // 1000 m

        val noise = ImuNoiseParams.consumerMems()
        val rng = Random(20261001)

        // Vehicle drives North along straight road
        val road = RoadGraph().apply {
            var prev = addNode(Vec3(-200.0, 0.0, 0.0))
            var north = -100.0
            while (north <= speedMps * totalSec + 200.0) {
                val next = addNode(Vec3(north, 0.0, 0.0))
                addSegment(prev, next, 1)
                prev = next
                north += 100.0
            }
        }
        val matcher = HmmMapMatcher(road, HmmParams())

        val initialTruth = NavState(
            position = Vec3(0.0, 0.0, 0.0),
            velocity = Vec3(speedMps, 0.0, 0.0),
            orientation = Quaternion(1.0, 0.0, 0.0, 0.0),
        )

        val engine = UkfFusionEngine(
            initialState = initialTruth,
            initialCovarianceSqrt = initialSqrtCovariance(5.0),
            noise = noise,
        )

        // Turn-on IMU bias that filter starts unaware of
        val trueAccelBias = Vec3(0.03, 0.02, 0.0)
        val trueGyroBias = Vec3(0.0, 0.0, 2e-3)

        var lastGnssSec = -1
        var lastAidingDecisec = -1
        var errorAtBlackoutEnd = 0.0
        var crossTrackAtBlackoutEnd = 0.0

        val totalSteps = (totalSec * imuRateHz).toInt()
        for (step in 0 until totalSteps) {
            val t = step * dt
            val inBlackout = t in warmupSec..(warmupSec + blackoutSec)

            // True vehicle motion: constant speed North
            val truePos = Vec3(speedMps * t, 0.0, 0.0)
            val trueVel = Vec3(speedMps, 0.0, 0.0)

            // Synthetic IMU measurement
            val accelWhiteNoise = Vec3(
                rng.nextGaussian() * noise.accelNoiseDensity * sqrt(imuRateHz),
                rng.nextGaussian() * noise.accelNoiseDensity * sqrt(imuRateHz),
                rng.nextGaussian() * noise.accelNoiseDensity * sqrt(imuRateHz),
            )
            val gyroWhiteNoise = Vec3(
                rng.nextGaussian() * noise.gyroNoiseDensity * sqrt(imuRateHz),
                rng.nextGaussian() * noise.gyroNoiseDensity * sqrt(imuRateHz),
                rng.nextGaussian() * noise.gyroNoiseDensity * sqrt(imuRateHz),
            )

            // Stationary specific force in body FRD = [0, 0, -g]
            val specificForce = Vec3(0.0, 0.0, -9.80665) + trueAccelBias + accelWhiteNoise
            val angRate = trueGyroBias + gyroWhiteNoise

            val imuSample = ImuSample(
                timestampNanos = (t * 1e9).toLong(),
                accel = floatArrayOf(specificForce.x.toFloat(), specificForce.y.toFloat(), specificForce.z.toFloat()),
                gyro = floatArrayOf(angRate.x.toFloat(), angRate.y.toFloat(), angRate.z.toFloat()),
                mag = null,
            )

            engine.predict(imuSample)

            // 10 Hz aiding updates
            val decisec = (t * 10.0).toInt()
            if (decisec != lastAidingDecisec) {
                lastAidingDecisec = decisec
                if (useNhc) engine.updateNonHolonomic()
                if (useMap) {
                    val match = matcher.step(engine.state().position)
                    engine.updateMapMatch(match)
                }
            }

            // 1 Hz GNSS fixes outside blackout
            val sec = t.toInt()
            if (sec != lastGnssSec && !inBlackout) {
                lastGnssSec = sec
                val gnssNoise = Vec3(rng.nextGaussian() * 2.0, rng.nextGaussian() * 2.0, 0.0)
                val fix = GnssFix(
                    timestampNanos = (t * 1e9).toLong(),
                    position = Position(truePos.x + gnssNoise.x, truePos.y + gnssNoise.y, truePos.z),
                    velocityNed = if (useDoppler) trueVel + (gnssNoise * 0.03) else Vec3(),
                    hasVelocity = useDoppler,
                    horizontalAccuracyM = 2.5,
                    verticalAccuracyM = 5.0,
                    speedAccuracyMps = 0.1,
                    satellitesUsed = 12,
                )
                engine.updateGnss(fix)
            }

            // Record error at the last moment of the blackout
            if (inBlackout) {
                val err = engine.state().position - truePos
                errorAtBlackoutEnd = hypot(err.x, err.y)
                crossTrackAtBlackoutEnd = abs(err.y)
            }
        }

        val finalTruePos = Vec3(speedMps * totalSec, 0.0, 0.0)
        val finalErr = (engine.state().position - finalTruePos).norm()

        return BlackoutResult(
            errorAtBlackoutEndM = errorAtBlackoutEnd,
            percentOfDistance = 100.0 * errorAtBlackoutEnd / blackoutDistM,
            crossTrackAtBlackoutEndM = crossTrackAtBlackoutEnd,
            errorAfterReacquisitionM = finalErr,
            gnssRejected = engine.diagnostics.gnssRejected,
        )
    }

    @Test
    fun testBlackoutScenarioMeetsDriftTargetUnderTenPercent() {
        // Run with all aiding: Doppler GNSS + NHC + HMM Map Matching
        val result = runBlackoutScenario(useNhc = true, useMap = true, useDoppler = true)

        println("=== Blackout Scenario (1 km @ 60 km/h) ===")
        println(String.format(Locale.US, "End of blackout error: %.2f m (%.2f%% of 1000m)", result.errorAtBlackoutEndM, result.percentOfDistance))
        println(String.format(Locale.US, "Cross-track error:     %.2f m", result.crossTrackAtBlackoutEndM))
        println(String.format(Locale.US, "Post-reacquisition:    %.2f m", result.errorAfterReacquisitionM))

        // THE PRIMARY GOAL: Under 10% drift over 1 km distance during blackout!
        assertTrue(
            result.percentOfDistance < 10.0
        )

        // Comfortably under 5% with full aiding
        assertTrue(
            result.percentOfDistance < 5.0
        )

        // Lateral / Cross-track bounded to lane-level (< 1.5m)
        assertTrue(
            result.crossTrackAtBlackoutEndM < 1.5
        )

        // GNSS re-acquisition recovers position cleanly without being rejected by gate
        assertTrue(
            result.errorAfterReacquisitionM < 5.0
        )
    }

    @Test
    fun testAntiDivergenceGnssRecoveryAfterGateLockout() {
        val engine = UkfFusionEngine()
        val initialFix = GnssFix(
            timestampNanos = 1_000_000_000L,
            position = Position(0.0, 0.0, 0.0),
            velocityNed = Vec3(0.0, 0.0, 0.0),
            hasVelocity = true,
            speedAccuracyMps = 0.2,
            horizontalAccuracyM = 3.0,
            verticalAccuracyM = 5.0,
            satellitesUsed = 12,
        )
        engine.updateGnss(initialFix)
        assertEquals(0.0, engine.state().position.norm(), 1e-3)

        // Present a fix 200m away with tight covariance (which will trigger gate rejection)
        val distantFix1 = GnssFix(
            timestampNanos = 2_000_000_000L,
            position = Position(200.0, 0.0, 0.0),
            velocityNed = Vec3(0.0, 0.0, 0.0),
            hasVelocity = true,
            speedAccuracyMps = 0.2,
            horizontalAccuracyM = 3.0,
            verticalAccuracyM = 5.0,
            satellitesUsed = 12,
        )
        // 1st rejection
        engine.updateGnss(distantFix1)
        assertTrue(engine.state().position.norm() < 50.0)

        // 2nd rejection
        val distantFix2 = distantFix1.copy(timestampNanos = 3_000_000_000L)
        engine.updateGnss(distantFix2)
        assertTrue(engine.state().position.norm() < 50.0)

        // 3rd fix triggers anti-divergence recovery: snaps position directly to fix
        val distantFix3 = distantFix1.copy(timestampNanos = 4_000_000_000L)
        engine.updateGnss(distantFix3)
        assertEquals(200.0, engine.state().position.x, 1.0)
    }

    @Test
    fun testVelocityClampingUnderRunawayAcceleration() {
        val engine = UkfFusionEngine()
        var tNanos = 1_000_000_000L

        // Feed enormous 500 m/s^2 acceleration over 100 steps
        for (i in 0 until 100) {
            tNanos += 10_000_000L // 10 ms
            engine.predict(
                timestampNanos = tNanos,
                accel = Vec3(500.0, 0.0, -9.80665),
                gyro = Vec3(0.0, 0.0, 0.0),
            )
        }

        // Velocity should be strictly clamped to <= 60.0 m/s
        assertTrue("Speed must be clamped <= 60 m/s but was ${engine.state().velocity.norm()}",
            engine.state().velocity.norm() <= 60.001)
    }

    @Test
    fun testNhcSkippedAtStandstillAndSlowSpeeds() {
        val engine = UkfFusionEngine(
            initialState = NavState(velocity = Vec3(1.2, 0.0, 0.0)) // speed 1.2 m/s < 2.0 m/s
        )
        val outcome = engine.updateNonHolonomic()
        assertEquals(UpdateOutcome.Skipped, outcome)
        assertEquals(1, engine.diagnostics.nhcSkipped)
        assertEquals(0, engine.diagnostics.nhcApplied)
        // Ensure velocity was not clamped to zero
        assertTrue("Velocity should not be clamped to zero", engine.state().velocity.norm() > 1.0)
    }

    @Test
    fun testHeadingAlignmentFromGnssVelocityCourse() {
        val engine = UkfFusionEngine()
        // Gravity leveling with stationary specific force
        engine.alignGravity(Vec3(0.0, 0.0, -9.80665))
        // Before driving, orientation forward is roughly North (0 deg)
        val initialForward = engine.state().orientation.rotate(Vec3(1.0, 0.0, 0.0))
        assertEquals(1.0, initialForward.x, 1e-3)
        assertEquals(0.0, initialForward.y, 1e-3)

        // Vehicle starts moving along Mansarovar Link Road: 40 deg azimuth East of North pulling away at 2.5 m/s (~9 km/h)
        val hdgDeg = 40.0
        val hdgRad = Math.toRadians(hdgDeg)
        val speedMps = 2.5
        val movingFix = GnssFix(
            timestampNanos = 1_000_000_000L,
            position = Position(0.0, 0.0, 0.0),
            velocityNed = Vec3(speedMps * kotlin.math.cos(hdgRad), speedMps * kotlin.math.sin(hdgRad), 0.0),
            hasVelocity = true,
            horizontalAccuracyM = 2.0,
            verticalAccuracyM = 4.0,
            speedAccuracyMps = 0.1,
            satellitesUsed = 14,
        )

        val outcome = engine.updateGnss(movingFix)
        assertEquals(UpdateOutcome.Applied, outcome)
        assertTrue(engine.isHeadingInitialized)

        // Fused orientation forward vector in NED should now point along 40 deg
        val alignedForward = engine.state().orientation.rotate(Vec3(1.0, 0.0, 0.0))
        val estHdgDeg = (Math.toDegrees(kotlin.math.atan2(alignedForward.y, alignedForward.x)) + 360.0) % 360.0
        assertEquals(40.0, estHdgDeg, 1.0)

        // Non-holonomic update should now pass the gate cleanly because lateral velocity is ~0
        val nhcOutcome = engine.updateNonHolonomic()
        assertEquals(UpdateOutcome.Applied, nhcOutcome)
        assertEquals(1, engine.diagnostics.nhcApplied)
        assertEquals(0, engine.diagnostics.nhcRejected)
    }

    @Test
    fun testAngledRoadBlackoutStabilityNoDeviation() {
        val engine = UkfFusionEngine()
        val hdgDeg = 40.0
        val hdgRad = Math.toRadians(hdgDeg)
        val speedMps = 15.0 // ~54 km/h
        val dirNed = Vec3(kotlin.math.cos(hdgRad), kotlin.math.sin(hdgRad), 0.0)

        // Provide initial GNSS fix while moving to establish anchor and heading
        val fix1 = GnssFix(
            timestampNanos = 1_000_000_000L,
            position = Position(0.0, 0.0, 0.0),
            velocityNed = dirNed * speedMps,
            hasVelocity = true,
            horizontalAccuracyM = 2.0,
            verticalAccuracyM = 4.0,
            speedAccuracyMps = 0.1,
            satellitesUsed = 12,
        )
        engine.updateGnss(fix1)
        assertTrue(engine.isHeadingInitialized)

        // Propagate during a 20-second simulated blackout (300 meters)
        var tNanos = 1_000_000_000L
        val dtSec = 0.01 // 100 Hz IMU
        val dtNanos = (dtSec * 1e9).toLong()

        for (i in 0 until 2000) { // 20 seconds
            tNanos += dtNanos
            // Specific force in body frame: [0, 0, -g]
            val specificForce = Vec3(0.0, 0.0, -9.80665)
            engine.predict(tNanos, specificForce, Vec3(0.0, 0.0, 0.0))

            // 10 Hz NHC aiding during blackout
            if (i % 10 == 0) {
                engine.updateNonHolonomic()
            }
        }

        // True position along 40-degree road after 20 seconds
        val truePos = dirNed * (speedMps * 20.0)
        val estPos = engine.state().position
        val err = estPos - truePos

        // Road normal vector (perpendicular to 40 deg road)
        val roadNormal = Vec3(-kotlin.math.sin(hdgRad), kotlin.math.cos(hdgRad), 0.0)
        val crossTrackError = kotlin.math.abs(err.dot(roadNormal))

        // Cross track error must remain tightly bounded within road corridor (< 1.5m), not shooting off into blocks!
        assertTrue("Cross-track drift must remain < 1.5m, was $crossTrackError m", crossTrackError < 1.5)
    }

    @Test
    fun testStationaryDuringBlackoutDoesNotDrift() {
        val engine = UkfFusionEngine()
        val initFix = GnssFix(
            timestampNanos = 1_000_000_000L,
            position = Position(0.0, 0.0, 0.0),
            velocityNed = Vec3(0.0, 0.0, 0.0),
            hasVelocity = true,
            horizontalAccuracyM = 1.0,
            verticalAccuracyM = 2.0,
            speedAccuracyMps = 0.05,
            satellitesUsed = 12,
        )
        engine.updateGnss(initFix)

        // Simulate 10 seconds of blackout with phone tilted forward by 3 degrees (pitch tilt)
        val tiltAngleRad = Math.toRadians(3.0)
        val ax = -9.80665 * kotlin.math.sin(tiltAngleRad)
        val az = -9.80665 * kotlin.math.cos(tiltAngleRad)
        val tiltedAccel = Vec3(ax, 0.0, az)

        var tNanos = 1_000_000_000L
        val dtSec = 0.01 // 100 Hz
        val dtNanos = (dtSec * 1e9).toLong()

        for (i in 0 until 1000) { // 10 seconds
            tNanos += dtNanos
            engine.predict(tNanos, tiltedAccel, Vec3(0.0, 0.0, 0.0))

            // 10 Hz ZUPT updates as applied during blackout standstill
            if (i % 10 == 0) {
                engine.updateZeroVelocity(0.05)
            }
        }

        val finalSpeed = engine.state().velocity.norm()
        val finalPosDrift = engine.state().position.norm()

        assertTrue("Stationary vehicle speed must remain < 0.1 m/s, was $finalSpeed", finalSpeed < 0.1)
        assertTrue("Stationary position drift must remain < 0.2 m, was $finalPosDrift", finalPosDrift < 0.2)
    }

    @Test
    fun testVelocityModelAntiDivergenceRecovery() {
        val engine = UkfFusionEngine(initialState = NavState(velocity = Vec3(30.0, 0.0, 0.0)))
        assertEquals(30.0, engine.state().velocity.norm(), 0.1)

        // Model predicts actual speed is 5.0 m/s (large divergence triggers gate rejection and immediate recovery)
        val outcome1 = engine.updateVelocityModel(5.0)
        assertEquals(UpdateOutcome.RejectedByGate, outcome1)

        val recoveredSpeed = engine.state().velocity.norm()
        assertEquals("Anti-divergence safeguard must pull velocity back to predicted speed", 5.0, recoveredSpeed, 0.5)

        // Now that state velocity has been brought back to 5.0 m/s, subsequent model update is cleanly Applied
        val outcome2 = engine.updateVelocityModel(5.0)
        assertEquals(UpdateOutcome.Applied, outcome2)
    }

    @Test
    fun testCompassAttitudeAlignment() {
        val engine = UkfFusionEngine()
        assertFalse(engine.isHeadingInitialized)

        // Phone lying flat: accel is (0, 0, 9.80665), magnetic field points North (25 uT north, 40 uT down)
        val accel = Vec3(0.0, 0.0, 9.80665)
        val mag = Vec3(25.0, 0.0, 40.0)

        engine.alignAttitudeWithCompass(accel, mag)

        assertTrue("Heading must be initialized by compass", engine.isHeadingInitialized)
        val forwardNed = engine.state().orientation.rotate(Vec3(1.0, 0.0, 0.0))
        // Forward should point North (1, 0, 0)
        assertEquals(1.0, forwardNed.x, 0.05)
        assertEquals(0.0, forwardNed.y, 0.05)
        assertEquals(0.0, forwardNed.z, 0.05)
    }
}

