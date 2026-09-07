package com.driftless.fusion

import com.driftless.frames.LocalTangentFrame
import com.driftless.math.norm
import com.driftless.sensors.ImuSample
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale
import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Closed-loop replay of a synthetic drive, used to test the fusion filter without a car.
 *
 * Why synthetic rather than a recorded log: the recorded log can only be scored against GNSS,
 * which is itself the noisy thing under test, and it cannot answer "what would have happened
 * with a different filter". Here the speed at every instant is known exactly, so a speed error
 * is unambiguous.
 *
 * The drive is a 1.25 km straight run due north with a speed profile that accelerates, cruises,
 * slows, cruises, accelerates again and stops -- ordinary traffic behaviour, and specifically
 * varied so a filter cannot look correct by accidentally holding one constant speed.
 *
 * The phone is mounted pitched 35 degrees nose-up, matching the real rig: at 35 degrees the
 * predicted stationary reading is accel.x = g*sin(35) = +5.63, accel.z = -g*cos(35) = -8.03,
 * against [+5.596, -0.025, -8.053] measured on the actual dash mount on 2026-09-07.
 *
 * The filter is then handed a deliberately WRONG initial attitude -- 32 degrees, a 3 degree
 * error, matching the ~3.8 degrees inferred from bias_a on that drive. That error is the whole
 * point: it projects g*sin(3 deg) = 0.51 m/s^2 of gravity onto the forward axis, which the
 * filter integrates as real acceleration. Over a 20 s GNSS blackout that is ~10 m/s of invented
 * speed, which is exactly the runaway seen on the road.
 */
class SyntheticDriveTest {

    // ---- drive definition -------------------------------------------------

    /** (duration s, start speed m/s, end speed m/s). Sums to 95 s and ~1250 m. */
    private val segments = listOf(
        Triple(10.0, 0.0, 16.0),
        Triple(20.0, 16.0, 16.0),
        Triple(10.0, 16.0, 8.0),
        Triple(15.0, 8.0, 8.0),
        Triple(10.0, 8.0, 19.0),
        Triple(20.0, 19.0, 19.0),
        Triple(10.0, 19.0, 0.0),
    )

    private val durationSec = segments.sumOf { it.first }

    /** Truth speed (m/s) and forward acceleration (m/s^2) at time t. */
    private fun truthAt(t: Double): Pair<Double, Double> {
        var start = 0.0
        for ((dur, v0, v1) in segments) {
            if (t < start + dur || start + dur >= durationSec && t >= start) {
                val frac = ((t - start) / dur).coerceIn(0.0, 1.0)
                return Pair(v0 + (v1 - v0) * frac, (v1 - v0) / dur)
            }
            start += dur
        }
        return Pair(0.0, 0.0)
    }

    private fun truthDistance(t: Double): Double {
        var start = 0.0
        var dist = 0.0
        for ((dur, v0, v1) in segments) {
            if (t <= start) break
            val span = minOf(t - start, dur)
            val vEnd = v0 + (v1 - v0) * (span / dur)
            dist += (v0 + vEnd) * 0.5 * span
            start += dur
        }
        return dist
    }

    // ---- IMU synthesis ----------------------------------------------------

    private val g = 9.80665

    /**
     * Specific force in body FRD for a vehicle accelerating forward along North with the phone
     * pitched [pitch] nose-up. The accelerometer measures f = a - g, so a level stationary phone
     * reads (0, 0, -9.81) -- which is what the real device reports.
     */
    private fun bodyAccel(forwardAccel: Double, pitch: Double): DoubleArray {
        val fNorth = forwardAccel
        val fDown = -g
        return doubleArrayOf(
            fNorth * cos(pitch) - fDown * sin(pitch),
            0.0,
            fNorth * sin(pitch) + fDown * cos(pitch),
        )
    }

    // ---- harness ----------------------------------------------------------

    private data class Sample(val tSec: Double, val truthSpeed: Double, val filterSpeed: Double, val posErrorM: Double)

    private data class Report(
        val samples: List<Sample>,
        val gnssApplied: Int,
        val gnssRejected: Int,
        val snaps: Int,
        val blackoutFinalPosErrorM: Double,
        val blackoutMaxSpeedErrorMps: Double,
    ) {
        val medianSpeedError: Double
            get() = samples.map { abs(it.filterSpeed - it.truthSpeed) }.sorted()[samples.size / 2]
        val maxSpeedError: Double
            get() = samples.maxOf { abs(it.filterSpeed - it.truthSpeed) }
        val medianSpeedRatio: Double
            get() {
                val moving = samples.filter { it.truthSpeed > 2.0 }
                val r = moving.map { it.filterSpeed / it.truthSpeed }.sorted()
                return r[r.size / 2]
            }
    }

    /**
     * @param blackoutStart  seconds at which GNSS stops being delivered to the engine
     * @param blackoutEnd    seconds at which it resumes
     * @param attitudeErrorDeg  error injected into the filter's initial gravity alignment
     * @param velocityAiding whether to feed a velocity-model measurement at 10 Hz
     * @param velocitySigma  noise added to that measurement, m/s (0 = a perfect model)
     */
    private fun run(
        blackoutStart: Double,
        blackoutEnd: Double,
        attitudeErrorDeg: Double,
        velocityAiding: Boolean,
        velocitySigma: Double = 0.0,
        mountPitchDeg: Double = 35.0,
        nhcVerticalSigma: Double = UkfFusionEngine.NonHolonomicParams().verticalSigmaMps,
        snapDivergenceMps: Double = UkfFusionEngine.VelocityModelParams().snapDivergenceMps,
        seed: Long = 42L,
    ): Report {
        val mountPitchRad = Math.toRadians(mountPitchDeg)
        val rng = Random(seed)
        val frame = LocalTangentFrame.anchoredAt(26.8521692, 75.5623364, 353.6)
        val engine = UkfFusionEngine(
            frameProvider = { frame },
            config = UkfFusionEngine.Config(
                nonHolonomic = UkfFusionEngine.NonHolonomicParams(verticalSigmaMps = nhcVerticalSigma),
                velocityModel = UkfFusionEngine.VelocityModelParams(snapDivergenceMps = snapDivergenceMps),
            ),
        )

        // Align gravity against a deliberately wrong pitch, injecting a known attitude error.
        val alignPitch = mountPitchRad - Math.toRadians(attitudeErrorDeg)
        val alignAccel = bodyAccel(0.0, alignPitch)
        engine.alignGravity(Vec3(alignAccel[0], alignAccel[1], alignAccel[2]))
        engine.alignHeading(0.0) // travelling due north
        engine.resetPosition(Vec3(0.0, 0.0, 0.0))

        val imuHz = 100.0
        val imuDt = 1.0 / imuHz
        val t0 = 1_000_000_000L
        var nextGnss = 1.0
        var nextAiding = 0.1

        val samples = mutableListOf<Sample>()
        var blackoutFinalPosError = 0.0
        var blackoutMaxSpeedError = 0.0

        var t = 0.0
        while (t < durationSec) {
            val (vTruth, aTruth) = truthAt(t)
            val acc = bodyAccel(aTruth, mountPitchRad)
            val tsNanos = t0 + (t * 1e9).toLong()

            engine.predict(
                ImuSample(
                    timestampNanos = tsNanos,
                    accel = floatArrayOf(acc[0].toFloat(), acc[1].toFloat(), acc[2].toFloat()),
                    gyro = floatArrayOf(0f, 0f, 0f),
                    mag = null,
                ),
            )

            if (t >= nextGnss) {
                nextGnss += 1.0
                val inBlackout = t in blackoutStart..blackoutEnd
                if (!inBlackout) {
                    engine.updateGnss(
                        GnssFix(
                            timestampNanos = tsNanos,
                            position = Position(north = truthDistance(t), east = 0.0, down = 0.0),
                            velocityNed = Vec3(vTruth, 0.0, 0.0),
                            hasVelocity = true,
                            speedAccuracyMps = 0.2,
                            horizontalAccuracyM = 1.5,
                            verticalAccuracyM = 2.0,
                            satellitesUsed = 20,
                        ),
                    )
                }
            }

            if (t >= nextAiding) {
                nextAiding += 0.1
                if (velocityAiding) {
                    val noisy = vTruth + rng.nextGaussian() * velocitySigma
                    if (noisy <= 0.05) engine.updateZeroVelocity(0.05) else engine.updateVelocityModel(noisy)
                }
                engine.updateNonHolonomic()

                val st = engine.state()
                val filterSpeed = st.velocity.norm()
                val posErr = abs(st.position.x - truthDistance(t))
                samples += Sample(t, vTruth, filterSpeed, posErr)

                if (t in blackoutStart..blackoutEnd) {
                    blackoutMaxSpeedError = maxOf(blackoutMaxSpeedError, abs(filterSpeed - vTruth))
                    blackoutFinalPosError = posErr
                }
            }
            t += imuDt
        }

        val d = engine.diagnostics
        return Report(
            samples = samples,
            gnssApplied = d.gnssApplied,
            gnssRejected = d.gnssRejected,
            snaps = d.velocityModelSnaps,
            blackoutFinalPosErrorM = blackoutFinalPosError,
            blackoutMaxSpeedErrorMps = blackoutMaxSpeedError,
        )
    }

    private fun printReport(label: String, r: Report) {
        println(
            String.format(
                Locale.US,
                "%-34s speed err p50=%6.2f max=%6.2f m/s | ratio p50=%5.2fx | " +
                    "blackout dPos=%8.1f m dV=%6.2f m/s | GNSS %d/%d rej | snap=%d",
                label,
                r.medianSpeedError,
                r.maxSpeedError,
                r.medianSpeedRatio,
                r.blackoutFinalPosErrorM,
                r.blackoutMaxSpeedErrorMps,
                r.gnssRejected,
                r.gnssApplied + r.gnssRejected,
                r.snaps,
            ),
        )
    }

    @Test
    fun syntheticDriveScenarios() {
        println("\n=== synthetic drive: %.0f s, %.0f m, mount 35 deg, blackout 45-65 s ==="
            .format(durationSec, truthDistance(durationSec)))

        val clean = run(45.0, 65.0, attitudeErrorDeg = 0.0, velocityAiding = false)
        printReport("35deg mount, perfect att, no aid", clean)

        println("--- mount pitch sweep, perfect attitude, velocity aiding on ---")
        for (deg in listOf(0.0, 5.0, 10.0, 20.0, 35.0)) {
            printReport(
                "  mount %4.1f deg".format(deg),
                run(45.0, 65.0, attitudeErrorDeg = 0.0, velocityAiding = true, mountPitchDeg = deg),
            )
        }
        println("--- 35 deg mount: relax NHC vertical constraint (config only) ---")
        for (sig in listOf(0.3, 1.0, 3.0, 10.0, 100.0)) {
            printReport(
                "  vertSigma %6.1f".format(sig),
                run(45.0, 65.0, 0.0, velocityAiding = true, mountPitchDeg = 35.0, nhcVerticalSigma = sig),
            )
        }
        println("--- noisy model (sigma 2.2): snap threshold, 12 seeds, median [min-max] ---")
        for (snap in listOf(3.0, 4.4, 6.6, 8.8, 1e9)) {
            val runs = (1L..12L).map {
                run(45.0, 65.0, 3.0, velocityAiding = true, velocitySigma = 2.2,
                    snapDivergenceMps = snap, seed = it)
            }
            val d = runs.map { it.blackoutFinalPosErrorM }.sorted()
            val rej = runs.map { it.gnssRejected.toDouble() / (it.gnssApplied + it.gnssRejected) }.sorted()
            println(
                String.format(
                    Locale.US,
                    "  snapDiv %8.1f   blackout dPos median=%7.1f m  [%6.1f - %7.1f]   rej median=%4.0f%%",
                    snap, d[d.size / 2], d.first(), d.last(), rej[rej.size / 2] * 100,
                ),
            )
        }
        run {
            val runs = (1L..12L).map { run(45.0, 65.0, 3.0, velocityAiding = false, seed = it) }
            val d = runs.map { it.blackoutFinalPosErrorM }.sorted()
            println(String.format(Locale.US, "  %-8s          blackout dPos median=%7.1f m  [%6.1f - %7.1f]",
                "no aiding", d[d.size / 2], d.first(), d.last()))
        }
        println("--- 35 deg mount, attitude-error scenarios ---")

        val tilted = run(45.0, 65.0, attitudeErrorDeg = 3.0, velocityAiding = false)
        printReport("3 deg attitude error, no aiding", tilted)

        val aided = run(45.0, 65.0, attitudeErrorDeg = 3.0, velocityAiding = true)
        printReport("3 deg error + perfect model", aided)

        val noisy = run(45.0, 65.0, attitudeErrorDeg = 3.0, velocityAiding = true, velocitySigma = 2.2)
        printReport("3 deg error + noisy model (2.2)", noisy)

        println()

        // Regression guard on the mount-geometry fix. At the real 35 degree mount, the shipped
        // NonHolonomicParams must keep a 20 s blackout inside 25 m and reject under a fifth of
        // GNSS fixes. Measured 144.8 m and 71% when verticalSigmaMps was 0.3, which is what the
        // 2026-09-07 road drive did (74% rejected, track visibly zig-zagging as the filter
        // re-anchored every third fix). If this fails, someone has tightened verticalSigmaMps
        // back up without adding mount-rotation compensation.
        val shipped = run(45.0, 65.0, attitudeErrorDeg = 0.0, velocityAiding = true, mountPitchDeg = 35.0)
        assertTrue(
            "35 deg mount blackout drift ${shipped.blackoutFinalPosErrorM} m should stay under 25 m",
            shipped.blackoutFinalPosErrorM < 25.0,
        )
        assertTrue(
            "35 deg mount GNSS rejection ${shipped.gnssRejected}/${shipped.gnssApplied + shipped.gnssRejected} should stay under 20%",
            shipped.gnssRejected < 0.2 * (shipped.gnssApplied + shipped.gnssRejected),
        )

        // The attitude error must actually produce a worse blackout than perfect attitude,
        // otherwise the scenario is not exercising the mechanism under test at all.
        assertTrue(
            "3 deg attitude error should degrade blackout dead reckoning vs perfect attitude " +
                "(tilted=${tilted.blackoutFinalPosErrorM} clean=${clean.blackoutFinalPosErrorM})",
            tilted.blackoutFinalPosErrorM > clean.blackoutFinalPosErrorM,
        )

        // Velocity aiding must contain the runaway the attitude error causes.
        assertTrue(
            "velocity aiding should cut blackout position error vs unaided " +
                "(aided=${aided.blackoutFinalPosErrorM} unaided=${tilted.blackoutFinalPosErrorM})",
            aided.blackoutFinalPosErrorM < tilted.blackoutFinalPosErrorM,
        )
    }
}
