package com.driftless.fusion

import com.driftless.frames.LocalTangentFrame
import com.driftless.math.norm
import com.driftless.sensors.ImuSample
import org.junit.Assume
import org.junit.Test
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Replays a recorded NDJSON drive log through the fusion engine, so a filter change can be
 * scored against a real drive without driving again.
 *
 * Complements [SyntheticDriveTest]. The synthetic drive has exact ground truth but idealised
 * inputs -- noiseless IMU, a straight line, one fixed mount angle. This has none of that
 * cleanliness and all of the realism: genuine sensor noise, road vibration, a real 35-degree
 * dash mount, real GNSS accuracy, real turns, and the operator's own blackout button presses.
 * Truth here is GNSS, which is itself noisy, so absolute numbers mean less than the difference
 * between two configurations replayed over identical input.
 *
 * The log is not committed -- it is 3.7 MB of one drive, and the repo tracks code. Point the
 * test at one with -Ddriftless.replay.log=<path>; it skips when absent rather than failing, so
 * a checkout without the file still builds green.
 */
class LoggedDriveReplayTest {

    private val NL = System.lineSeparator()

    private data class Imu(val ts: Long, val ax: Double, val ay: Double, val az: Double, val gx: Double, val gy: Double, val gz: Double)
    private data class Gnss(
        val ts: Long, val n: Double, val e: Double, val d: Double,
        val sigmaH: Double, val sigmaV: Double,
        val hasVel: Boolean, val vn: Double, val ve: Double, val sigmaSpeed: Double, val sats: Int,
    )
    private data class Blackout(val ts: Long, val active: Boolean)

    private class Log(
        val imu: List<Imu>,
        val gnss: List<Gnss>,
        val blackouts: List<Blackout>,
        val anchorLat: Double,
        val anchorLon: Double,
        val anchorAlt: Double,
    )

    // ---- minimal flat-JSON extraction (no JSON lib on the unit-test classpath) ----

    private fun num(s: String, key: String): Double? {
        val i = s.indexOf("\"$key\":")
        if (i < 0) return null
        var j = i + key.length + 3
        while (j < s.length && s[j] == ' ') j++
        val start = j
        while (j < s.length && (s[j].isDigit() || s[j] == '-' || s[j] == '+' || s[j] == '.' || s[j] == 'e' || s[j] == 'E')) j++
        return s.substring(start, j).toDoubleOrNull()
    }

    private fun arr3(s: String, key: String): DoubleArray? {
        val i = s.indexOf("\"$key\":")
        if (i < 0) return null
        val open = s.indexOf('[', i)
        val close = s.indexOf(']', open)
        if (open < 0 || close < 0) return null
        val parts = s.substring(open + 1, close).split(',')
        if (parts.size < 3) return null
        return DoubleArray(3) { parts[it].trim().toDoubleOrNull() ?: return null }
    }

    private fun parse(file: File): Log {
        val imu = mutableListOf<Imu>()
        val gnss = mutableListOf<Gnss>()
        val blackouts = mutableListOf<Blackout>()
        var lat = 0.0; var lon = 0.0; var alt = 0.0

        file.forEachLine { raw ->
            val line = raw.trim()
            if (line.length < 8) return@forEachLine
            when {
                line.contains("\"t\": \"imu\"") || line.contains("\"t\":\"imu\"") -> {
                    val a = arr3(line, "a"); val g = arr3(line, "g"); val ts = num(line, "ts")
                    if (a != null && g != null && ts != null) {
                        imu += Imu(ts.toLong(), a[0], a[1], a[2], g[0], g[1], g[2])
                    }
                }
                line.contains("\"t\": \"gnss\"") || line.contains("\"t\":\"gnss\"") -> {
                    val ts = num(line, "ts") ?: return@forEachLine
                    gnss += Gnss(
                        ts = ts.toLong(),
                        n = num(line, "n") ?: 0.0,
                        e = num(line, "e") ?: 0.0,
                        d = num(line, "d") ?: 0.0,
                        sigmaH = num(line, "sigmaH") ?: 5.0,
                        sigmaV = num(line, "sigmaV") ?: 5.0,
                        hasVel = line.contains("\"hasVel\": true") || line.contains("\"hasVel\":true"),
                        vn = num(line, "vn") ?: 0.0,
                        ve = num(line, "ve") ?: 0.0,
                        sigmaSpeed = num(line, "sigmaSpeed") ?: 1.0,
                        sats = (num(line, "sats") ?: 0.0).toInt(),
                    )
                }
                line.contains("\"t\": \"blackout\"") || line.contains("\"t\":\"blackout\"") -> {
                    val ts = num(line, "ts") ?: return@forEachLine
                    blackouts += Blackout(
                        ts.toLong(),
                        line.contains("\"active\": true") || line.contains("\"active\":true"),
                    )
                }
                line.contains("\"t\": \"anchor\"") || line.contains("\"t\":\"anchor\"") -> {
                    lat = num(line, "lat") ?: 0.0
                    lon = num(line, "lon") ?: 0.0
                    alt = num(line, "alt") ?: 0.0
                }
            }
        }
        return Log(imu, gnss, blackouts, lat, lon, alt)
    }

    // ---- replay ----

    private data class Result(
        val label: String,
        val posErrNormal: List<Double>,
        val posErrBlackout: List<Double>,
        val speedRatio: List<Double>,
        val gnssApplied: Int,
        val gnssRejected: Int,
        val snaps: Int,
    ) {
        private fun List<Double>.med() = if (isEmpty()) Double.NaN else sorted()[size / 2]
        private fun List<Double>.p95() = if (isEmpty()) Double.NaN else sorted()[(size * 0.95).toInt().coerceAtMost(size - 1)]
        fun line(): String = String.format(
            Locale.US,
            "%-26s posErr p50=%6.2f p95=%7.2f m | blackout p50=%7.2f p95=%8.2f m | " +
                "speed ratio p50=%5.2fx | GNSS rej %3d/%3d = %3.0f%% | snap=%d",
            label,
            posErrNormal.med(), posErrNormal.p95(),
            posErrBlackout.med(), posErrBlackout.p95(),
            speedRatio.med(),
            gnssRejected, gnssApplied + gnssRejected,
            100.0 * gnssRejected / maxOf(1, gnssApplied + gnssRejected),
            snaps,
        )
    }

    /**
     * @param withholdGnssDuringBlackout replay the operator's real blackout presses, keeping the
     *   withheld fixes as truth. False replays the drive with GNSS never interrupted.
     */
    private fun replay(
        log: Log,
        label: String,
        nhcVerticalSigma: Double,
        withholdGnssDuringBlackout: Boolean = true,
        velocityAidingFromDoppler: Boolean = false,
        useGroundSpeed: Boolean = false,
    ): Result {
        val frame = LocalTangentFrame.anchoredAt(log.anchorLat, log.anchorLon, log.anchorAlt)
        val engine = UkfFusionEngine(
            frameProvider = { frame },
            config = UkfFusionEngine.Config(
                nonHolonomic = UkfFusionEngine.NonHolonomicParams(verticalSigmaMps = nhcVerticalSigma),
                velocityModel = UkfFusionEngine.VelocityModelParams(useGroundSpeed = useGroundSpeed),
            ),
        )

        engine.alignGravity(Vec3(log.imu[0].ax, log.imu[0].ay, log.imu[0].az))
        val first = log.gnss.first()
        engine.resetPosition(Vec3(first.n, first.e, first.d))

        val posErrNormal = mutableListOf<Double>()
        val posErrBlackout = mutableListOf<Double>()
        val speedRatio = mutableListOf<Double>()

        var gi = 0
        var bi = 0
        var blackoutActive = false
        var lastAiding = 0L
        var lastDopplerSpeed = 0.0

        for (s in log.imu) {
            while (bi < log.blackouts.size && log.blackouts[bi].ts <= s.ts) {
                blackoutActive = log.blackouts[bi].active
                bi++
            }

            engine.predict(
                ImuSample(
                    timestampNanos = s.ts,
                    accel = floatArrayOf(s.ax.toFloat(), s.ay.toFloat(), s.az.toFloat()),
                    gyro = floatArrayOf(s.gx.toFloat(), s.gy.toFloat(), s.gz.toFloat()),
                    mag = null,
                ),
            )

            while (gi < log.gnss.size && log.gnss[gi].ts <= s.ts) {
                val f = log.gnss[gi]
                gi++

                // Score BEFORE applying, so the fix is genuinely unseen at scoring time.
                val st = engine.state()
                val err = hypot(st.position.x - f.n, st.position.y - f.e)
                val truthSpeed = if (f.hasVel) hypot(f.vn, f.ve) else 0.0
                if (blackoutActive && withholdGnssDuringBlackout) posErrBlackout += err else posErrNormal += err
                if (truthSpeed > 2.0) speedRatio += st.velocity.norm() / truthSpeed

                if (f.hasVel) lastDopplerSpeed = truthSpeed

                if (!(blackoutActive && withholdGnssDuringBlackout)) {
                    engine.updateGnss(
                        GnssFix(
                            timestampNanos = f.ts,
                            position = Position(f.n, f.e, f.d),
                            velocityNed = Vec3(f.vn, f.ve, 0.0),
                            hasVelocity = f.hasVel,
                            speedAccuracyMps = f.sigmaSpeed,
                            horizontalAccuracyM = f.sigmaH,
                            verticalAccuracyM = f.sigmaV,
                            satellitesUsed = f.sats,
                        ),
                    )
                    if (!engine.isHeadingInitialized && f.hasVel && hypot(f.vn, f.ve) > 1.5) {
                        engine.alignHeading(atan2(f.ve, f.vn))
                    }
                }
            }

            if (s.ts - lastAiding >= 100_000_000L) {
                lastAiding = s.ts
                if (velocityAidingFromDoppler && lastDopplerSpeed > 0.05) {
                    engine.updateVelocityModel(lastDopplerSpeed)
                }
                engine.updateNonHolonomic()
            }
        }

        val d = engine.diagnostics
        return Result(label, posErrNormal, posErrBlackout, speedRatio, d.gnssApplied, d.gnssRejected, d.velocityModelSnaps)
    }

    @Test
    fun replayRecordedDrive() {
        // Not committed: a drive log is megabytes of one run, and the repo tracks code.
        // Supply one with -Ddriftless.replay.log=<path> or DRIFTLESS_REPLAY_LOG=<path>.
        val path = System.getProperty("driftless.replay.log")
            ?: System.getenv("DRIFTLESS_REPLAY_LOG")
        Assume.assumeTrue("no replay log configured; skipping", path != null)
        val file = File(path)
        Assume.assumeTrue("no replay log at $path; skipping", file.isFile)

        val log = parse(file)
        val span = (log.imu.last().ts - log.imu.first().ts) / 1e9
        println(
            "\n=== logged drive replay: %d imu (%.1f s), %d gnss, %d blackout markers ==="
                .format(log.imu.size, span, log.gnss.size, log.blackouts.size),
        )

        val before = replay(log, "vertSigma 0.3 (before)", 0.3)
        val after = replay(log, "vertSigma 10.0 (shipped)", 10.0)
        println(before.line())
        println(after.line())

        println(NL + "-- with velocity aiding during blackout (perfect-model stand-in) --")
        for (vs in listOf(0.3, 1.0, 3.0, 10.0, 30.0)) {
            println(replay(log, "vertSigma %.1f + vel aid".format(vs), vs, velocityAidingFromDoppler = true).line())
        }
        println(NL + "-- same, but h() = horizontal ground speed (mount-invariant) --")
        for (vs in listOf(0.3, 1.0, 3.0, 10.0, 30.0)) {
            println(replay(log, "vertSigma %.1f groundSpd".format(vs), vs, velocityAidingFromDoppler = true, useGroundSpeed = true).line())
        }

        println("\n-- no blackout withheld (whole drive GNSS-aided) --")
        println(replay(log, "vertSigma 0.3 (before)", 0.3, withholdGnssDuringBlackout = false).line())
        println(replay(log, "vertSigma 10.0 (shipped)", 10.0, withholdGnssDuringBlackout = false).line())
        println()
    }
}
