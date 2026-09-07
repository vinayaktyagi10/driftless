package com.driftless.logging

import android.content.Context
import android.os.Build
import android.os.SystemClock
import com.driftless.frames.LocalTangentFrame
import com.driftless.fusion.FusedPosition
import com.driftless.fusion.GnssFix
import com.driftless.sensors.ImuFrame
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

/**
 * Writes a run to newline-delimited JSON so it can be replayed later.
 *
 * The point is not diagnostics — the on-screen readout already covers that —
 * it is that a road test is expensive and non-repeatable. A logged commute
 * becomes a fixture the fusion engine can be run against as many times as
 * needed, which matters more than the drive itself when there are four days
 * left and every subsequent test would otherwise need another trip.
 *
 * Records are handed to a background writer through a [Channel]. The sampler
 * threads must never block on file I/O: a stalled write would show up as
 * dropped IMU samples, corrupting the very measurement being recorded.
 */
class TrackLogger(
    private val context: Context,
    private val scope: CoroutineScope,
) {

    private var writerJob: Job? = null
    private var channel: Channel<String>? = null

    /** Records the writer could not keep up with. Non-zero means a lossy log. */
    val droppedRecords = AtomicLong(0)

    var currentFile: File? = null
        private set

    private var nextImuDeadlineNanos = 0L

    /**
     * Opens a new log file and starts the writer. Returns the file, or null if
     * storage was unavailable — a failed log must not take the run down with it,
     * since the live test is still worth something without a recording.
     */
    fun start(sensorDelay: Int, gnssIntervalMillis: Long): File? {
        if (writerJob?.isActive == true) return currentFile

        val dir = File(context.getExternalFilesDir(null) ?: context.filesDir, "tracks")
        if (!dir.exists() && !dir.mkdirs()) return null

        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val file = File(dir, "track-$stamp.jsonl")

        val ch = Channel<String>(CAPACITY)
        channel = ch
        currentFile = file
        nextImuDeadlineNanos = 0L

        writerJob = scope.launch(Dispatchers.IO) {
            file.bufferedWriter().use { out ->
                out.appendLine(metaRecord(sensorDelay, gnssIntervalMillis))
                var sinceFlush = 0
                for (line in ch) {
                    out.appendLine(line)
                    // Flushed periodically rather than per record: a run that
                    // ends by the app being killed should lose a fraction of a
                    // second, not the whole file, and flushing every line would
                    // put an fsync in the path of every IMU sample.
                    if (++sinceFlush >= FLUSH_EVERY) {
                        out.flush()
                        sinceFlush = 0
                    }
                }
                out.flush()
            }
        }
        return file
    }

    fun stop() {
        channel?.close()
        channel = null
        writerJob = null
    }

    /**
     * The one line that makes the monotonic timestamps in every other record
     * interpretable. `elapsedRealtime` is only meaningful relative to boot, so
     * without this pairing a log cannot be aligned to anything — not to a wall
     * clock, not to another device, not to a second run.
     */
    private fun metaRecord(sensorDelay: Int, gnssIntervalMillis: Long): String = buildString {
        append("""{"t":"meta","schema":1""")
        append(""","device":"${Build.MANUFACTURER} ${Build.MODEL}"""")
        append(""","sdk":${Build.VERSION.SDK_INT}""")
        append(""","wallMillis":${System.currentTimeMillis()}""")
        append(""","monoNanos":${SystemClock.elapsedRealtimeNanos()}""")
        append(""","sensorDelay":$sensorDelay""")
        append(""","gnssIntervalMillis":$gnssIntervalMillis""")
        append(""","frames":"body FRD, nav local-tangent NED","accuracies":"1 sigma"}""")
    }

    /** Records the navigation frame's anchor. Emitted once, at the first fix. */
    fun logAnchor(frame: LocalTangentFrame) = offer(
        buildString {
            append("""{"t":"anchor"""")
            append(""","lat":${frame.originLatDeg}""")
            append(""","lon":${frame.originLonDeg}""")
            append(""","alt":${frame.originAltM}}""")
        }
    )

    fun logFix(fix: GnssFix) = offer(
        buildString {
            append("""{"t":"gnss","ts":${fix.timestampNanos}""")
            append(""","n":${r(fix.position.north)},"e":${r(fix.position.east)}""")
            append(""","d":${r(fix.position.down)}""")
            append(""","sigmaH":${r(fix.horizontalAccuracyM)}""")
            append(""","sigmaV":${r(fix.verticalAccuracyM)}""")
            append(""","hasVel":${fix.hasVelocity}""")
            if (fix.hasVelocity) {
                append(""","vn":${r(fix.velocityNed.x)},"ve":${r(fix.velocityNed.y)}""")
                append(""","sigmaSpeed":${r(fix.speedAccuracyMps)}""")
            }
            append(""","sats":${fix.satellitesUsed}}""")
        }
    )

    /**
     * The engine's own output, timestamped by the caller rather than carried
     * on [FusedPosition] itself (which has none) — this is what a blackout
     * drift analysis diffs against the raw fixes in [logFix], so it needs the
     * same monotonic clock those use.
     */
    fun logFused(fused: FusedPosition, timestampNanos: Long) = offer(
        buildString {
            append("""{"t":"fused","ts":$timestampNanos""")
            append(""","lat":${fused.lat},"lon":${fused.lon}""")
            append(""","hdg":${f(fused.headingDegrees)}""")
            append(""","spd":${f(fused.speedMetersPerSec)}""")
            append(""","conf":${f(fused.confidence)}}""")
        }
    )

    /**
     * Marks a software-injected GNSS blackout window. Written alongside — not
     * instead of — the raw fixes in [logFix], so post-drive analysis can slice
     * out exactly the window where fixes were withheld from the engine while
     * still having ground truth to score drift against.
     */
    fun logBlackout(active: Boolean, timestampNanos: Long) = offer(
        buildString {
            append("""{"t":"blackout","ts":$timestampNanos,"active":$active}""")
        }
    )

    /**
     * IMU is decimated to [IMU_LOG_HZ] on the sample's own timestamp rather than
     * by counting samples, so the rate holds even if the sensor delay setting
     * changes mid-run. Every record carries its own true timestamp, so a replay
     * never has to assume the nominal rate.
     *
     * Full rate is deliberately not logged. At 500 Hz this file would grow by
     * roughly 6 MB a minute, and the cost is not the disk — it is formatting
     * 500 strings a second on the sampler's path, which risks manufacturing the
     * dropped samples the log exists to rule out. 50 Hz is five times the rate
     * the velocity model consumes and preserves everything a replay needs.
     */
    fun logImu(frame: ImuFrame) {
        val ts = frame.timestampNanos
        if (ts < nextImuDeadlineNanos) return

        // The deadline advances by exactly one period rather than being reset to
        // the sample that crossed it. Resetting looks equivalent and is not: the
        // logged sample always lands a little past the deadline, and carrying
        // that overshoot forward biases every subsequent interval, costing ~9%
        // of the nominal rate at a 500 Hz source. Re-basing after a long stall
        // stops the deadline chasing a burst of catch-up records.
        nextImuDeadlineNanos =
            if (nextImuDeadlineNanos == 0L || ts - nextImuDeadlineNanos > IMU_LOG_PERIOD_NANOS) {
                ts + IMU_LOG_PERIOD_NANOS
            } else {
                nextImuDeadlineNanos + IMU_LOG_PERIOD_NANOS
            }
        offer(
            buildString {
                append("""{"t":"imu","ts":${frame.timestampNanos}""")
                append(""","a":[${f(frame.accel[0])},${f(frame.accel[1])},${f(frame.accel[2])}]""")
                append(""","g":[${f(frame.gyro[0])},${f(frame.gyro[1])},${f(frame.gyro[2])}]""")
                frame.mag?.let {
                    append(""","m":[${f(it[0])},${f(it[1])},${f(it[2])}]""")
                }
                frame.gravity?.let {
                    append(""","gr":[${f(it[0])},${f(it[1])},${f(it[2])}]""")
                }
                append("}")
            }
        )
    }

    private fun offer(line: String) {
        val ch = channel ?: return
        if (ch.trySend(line).isFailure) droppedRecords.incrementAndGet()
    }

    /** Six decimals: ~0.1 mm in metres, far finer than any sensor here. */
    private fun r(v: Double) = String.format(Locale.US, "%.6f", v)

    private fun f(v: Float) = String.format(Locale.US, "%.6f", v)

    private companion object {
        const val IMU_LOG_HZ = 50
        const val IMU_LOG_PERIOD_NANOS = 1_000_000_000L / IMU_LOG_HZ

        /** ~10 s of headroom at the logged rate, so a slow write is ridden out. */
        const val CAPACITY = 512

        const val FLUSH_EVERY = 100
    }
}
