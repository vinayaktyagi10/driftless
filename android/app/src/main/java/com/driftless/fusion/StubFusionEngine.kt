package com.driftless.fusion

import com.driftless.frames.LocalTangentFrame
import com.driftless.sensors.ImuSample
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.min

/**
 * Build-order step 4: the scaffold [UkfFusionEngine] replaces, and the fallback
 * if that port is not ready in time.
 *
 * **This is not a filter.** There is no state covariance, no innovation gate and
 * no sigma points, and it does not integrate the IMU at all — [predict] only
 * reads the sample's timestamp, because at 50 Hz that is the cheapest honest
 * clock in the process. What it does instead is the smallest thing that still
 * demonstrates dead reckoning: latch the receiver's Doppler velocity at each
 * fix and extrapolate along it in between. Doppler is measured rather than
 * differenced, so this coasts on real data and not on integrated noise.
 *
 * Two reasons it is worth having rather than a passthrough that echoes the last
 * fix. It keeps the marker moving when the sky closes, which is the entire
 * premise of the app and the one thing a passthrough cannot show. And it makes
 * the seam load-bearing early: the UI collects [observePosition] and never
 * touches [com.driftless.sensors.GnssSampler] directly, so swapping in the real
 * engine becomes a constructor change and nothing else.
 *
 * Coasting runs *between* ordinary 1 Hz fixes too, not only during blackouts.
 * That is deliberate: extrapolating for one fix interval lands 0.82 m from the
 * next fix at the median and 2.16 m at p95 (n=1132), both comfortably inside
 * the receiver's own ~2.75 m median sigma_h. So the marker glides instead of
 * stepping once a second, and the correction when the next fix arrives is
 * smaller than the noise on the fix itself.
 *
 * ### What the coast is and is not allowed to do
 *
 * A fix always resets position — it is a measurement and there is nothing
 * better. What varies is whether it leaves the coast *armed*, and the three
 * cases are deliberately not symmetric, because a missing velocity and a
 * reported slow velocity are different kinds of evidence:
 *
 *  - **Reported and fast** (>= [COAST_MIN_SPEED_MPS]) — arm, and re-date the
 *    latch.
 *  - **Reported and slow** — the receiver is affirmatively telling us we have
 *    nearly stopped. Disarm at once; hold position.
 *  - **Not reported at all** — no evidence either way, and at speed it happens
 *    on 1.56% of fixes for runs of at most 3. Keep the latch but let it age,
 *    and stop believing it after [LATCH_MAX_AGE_NANOS].
 *
 * Collapsing the last two into "no usable velocity, so disarm" was the first
 * version of this class, and it is wrong twice over: it throws away a good
 * velocity because of one dropout fix at 54 km/h, and it leaves the age check
 * unreachable, since every fix would either re-date the latch or clear it.
 *
 * Every threshold below is measured against `track-20260903-115325-FULL.jsonl`
 * — 1320 fixes over 22.0 min of Jaipur traffic — rather than guessed, and the
 * figure is quoted at each constant so it can be argued with.
 *
 * @param frameProvider the local-tangent anchor, read live rather than copied,
 *   because [com.driftless.sensors.GnssSampler] does not create it until the
 *   first fix arrives and this engine is constructed before that.
 */
class StubFusionEngine(
    private val frameProvider: () -> LocalTangentFrame?,
) : PositionFusionEngine {

    private val lock = Any()

    private val positions = MutableSharedFlow<FusedPosition>(
        // Replay so a collector that subscribes late -- the map, after a
        // lifecycle restart -- renders immediately instead of waiting for the
        // next sample.
        replay = 1,
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    /** Last GNSS position, navigation frame. Null until the first fix. */
    private var anchorPosition: Position? = null
    private var anchorNanos = 0L

    /**
     * Velocity we are entitled to coast along, or null if we are not.
     *
     * Null is the normal state at a standstill and is not a fault -- see
     * [COAST_MIN_SPEED_MPS] and [updateGnss].
     */
    private var latchedVelocity: Vec3? = null
    private var latchedNanos = 0L

    /** Held so a momentary loss of velocity does not spin the marker. */
    private var lastHeadingDegrees = 0f

    private var emittedNanos = 0L

    /**
     * Advances the coast clock and republishes the extrapolated position.
     *
     * The sample's contents are ignored. That is the difference between this
     * and a real dead-reckoning engine, and it is called out here because a
     * method named `predict` that quietly never reads its accelerometer is
     * exactly the kind of thing that gets mistaken for working inertial
     * navigation.
     */
    override fun predict(sample: ImuSample) {
        val fused = synchronized(lock) { extrapolateTo(sample.timestampNanos) }
        if (fused != null) positions.tryEmit(fused)
    }

    /**
     * Resets position to the fix, then re-evaluates whether the coast stays
     * armed. See the class comment for why the three cases differ.
     */
    override fun updateGnss(fix: GnssFix): UpdateOutcome = synchronized(lock) {
        anchorPosition = fix.position
        anchorNanos = fix.timestampNanos

        val velocity = fix.velocityNed
        val speed = hypot(velocity.x, velocity.y)

        when {
            fix.hasVelocity && speed >= COAST_MIN_SPEED_MPS -> {
                latchedVelocity = velocity
                latchedNanos = fix.timestampNanos
                lastHeadingDegrees = headingOf(velocity)
            }

            // Rule 1. A reported speed below the floor is the receiver saying
            // we have stopped, which is stronger evidence than silence. Drop
            // the latch now rather than coasting a pre-braking velocity into a
            // stop -- the 26 s traffic halt in the log would have earned 26 m
            // of travel that never happened.
            fix.hasVelocity -> latchedVelocity = null

            // Rule 2's other half: no velocity reported is not evidence of a
            // stop, so the latch survives and ages instead. coastVelocity()
            // decides when it has aged out.
            else -> Unit
        }

        val fused = extrapolateTo(fix.timestampNanos)
        if (fused != null) positions.tryEmit(fused)
        UpdateOutcome.Applied
    }

    /**
     * Always [UpdateOutcome.Skipped]. Map matching is Role 02's, and reporting
     * `Applied` here would let step 5 wire up a road-snapping path that
     * silently does nothing.
     */
    override fun updateMapMatch(match: MapMatchResult): UpdateOutcome =
        UpdateOutcome.Skipped

    override fun state(): NavState = synchronized(lock) {
        val now = maxOf(anchorNanos, emittedNanos)
        val position = coastedPosition(now) ?: Position()
        NavState(
            position = Vec3(position.north, position.east, position.down),
            velocity = coastVelocity() ?: Vec3(),
        )
    }

    override fun observePosition(): Flow<FusedPosition> = positions.asSharedFlow()

    // -- internals ---------------------------------------------------------

    /**
     * Where the coast has carried us by [nowNanos], or null before the first
     * fix -- there is no origin to coast from and nothing to draw.
     *
     * Rules 1 and 2 are both enforced by [coastVelocity] returning null, which
     * collapses this to "hold the last measured fix".
     */
    private fun coastedPosition(nowNanos: Long): Position? {
        val anchor = anchorPosition ?: return null
        val velocity = coastVelocity() ?: return anchor
        val seconds = coastSeconds(nowNanos)
        return Position(
            north = anchor.north + velocity.x * seconds,
            east = anchor.east + velocity.y * seconds,
            // The stub does not model vertical rate: the receiver does not
            // report it, and integrating it out of the IMU is the real
            // engine's job.
            down = anchor.down,
        )
    }

    /**
     * The latched velocity if it is still fresh enough to believe, else null.
     *
     * Rule 2, and the reason it exists: rule 1's floor catches a blackout that
     * begins at a standstill, but not one that begins two seconds after the
     * driver braked. The latch is aged against the *fix* that opened the
     * blackout rather than against now, so an expiry here means "we had already
     * lost Doppler before we lost the sky" -- not "this coast has run long",
     * which is [COAST_MAX_SECONDS]'s job.
     */
    private fun coastVelocity(): Vec3? {
        val velocity = latchedVelocity ?: return null
        val age = anchorNanos - latchedNanos
        return if (age <= LATCH_MAX_AGE_NANOS) velocity else null
    }

    private fun coastSeconds(nowNanos: Long): Double =
        min(secondsSinceFix(nowNanos), COAST_MAX_SECONDS)

    private fun secondsSinceFix(nowNanos: Long): Double =
        (nowNanos - anchorNanos).coerceAtLeast(0L) / NANOS_PER_SECOND

    private fun extrapolateTo(nowNanos: Long): FusedPosition? {
        val isFix = nowNanos == anchorNanos
        if (!isFix && nowNanos - emittedNanos < EMIT_INTERVAL_NANOS) return null

        val frame = frameProvider() ?: return null
        val position = coastedPosition(nowNanos) ?: return null
        emittedNanos = nowNanos

        val velocity = coastVelocity()
        val geodetic = frame.toGeodetic(position)
        return FusedPosition(
            lat = geodetic.latDeg,
            lon = geodetic.lonDeg,
            headingDegrees = lastHeadingDegrees,
            speedMetersPerSec =
                if (velocity == null) 0f else hypot(velocity.x, velocity.y).toFloat(),
            confidence = confidenceAt(nowNanos),
        )
    }

    /**
     * 1.0 while the position is no worse than the fix it came from, then a
     * linear ramp to 0.
     *
     * The knee is set where the two error curves cross rather than by taste: at
     * a 2 s horizon the coast lands 4.80 m from truth at p95, and the
     * receiver's own sigma_h is 4.83 m at p95 over the same run. Past that the
     * coast is the weaker source and should say so -- by 10 s it is at 58.6 m
     * p95, worse than the drift this app exists to remove.
     *
     * Deliberately driven by fix age alone, whether we are coasting or holding.
     * Ten seconds after the last fix we are guessing either way; a stationary
     * marker is not more trustworthy for being stationary, it just fails less
     * visibly.
     */
    private fun confidenceAt(nowNanos: Long): Float {
        if (anchorPosition == null) return 0f
        val elapsed = secondsSinceFix(nowNanos)
        if (elapsed <= COAST_CONFIDENT_SECONDS) return 1f
        val span = COAST_MAX_SECONDS - COAST_CONFIDENT_SECONDS
        return (1.0 - (elapsed - COAST_CONFIDENT_SECONDS) / span)
            .coerceIn(0.0, 1.0)
            .toFloat()
    }

    /** NED velocity to a compass bearing, degrees clockwise from north. */
    private fun headingOf(velocity: Vec3): Float {
        val degrees = Math.toDegrees(atan2(velocity.y, velocity.x))
        return ((degrees + 360.0) % 360.0).toFloat()
    }

    companion object {
        private const val NANOS_PER_SECOND = 1_000_000_000.0

        /**
         * Rule 1. Below this a *reported* velocity disarms the coast instead of
         * arming it.
         *
         * Set above the sampler's own 0.5 m/s bearing floor on purpose, so the
         * coast is strictly more cautious than the fix stream feeding it.
         * Between 0.5 and 1 m/s only 21% of fixes carry a usable velocity, and
         * a position-differenced speed estimate over that band disagreed with
         * itself by 2.5 percentage points depending on window width -- it is
         * noise.
         */
        const val COAST_MIN_SPEED_MPS = 1.0

        /**
         * Rule 2. How stale the latch may be at the moment the fixes stop.
         *
         * At a 2 s gap a vehicle doing 2 m/s or more has actually come to a
         * stop 2.0% of the time (n=1051), and the velocity itself has moved by
         * 2.93 m/s at p95; both roughly double by 5 s. Since fixes arrive at
         * 1 Hz this costs nothing in the normal case and fires exactly when it
         * should -- we lost Doppler for two or more fixes *before* losing the
         * sky. The longest such run observed while moving was 3 fixes, so 2 s
         * survives the common dropout and rejects the worst one seen.
         */
        const val LATCH_MAX_AGE_NANOS = 2_000_000_000L

        /** See [confidenceAt]: where coast error overtakes GNSS noise. */
        const val COAST_CONFIDENT_SECONDS = 2.0

        /**
         * Hard stop on extrapolation. Past this the marker holds position
         * rather than free-running, because a coast that keeps gliding
         * confidently through a 30 s tunnel is worse than one that visibly
         * gives up: p95 error is already 58.6 m at 10 s and grows faster than
         * linearly beyond it.
         */
        const val COAST_MAX_SECONDS = 10.0

        /**
         * Republish at 10 Hz, not the IMU's 50. Smooth well past what the eye
         * or the renderer's 2 m polyline thinning can resolve, at a fifth of
         * the wakeups.
         */
        const val EMIT_INTERVAL_NANOS = 100_000_000L
    }
}
