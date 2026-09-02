package com.driftless.fusion

import com.driftless.sensors.ImuSample
import kotlinx.coroutines.flow.Flow

/**
 * The Role 01 <-> Role 02 boundary.
 *
 * This is a direct mirror of `edge-engine/include/driftless/types.h`. The C++
 * engine is the reference implementation and the Kotlin engine is a port of it
 * ("not the other way round", per `ukf_fusion_engine.h`), so when the two
 * disagree, the header wins and this file is what changes.
 *
 * Frame conventions, which everything below assumes:
 *
 *   Navigation frame : local-tangent NED (x North, y East, z Down), origin
 *                      anchored at the first GNSS fix. Flat-earth.
 *   Body frame       : FRD (x Forward, y Right, z Down).
 *   Attitude         : Hamilton unit quaternion, body -> nav.
 *   Gravity          : g_nav = [0, 0, +9.80665], so a level stationary
 *                      accelerometer reads [0, 0, -9.80665].
 *
 * SensorManager does NOT deliver any of this. It delivers an ENU-ish body frame
 * (x right, y forward, z up) and a 68%-radius accuracy. Converting both is Role
 * 01's job and happens in the samplers, before anything reaches this file.
 */

/** Metres from the local-tangent origin. */
data class Position(
    val north: Double = 0.0,
    val east: Double = 0.0,
    val down: Double = 0.0,
)

/** Hamilton unit quaternion, body -> nav. */
data class Quaternion(
    val w: Double = 1.0,
    val x: Double = 0.0,
    val y: Double = 0.0,
    val z: Double = 0.0,
)

/** Three components, interpretation depends on the field that holds it. */
data class Vec3(
    val x: Double = 0.0,
    val y: Double = 0.0,
    val z: Double = 0.0,
)

/**
 * A GNSS observation, already projected into the local-tangent frame.
 *
 * Accuracies are **1-sigma**. Android's [android.location.Location.getAccuracy]
 * is a 68% radius, which is 1.5136 sigma for a 2D Rayleigh error — the sampler
 * divides it out. Getting this wrong tightens the filter's NIS gate until it
 * rejects honest fixes and the track free-runs on the IMU alone, which looks
 * exactly like the drift we are supposed to be eliminating.
 */
data class GnssFix(
    val timestampNanos: Long,
    val position: Position,
    /**
     * Doppler-derived velocity is genuinely independent of position and much
     * better than differencing fixes, but not every receiver reports it and it
     * is meaningless at standstill — hence opt-in per fix via [hasVelocity].
     */
    val velocityNed: Vec3 = Vec3(),
    val hasVelocity: Boolean = false,
    val speedAccuracyMps: Double = 0.0,
    val horizontalAccuracyM: Double = 0.0,
    val verticalAccuracyM: Double = 0.0,
    /**
     * Surfaced so fusion can see degradation coming instead of only reacting to
     * a full outage: a 4-satellite fix in an urban canyon deserves a very
     * different R than a 14-satellite one in the open.
     */
    val satellitesUsed: Int = 0,
)

/**
 * The nominal navigation state. This is integrated directly and is *not* what
 * the UKF covariance is over — see `ukf_fusion_engine.h` for why the error
 * state is separate.
 */
data class NavState(
    val position: Vec3 = Vec3(),
    val velocity: Vec3 = Vec3(),
    val orientation: Quaternion = Quaternion(),
    val accelBias: Vec3 = Vec3(),
    val gyroBias: Vec3 = Vec3(),
)

/** Mirror of `MapMatchResult` in `hmm_map_matcher.h`. */
data class MapMatchResult(
    val matched: Boolean = false,
    val segmentId: Int = -1,
    val wayId: Int = -1,
    val snappedPosition: Vec3 = Vec3(),
    /** Unit vector along the matched road, horizontal. */
    val segmentDirection: Vec3 = Vec3(x = 1.0),
    val distanceToRoadM: Double = 0.0,
)

/** What an update actually did. Mirrors `UpdateOutcome` in the C++ engine. */
enum class UpdateOutcome { Applied, RejectedByGate, Skipped, NumericalFailure }

/**
 * What the UI renders. Geodetic rather than NED because the map wants lat/lon;
 * the conversion back out of the local-tangent frame happens on this side.
 */
data class FusedPosition(
    val lat: Double,
    val lon: Double,
    val headingDegrees: Float,
    val speedMetersPerSec: Float,
    /** 0..1. Drives marker styling — dimmed during a GNSS blackout. */
    val confidence: Float,
)

/**
 * The stable seam. `UkfFusionEngine` (Role 02's Kotlin port) and
 * `StubFusionEngine` (build-order step 4) both implement this, and the sampler
 * and map code are written against it and nothing else.
 *
 * Keep it stable — Prasoon's port targets this signature.
 */
interface PositionFusionEngine {

    /** Dead-reckoning step. Called on every IMU sample. */
    fun predict(sample: ImuSample)

    /** Correction from a GNSS fix, when one is available and passes the gate. */
    fun updateGnss(fix: GnssFix): UpdateOutcome

    /** Correction from snapping to a road centreline. */
    fun updateMapMatch(match: MapMatchResult): UpdateOutcome

    /** Current nominal state, in the navigation frame. */
    fun state(): NavState

    /** Render stream for the map marker. */
    fun observePosition(): Flow<FusedPosition>
}
