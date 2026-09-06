package com.driftless.frames

import com.driftless.fusion.Position
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Geodetic lat/lon/alt <-> local-tangent **NED** metres, anchored at the first
 * GNSS fix.
 *
 * Flat-earth on purpose. `types.h`: *"over the distances a blackout covers,
 * earth-rate and Coriolis terms are far below this IMU's noise floor."* The
 * same reasoning covers the curvature this drops — the scale factors are
 * evaluated once at the origin latitude and held, which costs well under a
 * metre over the few kilometres a demo route spans, and buys a conversion with
 * no iteration and no trigonometry per fix.
 *
 * Anchoring at the **first** fix rather than a fixed datum is what keeps those
 * scale factors valid and keeps the numbers small enough that float precision
 * downstream is not a question.
 */
class LocalTangentFrame private constructor(
    val originLatDeg: Double,
    val originLonDeg: Double,
    val originAltM: Double,
    /** Metres north per degree of latitude, at the origin. */
    private val metresPerDegLat: Double,
    /** Metres east per degree of longitude, at the origin. */
    private val metresPerDegLon: Double,
) {

    /** Geodetic -> NED metres from the origin. */
    fun toNed(latDeg: Double, lonDeg: Double, altM: Double): Position = Position(
        north = (latDeg - originLatDeg) * metresPerDegLat,
        east = (lonDeg - originLonDeg) * metresPerDegLon,
        // Down is positive, altitude is up.
        down = -(altM - originAltM),
    )

    /** NED metres -> geodetic, for putting the fused track back on the map. */
    fun toGeodetic(position: Position): Geodetic = Geodetic(
        latDeg = originLatDeg + position.north / metresPerDegLat,
        lonDeg = originLonDeg + position.east / metresPerDegLon,
        altM = originAltM - position.down,
    )

    data class Geodetic(val latDeg: Double, val lonDeg: Double, val altM: Double)

    companion object {
        /** WGS-84 semi-major axis, metres. */
        const val WGS84_A = 6_378_137.0

        /** WGS-84 flattening. */
        const val WGS84_F = 1.0 / 298.257223563

        /** First eccentricity squared, e^2 = f(2 - f). */
        val WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

        /**
         * Anchor a frame at a fix.
         *
         * The two radii of curvature differ by ~1% between equator and pole and
         * by ~0.3% at Jaipur's latitude — small, but free to get right, and
         * using a single spherical radius instead would bias north and east
         * against each other in a way that looks like a systematic heading
         * error rather than like a scale error.
         */
        fun anchoredAt(latDeg: Double, lonDeg: Double, altM: Double): LocalTangentFrame {
            val latRad = Math.toRadians(latDeg)
            val sinLat = sin(latRad)
            val w2 = 1.0 - WGS84_E2 * sinLat * sinLat
            val w = sqrt(w2)

            // Meridian radius of curvature — north/south.
            val m = WGS84_A * (1.0 - WGS84_E2) / (w2 * w)
            // Prime vertical radius of curvature — east/west, scaled by cos(lat).
            val n = WGS84_A / w

            return LocalTangentFrame(
                originLatDeg = latDeg,
                originLonDeg = lonDeg,
                originAltM = altM,
                metresPerDegLat = Math.toRadians(1.0) * m,
                metresPerDegLon = Math.toRadians(1.0) * n * cos(latRad),
            )
        }
    }
}
