package com.driftless.frames

import com.driftless.fusion.Position
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalTangentFrameTest {

    // Jaipur — the demo city, so the scale factors are exercised at the
    // latitude they will actually run at rather than at the equator.
    private val jaipurLat = 26.9124
    private val jaipurLon = 75.7873
    private val jaipurAlt = 431.0

    private fun jaipur() =
        LocalTangentFrame.anchoredAt(jaipurLat, jaipurLon, jaipurAlt)

    @Test
    fun `the origin maps to zero`() {
        val ned = jaipur().toNed(jaipurLat, jaipurLon, jaipurAlt)

        assertEquals(0.0, ned.north, 1e-9)
        assertEquals(0.0, ned.east, 1e-9)
        assertEquals(0.0, ned.down, 1e-9)
    }

    @Test
    fun `north is positive going north and east positive going east`() {
        val ned = jaipur().toNed(jaipurLat + 0.001, jaipurLon + 0.001, jaipurAlt)

        assertTrue("increasing latitude must be +north", ned.north > 0)
        assertTrue("increasing longitude must be +east", ned.east > 0)
    }

    @Test
    fun `down is positive downwards`() {
        // The sign flip against altitude is the one that silently inverts the
        // vertical channel of every fix.
        val ned = jaipur().toNed(jaipurLat, jaipurLon, jaipurAlt + 50.0)

        assertEquals(-50.0, ned.down, 1e-9)
    }

    @Test
    fun `one degree of latitude is about 110 point 9 km at Jaipur`() {
        // Independent reference value: the meridian arc per degree runs from
        // ~110.574 km at the equator to ~111.694 km at the poles, and is
        // ~110.90 km at 27 deg N.
        val ned = jaipur().toNed(jaipurLat + 1.0, jaipurLon, jaipurAlt)

        assertEquals(110_900.0, ned.north, 150.0)
    }

    @Test
    fun `one degree of longitude is about 99 point 3 km at Jaipur`() {
        // cos(26.9124 deg) * 111.32 km/deg ~= 99.3 km.
        val ned = jaipur().toNed(jaipurLat, jaipurLon + 1.0, jaipurAlt)

        assertEquals(99_300.0, ned.east, 200.0)
    }

    @Test
    fun `north and east scale factors differ`() {
        // A single spherical radius for both would show up as a systematic
        // heading error rather than as an obvious scale error, so assert they
        // are actually distinct.
        val f = jaipur()
        val north = f.toNed(jaipurLat + 0.01, jaipurLon, jaipurAlt).north
        val east = f.toNed(jaipurLat, jaipurLon + 0.01, jaipurAlt).east

        assertTrue("north/east metres per degree must differ", north - east > 100.0)
    }

    @Test
    fun `geodetic round trip returns the original position`() {
        val f = jaipur()
        val lat = jaipurLat + 0.0123
        val lon = jaipurLon - 0.0456
        val alt = jaipurAlt + 17.5

        val back = f.toGeodetic(f.toNed(lat, lon, alt))

        assertEquals(lat, back.latDeg, 1e-9)
        assertEquals(lon, back.lonDeg, 1e-9)
        assertEquals(alt, back.altM, 1e-9)
    }

    @Test
    fun `NED round trip returns the original metres`() {
        val f = jaipur()
        val ned = Position(north = 1234.5, east = -678.9, down = 12.3)

        val g = f.toGeodetic(ned)
        val back = f.toNed(g.latDeg, g.lonDeg, g.altM)

        assertEquals(ned.north, back.north, 1e-6)
        assertEquals(ned.east, back.east, 1e-6)
        assertEquals(ned.down, back.down, 1e-6)
    }

    @Test
    fun `holding the meridian radius at the origin costs under a centimetre over two km`() {
        // Flat-earth is an approximation; this bounds what it actually costs
        // over the distance a demo route covers, rather than trusting the
        // argument in the doc comment.
        //
        // The reference integrates the meridian radius of curvature across the
        // interval, which is what LocalTangentFrame deliberately does not do —
        // it evaluates M once at the origin and holds it. The gap between the
        // two IS the approximation error.
        //
        // Note this is NOT comparable to a haversine distance on a mean-radius
        // sphere: R_mean is 6371.0 km while the meridian radius at 27 deg N is
        // 6348.5 km, so that reference disagrees by ~7 m over 2 km for reasons
        // that have nothing to do with this code.
        val f = jaipur()
        val dLatDeg = 0.018   // ~2 km north

        val ned = f.toNed(jaipurLat + dLatDeg, jaipurLon, jaipurAlt)
        val arc = meridianArcMetres(jaipurLat, jaipurLat + dLatDeg)

        assertEquals(arc, ned.north, 0.01)
    }

    /** WGS-84 meridian radius of curvature at a latitude, metres. */
    private fun meridianRadius(latDeg: Double): Double {
        val s = kotlin.math.sin(Math.toRadians(latDeg))
        val w2 = 1.0 - LocalTangentFrame.WGS84_E2 * s * s
        return LocalTangentFrame.WGS84_A * (1.0 - LocalTangentFrame.WGS84_E2) /
            (w2 * kotlin.math.sqrt(w2))
    }

    /** Simpson's rule over the meridian arc — the honest north distance. */
    private fun meridianArcMetres(fromDeg: Double, toDeg: Double): Double {
        val n = 1000   // even, as Simpson requires
        val stepDeg = (toDeg - fromDeg) / n
        var sum = 0.0
        for (i in 0..n) {
            val weight = when {
                i == 0 || i == n -> 1.0
                i % 2 == 1 -> 4.0
                else -> 2.0
            }
            sum += weight * meridianRadius(fromDeg + i * stepDeg)
        }
        return sum * Math.toRadians(stepDeg) / 3.0
    }

    @Test
    fun `scale factors track latitude`() {
        val equator = LocalTangentFrame.anchoredAt(0.0, 0.0, 0.0)
        val high = LocalTangentFrame.anchoredAt(60.0, 0.0, 0.0)

        val eastAtEquator = equator.toNed(0.0, 1.0, 0.0).east
        val eastAtHigh = high.toNed(60.0, 1.0, 0.0).east

        // cos(60) = 0.5, so a degree of longitude is about half as wide.
        assertEquals(0.5, eastAtHigh / eastAtEquator, 0.01)
    }
}
