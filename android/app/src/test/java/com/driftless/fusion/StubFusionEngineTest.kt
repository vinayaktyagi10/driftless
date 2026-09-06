package com.driftless.fusion

import com.driftless.frames.LocalTangentFrame
import com.driftless.sensors.ImuSample
import kotlinx.coroutines.flow.SharedFlow
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The three coast rules, each exercised on both sides of its threshold.
 *
 * Position is asserted through [StubFusionEngine.state] in metres NED rather
 * than through the emitted lat/lon, so a failure reads as "coasted 4 m too far"
 * instead of as a difference in the seventh decimal of a latitude.
 */
class StubFusionEngineTest {

    private val frame = LocalTangentFrame.anchoredAt(26.8770866, 75.7727424, 362.5)
    private val engine = StubFusionEngine { frame }

    /** t0 well away from zero, so an accidental absolute-time bug shows up. */
    private val t0 = 47_231_884_000_000L

    private fun nanos(seconds: Double) = t0 + (seconds * 1e9).toLong()

    private fun fix(
        atSeconds: Double,
        north: Double,
        east: Double,
        velocityNorth: Double = 0.0,
        velocityEast: Double = 0.0,
        hasVelocity: Boolean = true,
    ) = GnssFix(
        timestampNanos = nanos(atSeconds),
        position = Position(north = north, east = east, down = 0.0),
        velocityNed = Vec3(x = velocityNorth, y = velocityEast),
        hasVelocity = hasVelocity,
        horizontalAccuracyM = 2.75,
        satellitesUsed = 24,
    )

    private fun tick(atSeconds: Double) = engine.predict(
        ImuSample(
            timestampNanos = nanos(atSeconds),
            accel = floatArrayOf(0f, 0f, -9.80665f),
            gyro = floatArrayOf(0f, 0f, 0f),
            mag = null,
        ),
    )

    private fun north() = engine.state().position.x

    private fun east() = engine.state().position.y

    private fun latest(): FusedPosition? =
        @Suppress("UNCHECKED_CAST")
        (engine.observePosition() as SharedFlow<FusedPosition>).replayCache.lastOrNull()

    // -- before there is anything to say -----------------------------------

    @Test
    fun `emits nothing before the first fix`() {
        tick(0.0)
        tick(1.0)
        assertNull("no fix means no origin, so nothing may be drawn", latest())
    }

    @Test
    fun `emits nothing while the frame is unanchored`() {
        val unanchored = StubFusionEngine { null }
        unanchored.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        val flow = @Suppress("UNCHECKED_CAST")
        (unanchored.observePosition() as SharedFlow<FusedPosition>)
        assertNull("lat/lon is meaningless without an anchor", flow.replayCache.lastOrNull())
    }

    // -- the coast itself ---------------------------------------------------

    @Test
    fun `coasts along the latched velocity between fixes`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        tick(1.0)
        assertEquals(10.0, north(), 1e-9)
        assertEquals(0.0, east(), 1e-9)
    }

    @Test
    fun `every fix resets position, coast or no coast`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        tick(0.5)
        engine.updateGnss(fix(1.0, 100.0, 50.0, velocityNorth = 10.0))
        assertEquals("the fix is a measurement and wins", 100.0, north(), 1e-9)
        assertEquals(50.0, east(), 1e-9)
    }

    @Test
    fun `reports heading and speed from the latched velocity`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityEast = 12.0))
        val fused = latest()!!
        assertEquals("due east is 90 degrees", 90.0f, fused.headingDegrees, 1e-3f)
        assertEquals(12.0f, fused.speedMetersPerSec, 1e-3f)
    }

    // -- rule 1: the standstill floor --------------------------------------

    @Test
    fun `rule 1 - a reported speed below the floor disarms the coast`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        // 0.8 m/s clears the sampler's 0.5 m/s bearing gate but not our floor.
        engine.updateGnss(fix(1.0, 10.0, 0.0, velocityNorth = 0.8))
        tick(1.5)
        assertEquals("must hold, not coast the pre-braking velocity", 10.0, north(), 1e-9)
        assertEquals(0.0f, latest()!!.speedMetersPerSec, 1e-9f)
    }

    @Test
    fun `rule 1 - a reported speed at the floor still arms the coast`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        engine.updateGnss(fix(1.0, 10.0, 0.0, velocityNorth = 1.0))
        tick(2.0)
        assertEquals(11.0, north(), 1e-9)
    }

    // -- rule 2: latch expiry ----------------------------------------------

    @Test
    fun `rule 2 - the latch survives a single fix with no velocity`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        engine.updateGnss(fix(1.0, 10.0, 0.0, hasVelocity = false))
        tick(1.5)
        assertEquals(
            "one dropout fix at speed must not kill the coast",
            15.0,
            north(),
            1e-9,
        )
    }

    @Test
    fun `rule 2 - the latch is still believed at exactly two seconds old`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        engine.updateGnss(fix(1.0, 10.0, 0.0, hasVelocity = false))
        engine.updateGnss(fix(2.0, 20.0, 0.0, hasVelocity = false))
        tick(2.5)
        assertEquals(25.0, north(), 1e-9)
    }

    @Test
    fun `rule 2 - the latch expires past two seconds old`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        engine.updateGnss(fix(1.0, 10.0, 0.0, hasVelocity = false))
        engine.updateGnss(fix(2.0, 20.0, 0.0, hasVelocity = false))
        engine.updateGnss(fix(3.0, 30.0, 0.0, hasVelocity = false))
        tick(3.5)
        assertEquals("three fixes without Doppler is too stale", 30.0, north(), 1e-9)
        assertEquals(Vec3(), engine.state().velocity)
    }

    @Test
    fun `rule 2 - a fresh velocity re-arms an expired latch`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        engine.updateGnss(fix(3.0, 30.0, 0.0, hasVelocity = false))
        engine.updateGnss(fix(4.0, 40.0, 0.0, velocityNorth = 10.0))
        tick(4.5)
        assertEquals(45.0, north(), 1e-9)
    }

    // -- the coast horizon cap ---------------------------------------------

    @Test
    fun `stops extrapolating at the ten second cap`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        tick(10.0)
        assertEquals(100.0, north(), 1e-9)
        tick(20.0)
        assertEquals("held, not free-running", 100.0, north(), 1e-9)
    }

    // -- confidence ---------------------------------------------------------

    @Test
    fun `confidence holds while the coast beats the fix it replaced`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        assertEquals(1.0f, latest()!!.confidence, 1e-6f)
        tick(2.0)
        assertEquals("2 s is where coast p95 meets sigma_h p95", 1.0f, latest()!!.confidence, 1e-6f)
    }

    @Test
    fun `confidence ramps to zero across the coast horizon`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        tick(6.0)
        assertEquals(0.5f, latest()!!.confidence, 1e-6f)
        tick(10.0)
        assertEquals(0.0f, latest()!!.confidence, 1e-6f)
        tick(30.0)
        assertEquals(0.0f, latest()!!.confidence, 1e-6f)
    }

    @Test
    fun `confidence decays while holding position too`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 0.5))
        tick(6.0)
        assertEquals(
            "a stationary marker is not more trustworthy for being stationary",
            0.5f,
            latest()!!.confidence,
            1e-6f,
        )
    }

    // -- emission rate and the unimplemented seam ---------------------------

    @Test
    fun `republishes at ten hertz, not the imu rate`() {
        engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0))
        val first = latest()!!
        tick(0.02)
        assertTrue(
            "a 20 ms tick must not republish",
            latest()!!.lat == first.lat,
        )
        tick(0.12)
        assertTrue("a 120 ms tick must", latest()!!.lat != first.lat)
    }

    @Test
    fun `map matching is skipped rather than silently claimed`() {
        assertEquals(
            UpdateOutcome.Skipped,
            engine.updateMapMatch(MapMatchResult(matched = true, segmentId = 7)),
        )
    }

    @Test
    fun `a gnss update is always applied`() {
        assertEquals(
            UpdateOutcome.Applied,
            engine.updateGnss(fix(0.0, 0.0, 0.0, velocityNorth = 10.0)),
        )
    }
}
