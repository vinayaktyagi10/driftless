package com.driftless.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

/**
 * Guards the hand-written equals/hashCode on [ImuSample].
 *
 * Without the override these all fail: a data class holding FloatArray compares
 * array identity, so two samples with identical contents are unequal and a
 * sample rehashes differently after a copy. The fusion port is going to hold
 * these in buffers and assert on them in its own tests, so the behaviour is
 * pinned here rather than discovered there.
 */
class ImuSampleTest {

    private fun sample(
        t: Long = 1_000L,
        accel: FloatArray = floatArrayOf(0f, 0f, -9.80665f),
        gyro: FloatArray = floatArrayOf(0f, 0f, 0f),
        mag: FloatArray? = floatArrayOf(20f, 0f, 40f),
    ) = ImuSample(t, accel, gyro, mag)

    @Test
    fun `equal contents in distinct arrays are equal`() {
        assertEquals(sample(), sample())
    }

    @Test
    fun `equal contents hash alike`() {
        assertEquals(sample().hashCode(), sample().hashCode())
    }

    @Test
    fun `a differing accel component is not equal`() {
        assertNotEquals(sample(), sample(accel = floatArrayOf(0f, 0f, -9.8f)))
    }

    @Test
    fun `a differing timestamp is not equal`() {
        assertNotEquals(sample(), sample(t = 1_001L))
    }

    @Test
    fun `null magnetometer is handled on both sides`() {
        assertEquals(sample(mag = null), sample(mag = null))
        assertNotEquals(sample(mag = null), sample())
    }

    @Test
    fun `distinct samples survive a set`() {
        val set = setOf(sample(), sample(), sample(t = 2_000L))
        assertEquals(2, set.size)
    }
}
