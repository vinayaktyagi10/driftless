package com.driftless.frames

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The single most consequential conversion in the app. A sign or axis error
 * here mirrors or rotates every trajectory, which is plausible over ten metres
 * and unusable over two kilometres — so it is pinned against physical cases
 * with hand-known answers rather than against itself.
 */
class AxisConventionTest {

    private val g = 9.80665f
    private val eps = 1e-6f

    @Test
    fun `level stationary phone reads minus g on FRD down`() {
        // Phone flat, screen up. SensorManager's accelerometer measures specific
        // force, so it reports +g on device z (out of the screen).
        val device = floatArrayOf(0f, 0f, g)

        val frd = AxisConvention.deviceToFrd(device)

        // types.h: "g_nav = [0, 0, +9.80665] (down is positive z), so a level
        // stationary accelerometer reads [0, 0, -9.80665]." This is the one
        // assertion that ties the whole convention to the engine's contract.
        assertArrayEquals(floatArrayOf(0f, 0f, -g), frd, eps)
    }

    @Test
    fun `device forward becomes FRD x`() {
        assertArrayEquals(
            floatArrayOf(1f, 0f, 0f),
            AxisConvention.deviceToFrd(floatArrayOf(0f, 1f, 0f)),
            eps,
        )
    }

    @Test
    fun `device right becomes FRD y`() {
        assertArrayEquals(
            floatArrayOf(0f, 1f, 0f),
            AxisConvention.deviceToFrd(floatArrayOf(1f, 0f, 0f)),
            eps,
        )
    }

    @Test
    fun `device up becomes negative FRD z`() {
        assertArrayEquals(
            floatArrayOf(0f, 0f, -1f),
            AxisConvention.deviceToFrd(floatArrayOf(0f, 0f, 1f)),
            eps,
        )
    }

    @Test
    fun `yaw left about device up is negative about FRD down`() {
        // Right-hand rule: turning anticlockwise seen from above is +z in the
        // device frame. FRD z points down, so the same physical turn must come
        // out negative. Getting this backwards makes every left turn a right
        // turn, and a map-matched track would fight it silently.
        val frd = AxisConvention.deviceToFrd(floatArrayOf(0f, 0f, 0.5f))
        assertEquals(-0.5f, frd[2], eps)
    }

    @Test
    fun `the mapping is a rotation and not a reflection`() {
        // det(M) must be +1. Built column-wise by transforming the basis, so
        // this tests the actual function rather than a restatement of it.
        val c0 = AxisConvention.deviceToFrd(floatArrayOf(1f, 0f, 0f))
        val c1 = AxisConvention.deviceToFrd(floatArrayOf(0f, 1f, 0f))
        val c2 = AxisConvention.deviceToFrd(floatArrayOf(0f, 0f, 1f))

        val det =
            c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) -
                c1[0] * (c0[1] * c2[2] - c0[2] * c2[1]) +
                c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])

        assertEquals(1.0f, det, eps)
    }

    @Test
    fun `the mapping preserves vector length`() {
        val v = floatArrayOf(1.5f, -2.25f, 0.75f)
        val frd = AxisConvention.deviceToFrd(v)

        fun norm2(a: FloatArray) = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        assertEquals(norm2(v), norm2(frd), 1e-5f)
    }

    @Test
    fun `in-place form matches the allocating form`() {
        val v = floatArrayOf(0.1f, -0.2f, 0.3f)
        val expected = AxisConvention.deviceToFrd(v)

        AxisConvention.deviceToFrdInPlace(v)

        assertArrayEquals(expected, v, eps)
    }

    @Test
    fun `the mapping is its own inverse`() {
        // Swapping x with y twice, and negating z twice, both return the
        // original — so M * M = I. Worth pinning rather than assuming: it means
        // an accidental double application is invisible, and it is why the
        // matching-the-allocating-form test above is what actually guards the
        // in-place swap against clobbering a component before it is read.
        val v = floatArrayOf(1f, 2f, 3f)

        AxisConvention.deviceToFrdInPlace(v)
        assertArrayEquals(floatArrayOf(2f, 1f, -3f), v, eps)

        AxisConvention.deviceToFrdInPlace(v)
        assertArrayEquals(floatArrayOf(1f, 2f, 3f), v, eps)
    }
}
