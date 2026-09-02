package com.driftless.frames

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

class AccuracyTest {

    @Test
    fun `the constant is the Rayleigh radius holding one sigma of probability`() {
        // Derived independently here from the Rayleigh CDF rather than copied
        // from the source, so a typo in the constant fails rather than agrees
        // with itself: P(R <= r) = 1 - exp(-r^2 / 2).
        val r = Accuracy.HORIZONTAL_68_TO_SIGMA
        val enclosed = 1.0 - exp(-r * r / 2.0)

        assertEquals(0.6826895, enclosed, 1e-7)
    }

    @Test
    fun `horizontal accuracy is inflated into a larger sigma`() {
        // The direction matters: the reported 68% radius is LARGER than one
        // sigma, so sigma must come out smaller. Backwards would make the
        // filter over-confident and its NIS gate reject good fixes.
        val sigma = Accuracy.horizontalSigmaM(10f)

        assertEquals(6.5999068, sigma, 1e-6)
        assertTrue("sigma must be smaller than the 68% radius", sigma < 10.0)
    }

    @Test
    fun `vertical accuracy is already one sigma and passes through`() {
        assertEquals(4.0, Accuracy.verticalSigmaM(4f), 1e-9)
    }

    @Test
    fun `speed accuracy is already one sigma and passes through`() {
        assertEquals(0.35, Accuracy.speedSigmaMps(0.35f), 1e-6)
    }

    @Test
    fun `a perfect fix stays zero rather than dividing to NaN`() {
        assertEquals(0.0, Accuracy.horizontalSigmaM(0f), 0.0)
    }
}
