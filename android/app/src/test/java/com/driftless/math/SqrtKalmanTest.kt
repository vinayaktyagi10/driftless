package com.driftless.math

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

class SqrtKalmanTest {

    private fun randomSpd(n: Int, rng: Random): Matrix {
        val m = Matrix.zeros(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                m[i, j] = rng.nextGaussian()
            }
        }
        val spd = m * m.transpose()
        for (i in 0 until n) {
            spd[i, i] += n.toDouble()
        }
        return spd
    }

    private fun choleskyLlt(A: Matrix): Matrix {
        val n = A.rows
        val L = Matrix.zeros(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var sum = 0.0
                for (k in 0 until j) {
                    sum += L[i, k] * L[j, k]
                }
                if (i == j) {
                    val d = A[i, i] - sum
                    require(d > 0.0) { "Matrix not positive definite at $i: d=$d" }
                    L[i, j] = sqrt(d)
                } else {
                    L[i, j] = (A[i, j] - sum) / L[j, j]
                }
            }
        }
        return L
    }

    private fun maxAbsCoeff(m: Matrix): Double {
        var v = 0.0
        for (x in m.data) {
            v = max(v, abs(x))
        }
        return v
    }

    @Test
    fun testReproducesTextbookKalmanEquations() {
        val rng = Random(20260920)

        for (n in intArrayOf(2, 5, 15)) {
            for (m in intArrayOf(1, 2, 3, 6)) {
                if (m > n) continue
                val P = randomSpd(n, rng)
                val R = randomSpd(m, rng)
                val H = Matrix.zeros(m, n)
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        H[i, j] = rng.nextGaussian()
                    }
                }

                val S = choleskyLlt(P)
                val sqrtR = choleskyLlt(R)
                val result = SqrtKalman.arrayFormUpdate(S, H, sqrtR)

                // Innovation covariance S_nu * S_nu^T vs H * P * H^T + R
                val expectedInnovation = (H * P * H.transpose()) + R
                val actualInnovation = result.sqrtInnovationCovariance * result.sqrtInnovationCovariance.transpose()
                val diffInn = maxAbsCoeff(actualInnovation - expectedInnovation)
                assertTrue("n=$n m=$m diffInn=$diffInn", diffInn < 1e-9 * maxAbsCoeff(expectedInnovation))

                // Kalman gain: K = P * H^T * (expectedInnovation)^-1
                // By lower-triangular / upper-triangular solve on S_nu:
                // K_expected = P * H^T * (S_nu * S_nu^T)^-1
                val sNu = choleskyLlt(expectedInnovation)
                val sNuT = sNu.transpose()
                val pHT = P * H.transpose()
                // solve sNu * Y = pHT^T (m x n)
                val y1 = sNu.solveLowerTriangular(pHT.transpose())
                // solve sNuT * Y2 = y1 -> Y2 = (sNu*sNu^T)^-1 * pHT^T
                val y2 = sNuT.solveUpperTriangular(y1)
                val expectedGain = y2.transpose()

                val diffGain = maxAbsCoeff(result.gain - expectedGain)
                assertTrue("n=$n m=$m diffGain=$diffGain", diffGain < 1e-9 * max(1.0, maxAbsCoeff(expectedGain)))

                // Posterior covariance S+ * S+^T vs P - K * (H*P*H^T + R) * K^T
                val expectedPosterior = P - (expectedGain * expectedInnovation * expectedGain.transpose())
                val actualPosterior = result.sqrtCovariancePosterior * result.sqrtCovariancePosterior.transpose()
                val diffPost = maxAbsCoeff(actualPosterior - expectedPosterior)
                assertTrue("n=$n m=$m diffPost=$diffPost", diffPost < 1e-9 * maxAbsCoeff(P))
            }
        }
    }

    @Test
    fun testPosteriorFactorIsLowerTriangularAndPositiveDefinite() {
        val rng = Random(20260921)
        val n = 15
        val m = 3
        val S = choleskyLlt(randomSpd(n, rng))
        val H = Matrix.zeros(m, n)
        for (i in 0 until m) {
            H[i, i] = 1.0
        }

        // An extremely precise measurement
        val sqrtR = Matrix.identity(m) * 1e-8
        val result = SqrtKalman.arrayFormUpdate(S, H, sqrtR)

        for (i in 0 until n) {
            assertTrue("diagonal $i is not positive", result.sqrtCovariancePosterior[i, i] > 0.0)
            for (j in i + 1 until n) {
                assertEquals(0.0, result.sqrtCovariancePosterior[i, j], 1e-15)
            }
        }
    }

    @Test
    fun testUpdateReducesUncertaintyInTheMeasuredDirections() {
        val rng = Random(20260922)
        val n = 15
        val m = 3
        val P = randomSpd(n, rng)
        val S = choleskyLlt(P)
        val H = Matrix.zeros(m, n)
        for (i in 0 until m) {
            H[i, i] = 1.0
        }

        val result = SqrtKalman.arrayFormUpdate(S, H, Matrix.identity(m))
        val posterior = result.sqrtCovariancePosterior * result.sqrtCovariancePosterior.transpose()

        for (i in 0 until m) {
            assertTrue("state $i variance did not decrease", posterior[i, i] < P[i, i])
        }
        for (i in 0 until n) {
            assertTrue(posterior[i, i] <= P[i, i] + 1e-9)
        }
    }

    @Test
    fun testNormalizedInnovationSquaredMatchesQuadraticForm() {
        val rng = Random(20260923)
        for (m in intArrayOf(1, 2, 3, 6)) {
            val cov = randomSpd(m, rng)
            val sqrtCov = choleskyLlt(cov)
            val innovation = DoubleArray(m) { rng.nextGaussian() }

            // Expected NIS: innovation^T * cov^-1 * innovation = ||sqrtCov^-1 * innovation||^2
            val whitened = sqrtCov.solveLowerTriangular(innovation)
            var expected = 0.0
            for (x in whitened) {
                expected += x * x
            }

            val actual = SqrtKalman.normalizedInnovationSquared(sqrtCov, innovation)
            assertEquals(expected, actual, 1e-9 * max(1.0, expected))
        }
    }

    @Test
    fun testChiSquaredThresholdsMatchPublishedTable() {
        assertEquals(6.635, SqrtKalman.chiSquaredThreshold(1, 0.99), 1e-3)
        assertEquals(9.210, SqrtKalman.chiSquaredThreshold(2, 0.99), 1e-3)
        assertEquals(11.345, SqrtKalman.chiSquaredThreshold(3, 0.99), 1e-3)
        assertEquals(16.812, SqrtKalman.chiSquaredThreshold(6, 0.99), 1e-3)
        assertEquals(7.815, SqrtKalman.chiSquaredThreshold(3, 0.95), 1e-3)
        assertEquals(16.266, SqrtKalman.chiSquaredThreshold(3, 0.999), 1e-3)
        assertTrue(SqrtKalman.chiSquaredThreshold(4, 0.99) < 0.0)
    }
}
