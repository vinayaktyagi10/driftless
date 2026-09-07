package com.driftless.math

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

class CholeskyUpdateTest {

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

    private fun randomVec(n: Int, rng: Random): DoubleArray {
        val v = DoubleArray(n)
        for (i in 0 until n) {
            v[i] = rng.nextGaussian()
        }
        return v
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

    private fun naiveUpdatedFactor(A: Matrix, u: DoubleArray, sigma: Double): Matrix {
        val n = A.rows
        val updated = A.copy()
        for (i in 0 until n) {
            for (j in 0 until n) {
                updated[i, j] += sigma * u[i] * u[j]
            }
        }
        return choleskyLlt(updated)
    }

    private fun maxDiff(A: Matrix, B: Matrix): Double {
        var m = 0.0
        for (i in 0 until A.rows) {
            for (j in 0 until A.cols) {
                m = max(m, abs(A[i, j] - B[i, j]))
            }
        }
        return m
    }

    @Test
    fun testMatchesNaiveReferenceOnUpdate() {
        val rng = Random(20260901)
        for (n in intArrayOf(1, 2, 3, 7, 15)) {
            for (trial in 0 until 20) {
                val A = randomSpd(n, rng)
                val u = randomVec(n, rng)
                val S = choleskyLlt(A)

                assertTrue("cholUpdate failed for n=$n", CholeskyUpdate.cholUpdate(S, u, 1.0))
                val expected = naiveUpdatedFactor(A, u, 1.0)
                val diff = maxDiff(S, expected)
                assertTrue("n=$n trial=$trial diff=$diff", diff < 1e-9)
            }
        }
    }

    @Test
    fun testMatchesNaiveReferenceOnDowndate() {
        val rng = Random(20260902)
        for (n in intArrayOf(1, 2, 3, 7, 15)) {
            for (trial in 0 until 20) {
                val A = randomSpd(n, rng)
                val u = DoubleArray(n) { rng.nextGaussian() * 0.1 }
                val S = choleskyLlt(A)

                assertTrue("cholUpdate failed for n=$n", CholeskyUpdate.cholUpdate(S, u, -1.0))
                val expected = naiveUpdatedFactor(A, u, -1.0)
                val diff = maxDiff(S, expected)
                assertTrue("n=$n trial=$trial diff=$diff", diff < 1e-9)
            }
        }
    }

    @Test
    fun testHandlesArbitrarySigmaMagnitude() {
        val rng = Random(20260903)
        val sigmas = doubleArrayOf(0.001, 0.5, 3.7, 1000.0, -0.001, -0.05)
        for (sigma in sigmas) {
            val n = 15
            val A = randomSpd(n, rng)
            val u = DoubleArray(n) { rng.nextGaussian() * 0.05 }
            val S = choleskyLlt(A)

            assertTrue("sigma=$sigma failed", CholeskyUpdate.cholUpdate(S, u, sigma))
            val expected = naiveUpdatedFactor(A, u, sigma)
            val diff = maxDiff(S, expected)
            assertTrue("sigma=$sigma diff=$diff", diff < 1e-8)
        }
    }

    @Test
    fun testResultStaysLowerTriangularWithPositiveDiagonal() {
        val rng = Random(20260904)
        val n = 15
        val S = choleskyLlt(randomSpd(n, rng))
        for (step in 0 until 200) {
            val u = DoubleArray(n) { rng.nextGaussian() * 0.01 }
            assertTrue(CholeskyUpdate.cholUpdate(S, u, 1.0))
        }
        for (i in 0 until n) {
            assertTrue("diagonal $i is not positive", S[i, i] > 0.0)
            for (j in i + 1 until n) {
                assertEquals(0.0, S[i, j], 1e-15)
            }
        }
    }

    @Test
    fun testReportsFailureWhenDowndateBreaksPositiveDefiniteness() {
        val n = 4
        val S = Matrix.identity(n)
        val u = DoubleArray(n)
        u[0] = 10.0 // removes 100 from unit variance
        assertFalse(CholeskyUpdate.cholUpdate(S, u, -1.0))
    }

    @Test
    fun testQrToLowerTriangularReproducesFactorOfNormalEquations() {
        val rng = Random(20260905)
        for (n in intArrayOf(2, 5, 15)) {
            val rows = 3 * n
            val A = Matrix.zeros(rows, n)
            for (i in 0 until rows) {
                for (j in 0 until n) {
                    A[i, j] = rng.nextGaussian()
                }
            }

            val S = CholeskyUpdate.qrToLowerTriangular(A)
            val atA = A.transpose() * A
            val expected = choleskyLlt(atA)

            val diff = maxDiff(S, expected)
            assertTrue("n=$n diff=$diff", diff < 1e-9)

            for (i in 0 until n) {
                assertTrue(S[i, i] > 0.0)
                for (j in i + 1 until n) {
                    assertEquals(0.0, S[i, j], 1e-15)
                }
            }
        }
    }
}
