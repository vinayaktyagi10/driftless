package com.driftless.math

import kotlin.math.abs

data class SqrtKalmanUpdate(
    val sqrtInnovationCovariance: Matrix,
    val gain: Matrix,
    val sqrtCovariancePosterior: Matrix,
)

object SqrtKalman {

    private data class ChiSquaredEntry(
        val dof: Int,
        val at95: Double,
        val at99: Double,
        val at999: Double,
    )

    private val chiSquaredTable = arrayOf(
        ChiSquaredEntry(1, 3.841, 6.635, 10.828),
        ChiSquaredEntry(2, 5.991, 9.210, 13.816),
        ChiSquaredEntry(3, 7.815, 11.345, 16.266),
        ChiSquaredEntry(6, 12.592, 16.812, 22.458),
    )

    /**
     * Kailath array form of the linear Kalman update in square-root covariance coordinates.
     *
     * Pre-array:
     *     A = [ sqrt_R   H * S ]   ((m + n) x (m + n))
     *         [   0        S   ]
     *
     * Transposed and triangularized via qrToLowerTriangular:
     *     post_array = [ S_nu    0  ]
     *                  [ K_bar  S+  ]
     *
     * Gain: K = K_bar * S_nu^-1 = (S_nu^T \ K_bar^T)^T
     */
    fun arrayFormUpdate(sqrtCovariance: Matrix, H: Matrix, sqrtR: Matrix): SqrtKalmanUpdate {
        val n = sqrtCovariance.rows
        val m = H.rows

        val preArray = Matrix.zeros(n + m, n + m)
        preArray.setBlock(0, 0, sqrtR)
        preArray.setBlock(0, m, H * sqrtCovariance)
        preArray.setBlock(m, m, sqrtCovariance)

        val postArray = CholeskyUpdate.qrToLowerTriangular(preArray.transpose())

        val sqrtInnovationCovariance = postArray.topLeftCorner(m, m)
        val sqrtCovariancePosterior = postArray.bottomRightCorner(n, n)
        val gainBar = postArray.bottomLeftCorner(n, m)

        // Solve S_nu^T * Y = gainBar^T -> K = Y^T
        val sNuT = sqrtInnovationCovariance.transpose()
        val y = sNuT.solveUpperTriangular(gainBar.transpose())
        val gain = y.transpose()

        return SqrtKalmanUpdate(
            sqrtInnovationCovariance = sqrtInnovationCovariance,
            gain = gain,
            sqrtCovariancePosterior = sqrtCovariancePosterior,
        )
    }

    /**
     * Normalized Innovation Squared: nu^T * (S_nu * S_nu^T)^-1 * nu = ||S_nu^-1 * nu||^2
     */
    fun normalizedInnovationSquared(sqrtInnovationCovariance: Matrix, innovation: DoubleArray): Double {
        val whitened = sqrtInnovationCovariance.solveLowerTriangular(innovation)
        var sum = 0.0
        for (x in whitened) {
            sum += x * x
        }
        return sum
    }

    /**
     * Chi-squared upper-tail critical values.
     */
    fun chiSquaredThreshold(degreesOfFreedom: Int, confidence: Double): Double {
        for (entry in chiSquaredTable) {
            if (entry.dof != degreesOfFreedom) continue
            return when {
                confidence >= 0.999 - 1e-4 -> entry.at999
                confidence >= 0.99 - 1e-4 -> entry.at99
                confidence >= 0.95 - 1e-4 -> entry.at95
                else -> entry.at95
            }
        }
        return -1.0
    }
}
