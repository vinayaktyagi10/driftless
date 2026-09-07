package com.driftless.math

import kotlin.math.abs
import kotlin.math.sqrt

object CholeskyUpdate {

    /**
     * Rank-1 Cholesky update / downdate.
     *
     * Overwrites lower-triangular matrix S in-place so that:
     *   S_new * S_new^T = S * S^T + sigma * u * u^T
     *
     * @param S lower-triangular matrix (n x n), modified in place
     * @param u vector (size n)
     * @param sigma scalar weight (+ for update, - for downdate)
     * @return true if successful, false if downdate loses positive definiteness
     */
    fun cholUpdate(S: Matrix, u: DoubleArray, sigma: Double): Boolean {
        val n = S.rows
        if (S.cols != n || u.size != n) return false
        if (sigma == 0.0) return true

        val sign = if (sigma > 0.0) 1.0 else -1.0
        val w = DoubleArray(n)
        val sqrtSigma = sqrt(abs(sigma))
        for (i in 0 until n) {
            w[i] = sqrtSigma * u[i]
        }

        // Golub's rank-1 modification, column by column
        for (k in 0 until n) {
            val d = S[k, k]
            if (!(d > 0.0)) return false

            val rSquared = d * d + sign * w[k] * w[k]
            if (!(rSquared > 0.0)) return false

            val r = sqrt(rSquared)
            val c = r / d
            val s = w[k] / d
            S[k, k] = r

            for (i in k + 1 until n) {
                val sIk = (S[i, k] + sign * s * w[i]) / c
                w[i] = c * w[i] - s * sIk
                S[i, k] = sIk
            }
        }
        return true
    }

    /**
     * Lower-triangular factor L with L * L^T = A^T * A, computed via Householder QR
     * decomposition of A without ever forming A^T * A.
     *
     * @param A matrix (m x n) with m >= n
     * @return lower-triangular matrix L (n x n) with non-negative diagonal
     */
    fun qrToLowerTriangular(A: Matrix): Matrix {
        val m = A.rows
        val n = A.cols
        require(m >= n) { "qrToLowerTriangular requires rows ($m) >= cols ($n)" }

        // Work on a copy of A in column-friendly/row-friendly representation
        val aCopy = A.data.copyOf()
        val v = DoubleArray(m)

        for (k in 0 until n) {
            // Compute norm of column k from row k to m - 1
            var normSq = 0.0
            for (i in k until m) {
                val valI = aCopy[i * n + k]
                normSq += valI * valI
            }
            if (normSq <= 1e-30) continue

            val norm = sqrt(normSq)
            val x0 = aCopy[k * n + k]
            val sign = if (x0 >= 0.0) 1.0 else -1.0
            val v0 = x0 + sign * norm

            // Form Householder vector v
            v[k] = 1.0
            val invV0 = 1.0 / v0
            var vNormSq = 1.0
            for (i in k + 1 until m) {
                val vi = aCopy[i * n + k] * invV0
                v[i] = vi
                vNormSq += vi * vi
            }

            val beta = 2.0 / vNormSq

            // Apply Householder reflector H = I - beta * v * v^T to columns k..n-1
            for (j in k until n) {
                var dot = 0.0
                for (i in k until m) {
                    dot += v[i] * aCopy[i * n + j]
                }
                val factor = beta * dot
                for (i in k until m) {
                    aCopy[i * n + j] -= factor * v[i]
                }
            }
        }

        // Extract upper-triangular R (n x n) from top of aCopy, and transpose to lower-triangular L (n x n)
        val L = Matrix(n, n)
        for (i in 0 until n) {
            // Check sign of R(i, i)
            val rDiag = aCopy[i * n + i]
            val sign = if (rDiag < 0.0) -1.0 else 1.0

            for (j in 0..i) {
                // L(i, j) = R(j, i) with row j of R normalized by sign
                val rVal = aCopy[j * n + i]
                val rjDiag = aCopy[j * n + j]
                val jSign = if (rjDiag < 0.0) -1.0 else 1.0
                L[i, j] = rVal * jSign
            }
        }

        return L
    }
}
