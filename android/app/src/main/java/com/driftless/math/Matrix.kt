package com.driftless.math

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * High-performance, allocation-conscious dense matrix backed by a 1D flat DoubleArray.
 * Row-major storage: element (r, c) is at index r * cols + c.
 */
class Matrix(val rows: Int, val cols: Int, val data: DoubleArray = DoubleArray(rows * cols)) {

    init {
        require(rows >= 0 && cols >= 0) { "Matrix dimensions must be non-negative: rows=$rows, cols=$cols" }
        require(data.size == rows * cols) { "Data size ${data.size} does not match dimensions $rows x $cols" }
    }

    operator fun get(r: Int, c: Int): Double = data[r * cols + c]
    operator fun set(r: Int, c: Int, value: Double) {
        data[r * cols + c] = value
    }

    operator fun plus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols) { "Dimension mismatch: ($rows x $cols) + (${other.rows} x ${other.cols})" }
        val out = DoubleArray(data.size)
        for (i in data.indices) {
            out[i] = data[i] + other.data[i]
        }
        return Matrix(rows, cols, out)
    }

    operator fun minus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols) { "Dimension mismatch: ($rows x $cols) - (${other.rows} x ${other.cols})" }
        val out = DoubleArray(data.size)
        for (i in data.indices) {
            out[i] = data[i] - other.data[i]
        }
        return Matrix(rows, cols, out)
    }

    operator fun times(scalar: Double): Matrix {
        val out = DoubleArray(data.size)
        for (i in data.indices) {
            out[i] = data[i] * scalar
        }
        return Matrix(rows, cols, out)
    }

    operator fun times(other: Matrix): Matrix {
        require(cols == other.rows) { "Matrix multiplication dimension mismatch: ($rows x $cols) * (${other.rows} x ${other.cols})" }
        val out = DoubleArray(rows * other.cols)
        val otherCols = other.cols
        for (i in 0 until rows) {
            val iRowOffset = i * cols
            val outRowOffset = i * otherCols
            for (k in 0 until cols) {
                val a = data[iRowOffset + k]
                if (a == 0.0) continue
                val kRowOffset = k * otherCols
                for (j in 0 until otherCols) {
                    out[outRowOffset + j] += a * other.data[kRowOffset + j]
                }
            }
        }
        return Matrix(rows, other.cols, out)
    }

    operator fun times(vector: DoubleArray): DoubleArray {
        require(cols == vector.size) { "Matrix-vector dimension mismatch: ($rows x $cols) * (${vector.size})" }
        val out = DoubleArray(rows)
        for (i in 0 until rows) {
            var sum = 0.0
            val rowOffset = i * cols
            for (j in 0 until cols) {
                sum += data[rowOffset + j] * vector[j]
            }
            out[i] = sum
        }
        return out
    }

    fun transpose(): Matrix {
        val out = DoubleArray(rows * cols)
        for (i in 0 until rows) {
            val rowOffset = i * cols
            for (j in 0 until cols) {
                out[j * rows + i] = data[rowOffset + j]
            }
        }
        return Matrix(cols, rows, out)
    }

    fun copy(): Matrix = Matrix(rows, cols, data.copyOf())

    fun block(startRow: Int, startCol: Int, blockRows: Int, blockCols: Int): Matrix {
        require(startRow >= 0 && startCol >= 0 && startRow + blockRows <= rows && startCol + blockCols <= cols) {
            "Block ($startRow..${startRow + blockRows}, $startCol..${startCol + blockCols}) out of bounds for ($rows x $cols)"
        }
        val out = DoubleArray(blockRows * blockCols)
        for (i in 0 until blockRows) {
            val srcOffset = (startRow + i) * cols + startCol
            val dstOffset = i * blockCols
            System.arraycopy(data, srcOffset, out, dstOffset, blockCols)
        }
        return Matrix(blockRows, blockCols, out)
    }

    fun setBlock(startRow: Int, startCol: Int, block: Matrix) {
        require(startRow >= 0 && startCol >= 0 && startRow + block.rows <= rows && startCol + block.cols <= cols) {
            "Cannot set block (${block.rows} x ${block.cols}) at ($startRow, $startCol) in ($rows x $cols)"
        }
        for (i in 0 until block.rows) {
            val srcOffset = i * block.cols
            val dstOffset = (startRow + i) * cols + startCol
            System.arraycopy(block.data, srcOffset, data, dstOffset, block.cols)
        }
    }

    fun row(r: Int): DoubleArray {
        require(r in 0 until rows)
        val out = DoubleArray(cols)
        System.arraycopy(data, r * cols, out, 0, cols)
        return out
    }

    fun setRow(r: Int, values: DoubleArray) {
        require(r in 0 until rows && values.size == cols)
        System.arraycopy(values, 0, data, r * cols, cols)
    }

    fun col(c: Int): DoubleArray {
        require(c in 0 until cols)
        val out = DoubleArray(rows)
        for (i in 0 until rows) {
            out[i] = data[i * cols + c]
        }
        return out
    }

    fun setCol(c: Int, values: DoubleArray) {
        require(c in 0 until cols && values.size == rows)
        for (i in 0 until rows) {
            data[i * cols + c] = values[i]
        }
    }

    fun scaleRow(r: Int, factor: Double) {
        val offset = r * cols
        for (j in 0 until cols) {
            data[offset + j] *= factor
        }
    }

    fun topRows(n: Int): Matrix = block(0, 0, n, cols)
    fun bottomRows(n: Int): Matrix = block(rows - n, 0, n, cols)
    fun topLeftCorner(r: Int, c: Int): Matrix = block(0, 0, r, c)
    fun topRightCorner(r: Int, c: Int): Matrix = block(0, cols - c, r, c)
    fun bottomLeftCorner(r: Int, c: Int): Matrix = block(rows - r, 0, r, c)
    fun bottomRightCorner(r: Int, c: Int): Matrix = block(rows - r, cols - c, r, c)

    /**
     * Solves L * X = B for X, where this matrix L is lower-triangular (n x n).
     * B is (n x m), returns X (n x m).
     */
    fun solveLowerTriangular(B: Matrix): Matrix {
        require(rows == cols && rows == B.rows) { "Dimension mismatch for lower-triangular solve: ($rows x $cols) and (${B.rows} x ${B.cols})" }
        val n = rows
        val m = B.cols
        val X = Matrix(n, m)
        for (j in 0 until m) {
            for (i in 0 until n) {
                var sum = B[i, j]
                val rowOffset = i * cols
                for (k in 0 until i) {
                    sum -= data[rowOffset + k] * X[k, j]
                }
                val diag = data[rowOffset + i]
                require(abs(diag) > 1e-15) { "Singular matrix in lower-triangular solve at diagonal index $i (value=$diag)" }
                X[i, j] = sum / diag
            }
        }
        return X
    }

    /**
     * Solves L * x = b for vector x, where this matrix L is lower-triangular (n x n).
     */
    fun solveLowerTriangular(b: DoubleArray): DoubleArray {
        require(rows == cols && rows == b.size) { "Dimension mismatch for lower-triangular solve: ($rows x $cols) and (${b.size})" }
        val n = rows
        val x = DoubleArray(n)
        for (i in 0 until n) {
            var sum = b[i]
            val rowOffset = i * cols
            for (k in 0 until i) {
                sum -= data[rowOffset + k] * x[k]
            }
            val diag = data[rowOffset + i]
            require(abs(diag) > 1e-15) { "Singular matrix in lower-triangular solve at diagonal index $i (value=$diag)" }
            x[i] = sum / diag
        }
        return x
    }

    /**
     * Solves U * X = B for X, where this matrix U is upper-triangular (n x n).
     * B is (n x m), returns X (n x m).
     */
    fun solveUpperTriangular(B: Matrix): Matrix {
        require(rows == cols && rows == B.rows) { "Dimension mismatch for upper-triangular solve: ($rows x $cols) and (${B.rows} x ${B.cols})" }
        val n = rows
        val m = B.cols
        val X = Matrix(n, m)
        for (j in 0 until m) {
            for (i in n - 1 downTo 0) {
                var sum = B[i, j]
                val rowOffset = i * cols
                for (k in i + 1 until n) {
                    sum -= data[rowOffset + k] * X[k, j]
                }
                val diag = data[rowOffset + i]
                require(abs(diag) > 1e-15) { "Singular matrix in upper-triangular solve at diagonal index $i (value=$diag)" }
                X[i, j] = sum / diag
            }
        }
        return X
    }

    /**
     * Solves U * x = b for vector x, where this matrix U is upper-triangular (n x n).
     */
    fun solveUpperTriangular(b: DoubleArray): DoubleArray {
        require(rows == cols && rows == b.size) { "Dimension mismatch for upper-triangular solve: ($rows x $cols) and (${b.size})" }
        val n = rows
        val x = DoubleArray(n)
        for (i in n - 1 downTo 0) {
            var sum = b[i]
            val rowOffset = i * cols
            for (k in i + 1 until n) {
                sum -= data[rowOffset + k] * x[k]
            }
            val diag = data[rowOffset + i]
            require(abs(diag) > 1e-15) { "Singular matrix in upper-triangular solve at diagonal index $i (value=$diag)" }
            x[i] = sum / diag
        }
        return x
    }

    companion object {
        fun zeros(rows: Int, cols: Int): Matrix = Matrix(rows, cols, DoubleArray(rows * cols))

        fun identity(n: Int): Matrix {
            val m = Matrix(n, n)
            for (i in 0 until n) {
                m[i, i] = 1.0
            }
            return m
        }

        fun diagonal(values: DoubleArray): Matrix {
            val n = values.size
            val m = Matrix(n, n)
            for (i in 0 until n) {
                m[i, i] = values[i]
            }
            return m
        }

        fun fromRowVectors(rows: Array<DoubleArray>): Matrix {
            val r = rows.size
            if (r == 0) return Matrix(0, 0)
            val c = rows[0].size
            val m = Matrix(r, c)
            for (i in 0 until r) {
                m.setRow(i, rows[i])
            }
            return m
        }
    }
}
