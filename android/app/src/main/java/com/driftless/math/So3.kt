package com.driftless.math

import com.driftless.fusion.Quaternion
import com.driftless.fusion.Vec3
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

// --- Vector 3D Extensions & Operators ---

operator fun Vec3.plus(other: Vec3): Vec3 = Vec3(x + other.x, y + other.y, z + other.z)
operator fun Vec3.minus(other: Vec3): Vec3 = Vec3(x - other.x, y - other.y, z - other.z)
operator fun Vec3.unaryMinus(): Vec3 = Vec3(-x, -y, -z)
operator fun Vec3.times(scalar: Double): Vec3 = Vec3(x * scalar, y * scalar, z * scalar)
operator fun Double.times(v: Vec3): Vec3 = Vec3(this * v.x, this * v.y, this * v.z)
operator fun Vec3.div(scalar: Double): Vec3 = Vec3(x / scalar, y / scalar, z / scalar)

fun Vec3.dot(other: Vec3): Double = x * other.x + y * other.y + z * other.z
fun Vec3.cross(other: Vec3): Vec3 = Vec3(
    y * other.z - z * other.y,
    z * other.x - x * other.z,
    x * other.y - y * other.x
)

fun Vec3.norm(): Double = sqrt(x * x + y * y + z * z)
fun Vec3.squaredNorm(): Double = x * x + y * y + z * z

fun Vec3.normalized(): Vec3 {
    val n = norm()
    return if (n > 0.0) this / n else Vec3(1.0, 0.0, 0.0)
}

fun Vec3.toDoubleArray(): DoubleArray = doubleArrayOf(x, y, z)

fun DoubleArray.toVec3(startIndex: Int = 0): Vec3 =
    Vec3(this[startIndex], this[startIndex + 1], this[startIndex + 2])

// --- Quaternion Extensions & Operators ---

fun Quaternion.conjugate(): Quaternion = Quaternion(w, -x, -y, -z)

fun Quaternion.norm(): Double = sqrt(w * w + x * x + y * y + z * z)

fun Quaternion.normalized(): Quaternion {
    val n = norm()
    return if (n > 0.0) Quaternion(w / n, x / n, y / n, z / n) else Quaternion(1.0, 0.0, 0.0, 0.0)
}

/** Hamilton product: q1 * q2 */
operator fun Quaternion.times(other: Quaternion): Quaternion = Quaternion(
    w = w * other.w - x * other.x - y * other.y - z * other.z,
    x = w * other.x + x * other.w + y * other.z - z * other.y,
    y = w * other.y - x * other.z + y * other.w + z * other.x,
    z = w * other.z + x * other.y - y * other.x + z * other.w
)

/**
 * Rotates a 3D vector v by this unit quaternion: q * (0, v) * q^*
 */
fun Quaternion.rotate(v: Vec3): Vec3 {
    // Optimized Rodrigues / quaternion vector rotation:
    // v' = v + 2 * q_vec x (q_vec x v + q_w * v)
    val qv = Vec3(x, y, z)
    val t = 2.0 * qv.cross(v)
    return v + (w * t) + qv.cross(t)
}

operator fun Quaternion.times(v: Vec3): Vec3 = rotate(v)

// --- SO(3) Lie Group & Lie Algebra Operations ---

object So3 {
    private const val SMALL_ANGLE = 1e-8

    /**
     * Exponential map: a rotation vector (axis * angle, radians) to the unit
     * quaternion representing that rotation.
     */
    fun expMap(rotationVector: Vec3): Quaternion {
        val theta = rotationVector.norm()
        if (theta < SMALL_ANGLE) {
            // cos(t/2) ~ 1 - t^2/8, and sin(t/2)/t ~ 1/2 - t^2/48.
            val factor = 0.5 - theta * theta / 48.0
            val vx = rotationVector.x * factor
            val vy = rotationVector.y * factor
            val vz = rotationVector.z * factor
            return Quaternion(1.0 - theta * theta / 8.0, vx, vy, vz).normalized()
        }
        val half = 0.5 * theta
        val factor = sin(half) / theta
        return Quaternion(cos(half), rotationVector.x * factor, rotationVector.y * factor, rotationVector.z * factor)
    }

    /**
     * Logarithmic map: the inverse of expMap, returning the rotation vector with
     * the smaller of the two equivalent magnitudes (|phi| <= pi).
     */
    fun logMap(q: Quaternion): Vec3 {
        var n = q.normalized()
        if (n.w < 0.0) {
            n = Quaternion(-n.w, -n.x, -n.y, -n.z)
        }
        val vec = Vec3(n.x, n.y, n.z)
        val sinHalf = vec.norm()
        if (sinHalf < SMALL_ANGLE) {
            return (2.0 / n.w) * vec
        }
        val theta = 2.0 * atan2(sinHalf, n.w)
        return (theta / sinHalf) * vec
    }

    /**
     * Manifold addition: apply a local (body-frame) rotation-vector perturbation
     * to an orientation.
     */
    fun boxPlus(q: Quaternion, delta: Vec3): Quaternion = (q * expMap(delta)).normalized()

    /**
     * Manifold subtraction: the local rotation vector taking `reference` to `q`,
     * i.e. the delta d such that boxPlus(reference, d) == q.
     */
    fun boxMinus(q: Quaternion, reference: Quaternion): Vec3 = logMap(reference.conjugate() * q)

    /**
     * Computes the shortest arc rotation quaternion taking unit vector `from` to unit vector `to`.
     */
    fun fromTwoVectors(from: Vec3, to: Vec3): Quaternion {
        val u = from.normalized()
        val v = to.normalized()
        val d = u.dot(v)
        if (d >= 1.0 - 1e-8) {
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        }
        if (d <= -1.0 + 1e-8) {
            val axis = if (kotlin.math.abs(u.x) < 0.9) Vec3(0.0, 1.0, 0.0).cross(u).normalized() else Vec3(0.0, 0.0, 1.0).cross(u).normalized()
            return Quaternion(0.0, axis.x, axis.y, axis.z).normalized()
        }
        val k = u.cross(v)
        val s = kotlin.math.sqrt((1.0 + d) * 2.0)
        val invS = 1.0 / s
        return Quaternion(0.5 * s, k.x * invS, k.y * invS, k.z * invS).normalized()
    }
}
