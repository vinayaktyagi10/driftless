package com.driftless.math

import com.driftless.fusion.Quaternion
import com.driftless.fusion.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

class So3Test {

    @Test
    fun testExpLogRoundTripAcrossMagnitudes() {
        val rng = Random(20260906)
        val magnitudes = doubleArrayOf(1e-12, 1e-9, 1e-6, 1e-3, 0.1, 1.0, 3.0, PI - 1e-6)
        for (mag in magnitudes) {
            for (trial in 0 until 25) {
                val axis = Vec3(rng.nextGaussian(), rng.nextGaussian(), rng.nextGaussian())
                if (axis.norm() < 1e-9) continue
                val phi = axis.normalized() * mag
                val recovered = So3.logMap(So3.expMap(phi))
                val err = (recovered - phi).norm()
                assertTrue("Failed for magnitude $mag: err=$err", err < 1e-9 * max(1.0, mag))
            }
        }
    }

    @Test
    fun testExpMapProducesUnitQuaternions() {
        val rng = Random(20260907)
        for (trial in 0 until 500) {
            val phi = Vec3(rng.nextGaussian() * 2.0, rng.nextGaussian() * 2.0, rng.nextGaussian() * 2.0)
            val q = So3.expMap(phi)
            assertEquals(1.0, q.norm(), 1e-12)
        }
    }

    @Test
    fun testSmallAngleBranchAgreesWithTrigonometricBranch() {
        val axis = Vec3(1.0, -2.0, 0.5).normalized()
        val testAngles = doubleArrayOf(9e-9, 1.1e-8)
        for (t in testAngles) {
            val q = So3.expMap(axis * t)
            // Rotate canonical test vectors and compare with Rodrigues formula
            for (v in arrayOf(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0))) {
                val rotated = q.rotate(v)
                val skewV = axis.cross(v)
                val skew2V = axis.cross(skewV)
                val expected = v + (sin(t) * skewV) + ((1.0 - cos(t)) * skew2V)
                val diff = (rotated - expected).norm()
                assertTrue("Mismatch at t=$t: diff=$diff", diff < 1e-14)
            }
        }
    }

    @Test
    fun testBoxPlusAndBoxMinusAreInverses() {
        val rng = Random(20260909)
        for (trial in 0 until 300) {
            val q = So3.expMap(Vec3(rng.nextGaussian(), rng.nextGaussian(), rng.nextGaussian()))
            val delta = Vec3(rng.nextGaussian(), rng.nextGaussian(), rng.nextGaussian()) * 0.01
            val composed = So3.boxPlus(q, delta)
            val recovered = So3.boxMinus(composed, q)
            val err = (recovered - delta).norm()
            assertTrue("trial $trial failed with err=$err", err < 1e-12)
        }
    }

    @Test
    fun testBoxPlusIsALocalPerturbation() {
        val q = So3.expMap(Vec3(0.0, 0.0, PI / 2.0)) // 90 deg yaw
        val delta = Vec3(0.1, 0.0, 0.0)              // roll, body frame
        val local = So3.boxPlus(q, delta)
        val directLocal = (q * So3.expMap(delta)).normalized()
        assertEquals(directLocal.w, local.w, 1e-15)
        assertEquals(directLocal.x, local.x, 1e-15)
        assertEquals(directLocal.y, local.y, 1e-15)
        assertEquals(directLocal.z, local.z, 1e-15)

        val navFrame = (So3.expMap(delta) * q).normalized()
        val diff = abs(local.w - navFrame.w) + abs(local.x - navFrame.x) + abs(local.y - navFrame.y) + abs(local.z - navFrame.z)
        assertTrue(diff > 1e-3)
    }

    @Test
    fun testLogMapReturnsShortWayRoundForNegatedQuaternion() {
        val phi = Vec3(0.3, -0.7, 1.1)
        val q = So3.expMap(phi)
        val negated = Quaternion(-q.w, -q.x, -q.y, -q.z)
        val recovered = So3.logMap(negated)
        val err = (recovered - phi).norm()
        assertTrue(err < 1e-12)
    }
}
