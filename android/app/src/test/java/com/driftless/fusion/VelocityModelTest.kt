package com.driftless.fusion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class VelocityModelTest {

    @Test
    fun testGyroChannelSlotsMapping() {
        val model = VelocityModel()
        val accel = Vec3(0.1, 0.2, 9.8)
        val gyro = Vec3(0.123, 0.456, 0.789) // gx=0.123, gy=0.456, gz=0.789

        model.addSample(accel = accel, gyro = gyro)

        val frame = model.getLatestFrame()
        assertNotNull(frame)
        val f = frame!!

        // Slot 0..2: acc_x, acc_y, acc_z
        assertEquals(0.1f, f[0], 1e-4f)
        assertEquals(0.2f, f[1], 1e-4f)
        assertEquals(9.8f, f[2], 1e-4f)

        // Slot 3..5: augment.GYRO_XYZ_IDX is [3, 5, 4]
        // Physical X -> Slot 3 ("gyro_yaw" in dataset)
        // Physical Z -> Slot 4 ("gyro_pitch", vertical axis in dataset)
        // Physical Y -> Slot 5 ("gyro_roll")
        assertEquals("Physical X should be in slot 3", 0.123f, f[3], 1e-4f)
        assertEquals("Physical Z should be in slot 4", 0.789f, f[4], 1e-4f)
        assertEquals("Physical Y should be in slot 5", 0.456f, f[5], 1e-4f)
    }

    @Test
    fun testIsReadyRequiresFullWindow() {
        val model = VelocityModel()
        val accel = Vec3(0.0, 0.0, 9.80665)
        val gyro = Vec3(0.0, 0.0, 0.0)

        for (i in 0 until 10) {
            model.addSample(accel, gyro)
        }
        // Should NOT be ready at 10 samples (needs full 80 samples for 8s context)
        assertFalse("Model should not be ready at 10 samples", model.isReady)

        for (i in 10 until 79) {
            model.addSample(accel, gyro)
        }
        assertFalse("Model should not be ready at 79 samples", model.isReady)

        model.addSample(accel, gyro) // 80th sample
        assertTrue("Model should be ready at 80 samples", model.isReady)
    }

    @Test
    fun testStationaryVarianceRuleTrueStandstill() {
        val model = VelocityModel()
        val accel = Vec3(0.0, 0.0, 9.80665)
        val gyro = Vec3(0.0, 0.0, 0.0)

        // Need at least 20 samples (STATIONARY_WINDOW_SIZE)
        for (i in 0 until 25) {
            model.addSample(accel, gyro)
        }

        assertTrue("True standstill on desk must be classified as stationary", model.isStationary())
    }

    @Test
    fun testStationaryVarianceRuleCruisingNotClassifiedAsStandstill() {
        val model = VelocityModel()
        val rng = Random(42)

        // Cruising straight at 37 km/h:
        // |acc| is near g (~9.81 m/s^2), yaw rate is near zero,
        // but road vibrations cause variance: std(acc_norm) > 0.15, std(gyro) > 0.01
        for (i in 0 until 30) {
            val roadVibrationZ = rng.nextDouble(-0.5, 0.5) // std ~ 0.28
            val roadVibrationX = rng.nextDouble(-0.2, 0.2)
            val roadJitterGyro = rng.nextDouble(-0.03, 0.03) // std ~ 0.017

            val accel = Vec3(roadVibrationX, 0.0, 9.80665 + roadVibrationZ)
            val gyro = Vec3(roadJitterGyro, 0.0, 0.0)
            model.addSample(accel, gyro)
        }

        assertFalse("Cruising car with road vibrations must NOT be classified as stationary", model.isStationary())
    }
}
