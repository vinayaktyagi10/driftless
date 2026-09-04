package com.driftless.frames

/**
 * SensorManager's device axes -> the engine's body **FRD** axes.
 *
 * Android's device frame, with the phone held upright facing you:
 *   x = right across the screen, y = up the screen, z = out of the screen.
 * `types.h` calls this "ENU-ish (x right, y forward, z up)", which is the same
 * thing once the phone is laid into a dashboard mount with its top edge
 * pointing at the windscreen.
 *
 * The engine wants FRD: x Forward, y Right, z Down. So:
 *
 *     frd.x =  device.y      (forward)
 *     frd.y =  device.x      (right)
 *     frd.z = -device.z      (down)
 *
 * That is a swap of two axes composed with one negation — determinant
 * (-1) * (-1) = +1, so it is a proper rotation and not a reflection. A
 * reflection here would silently mirror every trajectory, which is exactly the
 * failure that looks fine over ten metres and is unusable over two kilometres.
 *
 * ## The mounting assumption
 *
 * This mapping assumes the phone is **portrait, top edge forward**. Landscape,
 * or a mount tilted more than a few degrees, breaks the raw-channel half of the
 * velocity model's input (9 of its 14 channels are raw body axes) even though
 * the five derived channels are built to survive it. Role 03 measured that
 * degradation deliberately. There is no runtime mount calibration yet — if the
 * demo mount is not portrait-forward, this is the constant to revisit, and it
 * should be revisited here rather than patched downstream.
 */
object AxisConvention {

    /** Allocating form. Fine at GNSS rates; prefer [deviceToFrdInPlace] at 200 Hz. */
    fun deviceToFrd(device: FloatArray): FloatArray =
        floatArrayOf(device[1], device[0], -device[2])

    /**
     * In-place form, for the IMU path where this runs on every sample and the
     * array has just been copied out of a recycled `SensorEvent.values`.
     */
    fun deviceToFrdInPlace(v: FloatArray) {
        val forward = v[1]
        v[1] = v[0]
        v[0] = forward
        v[2] = -v[2]
    }
}
