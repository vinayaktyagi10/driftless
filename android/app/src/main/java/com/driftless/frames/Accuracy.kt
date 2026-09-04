package com.driftless.frames

/**
 * Android's reported accuracies -> the 1-sigma values `types.h` requires.
 *
 * The distinction that matters, and that is easy to get wrong in both
 * directions: Android quotes everything "at 68% confidence", but the horizontal
 * figure is a **2-D radial** quantity and the other two are **1-D**.
 *
 * - [android.location.Location.getAccuracy] is the radius of a circle
 *   containing 68% of the error. For a 2-D isotropic Gaussian the radius is
 *   Rayleigh-distributed, so r_68 = sigma * sqrt(-2 ln(1 - 0.6827)) —
 *   [HORIZONTAL_68_TO_SIGMA] below. Feeding r_68 in as sigma understates the
 *   real uncertainty by ~51%, the NIS gate then rejects honest fixes, and the
 *   track free-runs on the IMU.
 * - `getVerticalAccuracyMeters()` and `getSpeedAccuracyMetersPerSecond()` are
 *   one-dimensional at 68%, which **is** 1 sigma already. Dividing those as
 *   well would make the filter over-confident.
 */
object Accuracy {

    /**
     * Rayleigh 68.27% radius, in units of sigma: sqrt(-2 * ln(1 - 0.6826895)).
     *
     * 0.6826895 rather than a rounded 0.68 because the intent is "the 2-D
     * radius equivalent to 1 sigma in 1-D", not "the 68% radius" — the rounded
     * figure gives 1.50959 and is wrong in the fourth digit.
     */
    const val HORIZONTAL_68_TO_SIGMA = 1.5151729

    /** 2-D radial 68% radius -> 1 sigma per axis. */
    fun horizontalSigmaM(accuracyMeters: Float): Double =
        accuracyMeters / HORIZONTAL_68_TO_SIGMA

    /** Already 1 sigma; present so the call sites read symmetrically and the
     *  reason is attached to the code rather than remembered. */
    fun verticalSigmaM(verticalAccuracyMeters: Float): Double =
        verticalAccuracyMeters.toDouble()

    /** Already 1 sigma, as above. */
    fun speedSigmaMps(speedAccuracyMps: Float): Double =
        speedAccuracyMps.toDouble()
}
