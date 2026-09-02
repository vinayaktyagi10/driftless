package com.driftless.sensors

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssStatus
import android.location.Location
import android.location.LocationManager
import android.os.Handler
import android.os.HandlerThread
import androidx.annotation.RequiresPermission
import com.driftless.frames.Accuracy
import com.driftless.frames.LocalTangentFrame
import com.driftless.fusion.GnssFix
import com.driftless.fusion.Vec3
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlin.math.cos
import kotlin.math.sin

/**
 * FusedLocationProviderClient wrapper. Also surfaces GNSS status
 * (satellite count, signal strength) so the fusion engine can detect
 * degradation before a full outage, not just react to one.
 *
 * Like [ImuSampler], this is a conversion boundary as much as a data source.
 * Everything it emits is already in the engine's terms: local-tangent NED
 * metres, 1-sigma accuracies, and timestamps on the monotonic clock.
 */
class GnssSampler(context: Context) {

    private val appContext = context.applicationContext
    private val fused = LocationServices.getFusedLocationProviderClient(appContext)
    private val locationManager =
        appContext.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    /**
     * The navigation frame, anchored at the first fix this sampler sees.
     *
     * Null until then. The UI needs it to convert fused NED positions back to
     * lat/lon for the map, which is why it is exposed rather than kept private.
     */
    @Volatile
    var frame: LocalTangentFrame? = null
        private set

    /** Satellites used in the most recent fix, from [GnssStatus]. */
    @Volatile
    var satellitesUsed: Int = 0
        private set

    /**
     * Cold flow of GNSS fixes, already projected into the navigation frame.
     *
     * @param intervalMillis desired update interval; the provider treats it as
     *   a hint and may deliver faster or slower.
     */
    @RequiresPermission(Manifest.permission.ACCESS_FINE_LOCATION)
    @SuppressLint("MissingPermission")
    fun fixes(intervalMillis: Long): Flow<GnssFix> = callbackFlow {
        val thread = HandlerThread("gnss-sampler").apply { start() }
        val handler = Handler(thread.looper)

        // Satellite count comes from the raw GNSS status, not from the fused
        // provider, which reports no such thing. It is read alongside each fix
        // rather than emitted on its own — the fusion engine wants it as an
        // attribute of a fix, per types.h.
        val statusCallback = object : GnssStatus.Callback() {
            override fun onSatelliteStatusChanged(status: GnssStatus) {
                var used = 0
                for (i in 0 until status.satelliteCount) {
                    if (status.usedInFix(i)) used++
                }
                satellitesUsed = used
            }
        }
        locationManager.registerGnssStatusCallback(statusCallback, handler)

        val locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val location = result.lastLocation ?: return
                if (trySend(toFix(location)).isFailure) {
                    // Unlike a dropped IMU sample, a dropped fix is only a
                    // missed correction, not a hole in an integral — the filter
                    // free-runs slightly longer and recovers on the next one.
                    // So this is not counted as a defect.
                }
            }
        }

        val request = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            intervalMillis,
        )
            // Accept anything the provider can give us early; the filter is
            // better served by more fixes than by an evenly spaced few.
            .setMinUpdateIntervalMillis(intervalMillis / 2)
            .setWaitForAccurateLocation(false)
            .build()

        fused.requestLocationUpdates(request, locationCallback, thread.looper)

        awaitClose {
            fused.removeLocationUpdates(locationCallback)
            locationManager.unregisterGnssStatusCallback(statusCallback)
            thread.quitSafely()
        }
    }.buffer(BUFFER_CAPACITY)

    /**
     * The conversion boundary itself: an Android [Location] in geodetic
     * degrees, 68%-radius accuracy and mixed clocks, out as a [GnssFix] in NED
     * metres, 1 sigma, on the monotonic clock.
     */
    private fun toFix(location: Location): GnssFix {
        // First fix anchors the navigation frame. Everything downstream is
        // relative to this point, so it is set exactly once.
        val f = frame ?: LocalTangentFrame
            .anchoredAt(location.latitude, location.longitude, location.altitude)
            .also { frame = it }

        val position = f.toNed(location.latitude, location.longitude, location.altitude)

        // Doppler velocity is opt-in per fix. Bearing is meaningless at a
        // standstill — the receiver reports whatever it last had — so below the
        // threshold the fix carries position only and the filter keeps its own
        // velocity estimate rather than being told a stale heading.
        // Two gates, because one is not enough. The absolute floor rejects the
        // obvious standstill case; the signal-to-noise gate rejects the case a
        // fixed threshold cannot see -- a receiver reporting 0.62 m/s with a
        // 0.64 m/s accuracy is reporting noise, and clears any threshold set
        // low enough to admit walking pace. Observed on a stationary phone
        // indoors at sigma_h 21 m, which is precisely the degraded-GNSS regime
        // this app exists to survive.
        val speedIsSignificant = !location.hasSpeedAccuracy() ||
            location.speed >= VELOCITY_MIN_SNR * location.speedAccuracyMetersPerSecond
        val usableVelocity =
            location.hasSpeed() && location.hasBearing() &&
                location.speed >= MIN_SPEED_FOR_BEARING && speedIsSignificant
        val velocityNed = if (usableVelocity) {
            val bearingRad = Math.toRadians(location.bearing.toDouble())
            Vec3(
                x = location.speed * cos(bearingRad),
                y = location.speed * sin(bearingRad),
                // Vertical rate is not reported; the filter models it.
                z = 0.0,
            )
        } else {
            Vec3()
        }

        return GnssFix(
            // The monotonic base, matching SensorEvent.timestamp. Using
            // location.time here instead — UTC epoch millis — is the mistake
            // that hands the filter a dt measured in decades.
            timestampNanos = location.elapsedRealtimeNanos,
            position = position,
            velocityNed = velocityNed,
            hasVelocity = usableVelocity,
            speedAccuracyMps = if (location.hasSpeedAccuracy()) {
                Accuracy.speedSigmaMps(location.speedAccuracyMetersPerSecond)
            } else {
                0.0
            },
            // The only one of the three that needs converting: it is a 2-D
            // radial 68% figure, where the other two are 1-D and already sigma.
            horizontalAccuracyM = Accuracy.horizontalSigmaM(location.accuracy),
            verticalAccuracyM = if (location.hasVerticalAccuracy()) {
                Accuracy.verticalSigmaM(location.verticalAccuracyMeters)
            } else {
                0.0
            },
            satellitesUsed = satellitesUsed,
        )
    }

    private companion object {
        /** m/s below which a reported bearing is not trustworthy. */
        const val MIN_SPEED_FOR_BEARING = 0.5f

        /**
         * How many times its own 1-sigma accuracy a reported speed must exceed
         * before the bearing derived from it is believed. Two sigma is the
         * usual "distinguishable from zero" bar and costs nothing at driving
         * speeds, where speed exceeds sigma by an order of magnitude.
         */
        const val VELOCITY_MIN_SNR = 2.0f

        /** Fixes arrive at ~1 Hz; this is minutes of headroom. */
        const val BUFFER_CAPACITY = 64
    }
}
