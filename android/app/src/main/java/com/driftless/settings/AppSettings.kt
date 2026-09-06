package com.driftless.settings

import android.content.Context
import android.content.SharedPreferences
import android.hardware.SensorManager
import androidx.preference.PreferenceManager

/**
 * Typed view over the preference screen, so the rest of the app never touches
 * raw preference keys or parses `ListPreference`'s string values by hand.
 *
 * `ListPreference` stores everything as a String even when the value is an
 * integer, which is the usual source of a `ClassCastException` on second
 * launch — reading these back through here keeps that in one place.
 */
class AppSettings(context: Context) {

    private val prefs: SharedPreferences =
        PreferenceManager.getDefaultSharedPreferences(context.applicationContext)

    /**
     * One of the `SensorManager.SENSOR_DELAY_*` constants. Default is FASTEST:
     * the fusion engine's predict step is tuned for roughly 200 Hz, and the
     * slower options exist for battery testing rather than because they are a
     * reasonable default.
     */
    val sensorDelay: Int
        get() = prefs.getString(KEY_SENSOR_RATE, null)?.toIntOrNull()
            ?: SensorManager.SENSOR_DELAY_FASTEST

    /** `LocationRequest` interval in milliseconds. */
    val gnssIntervalMillis: Long
        get() = prefs.getString(KEY_GNSS_INTERVAL_MS, null)?.toLongOrNull()
            ?: DEFAULT_GNSS_INTERVAL_MS

    /**
     * Whether the location permission has ever been requested.
     *
     * Android gives the same answer from `shouldShowRequestPermissionRationale`
     * for "never asked" and "permanently denied" — both false — so the two are
     * indistinguishable without remembering that we asked. This flag is that
     * memory, and it is why the UI can offer "Grant" the first time and "Open
     * app settings" after a permanent denial.
     */
    var hasRequestedLocation: Boolean
        get() = prefs.getBoolean(KEY_REQUESTED_LOCATION, false)
        set(value) = prefs.edit().putBoolean(KEY_REQUESTED_LOCATION, value).apply()

    companion object {
        const val KEY_SENSOR_RATE = "sensor_rate"
        const val KEY_GNSS_INTERVAL_MS = "gnss_interval_ms"
        private const val KEY_REQUESTED_LOCATION = "has_requested_location"

        const val DEFAULT_GNSS_INTERVAL_MS = 1000L
    }
}
