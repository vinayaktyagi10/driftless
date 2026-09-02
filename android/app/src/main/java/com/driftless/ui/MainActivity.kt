package com.driftless.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.net.Uri
import android.os.Bundle
import android.location.LocationManager
import android.provider.Settings
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.location.LocationManagerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.driftless.R
import com.driftless.databinding.ActivityMainBinding
import com.driftless.fusion.GnssFix
import com.driftless.sensors.GnssSampler
import com.driftless.sensors.ImuFrame
import com.driftless.sensors.ImuSampler
import com.driftless.settings.AppSettings
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Wires sensors -> fusion -> map-match -> map view. Owns the GNSS
 * deficit handler: switches the displayed position source between
 * GNSS-aided and dead-reckoning-only without a visible jump.
 *
 * Step 1 of the build order covers only the permission gate and the sensor
 * inventory below; the samplers, engine and map arrive in steps 2-5.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var settings: AppSettings
    private lateinit var imuSampler: ImuSampler
    private lateinit var gnssSampler: GnssSampler

    private var samplingJob: Job? = null

    // Step-2 diagnostics. Written from collector coroutines on the main
    // dispatcher and read by the ticker, so no synchronisation is needed.
    private var imuCount = 0L
    private var imuCountAtLastTick = 0L
    private var lastTickNanos = 0L
    private var measuredHz = 0.0
    private var latestFrame: ImuFrame? = null
    private var gnssCount = 0L
    private var latestFix: GnssFix? = null

    /**
     * The four states that actually need different UI. Android collapses
     * "never asked" and "permanently denied" into the same API answers, so
     * [AppSettings.hasRequestedLocation] is what separates them.
     */
    private enum class LocationAccess {
        Granted,
        /**
         * Permission held, but Location is off for the whole device. Android
         * reports this as a fully granted permission and then silently never
         * delivers a fix, which is the most misleading state of the five.
         */
        ServicesOff,
        /** Android 12+ lets the user grant approximate location only. */
        CoarseOnly,
        /** Not yet asked, or asked and dismissed — the prompt will still show. */
        Askable,
        /** Prompt will no longer show. Only app settings can fix this. */
        PermanentlyDenied,
    }

    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        // The result map is deliberately ignored: it reports what this one
        // dialog returned, whereas render() asks the system what the app
        // actually holds right now. Those differ if the user changed the
        // setting in another screen, and the system is the honest source.
        settings.hasRequestedLocation = true
        render()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        settings = AppSettings(this)
        imuSampler = ImuSampler(this)
        gnssSampler = GnssSampler(this)

        binding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        // TODO(step 3-5): feed these into the fusion engine and render the
        // result on the osmdroid MapView.
    }

    override fun onResume() {
        super.onResume()
        // Re-checked on every resume rather than only in onCreate, because the
        // user can revoke or grant the permission in system settings while the
        // activity is merely stopped, and Android does not restart us for it.
        render()
    }

    private fun locationAccess(): LocationAccess {
        val fine = ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        if (fine) {
            // Checked here rather than trusted: a granted permission says only
            // that this app may use location, not that the device will produce
            // any. With the master switch off the samplers run, the UI reports
            // "sensors ready", and no fix ever arrives.
            val manager = getSystemService(LOCATION_SERVICE) as LocationManager
            return if (LocationManagerCompat.isLocationEnabled(manager)) {
                LocationAccess.Granted
            } else {
                LocationAccess.ServicesOff
            }
        }

        val coarse = ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        if (coarse) return LocationAccess.CoarseOnly

        val canAskAgain = shouldShowRequestPermissionRationale(
            Manifest.permission.ACCESS_FINE_LOCATION
        )
        return when {
            canAskAgain -> LocationAccess.Askable
            !settings.hasRequestedLocation -> LocationAccess.Askable
            else -> LocationAccess.PermanentlyDenied
        }
    }

    private fun render() {
        renderSensorInventory()

        when (locationAccess()) {
            LocationAccess.Granted -> {
                binding.statusText.setText(R.string.perm_status_granted)
                binding.actionButton.visibility = android.view.View.GONE
                startSampling()
            }

            LocationAccess.ServicesOff -> {
                // The IMU is still worth running — dead reckoning is the whole
                // point and it needs no fix — so sampling starts anyway and the
                // banner explains why the GNSS half of the readout stays empty.
                binding.statusText.setText(R.string.perm_status_services_off)
                showAction(R.string.action_open_location_settings) {
                    startActivity(Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS))
                }
                startSampling()
            }

            LocationAccess.CoarseOnly -> {
                // Not treated as a soft success. Approximate location is
                // kilometre-scale, and the whole point of the app is metre-scale
                // drift, so proceeding on it would produce a demo that looks
                // like it works and is meaningless.
                binding.statusText.setText(R.string.perm_status_coarse_only)
                showAction(R.string.action_open_settings) { openAppSettings() }
            }

            LocationAccess.Askable -> {
                binding.statusText.setText(
                    if (settings.hasRequestedLocation) R.string.perm_status_rationale
                    else R.string.perm_status_needed
                )
                showAction(R.string.action_grant) { requestLocation() }
            }

            LocationAccess.PermanentlyDenied -> {
                binding.statusText.setText(R.string.perm_status_permanently_denied)
                showAction(R.string.action_open_settings) { openAppSettings() }
            }
        }
    }

    /**
     * Starts the sampler collectors. Idempotent — [render] runs on every resume
     * and would otherwise stack a second set of collectors on the sensors.
     *
     * `repeatOnLifecycle(STARTED)` is what actually releases the sensors and
     * the GNSS request when the app is backgrounded; without it the IMU keeps
     * running at 200 Hz behind a locked screen.
     */
    private fun startSampling() {
        if (samplingJob?.isActive == true) return

        samplingJob = lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    imuSampler.frames(settings.sensorDelay).collect { frame ->
                        imuCount++
                        latestFrame = frame
                    }
                }
                launch {
                    gnssSampler.fixes(settings.gnssIntervalMillis).collect { fix ->
                        gnssCount++
                        latestFix = fix
                    }
                }
                launch { tickDiagnostics() }
            }
        }
    }

    /**
     * Repaints the readout twice a second. The IMU is sampled far faster than
     * a display can show, and rendering per sample would make the UI thread the
     * bottleneck that causes the dropped samples it is meant to reveal.
     */
    private suspend fun tickDiagnostics() {
        lastTickNanos = System.nanoTime()
        while (true) {
            delay(500)

            val now = System.nanoTime()
            val elapsedSec = (now - lastTickNanos) / 1e9
            if (elapsedSec > 0) {
                measuredHz = (imuCount - imuCountAtLastTick) / elapsedSec
            }
            imuCountAtLastTick = imuCount
            lastTickNanos = now

            val text = diagnosticsText()
            binding.diagnosticsText.text = text
            // Mirrored to logcat so the readout can be verified over adb without
            // depending on a screenshot — the phone blanks its display for
            // reasons that have nothing to do with this app.
            Log.d(DIAG_TAG, text.replace('\n', '|'))
        }
    }

    private fun diagnosticsText(): String = buildString {
        val f = latestFrame
        appendLine("IMU   ${"%.0f".format(Locale.US, measuredHz)} Hz   n=$imuCount")
        appendLine("      dropped=${imuSampler.droppedSamples.get()}")
        if (f == null) {
            appendLine("      waiting for accel+gyro…")
        } else {
            appendLine("  accel FRD  ${vec(f.accel)}  m/s^2")
            appendLine("  gyro  FRD  ${vec(f.gyro)}  rad/s")
            appendLine("  mag        ${f.mag?.let { vec(it) } ?: "absent"}")
            appendLine("  gravity    ${f.gravity?.let { vec(it) } ?: "absent"}")
            appendLine("  t_mono     ${f.timestampNanos}")
        }
        appendLine()

        val fix = latestFix
        appendLine("GNSS  n=$gnssCount   sats=${gnssSampler.satellitesUsed}")
        if (fix == null) {
            appendLine("      waiting for first fix…")
        } else {
            val p = fix.position
            appendLine(
                "  NED  N=%.1f E=%.1f D=%.1f m".format(
                    Locale.US, p.north, p.east, p.down,
                )
            )
            appendLine(
                "  sigma_h=%.2f m  sigma_v=%.2f m".format(
                    Locale.US, fix.horizontalAccuracyM, fix.verticalAccuracyM,
                )
            )
            appendLine(
                if (fix.hasVelocity) {
                    "  vel  N=%.2f E=%.2f m/s  sigma=%.2f".format(
                        Locale.US,
                        fix.velocityNed.x, fix.velocityNed.y, fix.speedAccuracyMps,
                    )
                } else {
                    "  vel  none (standstill or no doppler)"
                }
            )
            appendLine("  t_mono     ${fix.timestampNanos}")

            // The check that matters most and is invisible in any single value:
            // both clocks must be the same base. A large gap here means one of
            // them is UTC epoch and the filter is about to be handed a
            // nonsensical dt.
            if (f != null) {
                val skewMs = (f.timestampNanos - fix.timestampNanos) / 1e6
                appendLine("  clock skew IMU-GNSS  %.0f ms".format(Locale.US, skewMs))
            }
        }
        gnssSampler.frame?.let {
            appendLine(
                "  anchor %.6f, %.6f".format(Locale.US, it.originLatDeg, it.originLonDeg)
            )
        }
    }

    private companion object {
        const val DIAG_TAG = "DriftlessDiag"
    }

    private fun vec(v: FloatArray) =
        "[%+7.3f %+7.3f %+7.3f]".format(Locale.US, v[0], v[1], v[2])

    private fun showAction(labelRes: Int, onClick: () -> Unit) {
        binding.actionButton.visibility = android.view.View.VISIBLE
        binding.actionButton.setText(labelRes)
        binding.actionButton.setOnClickListener { onClick() }
    }

    private fun requestLocation() {
        // Both are requested together on purpose: asking for FINE alone still
        // shows the user a Precise/Approximate choice on Android 12+, and
        // omitting COARSE means an approximate grant leaves the app with
        // nothing at all rather than with something we can report.
        requestPermissions.launch(
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION,
            )
        )
    }

    private fun openAppSettings() {
        startActivity(
            Intent(
                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.fromParts("package", packageName, null)
            )
        )
    }

    /**
     * Reports which inertial sensors this handset actually has. The manifest
     * marks all of them `required="false"` so the app installs anywhere, which
     * makes this the only place a missing gyroscope becomes visible before the
     * fusion engine quietly produces nonsense from it.
     */
    private fun renderSensorInventory() {
        val manager = getSystemService(SENSOR_SERVICE) as SensorManager
        fun present(type: Int) = getString(
            if (manager.getDefaultSensor(type) != null) R.string.sensor_present
            else R.string.sensor_absent
        )

        binding.sensorText.text = getString(
            R.string.sensor_availability_fmt,
            present(Sensor.TYPE_ACCELEROMETER),
            present(Sensor.TYPE_GYROSCOPE),
            present(Sensor.TYPE_MAGNETIC_FIELD),
            // TYPE_GRAVITY feeds grav_x/y/z, three of the velocity model's
            // fourteen input channels. Where it is absent the sampler has to
            // low-pass the accelerometer instead, exactly as training does.
            present(Sensor.TYPE_GRAVITY),
        )
    }
}
