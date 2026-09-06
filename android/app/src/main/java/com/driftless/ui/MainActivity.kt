package com.driftless.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import android.view.View
import android.view.WindowManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.location.LocationManagerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.driftless.R
import com.driftless.databinding.ActivityMainBinding
import com.driftless.fusion.FusedPosition
import com.driftless.fusion.GnssFix
import com.driftless.fusion.ImuNoiseParams
import com.driftless.fusion.UkfFusionEngine
import com.driftless.fusion.Vec3
import com.driftless.fusion.VelocityModel
import com.driftless.math.norm
import com.driftless.logging.TrackLogger
import com.driftless.mapmatch.HmmMapMatcher
import com.driftless.mapmatch.OsmReader
import com.driftless.mapmatch.RoadGraph
import com.driftless.sensors.GnssSampler
import com.driftless.sensors.ImuFrame
import com.driftless.sensors.ImuSampler
import com.driftless.settings.AppSettings
import kotlinx.coroutines.Job
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.osmdroid.events.DelayedMapListener
import org.osmdroid.events.MapListener
import org.osmdroid.events.ScrollEvent
import org.osmdroid.events.ZoomEvent
import java.util.Locale
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max

/**
 * Wires sensors -> UKF fusion engine -> TFLite velocity model -> HMM map matcher -> map view.
 *
 * Implements the full Role 02 Position Fusion Pipeline:
 * - 15-state Error-State Square-Root UKF with SO(3) Lie algebra kinematics.
 * - TFLite 14-channel Learned Forward Velocity Model.
 * - Newson & Krumm HMM Map Matcher against offline OSM road graph.
 * - Live GPS Outage / Blackout Simulation Mode with live drift characterization.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var settings: AppSettings
    private lateinit var imuSampler: ImuSampler
    private lateinit var gnssSampler: GnssSampler
    private lateinit var engine: UkfFusionEngine
    private lateinit var velocityModel: VelocityModel
    private var roadGraph: RoadGraph? = null
    private var hmmMatcher: HmmMapMatcher? = null
    private lateinit var track: TrackRenderer
    private lateinit var logger: TrackLogger

    private var samplingJob: Job? = null

    // Live diagnostics
    private var imuCount = 0L
    private var imuCountAtLastTick = 0L
    private var lastTickNanos = 0L
    private var measuredHz = 0.0
    private var latestFrame: ImuFrame? = null
    private var gnssCount = 0L
    private var latestFix: GnssFix? = null
    private var diagnosticsExpanded = true
    private var anchorLogged = false
    private var lastFixRealtimeNanos = 0L
    private var lastVelocityModelNanos = 0L

    // Filter output
    private var fusedCount = 0L
    private var latestFused: FusedPosition? = null

    // Blackout simulation state
    private var isSimulatedBlackout = false
    private var blackoutStartNanos = 0L
    private var blackoutDistanceM = 0.0
    private var lastBlackoutFusedNanos = 0L

    private enum class LocationAccess {
        Granted,
        ServicesOff,
        CoarseOnly,
        Askable,
        PermanentlyDenied,
    }

    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        settings.hasRequestedLocation = true
        render()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.i(
            DIAG_TAG,
            "onCreate activity=${System.identityHashCode(this)} " +
                "recreated=${savedInstanceState != null}",
        )

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        settings = AppSettings(this)
        imuSampler = ImuSampler(this)
        gnssSampler = GnssSampler(this)

        // Instantiate UKF Position Fusion Engine with consumer MEMS IMU parameters
        engine = UkfFusionEngine(
            noise = ImuNoiseParams.consumerMems(),
            frameProvider = { gnssSampler.frame }
        )

        // Instantiate TFLite Learned Velocity Model
        velocityModel = VelocityModel(this)

        track = TrackRenderer(binding.mapView)
        logger = TrackLogger(this, lifecycleScope)

        binding.mapView.addMapListener(
            DelayedMapListener(
                object : MapListener {
                    override fun onScroll(event: ScrollEvent?): Boolean {
                        track.followEnabled = false
                        return false
                    }

                    override fun onZoom(event: ZoomEvent?): Boolean = false
                },
                MAP_GESTURE_DEBOUNCE_MS,
            )
        )

        binding.diagnosticsText.setOnClickListener { diagnosticsExpanded = false; renderDiagnostics() }
        binding.diagnosticsBadge.setOnClickListener { diagnosticsExpanded = true; renderDiagnostics() }

        binding.recentreButton.setOnClickListener { track.recentre() }
        binding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        binding.blackoutButton.setOnClickListener {
            isSimulatedBlackout = !isSimulatedBlackout
            if (isSimulatedBlackout) {
                blackoutStartNanos = SystemClock.elapsedRealtimeNanos()
                blackoutDistanceM = 0.0
                lastBlackoutFusedNanos = blackoutStartNanos
                binding.blackoutButton.text = getString(R.string.action_blackout_stop)
                binding.blackoutBanner.visibility = View.VISIBLE
                updateBlackoutBanner()
            } else {
                binding.blackoutButton.text = getString(R.string.action_blackout_start)
                binding.blackoutBanner.visibility = View.GONE
            }
        }
    }

    override fun onResume() {
        super.onResume()
        binding.mapView.onResume()
        render()
    }

    override fun onPause() {
        binding.mapView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        velocityModel.close()
        super.onDestroy()
    }

    private fun locationAccess(): LocationAccess {
        val fine = ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        if (fine) {
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
                binding.actionButton.visibility = View.GONE
                startSampling()
            }

            LocationAccess.ServicesOff -> {
                binding.statusText.setText(R.string.perm_status_services_off)
                showAction(R.string.action_open_location_settings) {
                    startActivity(Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS))
                }
                startSampling()
            }

            LocationAccess.CoarseOnly -> {
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
     * Starts the sensor collectors and fusion pipelines.
     */
    private fun startSampling() {
        if (samplingJob?.isActive == true) return

        samplingJob = lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                val file = logger.start(settings.sensorDelay, settings.gnssIntervalMillis)
                Log.i(DIAG_TAG, "logging to ${file?.absolutePath ?: "<unavailable>"}")

                try {
                    // 1. IMU Propagation Loop (200 - 500 Hz)
                    launch {
                        imuSampler.frames(settings.sensorDelay).collect { frame ->
                            imuCount++
                            latestFrame = frame

                            val sample = frame.toImuSample()
                            engine.predict(sample)

                            // Feed sample to TFLite Velocity Model context window at 10 Hz (100 ms).
                            // The model expects 80 samples over 8s of driving context; feeding at 400 Hz shrinks context to 0.2s.
                            val nowNanos = sample.timestampNanos
                            if (lastVelocityModelNanos == 0L || (nowNanos - lastVelocityModelNanos) >= 95_000_000L) {
                                val dtSec = if (lastVelocityModelNanos == 0L) 0.1 else (nowNanos - lastVelocityModelNanos) / 1e9
                                lastVelocityModelNanos = nowNanos
                                velocityModel.addSample(
                                    accel = Vec3(sample.accel[0].toDouble(), sample.accel[1].toDouble(), sample.accel[2].toDouble()),
                                    gyro = Vec3(sample.gyro[0].toDouble(), sample.gyro[1].toDouble(), sample.gyro[2].toDouble()),
                                    gravity = frame.gravity?.let { Vec3(it[0].toDouble(), it[1].toDouble(), it[2].toDouble()) },
                                    dtSeconds = dtSec,
                                )
                            }

                            logger.logImu(frame)
                        }
                    }

                    // 2. GNSS Correction Loop (1 - 5 Hz)
                    launch {
                        gnssSampler.fixes(settings.gnssIntervalMillis).collect { fix ->
                            gnssCount++
                            latestFix = fix
                            lastFixRealtimeNanos = SystemClock.elapsedRealtimeNanos()

                            // Pass fix to filter only if not in simulated blackout mode
                            if (!isSimulatedBlackout) {
                                engine.updateGnss(fix)
                            }

                            // Initialize Road Graph if tangent frame is established
                            val frame = gnssSampler.frame
                            if (frame != null) {
                                if (!anchorLogged) {
                                    logger.logAnchor(frame)
                                    anchorLogged = true
                                }
                                if (roadGraph == null) {
                                    try {
                                        val xml = assets.open("maps/grid.osm").bufferedReader().use { it.readText() }
                                        // Load with autoOffset = false so synthetic test fixture is not transplanted onto live real-world drives
                                        val osmNetwork = OsmReader.loadOsmXml(xml, frame, autoOffset = false)
                                        val graph = osmNetwork.graph
                                        if (graph.segmentCount > 0) {
                                            roadGraph = graph
                                            hmmMatcher = HmmMapMatcher(graph)
                                            Log.i(DIAG_TAG, "Initialized OSM RoadGraph with ${graph.segmentCount} segments")
                                        }
                                    } catch (e: Exception) {
                                        Log.w(DIAG_TAG, "No bundled map loaded: ${e.message}")
                                    }
                                }
                            }

                            logger.logFix(fix)
                        }
                    }

                    // 3. TFLite Velocity Model & NHC Aiding Loop (10 Hz)
                    launch {
                        while (true) {
                            try {
                                delay(100)
                                val fix = latestFix
                                val gnssAgeSec = if (fix != null) (SystemClock.elapsedRealtimeNanos() - lastFixRealtimeNanos) / 1e9 else 999.0
                                val gnssIsActive = !isSimulatedBlackout && fix != null && gnssAgeSec in 0.0..2.0
                                val isGnssStationary = gnssIsActive && (!fix!!.hasVelocity || fix.velocityNed.norm() < 0.3)

                                if (isGnssStationary) {
                                    // Active GNSS confirms device is stationary: strictly enforce zero velocity
                                    engine.updateZeroVelocity(0.05)
                                } else if (velocityModel.isReady) {
                                    val pred = velocityModel.predict()
                                    if (pred != null) {
                                        if (pred.speedMps <= 0.05f) {
                                            engine.updateZeroVelocity(0.05)
                                        } else {
                                            engine.updateVelocityModel(pred.speedMps.toDouble())
                                        }
                                    }
                                }
                                engine.updateNonHolonomic()
                            } catch (e: Exception) {
                                Log.w(DIAG_TAG, "Aiding loop tick error: ${e.message}")
                            }
                        }
                    }

                    // 4. Fused State & Map Rendering Loop (10 Hz)
                    launch {
                        engine.observePosition().collect { fused ->
                            fusedCount++
                            latestFused = fused

                            // HMM Map Matching update
                            val matcher = hmmMatcher
                            if (matcher != null) {
                                val state = engine.state()
                                val match = matcher.step(state.position)
                                if (match.matched) {
                                    engine.updateMapMatch(match)
                                }
                            }

                            track.addFusedPoint(fused)

                            // Accumulate blackout dead-reckoning metrics
                            if (isSimulatedBlackout) {
                                val now = SystemClock.elapsedRealtimeNanos()
                                val dtSec = if (lastBlackoutFusedNanos > 0L) (now - lastBlackoutFusedNanos) / 1e9 else 0.0
                                lastBlackoutFusedNanos = now
                                val speed = fused.speedMetersPerSec
                                if (speed > 0.1f && dtSec in 0.0..1.0) {
                                    blackoutDistanceM += speed * dtSec
                                }
                                updateBlackoutBanner()
                            }
                        }
                    }

                    // 5. Diagnostics UI Ticker (2 Hz)
                    launch { tickDiagnostics() }
                    awaitCancellation()
                } finally {
                    logger.stop()
                }
            }
        }
    }

    private fun updateBlackoutBanner() {
        if (!isSimulatedBlackout) return
        val blackoutSec = (SystemClock.elapsedRealtimeNanos() - blackoutStartNanos) / 1e9
        val s = engine.covarianceSqrt()
        val horizVar = s[0, 0] * s[0, 0] + s[0, 1] * s[0, 1] + s[1, 0] * s[1, 0] + s[1, 1] * s[1, 1]
        val sigmaH = kotlin.math.sqrt(max(horizVar, 0.01))
        val driftEstimateM = 2.0 * sigmaH
        val driftPct = if (blackoutDistanceM > 5.0) {
            (driftEstimateM / blackoutDistanceM) * 100.0
        } else {
            0.0
        }
        binding.blackoutBanner.text = String.format(
            Locale.US,
            getString(R.string.blackout_banner_fmt),
            blackoutSec,
            blackoutDistanceM,
            driftPct,
        )
    }

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
            renderDiagnostics()
            if (isSimulatedBlackout) {
                updateBlackoutBanner()
            }
            Log.d(DIAG_TAG, text.replace('\n', '|'))
        }
    }

    private fun renderDiagnostics() {
        binding.diagnosticsText.visibility =
            if (diagnosticsExpanded) View.VISIBLE else View.GONE
        binding.diagnosticsBadge.visibility =
            if (diagnosticsExpanded) View.GONE else View.VISIBLE

        binding.diagnosticsBadge.text = String.format(
            Locale.US,
            getString(R.string.diagnostics_badge_fmt),
            measuredHz.toInt(),
            gnssSampler.satellitesUsed,
            latestFix?.horizontalAccuracyM ?: Double.NaN,
            fixAgeSeconds(),
        )
    }

    private fun fixAgeSeconds(): Double =
        if (lastFixRealtimeNanos == 0L) -1.0
        else (SystemClock.elapsedRealtimeNanos() - lastFixRealtimeNanos) / 1e9

    private fun diagnosticsText(): String = buildString {
        val f = latestFrame
        appendLine("IMU   ${String.format(Locale.US, "%.0f", measuredHz)} Hz   n=$imuCount")
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
        appendLine(
            String.format(
                Locale.US,
                "GNSS  n=%d   sats=%d   age=%.1fs%s",
                gnssCount,
                gnssSampler.satellitesUsed,
                fixAgeSeconds(),
                if (isSimulatedBlackout) " [BLACKOUT SIMULATION]" else ""
            )
        )
        appendLine("      log dropped=${logger.droppedRecords.get()}")
        if (fix == null) {
            appendLine("      waiting for first fix…")
        } else {
            val p = fix.position
            appendLine(
                String.format(Locale.US, "  NED  N=%.1f E=%.1f D=%.1f m", p.north, p.east, p.down)
            )
            appendLine(
                String.format(
                    Locale.US,
                    "  sigma_h=%.2f m  sigma_v=%.2f m",
                    fix.horizontalAccuracyM,
                    fix.verticalAccuracyM,
                )
            )
            appendLine(
                if (fix.hasVelocity) {
                    String.format(
                        Locale.US,
                        "  vel  N=%.2f E=%.2f m/s  sigma=%.2f",
                        fix.velocityNed.x,
                        fix.velocityNed.y,
                        fix.speedAccuracyMps,
                    )
                } else {
                    "  vel  none (standstill or no doppler)"
                }
            )
        }
        gnssSampler.frame?.let {
            appendLine(
                String.format(Locale.US, "  anchor %.6f, %.6f", it.originLatDeg, it.originLonDeg)
            )
        }
        appendLine()

        // UKF Filter Diagnostics & Biases
        val diag = engine.diagnostics
        val state = engine.state()
        val s = engine.covarianceSqrt()
        val horizVar = s[0, 0] * s[0, 0] + s[0, 1] * s[0, 1] + s[1, 0] * s[1, 0] + s[1, 1] * s[1, 1]
        val vertVar = s[2, 2] * s[2, 2]
        val filterSigmaH = kotlin.math.sqrt(max(horizVar, 0.0001))
        val filterSigmaV = kotlin.math.sqrt(max(vertVar, 0.0001))

        appendLine("UKF 15-STATE SR-UKF")
        appendLine(
            String.format(
                Locale.US,
                "  updates: GNSS applied=%d (rej=%d)  NHC=%d  TFLite=%d  Map=%d",
                diag.gnssApplied,
                diag.gnssRejected,
                diag.nhcApplied,
                diag.velocityModelApplied,
                diag.mapMatchApplied,
            )
        )
        appendLine(
            String.format(
                Locale.US,
                "  bias_a=[%+.3f, %+.3f, %+.3f] m/s^2",
                state.accelBias.x,
                state.accelBias.y,
                state.accelBias.z,
            )
        )
        appendLine(
            String.format(
                Locale.US,
                "  bias_g=[%+.4f, %+.4f, %+.4f] deg/s",
                Math.toDegrees(state.gyroBias.x),
                Math.toDegrees(state.gyroBias.y),
                Math.toDegrees(state.gyroBias.z),
            )
        )
        appendLine(
            String.format(
                Locale.US,
                "  uncert: sigma_h=%.2f m  sigma_v=%.2f m",
                filterSigmaH,
                filterSigmaV,
            )
        )
        appendLine()

        val fused = latestFused
        appendLine("FUSED OUTPUT n=$fusedCount")
        if (fused == null) {
            appendLine("      waiting for engine output…")
        } else {
            appendLine(
                String.format(Locale.US, "  pos  %.6f, %.6f", fused.lat, fused.lon)
            )
            appendLine(
                String.format(
                    Locale.US,
                    "  hdg=%.0f deg  speed=%.2f m/s (%.1f km/h)",
                    fused.headingDegrees,
                    fused.speedMetersPerSec,
                    fused.speedMetersPerSec * 3.6f,
                )
            )
            val isGnssActive = !isSimulatedBlackout && latestFix != null && fixAgeSeconds() in 0.0..3.0
            appendLine(
                if (isGnssActive) {
                    String.format(
                        Locale.US,
                        "  mode=GNSS (conf=%.2f)",
                        fused.confidence,
                    )
                } else {
                    String.format(
                        Locale.US,
                        "  mode=DEAD_RECKONING (conf=%.2f, outage=%.1fs)",
                        fused.confidence,
                        if (isSimulatedBlackout) (SystemClock.elapsedRealtimeNanos() - blackoutStartNanos) / 1e9 else fixAgeSeconds(),
                    )
                }
            )
        }
    }

    private companion object {
        const val DIAG_TAG = "DriftlessDiag"
        const val MAP_GESTURE_DEBOUNCE_MS = 200L
    }

    private fun vec(v: FloatArray) =
        String.format(Locale.US, "[%+7.3f %+7.3f %+7.3f]", v[0], v[1], v[2])

    private fun showAction(labelRes: Int, onClick: () -> Unit) {
        binding.actionButton.visibility = View.VISIBLE
        binding.actionButton.setText(labelRes)
        binding.actionButton.setOnClickListener { onClick() }
    }

    private fun requestLocation() {
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
            present(Sensor.TYPE_GRAVITY),
        )
    }
}
