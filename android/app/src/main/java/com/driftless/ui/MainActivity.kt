package com.driftless.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * Wires sensors -> fusion -> map-match -> map view. Owns the GNSS
 * deficit handler: switches the displayed position source between
 * GNSS-aided and dead-reckoning-only without a visible jump.
 */
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // TODO: start ImuSampler + GnssSampler, feed UkfFusionEngine, render on osmdroid MapView
    }
}
