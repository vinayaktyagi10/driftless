package com.driftless

import android.app.Application
import org.osmdroid.config.Configuration
import java.io.File

/**
 * Exists for one reason: osmdroid has to be configured **before the first
 * [org.osmdroid.views.MapView] is constructed**, and an Activity's `onCreate`
 * is already too late once a MapView is inflated from its layout.
 *
 * Both settings below are failure modes rather than preferences, and both fail
 * silently — a blank grey map with nothing in logcat, which is indistinguishable
 * from "the tile server is slow" until hours have been spent on it.
 */
class DriftlessApp : Application() {

    override fun onCreate() {
        super.onCreate()

        val config = Configuration.getInstance()

        // OSM's tile servers reject osmdroid's built-in default user agent
        // outright — the library is widely abused and the default is banned by
        // policy, not by rate limit. Every request 403s and the map stays empty.
        config.userAgentValue = packageName

        // osmdroid 6.1.20 still reaches for public external storage by default.
        // On Android 10+ scoped storage that either throws or, worse, quietly
        // caches nothing, so every pan re-downloads and an offline demo is
        // impossible. These paths are app-private and need no permission.
        val base = File(getExternalFilesDir(null) ?: filesDir, "osmdroid")
        config.osmdroidBasePath = base
        config.osmdroidTileCache = File(base, "tiles").apply { mkdirs() }
    }
}
