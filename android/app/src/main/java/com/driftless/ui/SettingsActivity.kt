package com.driftless.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.PreferenceFragmentCompat
import com.driftless.R

/**
 * Sensor-rate and GNSS-interval settings. `PreferenceFragmentCompat` gives the
 * whole screen — layout, persistence, summaries — off one XML file, which is
 * the right trade for a shell whose settings are two dropdowns.
 *
 * Requires `preferenceTheme` on the activity theme; see `styles.xml`.
 */
class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        // Guarded on savedInstanceState so a rotation does not stack a second
        // copy of the fragment on top of the restored one.
        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.settingsContainer, SettingsFragment())
                .commit()
        }
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey)
        }
    }
}
