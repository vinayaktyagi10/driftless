plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.driftless"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.driftless"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.gms:play-services-location:21.3.0")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.osmdroid:osmdroid-android:6.1.20")

    // Settings screen. PreferenceFragmentCompat gives the whole screen with
    // SharedPreferences backing; hand-rolling it would cost a day we don't have.
    implementation("androidx.preference:preference-ktx:1.2.1")

    // repeatOnLifecycle, so sampler Flows stop collecting when the app is
    // backgrounded instead of holding the sensors open and draining the battery.
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    testImplementation("junit:junit:4.13.2")
}
