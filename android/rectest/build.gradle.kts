// Stand-in for a phone's built-in voice recorder when testing the virtual microphone:
// a separate app (separate uid) that records with a chosen capture preset and saves a wav.
plugins {
    id("com.android.application")
}

android {
    namespace = "com.mobilervc.rectest"
    compileSdk {
        version = release(37) { minorApiLevel = 2 }
    }
    defaultConfig {
        applicationId = "com.mobilervc.rectest"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
