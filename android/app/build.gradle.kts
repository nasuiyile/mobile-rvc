plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Base models (HuBERT/ContentVec, FCPE, synthesizer templates) produced by
// tools/build_assets.sh. Override with -PrvcAssetsDir=/path/to/apk_assets
val rvcAssetsDir: String = (project.findProperty("rvcAssetsDir") as String?)
    ?: rootProject.file("../build/apk_assets").absolutePath

val ortVersionStandard = "1.32.0"
val ortVersionQnn = "1.29.0"   // pulls com.qualcomm.qti:qnn-runtime transitively

android {
    namespace = "com.mobilervc"
    compileSdk {
        version = release(37) { minorApiLevel = 2 }
    }
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "com.mobilervc"
        minSdk = 29          // AAudio input presets (VOICE_PERFORMANCE), AudioPolicy injection
        targetSdk = 36
        versionCode = 5
        versionName = "0.1.4"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                targets += "rvc"
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }
    }

    flavorDimensions += "accel"
    productFlavors {
        create("standard") {
            dimension = "accel"
            externalNativeBuild { cmake { arguments += "-DORT_ROOT=${layout.buildDirectory.get().asFile}/ort/standard" } }
        }
        create("qnn") {
            dimension = "accel"
            applicationIdSuffix = ".qnn"
            versionNameSuffix = "-qnn"
            externalNativeBuild { cmake { arguments += "-DORT_ROOT=${layout.buildDirectory.get().asFile}/ort/qnn" } }
        }
    }

    sourceSets {
        getByName("main") {
            if (file(rvcAssetsDir).isDirectory) assets.srcDir(rvcAssetsDir)
        }
    }

    // Release keystore from the environment (set by CI, see .github/workflows/release.yml).
    // RVC_KEYSTORE is the .jks path; the passwords and alias come from the other RVC_* vars.
    val releaseKeystore = providers.environmentVariable("RVC_KEYSTORE").orNull
    signingConfigs {
        if (releaseKeystore != null) create("release") {
            storeFile = file(releaseKeystore)
            storePassword = providers.environmentVariable("RVC_KEYSTORE_PASSWORD").orNull
            keyAlias = providers.environmentVariable("RVC_KEY_ALIAS").orNull
            keyPassword = providers.environmentVariable("RVC_KEY_PASSWORD").orNull
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Self-signed builds: debug key unless a keystore is configured (RVC_KEYSTORE)
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        compose = true
        aidl = true
        buildConfig = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    packaging {
        jniLibs {
            // extracted to nativeLibraryDir: the Shizuku shell service loads nothing native,
            // but QNN needs its skel libraries as real files
            useLegacyPackaging = true
            excludes += listOf("**/libonnxruntime4j_jni.so", "**/libQnnSaver.so", "**/libQnnCpu.so")
        }
    }

    androidResources {
        // weights barely compress; keep them stored for fast first-run extraction
        noCompress += listOf("bin", "onnx")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

val ortStandard: Configuration by configurations.creating { isTransitive = false }
val ortQnn: Configuration by configurations.creating { isTransitive = false }

dependencies {
    "standardImplementation"("com.microsoft.onnxruntime:onnxruntime-android:$ortVersionStandard")
    "qnnImplementation"("com.microsoft.onnxruntime:onnxruntime-android-qnn:$ortVersionQnn")
    ortStandard("com.microsoft.onnxruntime:onnxruntime-android:$ortVersionStandard@aar")
    ortQnn("com.microsoft.onnxruntime:onnxruntime-android-qnn:$ortVersionQnn@aar")

    implementation(platform("androidx.compose:compose-bom:2026.11.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("dev.rikka.shizuku:api:13.1.5")
    implementation("dev.rikka.shizuku:provider:13.1.5")
    implementation("org.lsposed.hiddenapibypass:hiddenapibypass:6.1")
}

// Headers + libonnxruntime.so for CMake, taken from the same AAR that is packaged.
fun registerOrtExtract(name: String, conf: Configuration) = tasks.register<Copy>("extractOrt$name") {
    from({ conf.resolve().map { zipTree(it) } }) {
        include("headers/**", "jni/arm64-v8a/libonnxruntime.so")
    }
    into(layout.buildDirectory.dir("ort/${name.lowercase()}"))
}
val extractOrtStandard = registerOrtExtract("Standard", ortStandard)
val extractOrtQnn = registerOrtExtract("Qnn", ortQnn)

tasks.configureEach {
    if (name.startsWith("configureCMake") || name.startsWith("buildCMake")) {
        when {
            name.contains("Standard") -> dependsOn(extractOrtStandard)
            name.contains("Qnn") -> dependsOn(extractOrtQnn)
            else -> dependsOn(extractOrtStandard, extractOrtQnn)
        }
    }
}
