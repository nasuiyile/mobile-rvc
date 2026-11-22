package com.mobilervc.shizuku

import android.annotation.SuppressLint
import android.app.Application
import android.content.AttributionSource
import android.content.Context
import android.content.ContextWrapper
import android.content.pm.ApplicationInfo
import android.media.AudioManager
import android.os.Build
import android.os.Process
import org.lsposed.hiddenapibypass.HiddenApiBypass

/**
 * Process environment for code running as the shell (or root) user inside Shizuku's
 * app_process. Framework audio APIs derive the caller's package from the Context /
 * ActivityThread; there they must report the package that really owns this uid, otherwise
 * AppOps rejects recording and playback.
 */
@SuppressLint("PrivateApi", "DiscouragedPrivateApi")
object ShellEnv {
    /** Set before init() when running under another uid (diagnostics via run-as). */
    var packageOverride: String? = null

    val packageName: String
        get() = packageOverride ?: when (Process.myUid()) {
            0 -> "root"
            else -> "com.android.shell"
        }

    lateinit var context: Context
        private set
    private var ready = false

    @Synchronized
    fun init() {
        if (ready) return
        if (Build.VERSION.SDK_INT >= 28) runCatching { HiddenApiBypass.addHiddenApiExemptions("") }
        val atClass = Class.forName("android.app.ActivityThread")
        val at = atClass.getMethod("currentActivityThread").invoke(null)
            ?: atClass.getMethod("systemMain").invoke(null)
        val sys = atClass.getMethod("getSystemContext").invoke(at) as Context
        val ctx = ShellContext(sys)
        context = ctx
        // ActivityThread.currentOpPackageName() / currentApplication() are used by
        // AudioRecord / AudioTrack on older releases: point them at a shell "application".
        runCatching {
            val bindData = Class.forName("android.app.ActivityThread\$AppBindData").getDeclaredConstructor()
                .apply { isAccessible = true }.newInstance()
            val ai = ApplicationInfo().apply { packageName = this@ShellEnv.packageName; uid = Process.myUid() }
            bindData.javaClass.getDeclaredField("appInfo").apply { isAccessible = true }.set(bindData, ai)
            atClass.getDeclaredField("mBoundApplication").apply { isAccessible = true }.set(at, bindData)
        }
        runCatching {
            val app = Application()
            Application::class.java.superclass!!.getDeclaredMethod("attachBaseContext", Context::class.java)
                .apply { isAccessible = true }.invoke(app, ctx)
            atClass.getDeclaredField("mInitialApplication").apply { isAccessible = true }.set(at, app)
        }
        ready = true
    }

    /** AudioManager bound to the shell context (hidden constructor AudioManager(Context)). */
    fun audioManager(): AudioManager =
        runCatching {
            AudioManager::class.java.getDeclaredConstructor(Context::class.java).apply { isAccessible = true }
                .newInstance(context)
        }.getOrElse { context.getSystemService(AudioManager::class.java) }
}

class ShellContext(base: Context) : ContextWrapper(base) {
    override fun getPackageName(): String = ShellEnv.packageName
    override fun getOpPackageName(): String = ShellEnv.packageName
    override fun getApplicationContext(): Context = this

    override fun getAttributionSource(): AttributionSource =
        AttributionSource.Builder(Process.myUid()).setPackageName(ShellEnv.packageName).build()

    override fun getSystemService(name: String): Any? =
        if (name == Context.AUDIO_SERVICE) ShellEnv.audioManager() else super.getSystemService(name)
}
