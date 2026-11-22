package com.mobilervc.shizuku

import android.content.ComponentName
import android.content.Context
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.mobilervc.BuildConfig
import rikka.shizuku.Shizuku
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** App-side Shizuku handling: permission and the shell-uid [ShellAudioService]. */
object ShizukuBridge {
    enum class State { NOT_RUNNING, NO_PERMISSION, READY, CONNECTED }

    private const val REQUEST_CODE = 0x52564300
    var state by mutableStateOf(State.NOT_RUNNING)
        private set
    @Volatile var service: IShellAudio? = null
        private set
    var serviceUid by mutableStateOf(-1)
        private set
    private var latch = CountDownLatch(1)
    private var listening = false

    val stateText: String
        get() = when (state) {
            State.NOT_RUNNING -> "Shizuku 未运行"
            State.NO_PERMISSION -> "未授权"
            State.READY -> "已授权"
            State.CONNECTED -> if (serviceUid == 0) "已连接 (root)" else "已连接 (shell)"
        }

    private fun args(ctx: Context) = Shizuku.UserServiceArgs(ComponentName(ctx.packageName, ShellAudioService::class.java.name))
        .daemon(false)
        .processNameSuffix("shell_audio")
        .debuggable(BuildConfig.DEBUG)
        .version(BuildConfig.VERSION_CODE)

    private val conn = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            if (binder != null && binder.pingBinder()) {
                val s = IShellAudio.Stub.asInterface(binder)
                service = s
                serviceUid = runCatching { s.uid() }.getOrDefault(-1)
                state = State.CONNECTED
            }
            latch.countDown()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            refresh()
        }
    }

    fun init() {
        if (listening) return
        listening = true
        Shizuku.addBinderReceivedListenerSticky { refresh() }
        Shizuku.addBinderDeadListener {
            service = null
            refresh()
        }
        Shizuku.addRequestPermissionResultListener { code, _ -> if (code == REQUEST_CODE) refresh() }
    }

    fun refresh() {
        state = when {
            !runCatching { Shizuku.pingBinder() }.getOrDefault(false) -> State.NOT_RUNNING
            Shizuku.isPreV11() -> State.NO_PERMISSION
            Shizuku.checkSelfPermission() != PackageManager.PERMISSION_GRANTED -> State.NO_PERMISSION
            service != null -> State.CONNECTED
            else -> State.READY
        }
    }

    fun requestPermission() {
        if (runCatching { Shizuku.pingBinder() }.getOrDefault(false) && !Shizuku.isPreV11()) {
            Shizuku.requestPermission(REQUEST_CODE)
        }
    }

    /** Binds the shell service (blocking, call off the main thread). */
    fun connect(ctx: Context, timeoutMs: Long = 15000): IShellAudio? {
        service?.let { if (it.asBinder().pingBinder()) return it }
        refresh()
        if (state == State.NOT_RUNNING || state == State.NO_PERMISSION) return null
        latch = CountDownLatch(1)
        val l = latch
        Handler(Looper.getMainLooper()).post {
            runCatching { Shizuku.bindUserService(args(ctx), conn) }.onFailure { l.countDown() }
        }
        l.await(timeoutMs, TimeUnit.MILLISECONDS)
        return service
    }

    fun disconnect(ctx: Context) {
        runCatching { Shizuku.unbindUserService(args(ctx), conn, true) }
        service = null
        refresh()
    }
}
