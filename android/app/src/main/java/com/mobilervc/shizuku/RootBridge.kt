package com.mobilervc.shizuku

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.IBinder
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Runs [ShellAudioService] as root through `su` (Magisk, KernelSU, APatch...), so rooted devices
 * need no Shizuku. The root process is [RootServiceMain]; it sends its binder back in a broadcast
 * to a receiver that is not exported (only root, system and this app can reach it), tagged with a
 * per-launch token.
 */
object RootBridge {
    enum class State { NONE, AVAILABLE, CONNECTING, DENIED, CONNECTED }

    private const val TAG = "RootBridge"
    /** Long enough for the root manager's "allow?" dialog. */
    private const val CONNECT_TIMEOUT_MS = 60_000L

    var state by mutableStateOf(State.NONE)
        private set
    /** Why the last launch failed, for the UI and error messages. */
    var error by mutableStateOf<String?>(null)
        private set
    @Volatile var service: IShellAudio? = null
        private set
    private var process: Process? = null
    private val lock = Object()

    /** su is visible to this app. KernelSU / APatch only show it to apps allowed root. */
    val available: Boolean get() = state != State.NONE

    val stateText: String
        get() = when (state) {
            State.NONE -> "未检测到"
            State.AVAILABLE -> "开始时请求授权"
            State.CONNECTING -> "等待授权…"
            State.DENIED -> "授权失败"
            State.CONNECTED -> "已连接"
        }

    private val SU_DIRS = listOf("/system/bin", "/system/xbin", "/sbin", "/su/bin", "/system/sbin", "/vendor/bin", "/debug_ramdisk")

    private fun findSu(): String? =
        ((System.getenv("PATH")?.split(':') ?: emptyList()) + SU_DIRS).distinct()
            .map { File(it, "su") }.firstOrNull { runCatching { it.exists() }.getOrDefault(false) }?.path

    fun refresh() {
        // connect() runs on another thread and sets the outcome itself; the death recipient
        // clears [service]
        val cur = state
        if (cur == State.CONNECTING) return
        state = when {
            service?.asBinder()?.pingBinder() == true -> State.CONNECTED
            findSu() == null -> State.NONE
            cur == State.DENIED -> State.DENIED
            else -> State.AVAILABLE
        }
    }

    /**
     * Starts the root process (or reuses the running one) and waits for its binder. Blocking, call
     * off the main thread; the first call usually shows the root manager's permission dialog.
     */
    fun connect(ctx: Context): IShellAudio? = synchronized(lock) {
        service?.let { if (it.asBinder().pingBinder()) return it }
        service = null
        stopProcess()
        val su = findSu() ?: run {
            state = State.NONE
            error = "未找到 su"
            return null
        }
        state = State.CONNECTING
        val app = ctx.applicationContext
        val token = UUID.randomUUID().toString()
        val latch = CountDownLatch(1)
        var binder: IBinder? = null
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(c: Context, intent: Intent) {
                if (intent.getStringExtra(RootServiceMain.EXTRA_TOKEN) != token) return
                binder = intent.getBundleExtra(RootServiceMain.EXTRA_BUNDLE)?.getBinder(RootServiceMain.KEY_BINDER)
                latch.countDown()
            }
        }
        ContextCompat.registerReceiver(app, receiver, IntentFilter(RootServiceMain.action(app.packageName)),
            ContextCompat.RECEIVER_NOT_EXPORTED)
        val output = StringBuffer()
        try {
            val p = ProcessBuilder(su).redirectErrorStream(true).start()
            process = p
            Thread({
                runCatching {
                    p.inputStream.bufferedReader().forEachLine {
                        Log.i(TAG, "root: $it")
                        if (output.length < 4000) output.append(it).append('\n')
                    }
                }
            }, "rvc-root-out").apply { isDaemon = true }.start()
            // stdin stays open: the root process exits when it reaches EOF, i.e. when this app dies
            val cmd = "export CLASSPATH=${quote(app.applicationInfo.sourceDir)}\n" +
                "exec /system/bin/app_process /system/bin --nice-name=${app.packageName}:root " +
                "${RootServiceMain::class.java.name} ${app.packageName} $token ${android.os.Process.myUid()}\n"
            p.outputStream.apply { write(cmd.toByteArray()); flush() }
            val deadline = System.currentTimeMillis() + CONNECT_TIMEOUT_MS
            while (!latch.await(200, TimeUnit.MILLISECONDS)) {
                if (!p.isAlive) break  // denied, or the root process crashed
                if (System.currentTimeMillis() > deadline) break
            }
            val b = binder
            if (b == null || !b.pingBinder()) {
                val exit = if (p.isAlive) "超时" else "退出码 ${p.exitValue()}"
                Thread.sleep(100)  // let the reader collect the last lines
                val out = output.toString().trim().lines().takeLast(3).joinToString(" / ")
                error = if (out.isEmpty()) exit else "$exit：$out"
                state = State.DENIED
                stopProcess()
                return null
            }
            val s = IShellAudio.Stub.asInterface(b)
            b.linkToDeath({
                if (service === s) {
                    service = null
                    refresh()
                }
            }, 0)
            service = s
            error = null
            state = State.CONNECTED
            return s
        } catch (t: Throwable) {
            Log.w(TAG, "connect", t)
            error = t.message ?: t.toString()
            state = State.DENIED
            stopProcess()
            return null
        } finally {
            runCatching { app.unregisterReceiver(receiver) }
        }
    }

    /** Ends the root process. */
    fun disconnect() = synchronized(lock) {
        runCatching { service?.destroy() }
        service = null
        stopProcess()
        refresh()
    }

    private fun stopProcess() {
        process?.let { p -> runCatching { p.outputStream.close() }; runCatching { p.destroy() } }
        process = null
    }

    private fun quote(s: String) = "'" + s.replace("'", "'\\''") + "'"
}
