package com.mobilervc.shizuku

import android.content.Context

/**
 * Picks where the privileged [ShellAudioService] runs: under Shizuku when it is running and this
 * app is authorised, otherwise as root through `su` ([RootBridge]), so rooted devices do not need
 * Shizuku. A backend that is already connected is kept.
 */
object ShellBridge {
    enum class Backend(val label: String) { SHIZUKU("Shizuku"), ROOT("Root") }

    @Volatile var backend: Backend? = null
        private set
    /** Why the last [connect] failed. */
    var lastError: String? = null
        private set

    val service: IShellAudio?
        get() = when (backend) {
            Backend.SHIZUKU -> ShizukuBridge.service
            Backend.ROOT -> RootBridge.service
            null -> null
        }

    /** Name of the backend in use (or of both while none is), for messages. */
    val name: String get() = backend?.label ?: "Shizuku / Root"

    val shizukuUsable: Boolean
        get() = ShizukuBridge.state == ShizukuBridge.State.READY || ShizukuBridge.state == ShizukuBridge.State.CONNECTED

    /** Something can host the service (or already does). */
    val usable: Boolean get() = shizukuUsable || RootBridge.available

    /** Shizuku's state without the "Shizuku " some of its texts start with. */
    val shizukuText: String get() = ShizukuBridge.stateText.removePrefix("Shizuku ")

    val stateText: String
        get() = when {
            backend == Backend.ROOT && RootBridge.state == RootBridge.State.CONNECTED -> "Root 已连接"
            backend == Backend.SHIZUKU && ShizukuBridge.state == ShizukuBridge.State.CONNECTED -> "Shizuku $shizukuText"
            shizukuUsable -> "Shizuku 已授权"
            RootBridge.state == RootBridge.State.CONNECTING -> "等待 Root 授权…"
            RootBridge.state == RootBridge.State.AVAILABLE -> "使用 Root"
            RootBridge.state == RootBridge.State.DENIED -> "Root 授权失败"
            else -> "Shizuku $shizukuText"
        }

    fun init() {
        ShizukuBridge.init()
        RootBridge.refresh()
    }

    fun refresh() {
        ShizukuBridge.refresh()
        RootBridge.refresh()
    }

    /** Connects (blocking, call off the main thread). Returns null and sets [lastError] on failure. */
    fun connect(ctx: Context): IShellAudio? {
        service?.let { if (it.asBinder().pingBinder()) return it }
        ShizukuBridge.refresh()
        val errors = mutableListOf<String>()
        if (shizukuUsable) {
            ShizukuBridge.connect(ctx)?.let {
                backend = Backend.SHIZUKU
                return it
            }
            errors += "Shizuku 服务连接失败"
        }
        RootBridge.refresh()
        if (RootBridge.available) {
            connectRoot(ctx)?.let { return it }
            errors += "Root：${RootBridge.error}"
        }
        if (errors.isEmpty()) errors += "需要 Shizuku 或 Root（Shizuku $shizukuText，未检测到 Root）"
        lastError = errors.joinToString("；")
        return null
    }

    /** Starts the root service now (the root manager asks for permission the first time). */
    fun connectRoot(ctx: Context): IShellAudio? =
        RootBridge.connect(ctx)?.also { backend = Backend.ROOT }
}
