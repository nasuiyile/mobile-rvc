package com.mobilervc.shizuku

import android.content.Intent
import android.os.Bundle
import android.os.Looper
import android.os.UserHandle
import kotlin.system.exitProcess

/**
 * Entry point of the root process that replaces Shizuku on rooted devices. [RootBridge] starts it
 * through `su`:
 *
 *   CLASSPATH=<apk> app_process /system/bin com.mobilervc.shizuku.RootServiceMain <package> <token> <app uid>
 *
 * It creates the same [ShellAudioService] Shizuku would host and hands its binder to the app in a
 * broadcast. The process lives as long as the app: `su` connects its stdin to the app, so when the
 * app dies, stdin reaches EOF and the process exits (Shizuku's non-daemon user services behave the
 * same way). The app can also end it with IShellAudio.destroy().
 */
object RootServiceMain {
    const val EXTRA_TOKEN = "token"
    const val EXTRA_BUNDLE = "bundle"
    const val KEY_BINDER = "binder"
    /** Intent.FLAG_RECEIVER_FROM_SHELL (hidden): the broadcast comes from shell / root. */
    private const val FLAG_RECEIVER_FROM_SHELL = 0x00400000

    fun action(packageName: String) = "$packageName.ROOT_SHELL_AUDIO"

    @JvmStatic
    fun main(args: Array<String>) {
        if (args.size < 3) {
            System.err.println("usage: RootServiceMain <package> <token> <app uid>")
            exitProcess(2)
        }
        val (pkg, token, appUid) = Triple(args[0], args[1], args[2].toInt())
        Looper.prepareMainLooper()
        val svc = try {
            ShellEnv.init()
            val svc = ShellAudioService()
            val intent = Intent(action(pkg))
                .setPackage(pkg)
                .putExtra(EXTRA_TOKEN, token)
                .putExtra(EXTRA_BUNDLE, Bundle().apply { putBinder(KEY_BINDER, svc) })
                .addFlags(Intent.FLAG_RECEIVER_FOREGROUND or FLAG_RECEIVER_FROM_SHELL)
            ShellEnv.context.sendBroadcastAsUser(intent, UserHandle.getUserHandleForUid(appUid))
            svc
        } catch (t: Throwable) {
            t.printStackTrace()
            exitProcess(1)
        }
        println("ok uid=${android.os.Process.myUid()} pid=${android.os.Process.myPid()}")
        Thread({
            runCatching { while (System.`in`.read() >= 0) Unit }
            svc.destroy()
        }, "rvc-stdin").apply { isDaemon = true }.start()
        Looper.loop()
    }
}
