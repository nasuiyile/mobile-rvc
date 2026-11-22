package com.mobilervc.shizuku

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.system.Os
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Diagnostic for the virtual-microphone path without Shizuku (same app_process setup):
 *
 *   adb shell CLASSPATH=$(pm path com.mobilervc | cut -d: -f2) \
 *       app_process /system/bin com.mobilervc.shizuku.ShellSelfTest
 *
 * Starts [ShellAudioService], feeds a 440 Hz tone into the injection pipe and records with
 * the MIC preset (which is intercepted) to check that the tone arrives, while also reading
 * the real-microphone capture pipe.
 */
@SuppressLint("MissingPermission")
object ShellSelfTest {
    @JvmStatic
    fun main(args: Array<String>) {
        if (Looper.myLooper() == null) Looper.prepareMainLooper()
        val sr = 48000
        val tStart = System.currentTimeMillis()
        val svc = ShellAudioService()
        val cap = Os.pipe()
        val play = Os.pipe()
        // --monitor N: run the service for N seconds with app_uid/a11y set, logging capture-mode
        // changes and status, so recordings started from other uids can be observed
        val monitorSecs = args.indexOf("--monitor").takeIf { it >= 0 }?.let { args[it + 1].toInt() }
        val appUid = args.indexOf("--app-uid").takeIf { it >= 0 }?.let { args[it + 1].toInt() } ?: -1
        val a11y = args.indexOf("--a11y").takeIf { it >= 0 }?.let { args[it + 1] } ?: ""
        val cfg = """{"sample_rate":$sr,"capture_source":10,"intercept":[1,5,6,7,9],"max_backlog_ms":600,""" +
            """"app_uid":$appUid,"a11y_component":"$a11y"}"""
        // --as-uid N: run start() under a foreign Binder calling identity, as when the app calls
        // the service through Shizuku (Binder.getCallingUid() is then the app, not shell)
        val asUid = args.indexOf("--as-uid").takeIf { it >= 0 }?.let { args[it + 1].toInt() }
        val saved = android.os.Binder.clearCallingIdentity()
        if (asUid != null) android.os.Binder.restoreCallingIdentity((asUid.toLong() shl 32) or android.os.Process.myPid().toLong())
        val st = try {
            svc.start(ParcelFileDescriptor.dup(cap[1]), ParcelFileDescriptor.dup(play[0]), cfg, object : IShellCallback.Stub() {
                override fun onCaptureMode(mode: Int, detail: String?) {
                    println("[callback] t=%.1fs mode=%d (%s)".format((System.currentTimeMillis() - tStart) / 1000.0, mode, detail))
                }
            })
        } finally {
            android.os.Binder.restoreCallingIdentity(saved)
        }
        if (asUid != null) println("start() called with Binder.getCallingUid() = $asUid")
        println("uid=${android.os.Process.myUid()} sdk=${Build.VERSION.SDK_INT} start -> $st")
        if (st.contains("error")) {
            System.exit(1)
        }
        val running = java.util.concurrent.atomic.AtomicBoolean(true)
        // producer: 440 Hz tone, paced in real time
        val producer = Thread {
            val n = sr / 100
            val bb = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder())
            var phase = 0.0
            var next = System.nanoTime()
            while (running.get()) {
                bb.clear()
                for (i in 0 until n) {
                    bb.putFloat((0.5 * sin(phase)).toFloat())
                    phase += 2 * PI * 440 / sr
                }
                bb.flip()
                runCatching { while (bb.hasRemaining()) Os.write(play[1], bb) }
                next += 10_000_000L
                val sleep = (next - System.nanoTime()) / 1_000_000
                if (sleep > 0) Thread.sleep(sleep)
            }
        }.apply { start() }
        // drain the real-mic capture pipe
        var micSamples = 0L
        val drain = Thread {
            val bb = ByteBuffer.allocateDirect(4096)
            while (running.get()) {
                bb.clear()
                val r = runCatching { Os.read(cap[0], bb) }.getOrDefault(-1)
                if (r <= 0) break
                micSamples += r / 4
            }
        }.apply { start() }
        Thread.sleep(500)
        if (monitorSecs != null) {
            val end = System.currentTimeMillis() + monitorSecs * 1000L
            while (System.currentTimeMillis() < end) {
                Thread.sleep(1000)
                println("[status] t=%.1fs %s".format((System.currentTimeMillis() - tStart) / 1000.0, svc.status()))
            }
            running.set(false)
            svc.stop()
            System.exit(0)
        }
        // a "victim" recorder using the intercepted MIC preset
        val fmt = AudioFormat.Builder().setSampleRate(sr).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build()
        val b = AudioRecord.Builder().setAudioSource(MediaRecorder.AudioSource.MIC).setAudioFormat(fmt)
        if (Build.VERSION.SDK_INT >= 31) b.setContext(ShellEnv.context)
        val rec = b.build()
        rec.startRecording()
        val pcm = ShortArray(sr)
        var got = 0
        val t0 = System.currentTimeMillis()
        while (got < pcm.size && System.currentTimeMillis() - t0 < 4000) {
            val r = rec.read(pcm, got, pcm.size - got)
            if (r > 0) got += r else break
        }
        rec.stop()
        rec.release()
        // analyse the second half: RMS and dominant frequency via zero crossings
        val seg = pcm.copyOfRange(got / 2, got)
        var sum = 0.0
        var zc = 0
        for (i in seg.indices) {
            sum += seg[i].toDouble() * seg[i]
            if (i > 0 && (seg[i - 1] < 0) != (seg[i] < 0)) zc++
        }
        val rms = sqrt(sum / maxOf(1, seg.size)) / 32768.0
        val freq = zc / 2.0 / (seg.size.toDouble() / sr)
        println("victim MIC recording: samples=$got rms=%.3f freq≈%.0f Hz".format(rms, freq))
        println("real mic capture pipe: $micSamples samples in ~${(System.currentTimeMillis() - t0 + 500) / 1000.0}s")
        println("service status: ${svc.status()}")
        val ok = rms > 0.1 && freq in 400.0..480.0
        println(if (ok) "RESULT: injection OK" else "RESULT: injection NOT detected")
        running.set(false)
        svc.stop()
        System.exit(if (ok) 0 else 2)
    }
}
