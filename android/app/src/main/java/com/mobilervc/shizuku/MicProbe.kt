package com.mobilervc.shizuku

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.os.Build
import android.os.Looper
import kotlin.math.sqrt

/**
 * Diagnostic: records for N seconds and prints, every 250 ms, whether Android's concurrent-
 * capture policy has silenced this client. Run it under different uids to see who wins:
 *
 *   adb shell su 2000 'CLASSPATH=<apk> app_process /system/bin com.mobilervc.shizuku.MicProbe 10 6 shell'
 *   adb shell su 0    'CLASSPATH=<apk> app_process /system/bin com.mobilervc.shizuku.MicProbe 1 3 victim'
 *   adb shell run-as com.mobilervc sh -c 'CLASSPATH=<apk> app_process /system/bin com.mobilervc.shizuku.MicProbe 10 6 app com.mobilervc'
 */
@SuppressLint("MissingPermission")
object MicProbe {
    @JvmStatic
    fun main(args: Array<String>) {
        if (Looper.myLooper() == null) Looper.prepareMainLooper()
        val source = args.getOrNull(0)?.toInt() ?: 1
        val seconds = args.getOrNull(1)?.toDouble() ?: 3.0
        val label = args.getOrNull(2) ?: "probe"
        ShellEnv.packageOverride = args.getOrNull(3)
        ShellEnv.init()
        val sr = 48000
        val fmt = AudioFormat.Builder().setSampleRate(sr).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build()
        val b = AudioRecord.Builder().setAudioSource(source).setAudioFormat(fmt)
        if (Build.VERSION.SDK_INT >= 31) b.setContext(ShellEnv.context)
        val rec = b.build()
        rec.startRecording()
        val t0 = System.currentTimeMillis()
        val buf = ShortArray(sr / 4)
        while (System.currentTimeMillis() - t0 < seconds * 1000) {
            val n = rec.read(buf, 0, buf.size)
            var s = 0.0
            for (i in 0 until maxOf(n, 0)) s += buf[i].toDouble() * buf[i]
            val silenced = rec.activeRecordingConfiguration?.isClientSilenced
            println("[%s uid=%d src=%d] t=%.2fs silenced=%s rms=%.4f".format(label, android.os.Process.myUid(), source,
                (System.currentTimeMillis() - t0) / 1000.0, silenced, sqrt(s / maxOf(1, n)) / 32768))
        }
        rec.stop()
        rec.release()
        System.exit(0)
    }
}
