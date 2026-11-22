package com.mobilervc.rectest

import android.annotation.SuppressLint
import android.app.Activity
import android.media.AudioFormat
import android.media.AudioRecord
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import java.io.File
import java.io.RandomAccessFile
import kotlin.concurrent.thread
import kotlin.math.sqrt

/**
 * Records like a typical voice-message / recorder app (16 kHz mono PCM16) and logs, every
 * 250 ms, the level and whether the system silenced the recording. Extras:
 *   source  (int, default 1 = MIC), seconds (int, default 4)
 * Output: logcat tag "RecTest" and files/rec_<source>.wav
 */
class RecordActivity : Activity() {
    @SuppressLint("MissingPermission", "SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this).apply { textSize = 16f; setPadding(32, 96, 32, 32) }
        setContentView(tv)
        val source = intent.getIntExtra("source", 1)
        val seconds = intent.getIntExtra("seconds", 4)
        thread {
            val sr = 16000
            val rec = AudioRecord.Builder().setAudioSource(source)
                .setAudioFormat(AudioFormat.Builder().setSampleRate(sr).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build()).build()
            val all = ShortArray(sr * seconds)
            var got = 0
            rec.startRecording()
            val sb = StringBuilder("source=$source\n")
            while (got < all.size) {
                val n = rec.read(all, got, minOf(sr / 10, all.size - got))
                if (n <= 0) break
                var s = 0.0
                for (i in got until got + n) s += all[i].toDouble() * all[i]
                got += n
                // routed device: 15 = built-in mic (the real voice!), 25 = remote submix (injected)
                val line = "t=%.2fs dev=%s silenced=%s rms=%.4f".format(got.toDouble() / sr, rec.routedDevice?.type,
                    rec.activeRecordingConfiguration?.isClientSilenced, sqrt(s / n) / 32768)
                Log.i("RecTest", line)
                sb.append(line).append('\n')
                runOnUiThread { tv.text = sb }
            }
            rec.stop()
            rec.release()
            val f = File(filesDir, "rec_$source.wav")
            writeWav(f, all, got, sr)
            Log.i("RecTest", "saved ${f.absolutePath} ($got samples)")
            runOnUiThread { tv.text = "$sb\n已保存 ${f.name}" }
        }
    }

    private fun writeWav(f: File, pcm: ShortArray, n: Int, sr: Int) {
        RandomAccessFile(f, "rw").use { r ->
            r.setLength(0)
            fun i32(v: Int) = r.write(byteArrayOf(v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte()))
            fun i16(v: Int) = r.write(byteArrayOf(v.toByte(), (v shr 8).toByte()))
            r.writeBytes("RIFF"); i32(36 + n * 2); r.writeBytes("WAVEfmt "); i32(16); i16(1); i16(1)
            i32(sr); i32(sr * 2); i16(2); i16(16); r.writeBytes("data"); i32(n * 2)
            val b = ByteArray(n * 2)
            for (i in 0 until n) { b[2 * i] = pcm[i].toByte(); b[2 * i + 1] = (pcm[i].toInt() shr 8).toByte() }
            r.write(b)
        }
    }
}
