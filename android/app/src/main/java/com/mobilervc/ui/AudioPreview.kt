package com.mobilervc.ui

import android.media.AudioAttributes
import android.media.MediaPlayer
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.mobilervc.engine.RvcController
import java.io.File

/** Plays one file at a time (the input or a result) for listening in the app. */
object AudioPreview {
    var playing by mutableStateOf<File?>(null)
        private set
    var positionMs by mutableIntStateOf(0)
        private set
    var durationMs by mutableIntStateOf(0)
        private set
    private var player: MediaPlayer? = null

    fun toggle(f: File) {
        val same = playing == f
        stop()
        if (same) return
        try {
            player = MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build(),
                )
                setDataSource(f.absolutePath)
                prepare()
                setOnCompletionListener { stop() }
                start()
            }
            durationMs = player?.duration ?: 0
            playing = f
        } catch (t: Throwable) {
            stop()
            RvcController.message = "无法播放：${t.message ?: t}"
        }
    }

    fun stop() {
        player?.let { p ->
            runCatching { p.stop() }
            p.release()
        }
        player = null
        playing = null
        positionMs = 0
    }

    /** Called while something plays, to move the progress bar. */
    fun tick() {
        positionMs = runCatching { player?.currentPosition }.getOrNull() ?: 0
    }
}
