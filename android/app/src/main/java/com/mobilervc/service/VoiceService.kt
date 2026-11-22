package com.mobilervc.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.ServiceCompat
import com.mobilervc.R
import com.mobilervc.engine.RvcController
import com.mobilervc.ui.MainActivity

/** Keeps realtime conversion alive in the background (screen off, other app in front). */
class VoiceService : Service() {
    private var wake: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            RvcController.stop()
            stopSelf()
            return START_NOT_STICKY
        }
        createChannel()
        val type = if (Build.VERSION.SDK_INT >= 30)
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE or ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK else 0
        ServiceCompat.startForeground(this, 1, notification(), type)
        wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "mobilervc:voice").apply { acquire(6 * 3600_000L) }
        RvcController.start { err -> if (err != null) stopSelf() }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        RvcController.stop()
        wake?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    private fun createChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, getString(R.string.channel_name), NotificationManager.IMPORTANCE_LOW))
    }

    private fun notification(): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(
            this, 1, Intent(this, VoiceService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_voice)
            .setContentTitle(getString(R.string.notif_title))
            .setContentText(getString(R.string.notif_text))
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, getString(R.string.stop), stop).build())
            .build()
    }

    companion object {
        private const val CHANNEL = "voice"
        private const val ACTION_STOP = "com.mobilervc.STOP"

        fun start(ctx: Context) {
            ctx.startForegroundService(Intent(ctx, VoiceService::class.java))
        }

        fun stop(ctx: Context) {
            ctx.startService(Intent(ctx, VoiceService::class.java).setAction(ACTION_STOP))
        }
    }
}
