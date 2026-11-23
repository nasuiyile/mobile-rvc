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
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import androidx.core.app.ServiceCompat
import com.mobilervc.R
import com.mobilervc.engine.Converter
import com.mobilervc.ui.MainActivity

/**
 * Keeps a file conversion running when the app goes to the background or the screen turns off
 * (a backgrounded process may be frozen or killed), with its progress in a notification.
 */
class ConvertService : Service() {
    private val handler = Handler(Looper.getMainLooper())
    private var wake: PowerManager.WakeLock? = null
    private val tick = object : Runnable {
        override fun run() {
            if (Converter.converting) {
                nm.notify(ID, progressNotification())
                handler.postDelayed(this, 1000)
            } else {
                done()
            }
        }
    }
    private val nm get() = getSystemService(NotificationManager::class.java)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            Converter.cancel()
            return START_NOT_STICKY
        }
        nm.createNotificationChannel(NotificationChannel(CHANNEL, getString(R.string.convert_channel), NotificationManager.IMPORTANCE_LOW))
        val type = if (Build.VERSION.SDK_INT >= 35) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        else ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        // not ServiceCompat: its androidx version drops types newer than it knows (mediaProcessing)
        startForeground(ID, progressNotification(), type)
        if (wake == null) {
            wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "mobilervc:convert").apply { acquire(3 * 3600_000L) }
        }
        handler.removeCallbacks(tick)
        handler.post(tick)
        return START_NOT_STICKY
    }

    /** Conversion over: say so (unless the app is in front showing it anyway) and stop. */
    private fun done() {
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        val r = Converter.finished
        if (r != null && !MainActivity.visible) {
            nm.notify(
                ID_DONE,
                Notification.Builder(this, CHANNEL)
                    .setSmallIcon(R.drawable.ic_stat_voice)
                    .setContentTitle(getString(R.string.convert_done))
                    .setContentText(r.file.name)
                    .setContentIntent(openApp())
                    .setAutoCancel(true)
                    .build(),
            )
        }
        stopSelf()
    }

    override fun onDestroy() {
        handler.removeCallbacks(tick)
        wake?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        Converter.cancel()  // mediaProcessing is limited to 6 h a day
        stopSelf()
    }

    private fun openApp() = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)

    private fun progressNotification(): Notification {
        val p = Converter.progress
        val cancel = PendingIntent.getService(this, 2, Intent(this, ConvertService::class.java).setAction(ACTION_CANCEL), PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_voice)
            .setContentTitle(getString(R.string.convert_running))
            .setContentText(Converter.input?.name ?: "")
            .setProgress(100, (p * 100).toInt(), p <= 0f)
            .setContentIntent(openApp())
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(Notification.Action.Builder(null, getString(R.string.cancel), cancel).build())
            .build()
    }

    companion object {
        private const val CHANNEL = "convert"
        private const val ID = 2
        private const val ID_DONE = 3
        private const val ACTION_CANCEL = "com.mobilervc.CANCEL_CONVERT"

        fun start(ctx: Context) {
            ctx.startForegroundService(Intent(ctx, ConvertService::class.java))
        }
    }
}
