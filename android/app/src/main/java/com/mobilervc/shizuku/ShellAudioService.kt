package com.mobilervc.shizuku

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioRecordingConfiguration
import android.media.AudioTrack
import android.os.Binder
import android.os.Build
import android.os.HandlerThread
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.Process
import android.os.SystemClock
import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.FileDescriptor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Shizuku UserService: runs as uid 2000 (shell) or 0 (root), never inside the app process.
 *
 * 1. Injection: an AudioPolicy loop-back MIX_TYPE_RECORDERS mix redirects other apps'
 *    recordings (MIC, VOICE_COMMUNICATION, ...) to a remote submix fed by our injection
 *    AudioTrack, so they record the converted voice. Needs MODIFY_AUDIO_ROUTING (shell has it).
 *
 * 2. Capture, only while such a recording is active. Android silences all but one concurrent
 *    capture (AudioPolicyService::updateUidStates_l):
 *      - against a normal recording (e.g. WeChat voice message, source MIC) the latest-started
 *        foreground client wins, which would silence a shell capture; an *accessibility* app's
 *        capture is exempt, so the app records the mic itself (mode APP);
 *      - against a privacy-sensitive recording / call (VOICE_COMMUNICATION) only a client with
 *        CAPTURE_AUDIO_OUTPUT may also capture, so this shell process records (mode SHELL).
 *    The choice is guessed from the victim's source and corrected if our capture is silenced.
 *    Nothing records while no app records, so apps no longer see a background recorder.
 */
@SuppressLint("PrivateApi", "MissingPermission")
class ShellAudioService() : IShellAudio.Stub() {
    @Suppress("unused")
    constructor(context: Context) : this()

    companion object {
        const val MODE_NONE = 0
        const val MODE_APP = 1
        const val MODE_SHELL = 2
        private const val TAG = "RvcShellAudio"
        private val CALL_SOURCES = setOf(5, 7)  // CAMCORDER, VOICE_COMMUNICATION: privacy-sensitive by default
    }

    @Volatile private var running = false
    private var track: AudioTrack? = null
    private var policy: Any? = null
    private var injectThread: Thread? = null
    private var monitorThread: Thread? = null
    private var captureFd: ParcelFileDescriptor? = null
    private var playbackFd: ParcelFileDescriptor? = null
    private var callback: IShellCallback? = null
    private val looperThread by lazy { HandlerThread("rvc-policy").apply { start() } }

    // shell capture (MODE_SHELL only)
    private val captureLock = Object()
    @Volatile private var record: AudioRecord? = null
    private var captureThread: Thread? = null
    @Volatile private var capturing = false

    // config
    private var sampleRate = 48000
    private var captureSource = 10
    private var intercept = setOf(1, 5, 6, 7, 9)
    private var appUid = -1
    private var a11yComponent = ""

    // monitor state
    private val monitorWake = Object()
    @Volatile private var flushPlayback = false
    @Volatile private var resetTrack = false
    private var recordingCallback: AudioManager.AudioRecordingCallback? = null
    @Volatile private var mode = MODE_NONE
    @Volatile private var victimsJson = JSONArray()
    @Volatile private var a11yOk = false
    @Volatile private var shellSilenced: Boolean? = null
    @Volatile private var appSilenced: Boolean? = null
    @Volatile private var captured = 0L
    @Volatile private var injected = 0L
    @Volatile private var dropped = 0L
    @Volatile private var lastError: String? = null

    override fun destroy() {
        stopInternal()
        System.exit(0)
    }

    override fun uid(): Int = Process.myUid()

    /**
     * Calls arrive from the app over Binder, so Binder.getCallingUid() is the *app* uid. Framework
     * code that checks permissions locally (AudioPolicy.createAudioTrackSource ->
     * checkCallingOrSelfPermission(MODIFY_AUDIO_ROUTING)) would then test the app instead of
     * shell and fail. Run our work under this process's own identity.
     */
    private inline fun <T> asShell(block: () -> T): T {
        val token = Binder.clearCallingIdentity()
        try {
            return block()
        } finally {
            Binder.restoreCallingIdentity(token)
        }
    }

    override fun start(captureOut: ParcelFileDescriptor?, playbackIn: ParcelFileDescriptor?, configJson: String?,
                       cb: IShellCallback?): String =
        asShell { startInternal(captureOut, playbackIn, configJson, cb) }

    private fun startInternal(captureOut: ParcelFileDescriptor?, playbackIn: ParcelFileDescriptor?, configJson: String?,
                              cb: IShellCallback?): String =
        try {
            stopInternal()
            ShellEnv.init()
            val cfg = JSONObject(configJson ?: "{}")
            sampleRate = cfg.optInt("sample_rate", 48000)
            captureSource = cfg.optInt("capture_source", 10)
            intercept = cfg.optJSONArray("intercept")?.let { a -> (0 until a.length()).map { a.getInt(it) }.toSet() } ?: intercept
            appUid = cfg.optInt("app_uid", -1)
            a11yComponent = cfg.optString("a11y_component")
            val maxBacklog = cfg.optInt("max_backlog_ms", 800) * sampleRate / 1000
            captureFd = captureOut
            playbackFd = playbackIn
            callback = cb
            cb?.asBinder()?.linkToDeath({ asShell { stopInternal() } }, 0)
            captureOut?.let { Os.fcntlInt(it.fileDescriptor, OsConstants.F_SETFL, Os.fcntlInt(it.fileDescriptor, OsConstants.F_GETFL, 0) or OsConstants.O_NONBLOCK) }
            running = true
            if (playbackIn != null) startInjection(playbackIn.fileDescriptor, maxBacklog)
            if (captureOut != null) prepareRecord()
            // open the remote-submix input path once now (and clear it), so the first recording
            // after starting does not pay for setting up the audio patch
            if (playbackIn != null) drainSubmix()
            monitorThread = Thread({ monitorLoop() }, "rvc-monitor").apply { start() }
            status()
        } catch (t: Throwable) {
            Log.e(TAG, "start", t)
            stopInternal()
            JSONObject().put("error", "${t.javaClass.simpleName}: ${t.message ?: t.cause?.message}").toString()
        }

    // ---------------------------------------------------------------- recording monitor
    private val getClientUid by lazy {
        runCatching { AudioRecordingConfiguration::class.java.getMethod("getClientUid") }.getOrNull()
    }

    private fun clientUid(c: AudioRecordingConfiguration): Int =
        runCatching { getClientUid?.invoke(c) as Int }.getOrDefault(-1)

    private fun monitorLoop() {
        val am = ShellEnv.audioManager()
        // event-driven: re-evaluate the moment any app starts/stops recording (poll as fallback)
        recordingCallback = object : AudioManager.AudioRecordingCallback() {
            override fun onRecordingConfigChanged(configs: MutableList<AudioRecordingConfiguration>?) {
                synchronized(monitorWake) { monitorWake.notifyAll() }
            }
        }.also { runCatching { am.registerAudioRecordingCallback(it, android.os.Handler(looperThread.looper)) } }
        val myUid = Process.myUid()
        var victimKey = ""
        val tried = mutableSetOf<Int>()
        var silencedSince = 0L
        var idleSince = 0L
        var a11yCheckedAt = 0L
        while (running) {
            try {
                val now = SystemClock.uptimeMillis()
                if (now - a11yCheckedAt > 3000) {
                    a11yOk = a11yComponent.isNotEmpty() && readSecure("enabled_accessibility_services").split(':').any {
                        it.equals(a11yComponent, ignoreCase = true)
                    }
                    a11yCheckedAt = now
                }
                var shellCfg: AudioRecordingConfiguration? = null
                var appCfg: AudioRecordingConfiguration? = null
                val victims = mutableListOf<Pair<Int, Int>>()
                for (c in am.activeRecordingConfigurations) {
                    val uid = clientUid(c)
                    when {
                        uid == myUid -> shellCfg = c
                        uid == appUid -> appCfg = c
                        c.clientAudioSource in intercept -> victims += uid to c.clientAudioSource
                    }
                }
                shellSilenced = shellCfg?.isClientSilenced
                appSilenced = appCfg?.isClientSilenced
                victimsJson = JSONArray(victims.map { JSONObject().put("uid", it.first).put("source", it.second) })
                if (victims.isEmpty()) {
                    victimKey = ""
                    if (mode != MODE_NONE) {
                        if (idleSince == 0L) idleSince = now
                        if (now - idleSince > 1200) setMode(MODE_NONE, "no app is recording")
                    }
                } else {
                    idleSince = 0L
                    val key = victims.map { "${it.first}:${it.second}" }.sorted().joinToString()
                    if (mode == MODE_NONE || key != victimKey) {
                        victimKey = key
                        tried.clear()
                        val inCall = am.mode == AudioManager.MODE_IN_COMMUNICATION || am.mode == AudioManager.MODE_IN_CALL
                        val callLike = inCall || victims.any { it.second in CALL_SOURCES }
                        val m = if (callLike || !a11yOk) MODE_SHELL else MODE_APP
                        tried += m
                        silencedSince = 0L
                        setMode(m, if (callLike) "call / privacy-sensitive recording" else "normal recording")
                    } else {
                        val sil = if (mode == MODE_SHELL) shellSilenced else appSilenced
                        if (sil == true) {
                            if (silencedSince == 0L) silencedSince = now
                            val other = if (mode == MODE_SHELL) MODE_APP else MODE_SHELL
                            if (now - silencedSince > 350 && other !in tried && (other != MODE_APP || a11yOk)) {
                                tried += other
                                silencedSince = 0L
                                setMode(other, "capture was silenced by the system, switched")
                            }
                        } else {
                            silencedSince = 0L
                        }
                    }
                }
            } catch (t: Throwable) {
                lastError = "monitor: ${t.message}"
                Log.w(TAG, "monitor", t)
            }
            synchronized(monitorWake) { monitorWake.wait(100) }
        }
        recordingCallback?.let { cb -> runCatching { am.unregisterAudioRecordingCallback(cb) } }
    }

    private fun setMode(m: Int, why: String) {
        if (m == mode) return
        // converted audio still queued from the previous recording must not leak into the next one
        if (mode == MODE_NONE || m == MODE_NONE) flushPlayback = true
        if (m == MODE_NONE) {
            resetTrack = true
            drainSubmix()
        }
        if (m == MODE_SHELL) startCapture() else stopCapture()
        mode = m
        Log.i(TAG, "capture mode -> $m ($why)")
        runCatching { callback?.onCaptureMode(m, why) }
    }

    /**
     * After a recording ends, the remote submix keeps the last converted audio buffered (the HAL
     * input stream stays open in standby and holds the old frames) and would hand it to the next
     * recording first. Read it out with a short recording of our own through the same mix.
     */
    private fun drainSubmix() {
        val preset = intercept.firstOrNull() ?: return
        Thread({
            runCatching {
                val fmt = AudioFormat.Builder().setSampleRate(sampleRate).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build()
                val b = AudioRecord.Builder().setAudioSource(preset).setAudioFormat(fmt)
                if (Build.VERSION.SDK_INT >= 31) b.setContext(ShellEnv.context)
                val r = b.build()
                if (r.state == AudioRecord.STATE_INITIALIZED) {
                    r.startRecording()
                    val buf = ShortArray(sampleRate / 50)
                    val end = SystemClock.uptimeMillis() + 250
                    while (running && mode == MODE_NONE && SystemClock.uptimeMillis() < end) r.read(buf, 0, buf.size)
                    r.stop()
                }
                r.release()
            }.onFailure { Log.w(TAG, "drain", it) }
        }, "rvc-drain").start()
    }

    private fun readSecure(key: String): String = runCatching {
        val p = ProcessBuilder("settings", "get", "secure", key).redirectErrorStream(true).start()
        p.inputStream.bufferedReader().readText().trim().also { p.waitFor() }
    }.getOrDefault("")

    override fun ensureAccessibility(component: String?): String = asShell {
        try {
            val c = component ?: return@asShell JSONObject().put("error", "no component").toString()
            val cur = readSecure("enabled_accessibility_services").takeIf { it.isNotEmpty() && it != "null" }
            val list = cur?.split(':')?.filter { it.isNotBlank() }?.toMutableList() ?: mutableListOf()
            if (list.none { it.equals(c, ignoreCase = true) }) {
                list += c
                ProcessBuilder("settings", "put", "secure", "enabled_accessibility_services", list.joinToString(":")).start().waitFor()
            }
            ProcessBuilder("settings", "put", "secure", "accessibility_enabled", "1").start().waitFor()
            a11yComponent = c
            a11yOk = readSecure("enabled_accessibility_services").split(':').any { it.equals(c, ignoreCase = true) }
            JSONObject().put("enabled", a11yOk).toString()
        } catch (t: Throwable) {
            JSONObject().put("error", t.toString()).toString()
        }
    }

    // ---------------------------------------------------------------- shell capture (MODE_SHELL)
    /** Built once and kept stopped (an inactive AudioRecord is invisible to other apps). */
    private fun prepareRecord(): AudioRecord? {
        record?.let { return it }
        val sr = sampleRate
        val fmt = AudioFormat.Builder().setSampleRate(sr).setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build()
        val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val b = AudioRecord.Builder().setAudioSource(captureSource).setAudioFormat(fmt)
            .setBufferSizeInBytes(maxOf(min, sr / 25 * 4) * 2)
        if (Build.VERSION.SDK_INT >= 31) b.setContext(ShellEnv.context)
        if (Build.VERSION.SDK_INT >= 30) b.setPrivacySensitive(false)
        val r = try {
            b.build()
        } catch (t: Throwable) {
            lastError = "AudioRecord: ${t.message}"
            return null
        }
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            lastError = "AudioRecord init failed (source $captureSource)"
            r.release()
            return null
        }
        record = r
        return r
    }

    private fun startCapture() {
        val out = captureFd?.fileDescriptor ?: return
        synchronized(captureLock) {
            if (capturing) return
            val r = prepareRecord() ?: return
            val sr = sampleRate
            capturing = true
            r.startRecording()
            captureThread = Thread({
                Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
                val frames = sr / 100
                val buf = ByteBuffer.allocateDirect(frames * 4).order(ByteOrder.nativeOrder())
                while (capturing && running) {
                    buf.clear()
                    val n = r.read(buf, frames * 4, AudioRecord.READ_BLOCKING)
                    if (n <= 0) {
                        if (n < 0) lastError = "AudioRecord.read=$n"
                        if (!capturing) break
                        continue
                    }
                    buf.limit(n)
                    captured += n / 4
                    try {
                        while (buf.hasRemaining()) Os.write(out, buf)
                    } catch (e: ErrnoException) {
                        if (e.errno == OsConstants.EAGAIN) dropped += buf.remaining() / 4
                        else {
                            lastError = "capture pipe: ${e.message}"
                            break
                        }
                    }
                }
            }, "rvc-capture").apply { start() }
        }
    }

    private fun stopCapture(release: Boolean = false) {
        synchronized(captureLock) {
            capturing = false
            runCatching { record?.stop() }
            captureThread?.join(500)
            captureThread = null
            if (release) {
                runCatching { record?.release() }
                record = null
            }
        }
    }

    // ---------------------------------------------------------------- injection
    private fun captureAttr(preset: Int): AudioAttributes {
        val b = AudioAttributes.Builder()
        AudioAttributes.Builder::class.java.getMethod("setCapturePreset", Int::class.javaPrimitiveType).invoke(b, preset)
        return b.build()
    }

    private fun startInjection(input: FileDescriptor, maxBacklog: Int) {
        val sr = sampleRate
        val ruleCls = Class.forName("android.media.audiopolicy.AudioMixingRule")
        val ruleB = Class.forName("android.media.audiopolicy.AudioMixingRule\$Builder").getConstructor().newInstance()
        val addRule = ruleB.javaClass.getMethod("addMixRule", Int::class.javaPrimitiveType, Any::class.java)
        val ruleMatchCapturePreset = 0x2  // AudioMixingRule.RULE_MATCH_ATTRIBUTE_CAPTURE_PRESET
        for (p in intercept) addRule.invoke(ruleB, ruleMatchCapturePreset, captureAttr(p))
        val rule = ruleB.javaClass.getMethod("build").invoke(ruleB)

        val fmt = AudioFormat.Builder().setSampleRate(sr).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build()
        val mixCls = Class.forName("android.media.audiopolicy.AudioMix")
        val mixB = Class.forName("android.media.audiopolicy.AudioMix\$Builder").getConstructor(ruleCls).newInstance(rule)
        mixB.javaClass.getMethod("setFormat", AudioFormat::class.java).invoke(mixB, fmt)
        mixB.javaClass.getMethod("setRouteFlags", Int::class.javaPrimitiveType).invoke(mixB, 0x2)  // ROUTE_FLAG_LOOP_BACK
        val mix = mixB.javaClass.getMethod("build").invoke(mixB)

        val policyCls = Class.forName("android.media.audiopolicy.AudioPolicy")
        val pb = Class.forName("android.media.audiopolicy.AudioPolicy\$Builder").getConstructor(Context::class.java)
            .newInstance(ShellEnv.context)
        pb.javaClass.getMethod("addMix", mixCls).invoke(pb, mix)
        pb.javaClass.getMethod("setLooper", Looper::class.java).invoke(pb, looperThread.looper)
        val pol = pb.javaClass.getMethod("build").invoke(pb)
        val am: AudioManager = ShellEnv.audioManager()
        val res = AudioManager::class.java.getMethod("registerAudioPolicy", policyCls).invoke(am, pol) as Int
        if (res != 0) throw IllegalStateException("registerAudioPolicy failed ($res)")
        policy = pol
        val t = policyCls.getMethod("createAudioTrackSource", mixCls).invoke(pol, mix) as AudioTrack?
            ?: throw IllegalStateException("createAudioTrackSource returned null")
        track = t
        t.play()
        injectThread = Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
            val frames = sr / 100
            val buf = ByteBuffer.allocateDirect(frames * 4 * 8).order(ByteOrder.nativeOrder())
            val pcm = ShortArray(frames * 8)
            val fis = java.io.FileInputStream(input)
            val silence = ShortArray(sr / 200)  // 5 ms
            var written = 0L  // frames written to the track (silence + audio)
            val pfd = arrayOf(android.system.StructPollfd().apply { fd = input; events = OsConstants.POLLIN.toShort() })
            while (running) {
                try {
                    // keep latency bounded if we fall behind
                    val backlog = fis.available() / 4
                    if (backlog > maxBacklog) {
                        val skip = (backlog - maxBacklog / 2) * 4L
                        var left = skip
                        while (left > 0) left -= fis.skip(left)
                        dropped += skip / 4
                    }
                    // No converted audio yet (standby, or the first block of a new recording is
                    // still being computed): keep the injection track fed with silence. A starved
                    // track is made inactive by AudioFlinger, the policy then marks the remote
                    // submix input unavailable and a newly started recording is routed to the real
                    // microphone - the other side would hear the unprocessed voice for ~1 s.
                    if (flushPlayback) {
                        flushPlayback = false
                        var left = fis.available().toLong()
                        while (left > 0) left -= fis.skip(left)
                    }
                    if (resetTrack) {
                        // recording ended: drop the unplayed tail and push ~150 ms of silence through,
                        // which also replaces the remote submix pipe content (with no reader attached
                        // it keeps the newest frames) - the next recording must not start with it
                        resetTrack = false
                        runCatching { t.pause(); t.flush(); t.play() }
                        written = (t.playbackHeadPosition.toLong() and 0xFFFFFFFFL)
                        val flushSilence = ShortArray(sr * 3 / 20)
                        t.write(flushSilence, 0, flushSilence.size, AudioTrack.WRITE_BLOCKING)
                        written += flushSilence.size
                    }
                    pfd[0].revents = 0
                    if (Os.poll(pfd, 2) == 0) {
                        // only top up when almost drained, so real audio never queues behind silence
                        // 32-bit modular difference (the head position wraps after ~24 h at 48 kHz)
                        val queued = ((written and 0xFFFFFFFFL) - (t.playbackHeadPosition.toLong() and 0xFFFFFFFFL)) and 0xFFFFFFFFL
                        if (queued < sr / 100) {
                            t.write(silence, 0, silence.size, AudioTrack.WRITE_BLOCKING)
                            written += silence.size
                        }
                        continue
                    }
                    if ((pfd[0].revents.toInt() and (OsConstants.POLLHUP or OsConstants.POLLERR)) != 0 &&
                        (pfd[0].revents.toInt() and OsConstants.POLLIN) == 0) break
                    buf.clear()
                    val n = Os.read(input, buf)
                    if (n <= 0) {
                        if (n == 0) break
                        continue
                    }
                    while (buf.position() % 4 != 0) Os.read(input, buf)  // keep float alignment
                    if (mode == MODE_NONE) continue  // late output of a finished recording: discard
                    val samples = buf.position() / 4
                    buf.flip()
                    val fb = buf.asFloatBuffer()
                    for (i in 0 until samples) pcm[i] = (fb.get(i).coerceIn(-1f, 1f) * 32767f).roundToInt().toShort()
                    t.write(pcm, 0, samples, AudioTrack.WRITE_BLOCKING)
                    written += samples
                    injected += samples
                } catch (e: ErrnoException) {
                    if (e.errno != OsConstants.EINTR) {
                        lastError = "playback pipe: ${e.message}"
                        break
                    }
                }
            }
        }, "rvc-inject").apply { start() }
    }

    // ---------------------------------------------------------------- control
    override fun stop() = asShell { stopInternal() }

    private fun stopInternal() {
        running = false
        synchronized(monitorWake) { monitorWake.notifyAll() }
        monitorThread?.join(500)
        monitorThread = null
        stopCapture(release = true)
        mode = MODE_NONE
        runCatching { playbackFd?.close() }
        runCatching { captureFd?.close() }
        injectThread?.join(500)
        runCatching { track?.stop(); track?.release() }
        track = null
        policy?.let { p ->
            runCatching {
                AudioManager::class.java.getMethod("unregisterAudioPolicy", p.javaClass).invoke(ShellEnv.audioManager(), p)
            }.onFailure { e -> Log.w(TAG, "unregisterAudioPolicy", e) }
        }
        policy = null
        injectThread = null
        captureFd = null
        playbackFd = null
        callback = null
    }

    override fun status(): String = JSONObject()
        .put("uid", Process.myUid())
        .put("mode", mode)
        .put("injecting", track != null)
        .put("capturing", capturing)
        .put("victims", victimsJson)
        .put("a11y", a11yOk)
        .put("shell_silenced", shellSilenced ?: JSONObject.NULL)
        .put("app_silenced", appSilenced ?: JSONObject.NULL)
        .put("captured", captured)
        .put("injected", injected)
        .put("dropped", dropped)
        .put("sdk", Build.VERSION.SDK_INT)
        .apply { lastError?.let { put("error", it) } }
        .toString()
}
