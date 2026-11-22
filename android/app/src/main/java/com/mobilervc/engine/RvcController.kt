package com.mobilervc.engine

import android.content.Context
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.mobilervc.shizuku.ShizukuBridge
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors

/**
 * App-wide owner of the native engine. The UI observes the Compose state below; the
 * foreground service calls start()/stop(). All native calls that may block run on [worker].
 */
object RvcController {
    private const val TAG = "RvcController"
    private lateinit var app: Context
    private val worker = Executors.newSingleThreadExecutor()
    private var handle = 0L
    private var loadedKey: String? = null
    private var shellActive = false

    var settings by mutableStateOf(Settings())
        private set
    var models by mutableStateOf(emptyList<VoiceModel>())
        private set
    var assetsReady by mutableStateOf(false)
        private set
    var installProgress by mutableStateOf<Float?>(null)
        private set
    var running by mutableStateOf(false)
        private set
    var busy by mutableStateOf<String?>(null)
        private set
    var status by mutableStateOf<JSONObject?>(null)
        private set
    var message by mutableStateOf<String?>(null)
    var benchResult by mutableStateOf<String?>(null)
        private set
    var device: JSONObject? = null
        private set

    val store by lazy { ModelStore(app) }

    fun init(ctx: Context) {
        if (::app.isInitialized) return
        app = ctx.applicationContext
        // QNN HTP loads its DSP skeleton libraries via FastRPC from these directories
        runCatching {
            val lib = app.applicationInfo.nativeLibraryDir
            android.system.Os.setenv("ADSP_LIBRARY_PATH", "$lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp", true)
        }
        settings = Settings.load(app)
        device = runCatching { JSONObject(Native.deviceInfo()) }.getOrNull()
        models = store.list()
        worker.execute { installAssets() }
    }

    private fun installAssets() {
        try {
            if (!AssetInstaller.isReady(app) || AssetInstaller.apkHasAssets(app)) {
                installProgress = 0f
                AssetInstaller.ensure(app) { done, total -> if (total > 0) installProgress = done.toFloat() / total }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "asset install", t)
            message = "基础模型安装失败: ${t.message}"
        } finally {
            installProgress = null
        }
        assetsReady = AssetInstaller.isReady(app)
        if (assetsReady && handle == 0L) handle = Native.create(AssetInstaller.baseDir(app).absolutePath)
    }

    fun importBasePack(open: () -> java.io.InputStream) = worker.execute {
        busy = "导入基础模型包…"
        try {
            open().use { AssetInstaller.installZip(app, it) }
            installAssets()
            message = "基础模型包已导入"
        } catch (t: Throwable) {
            message = "导入失败: ${t.message}"
        } finally {
            busy = null
        }
    }

    fun update(change: (Settings) -> Settings) {
        val s = change(settings)
        settings = s
        Settings.save(app, s)
        pushLive()
    }

    private fun pushLive() {
        val s = settings
        if (handle != 0L) {
            Native.setParams(handle, s.pitch, s.formant, s.indexRate, s.rmsMixRate, s.threshold, s.inputNoiseReduce, s.outputNoiseReduce)
        }
    }

    fun refreshModels() {
        models = store.list()
    }

    // ------------------------------------------------------------------ files opened/shared from other apps
    /** .pth / .index received via "open with" or "share" (QQ, WeChat, file managers), staged in cache. */
    data class SharedFiles(val pth: java.io.File?, val index: java.io.File?)

    var shared by mutableStateOf<SharedFiles?>(null)
        private set

    private val incomingDir get() = java.io.File(app.cacheDir, "incoming")

    /** Copies the files right away: URI grants from the sending app are only temporary. */
    fun receiveShared(uris: List<android.net.Uri>) = worker.execute {
        busy = "接收文件…"
        try {
            val dir = incomingDir.apply { deleteRecursively(); mkdirs() }
            var pth: java.io.File? = null
            var idx: java.io.File? = null
            val rejected = mutableListOf<String>()
            for (u in uris) {
                val name = runCatching { store.displayName(u) }.getOrDefault("file").substringAfterLast('/').ifBlank { "file" }
                val f = java.io.File(dir, name)
                app.contentResolver.openInputStream(u)?.use { i -> f.outputStream().use { i.copyTo(it, 1 shl 20) } }
                    ?: throw IllegalStateException("无法打开 $name")
                when (ModelStore.kindOf(f, name)) {
                    ModelStore.Kind.PTH -> pth = f
                    ModelStore.Kind.INDEX -> idx = f
                    ModelStore.Kind.OTHER -> {
                        rejected += name
                        f.delete()
                    }
                }
            }
            if (pth == null && idx == null) {
                message = "不是 RVC 模型文件：${rejected.joinToString()}\n请分享 RVC 导出的 .pth 模型或 added_*.index 索引。"
            } else {
                shared = SharedFiles(pth, idx)
                if (rejected.isNotEmpty()) message = "已忽略非模型文件：${rejected.joinToString()}"
            }
        } catch (t: Throwable) {
            Log.e(TAG, "receiveShared", t)
            message = "读取分享的文件失败: ${t.message}"
        } finally {
            busy = null
        }
    }

    /** Dialog dismissed: drop staged files (on the worker, after any queued import used them). */
    fun clearShared() {
        shared = null
        worker.execute { incomingDir.deleteRecursively() }
    }

    fun importModel(pth: android.net.Uri, index: android.net.Uri?) = worker.execute {
        busy = "导入模型…"
        try {
            val m = store.import(pth, index)
            models = store.list()
            update { it.copy(modelId = m.id) }
            message = "已导入 ${m.name} (${m.version}, ${m.sampleRate / 1000}k${if (m.hasIndex) ", 含索引" else ""})"
        } catch (t: Throwable) {
            message = "导入失败: ${t.message}"
        } finally {
            busy = null
            incomingDir.deleteRecursively()
        }
    }

    fun attachIndex(model: VoiceModel, index: android.net.Uri) = worker.execute {
        busy = "转换索引…"
        try {
            store.attachIndex(model, index)
            models = store.list()
            loadedKey = null
            message = "索引已添加到 ${model.name}"
        } catch (t: Throwable) {
            message = "索引导入失败: ${t.message}"
        } finally {
            busy = null
            incomingDir.deleteRecursively()
        }
    }

    fun deleteModel(model: VoiceModel) = worker.execute {
        if (settings.modelId == model.id) {
            stopBlocking()
            loadedKey = null
        }
        store.delete(model)
        models = store.list()
    }

    /** Loads the selected model if the model or performance settings changed. Returns error or null. */
    private fun ensureLoaded(): String? {
        if (handle == 0L) return "基础模型未就绪"
        val s = settings
        val model = models.firstOrNull { it.id == s.modelId } ?: return "请先选择/导入模型"
        val cfg = s.engineConfig(app, device)
        val key = model.id + cfg.toString()
        if (key == loadedKey) return null
        busy = "加载模型（首次使用加速器时需编译，可能较慢）…"
        val r = JSONObject(Native.load(handle, model.dir.absolutePath, cfg.toString()))
        busy = null
        if (r.has("error")) {
            loadedKey = null
            return r.getString("error")
        }
        loadedKey = key
        pushLive()
        r.optJSONArray("warnings")?.let { w -> if (w.length() > 0) message = (0 until w.length()).joinToString("\n") { w.getString(it) } }
        return null
    }

    fun start(onResult: (String?) -> Unit = {}) = worker.execute {
        val err = runCatching { startBlocking() }.getOrElse { it.message ?: it.toString() }
        if (err != null) message = err
        onResult(err)
    }

    /** Capture presets redirected to the converted voice in virtual-mic mode (DEFAULT maps to MIC). */
    private val INTERCEPT = listOf(1, 5, 6, 7, 9)  // MIC, CAMCORDER, VOICE_RECOGNITION, VOICE_COMMUNICATION, UNPROCESSED
    /** Our own capture uses VOICE_PERFORMANCE: low latency, and never redirected. */
    private const val CAPTURE_PRESET = 10

    /** 0 idle, 1 app captures (voice messages), 2 shell captures (calls); see IShellCallback. */
    var captureMode by mutableStateOf(0)
        private set

    private val shellCallback = object : com.mobilervc.shizuku.IShellCallback.Stub() {
        override fun onCaptureMode(mode: Int, detail: String?) {
            captureMode = mode
            if (handle != 0L) Native.setCaptureMode(handle, mode)
            Log.i(TAG, "capture mode $mode: $detail")
        }
    }

    private fun startBlocking(): String? {
        stopBlocking()
        ensureLoaded()?.let { return it }
        // Native.start() runs two silent blocks first, so the slow first inferences (GPU shader
        // compilation, NPU wake-up) are over before audio starts
        busy = "预热模型…"
        try {
            return startStreams()
        } finally {
            busy = null
        }
    }

    private fun startStreams(): String? {
        val s = settings
        val opts = JSONObject()
            .put("input_preset", CAPTURE_PRESET)
            .put("passthrough", s.passthrough)
            .put("output_gain", s.outputGain.toDouble())
            .put("sample_rate", 48000)
        // test hook: a 48 kHz wav placed at files/debug_input.wav replaces the microphone signal
        java.io.File(app.filesDir, "debug_input.wav").takeIf { it.isFile }?.let { opts.put("debug_input", it.absolutePath) }
        val r: JSONObject
        if (s.mode == AudioMode.LOCAL) {
            opts.put("local_input", true).put("local_output", true)
            r = JSONObject(Native.start(handle, opts.toString(), -1, -1))
        } else {
            val shell = ShizukuBridge.connect(app) ?: return "无法连接 Shizuku 服务：${ShizukuBridge.stateText}"
            // voice messages need this app's capture to count as accessibility capture
            val a11y = "${app.packageName}/${com.mobilervc.service.MicAccessibilityService::class.java.name}"
            runCatching { shell.ensureAccessibility(a11y) }
            val cap = Native.createPipe() ?: return "pipe 创建失败"
            val play = Native.createPipe() ?: return "pipe 创建失败"
            val capW = ParcelFileDescriptor.adoptFd(cap[1])
            val playR = ParcelFileDescriptor.adoptFd(play[0])
            val shellCfg = JSONObject()
                .put("sample_rate", 48000)
                .put("capture_source", CAPTURE_PRESET)
                .put("intercept", JSONArray(INTERCEPT))
                .put("app_uid", android.os.Process.myUid())
                .put("a11y_component", a11y)
                .put("max_backlog_ms", (s.blockTime * 2000).toInt() + 100)
                .put("comfort_noise_db", if (s.comfortNoise) s.comfortNoiseDb.toDouble() else 0.0)
            // the driver must be running before the shell may ask for local capture
            opts.put("local_input", false).put("local_output", s.monitor).put("input_exclusive", false)
                .put("virtual_capture", true)
            r = JSONObject(Native.start(handle, opts.toString(), cap[0], play[1]))
            if (r.has("error")) {
                capW.close()
                playR.close()
                return r.getString("error")
            }
            val sr = try {
                JSONObject(shell.start(capW, playR, shellCfg.toString(), shellCallback))
            } finally {
                capW.close()
                playR.close()
            }
            if (sr.has("error")) {
                Native.stop(handle)
                return "Shizuku 端启动失败: ${sr.getString("error")}"
            }
            shellActive = true
            captureMode = 0
        }
        if (r.has("error")) {
            stopBlocking()
            return r.getString("error")
        }
        running = true
        return null
    }

    fun stop() = worker.execute { stopBlocking() }

    private fun stopBlocking() {
        if (shellActive) {
            runCatching { ShizukuBridge.service?.stop() }
            shellActive = false
        }
        if (handle != 0L) Native.stop(handle)
        captureMode = 0
        running = false
    }

    fun pollStatus() {
        if (handle == 0L) return
        val j = runCatching { JSONObject(Native.status(handle)) }.getOrNull() ?: return
        if (shellActive) runCatching { ShizukuBridge.service?.status() }.getOrNull()?.let { j.put("shell", JSONObject(it)) }
        status = j
    }

    fun bench() = worker.execute {
        if (running) {
            message = "请先停止实时变声"
            return@execute
        }
        ensureLoaded()?.let {
            message = it
            return@execute
        }
        busy = "性能测试中…"
        val r = JSONObject(Native.bench(handle, 20))
        busy = null
        benchResult = if (r.has("error")) r.getString("error") else {
            val be = r.optJSONObject("engine")?.optJSONObject("backends")
            val indep = r.optBoolean("independent_stages")
            val suggest = r.optDouble("suggest_block_ms", 0.0)
            "每块 %.0f ms：平均 %.1f ms（HuBERT %.1f / 声码器 %.1f），最慢 %.1f ms，实时率 %.2f\n".format(
                r.getDouble("block_ms"), r.getDouble("avg_ms"), r.getDouble("hubert_ms"), r.getDouble("dec_ms"),
                r.getDouble("worst_ms"), r.getDouble("rtf"),
            ) + "分析 %.1f ms / 合成 %.1f ms，%s\n".format(
                r.optDouble("analyze_ms"), r.optDouble("synth_ms"),
                if (indep) "两段在不同处理器上，实时运行时并行" else "两段在同一处理器上，顺序执行",
            ) + (if (suggest > 0 && suggest < r.getDouble("block_ms") - 5)
                "采样长度可以试着降到 %.2f s 左右以降低延迟（降低后再测一次，实时率保持在 0.8 以下）\n".format(suggest / 1000.0)
            else "") + "后端：${be?.toString() ?: "-"}"
        }
    }
}
