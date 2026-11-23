package com.mobilervc.engine

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import com.mobilervc.service.ConvertService
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

/**
 * File conversion (RVC WebUI 模型推理, natively): an audio file picked in the app or opened /
 * shared from another app is converted with the selected model; results are kept in
 * filesDir/convert/out to play, share or save to Music.
 */
object Converter {
    private const val TAG = "Converter"
    private lateinit var app: Context
    /** Copies and conversions; not RvcController's worker, which must stay free for imports. */
    private val io = Executors.newSingleThreadExecutor()

    /** The file to convert, copied into the app (shares are only readable for a short while). */
    data class Input(val file: File, val name: String, val durationS: Double, val info: String)

    data class Result(
        val file: File, val source: String, val model: String, val pitch: Int, val format: String,
        val durationS: Double, val created: Long, val size: Long,
    )

    var input by mutableStateOf<Input?>(null)
        private set
    var staging by mutableStateOf(false)
        private set
    var converting by mutableStateOf(false)
        private set
    /** 0..1; 0 while the models load. */
    var progress by mutableStateOf(0f)
        private set
    var startedAt = 0L
        private set
    var results by mutableStateOf(emptyList<Result>())
        private set
    /** The conversion that just finished (the completion dialog). */
    var finished by mutableStateOf<Result?>(null)

    private val inDir get() = File(app.filesDir, "convert/input").apply { mkdirs() }
    private val outDir get() = File(app.filesDir, "convert/out").apply { mkdirs() }

    fun init(ctx: Context) {
        if (::app.isInitialized) return
        app = ctx.applicationContext
        io.execute {
            // temporary files of a conversion the system killed
            outDir.listFiles()?.filter { it.name.startsWith(".") }?.forEach { it.delete() }
            results = list()
            // the last input survives restarts, to convert it again with other settings
            inDir.listFiles()?.firstOrNull { !it.name.endsWith(".json") }?.let { f ->
                runCatching { input = inputOf(f, JSONObject(File(f.path + ".json").readText()).getString("name")) }
            }
        }
    }

    private fun inputOf(f: File, name: String): Input {
        val p = JSONObject(Native.probeAudio(f.absolutePath))
        if (p.has("error")) throw IllegalArgumentException(p.getString("error"))
        val ch = p.optInt("channels")
        // the extension says more than the track's MIME type (WAV and FLAC both decode as "audio/raw")
        val kind = name.substringAfterLast('.', "").takeIf { it.length in 2..5 }?.uppercase()
            ?: p.optString("mime").substringAfter('/')
        val info = "$kind · ${p.optInt("sample_rate") / 1000.0} kHz · " + if (ch == 1) "单声道" else "$ch 声道"
        return Input(f, name, p.optDouble("duration_s"), info)
    }

    /** Can the native decoder read it? (null) Else why not, for the user. */
    fun probe(f: File): String? {
        val p = JSONObject(Native.probeAudio(f.absolutePath))
        return if (p.has("error")) p.getString("error") else null
    }

    /** Makes `f` (already a private copy) the input; it is moved, not copied. Any thread. */
    fun adopt(f: File, name: String) {
        clearInputs()
        val dst = File(inDir, "input." + name.substringAfterLast('.', "bin").take(8))
        if (!f.renameTo(dst)) {
            f.copyTo(dst, overwrite = true)
            f.delete()
        }
        File(dst.path + ".json").writeText(JSONObject().put("name", name).toString())
        input = inputOf(dst, name)
    }

    /** A file picked in the app. */
    fun pick(uri: Uri) = io.execute {
        staging = true
        try {
            val name = RvcController.store.displayName(uri)
            val tmp = File(app.cacheDir, "pick_" + System.nanoTime())
            app.contentResolver.openInputStream(uri)?.use { i -> tmp.outputStream().use { i.copyTo(it, 1 shl 20) } }
                ?: throw IllegalStateException("无法打开 $name")
            try {
                adopt(tmp, name)
            } finally {
                tmp.delete()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "pick", t)
            RvcController.message = t.message ?: t.toString()
        } finally {
            staging = false
        }
    }

    private fun clearInputs() {
        inDir.listFiles()?.forEach { it.delete() }
        input = null
    }

    // ------------------------------------------------------------------ conversion
    /** Starts converting [input] with the current settings; null, or why it cannot start. */
    fun start(ctx: Context): String? {
        val c = RvcController
        val inp = input ?: return "请先选择音频文件"
        if (converting) return null
        if (c.running) return "请先停止实时变声"
        if (!c.assetsReady) return "基础模型未就绪"
        val model = c.models.firstOrNull { it.id == c.settings.modelId } ?: return "请先选择或导入模型"
        val s = c.settings
        converting = true
        progress = 0f
        startedAt = SystemClock.elapsedRealtime()
        ConvertService.start(ctx)
        io.execute { run(inp, model, s) }
        return null
    }

    fun cancel() {
        if (converting) RvcController.nativeHandle().takeIf { it != 0L }?.let { Native.cancelOffline(it) }
    }

    /** Leaving the conversion screen: give the memory of its models back. */
    fun release() = io.execute {
        RvcController.nativeHandle().takeIf { it != 0L }?.let { Native.releaseOffline(it) }
    }

    private fun run(inp: Input, model: VoiceModel, s: Settings) {
        val handle = RvcController.nativeHandle()
        val ext = if (s.convFormat == "m4a") "m4a" else "wav"
        val base = inp.name.substringBeforeLast('.').ifBlank { "audio" }
        val stem = sanitize("${base}_${model.name}" + if (s.convPitch != 0) "_%+d".format(s.convPitch) else "")
        val out = generateSequence(1) { it + 1 }.map { File(outDir, if (it == 1) "$stem.$ext" else "$stem ($it).$ext") }.first { !it.exists() }
        val part = File(outDir, ".${out.name}.part")
        val poll = Thread {
            while (converting) {
                progress = Native.offlineProgress(handle)
                Thread.sleep(200)
            }
        }
        try {
            if (handle == 0L) throw IllegalStateException("基础模型未就绪")
            RvcController.realtimeModelsUnloaded()  // the native side frees them for the offline models
            poll.start()
            val r = JSONObject(
                Native.convertFile(handle, model.dir.absolutePath, s.offlineConfig(RvcController.device).toString(),
                    s.offlineParams().toString(), inp.file.absolutePath, part.absolutePath, ext),
            )
            if (r.has("error")) {
                val e = r.getString("error")
                if (e != "cancelled") RvcController.message = "转换失败：$e"
                return
            }
            if (!part.renameTo(out)) throw IllegalStateException("无法保存 ${out.name}")
            File(out.path + ".json").writeText(
                JSONObject().put("source", inp.name).put("model", model.name).put("pitch", s.convPitch).put("format", ext)
                    .put("duration_s", r.optDouble("duration_s")).put("created", System.currentTimeMillis())
                    .put("convert_s", r.optDouble("total_s")).toString(),
            )
            results = list()
            finished = results.firstOrNull { it.file == out }
            Log.i(TAG, "converted ${inp.name} -> ${out.name}: $r")
        } catch (t: Throwable) {
            Log.e(TAG, "convert", t)
            RvcController.message = "转换失败：${t.message ?: t}"
        } finally {
            part.delete()
            RvcController.realtimeModelsUnloaded()
            converting = false
        }
    }

    private fun sanitize(s: String) = s.replace(Regex("[\\\\/:*?\"<>|\\p{Cntrl}]"), "_").trim().take(80)

    private fun list(): List<Result> = outDir.listFiles()
        ?.filter { !it.name.endsWith(".json") && !it.name.startsWith(".") }
        ?.map { f ->
            val j = runCatching { JSONObject(File(f.path + ".json").readText()) }.getOrDefault(JSONObject())
            Result(
                file = f, source = j.optString("source", f.name), model = j.optString("model"), pitch = j.optInt("pitch"),
                format = f.extension, durationS = j.optDouble("duration_s", 0.0), created = j.optLong("created", f.lastModified()),
                size = f.length(),
            )
        }
        ?.sortedByDescending { it.created } ?: emptyList()

    fun delete(r: Result) {
        r.file.delete()
        File(r.file.path + ".json").delete()
        if (finished?.file == r.file) finished = null
        results = list()
    }

    // ------------------------------------------------------------------ sharing
    private fun mime(r: Result) = if (r.format == "m4a") "audio/mp4" else "audio/wav"

    /** System share sheet (WeChat, QQ, ...) with the file itself. */
    fun share(ctx: Context, r: Result) {
        val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", r.file)
        val send = Intent(Intent.ACTION_SEND).setType(mime(r)).putExtra(Intent.EXTRA_STREAM, uri)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        send.clipData = android.content.ClipData.newRawUri(r.file.name, uri)
        ctx.startActivity(Intent.createChooser(send, "分享 ${r.file.name}").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
    }

    /** Copies the result into the shared Music/RVC folder (no storage permission needed). */
    fun saveToMusic(r: Result) = io.execute {
        try {
            val cr = app.contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Audio.Media.DISPLAY_NAME, r.file.name)
                put(MediaStore.Audio.Media.MIME_TYPE, mime(r))
                put(MediaStore.Audio.Media.RELATIVE_PATH, "Music/RVC")
                put(MediaStore.Audio.Media.IS_PENDING, 1)
            }
            val collection = MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val uri = cr.insert(collection, values) ?: throw IllegalStateException("系统拒绝创建文件")
            try {
                cr.openOutputStream(uri)!!.use { o -> r.file.inputStream().use { it.copyTo(o, 1 shl 20) } }
                cr.update(uri, ContentValues().apply { put(MediaStore.Audio.Media.IS_PENDING, 0) }, null, null)
            } catch (t: Throwable) {
                cr.delete(uri, null, null)
                throw t
            }
            val shown = cr.query(uri, arrayOf(MediaStore.Audio.Media.DISPLAY_NAME), null, null, null)?.use {
                if (it.moveToFirst()) it.getString(0) else null
            } ?: r.file.name
            RvcController.message = "已保存到 音乐/RVC/$shown"
        } catch (t: Throwable) {
            Log.e(TAG, "save", t)
            RvcController.message = "保存失败：${t.message ?: t}"
        }
    }

    /** "1:05" */
    fun clock(seconds: Double): String {
        val t = seconds.toLong().coerceAtLeast(0)
        return if (t >= 3600) "%d:%02d:%02d".format(t / 3600, t / 60 % 60, t % 60) else "%d:%02d".format(t / 60, t % 60)
    }
}
