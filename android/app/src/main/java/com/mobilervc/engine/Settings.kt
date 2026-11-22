package com.mobilervc.engine

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** Compute backend choices exposed in the UI (native names understood by the engine). */
enum class Backend(val id: String, val label: String, val short: String) {
    CPU("cpu", "CPU (ONNX Runtime)", "CPU"),
    NATIVE("native", "CPU 原生 NEON", "NEON"),
    XNNPACK("xnnpack", "CPU XNNPACK", "XNNPACK"),
    GPU("webgpu", "GPU (WebGPU/Vulkan)", "GPU"),
    NNAPI("nnapi", "NNAPI (GPU/NPU)", "NNAPI"),
    QNN_HTP("qnn_htp", "NPU 骁龙 HTP (QNN)", "NPU"),
    QNN_GPU("qnn_gpu", "GPU 骁龙 Adreno (QNN)", "GPU");

    companion object {
        fun of(id: String) = entries.firstOrNull { it.id == id } ?: CPU
    }
}

enum class AudioMode { LOCAL, VIRTUAL_MIC }

/**
 * User settings, mirroring RVC's realtime GUI (常规设置 / 性能设置) plus mobile options.
 * Defaults favour realtime on phones: shorter extra context than the PC default of 2.5 s.
 */
data class Settings(
    val modelId: String = "",
    // live parameters
    val threshold: Float = -60f,
    val pitch: Float = 0f,
    val formant: Float = 0f,
    val indexRate: Float = 0f,
    val rmsMixRate: Float = 0f,
    val inputNoiseReduce: Boolean = false,
    val outputNoiseReduce: Boolean = false,
    // performance (restart required)
    val blockTime: Float = 0.30f,
    val crossfadeTime: Float = 0.05f,
    val extraTime: Float = 0.50f,
    val hubertBackend: String = Backend.CPU.id,
    val decBackend: String = Backend.NATIVE.id,
    val fp16: Boolean = true,
    val threads: Int = 0,          // 0 = number of big cores
    val phaseLock: Boolean = false, // experimental (RVC restarts the sine phase every block)
    val sid: Int = 0,
    // audio
    val mode: AudioMode = AudioMode.LOCAL,
    val monitor: Boolean = true,   // play converted voice locally
    val outputGain: Float = 1f,
    val passthrough: Boolean = false, // RVC "输入监听"
    // virtual mic: a faint noise floor instead of digital silence (apps abort "dead" recordings)
    val comfortNoise: Boolean = true,
    val comfortNoiseDb: Float = -60f, // dBFS RMS
) {
    fun engineConfig(ctx: Context, device: JSONObject?): JSONObject {
        val big = device?.optJSONArray("big_cores") ?: JSONArray()
        val nThreads = if (threads > 0) threads else maxOf(1, minOf(6, big.length()))
        fun be(id: String) = JSONObject().put("backend", id).put("fp16", fp16).put("threads", nThreads)
        return JSONObject()
            .put("sample_rate", 48000)
            .put("block_time", blockTime.toDouble())
            .put("crossfade_time", crossfadeTime.toDouble())
            .put("extra_time", extraTime.toDouble())
            .put("sid", sid)
            .put("phase_lock", phaseLock)
            .put("threads", nThreads)
            .put("affinity", big)
            .put("cache_dir", ctx.cacheDir.resolve("accel_cache").absolutePath)
            .put("hubert", be(hubertBackend))
            .put("fcpe", JSONObject().put("backend", "cpu").put("threads", minOf(2, nThreads)))
            .put("synth", JSONObject().put("backend", "cpu").put("threads", nThreads))
            .put("dec", be(decBackend))
    }

    companion object {
        private const val PREFS = "settings"

        fun load(ctx: Context): Settings {
            val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val d = Settings()
            return Settings(
                modelId = p.getString("modelId", d.modelId)!!,
                threshold = p.getFloat("threshold", d.threshold),
                pitch = p.getFloat("pitch", d.pitch),
                formant = p.getFloat("formant", d.formant),
                indexRate = p.getFloat("indexRate", d.indexRate),
                rmsMixRate = p.getFloat("rmsMixRate", d.rmsMixRate),
                inputNoiseReduce = p.getBoolean("inputNoiseReduce", d.inputNoiseReduce),
                outputNoiseReduce = p.getBoolean("outputNoiseReduce", d.outputNoiseReduce),
                blockTime = p.getFloat("blockTime", d.blockTime),
                crossfadeTime = p.getFloat("crossfadeTime", d.crossfadeTime),
                extraTime = p.getFloat("extraTime", d.extraTime),
                hubertBackend = p.getString("hubertBackend", d.hubertBackend)!!,
                decBackend = p.getString("decBackend", d.decBackend)!!,
                fp16 = p.getBoolean("fp16", d.fp16),
                threads = p.getInt("threads", d.threads),
                phaseLock = p.getBoolean("phaseLock", d.phaseLock),
                sid = p.getInt("sid", d.sid),
                mode = runCatching { AudioMode.valueOf(p.getString("mode", d.mode.name)!!) }.getOrDefault(d.mode),
                monitor = p.getBoolean("monitor", d.monitor),
                outputGain = p.getFloat("outputGain", d.outputGain),
                passthrough = p.getBoolean("passthrough", d.passthrough),
                comfortNoise = p.getBoolean("comfortNoise", d.comfortNoise),
                comfortNoiseDb = p.getFloat("comfortNoiseDb", d.comfortNoiseDb),
            )
        }

        fun save(ctx: Context, s: Settings) {
            ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString("modelId", s.modelId)
                .putFloat("threshold", s.threshold).putFloat("pitch", s.pitch).putFloat("formant", s.formant)
                .putFloat("indexRate", s.indexRate).putFloat("rmsMixRate", s.rmsMixRate)
                .putBoolean("inputNoiseReduce", s.inputNoiseReduce).putBoolean("outputNoiseReduce", s.outputNoiseReduce)
                .putFloat("blockTime", s.blockTime).putFloat("crossfadeTime", s.crossfadeTime)
                .putFloat("extraTime", s.extraTime)
                .putString("hubertBackend", s.hubertBackend).putString("decBackend", s.decBackend)
                .putBoolean("fp16", s.fp16).putInt("threads", s.threads).putBoolean("phaseLock", s.phaseLock)
                .putInt("sid", s.sid).putString("mode", s.mode.name)
                .putBoolean("monitor", s.monitor).putFloat("outputGain", s.outputGain)
                .putBoolean("passthrough", s.passthrough)
                .putBoolean("comfortNoise", s.comfortNoise).putFloat("comfortNoiseDb", s.comfortNoiseDb)
                .apply()
        }
    }
}
