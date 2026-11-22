package com.mobilervc.engine

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONObject
import java.io.File

data class VoiceModel(
    val id: String,
    val dir: File,
    val name: String,
    val version: String,
    val sampleRate: Int,
    val hasIndex: Boolean,
    val speakers: Int,
)

/** Imported voice models live in filesDir/models/<id>/ (created by the native importer). */
class ModelStore(private val ctx: Context) {
    enum class Kind { PTH, INDEX, OTHER }

    companion object {
        /** By extension, else by content: torch zip checkpoint ("PK..") or faiss index (fourcc "I???"). */
        fun kindOf(f: File, name: String): Kind {
            val lower = name.lowercase()
            if (lower.endsWith(".pth")) return Kind.PTH
            if (lower.endsWith(".index")) return Kind.INDEX
            val head = ByteArray(4)
            val n = runCatching { f.inputStream().use { it.read(head) } }.getOrDefault(0)
            if (n < 4 || f.length() < 1024) return Kind.OTHER
            if (head[0] == 'P'.code.toByte() && head[1] == 'K'.code.toByte() && head[2] == 3.toByte() && head[3] == 4.toByte()) return Kind.PTH
            if (head[0] == 'I'.code.toByte() && (1..3).all { head[it].toInt().toChar().isLetterOrDigit() }) return Kind.INDEX
            return Kind.OTHER
        }
    }

    private val root = File(ctx.filesDir, "models").apply { mkdirs() }

    fun list(): List<VoiceModel> = root.listFiles()?.filter { File(it, "model.json").isFile }?.mapNotNull { dir ->
        runCatching {
            val j = JSONObject(File(dir, "model.json").readText())
            VoiceModel(
                id = dir.name, dir = dir, name = j.optString("name", dir.name), version = j.optString("version"),
                sampleRate = j.optInt("tgt_sr"), hasIndex = j.optBoolean("has_index"), speakers = j.optInt("n_spk", 1),
            )
        }.getOrNull()
    }?.sortedBy { it.name.lowercase() } ?: emptyList()

    fun displayName(uri: Uri): String {
        ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) return c.getString(0) ?: "model"
        }
        return uri.lastPathSegment ?: "model"
    }

    private fun copyToCache(uri: Uri, name: String): File {
        val dir = File(ctx.cacheDir, "import").apply { mkdirs() }
        val f = File(dir, name)
        ctx.contentResolver.openInputStream(uri)!!.use { input -> f.outputStream().use { input.copyTo(it, 1 shl 20) } }
        return f
    }

    /** Imports an original RVC .pth (+ optional .index). Throws with the native error message. */
    fun import(pth: Uri, index: Uri?): VoiceModel {
        val assets = AssetInstaller.baseDir(ctx).absolutePath
        val base = displayName(pth).removeSuffix(".pth")
        // the same file shared twice should still be distinguishable in the model list
        val taken = list().map { it.name }.toSet()
        val name = if (base !in taken) base else (2..999).map { "$base ($it)" }.first { it !in taken }
        val pthFile = copyToCache(pth, "model.pth")
        val idxFile = index?.let { copyToCache(it, "model.index") }
        val id = base.replace(Regex("[^A-Za-z0-9_\\-\\u4e00-\\u9fa5]"), "_") + "_" + System.currentTimeMillis() % 100000
        val dir = File(root, id)
        try {
            val r = JSONObject(Native.importModel(assets, pthFile.absolutePath, idxFile?.absolutePath ?: "", dir.absolutePath, name))
            if (r.has("error")) throw IllegalArgumentException(r.getString("error"))
        } catch (e: Throwable) {
            dir.deleteRecursively()
            throw e
        } finally {
            pthFile.delete()
            idxFile?.delete()
        }
        return list().first { it.id == id }
    }

    fun attachIndex(model: VoiceModel, index: Uri) {
        val f = copyToCache(index, "model.index")
        try {
            val r = JSONObject(Native.attachIndex(model.dir.absolutePath, f.absolutePath))
            if (r.has("error")) throw IllegalArgumentException(r.getString("error"))
        } finally {
            f.delete()
        }
    }

    fun delete(model: VoiceModel) {
        model.dir.deleteRecursively()
    }
}
