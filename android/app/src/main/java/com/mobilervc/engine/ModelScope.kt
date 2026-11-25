package com.mobilervc.engine

import org.json.JSONObject
import java.net.URLEncoder

/**
 * The online model library: RVC models on ModelScope (魔搭, modelscope.cn), which is reachable
 * from mainland China and abroad and needs no account for public models. Most results are
 * "sail-rvc/<name>" repos (a mirror of community RVC models: model.pth + model.index each); others
 * are collections with one folder per voice, browsed folder by folder.
 */
object ModelScope {
    private const val HOST = "https://www.modelscope.cn"
    const val PAGE_SIZE = 20

    data class Repo(val id: String, val title: String, val downloads: Int, val size: Long)
    data class Page(val repos: List<Repo>, val total: Int)

    sealed class Item {
        abstract val name: String
        data class Folder(override val name: String, val path: String) : Item()
        /** A model: .pth plus the index that belongs to it, if the folder has one. [title] names the imported model. */
        data class Voice(override val name: String, val title: String, val pth: String, val pthSize: Long, val index: String?, val indexSize: Long) : Item()
        data class Archive(override val name: String, val path: String, val size: Long) : Item()
    }

    private fun get(url: String): JSONObject {
        val conn = ModelDownloader.open(url, timeoutMs = 15_000)
        try {
            return JSONObject(conn.inputStream.bufferedReader().readText())
        } finally {
            conn.disconnect()
        }
    }

    private fun enc(s: String) = URLEncoder.encode(s, "UTF-8").replace("+", "%20")

    /** RVC repos matching [keyword] (all words must match), most downloaded first. [page] starts at 1. */
    fun search(keyword: String, page: Int): Page {
        val words = keyword.trim()
        val q = if (words.contains("rvc", ignoreCase = true)) words else "rvc $words".trim()
        val d = get("$HOST/openapi/v1/models?search=${enc(q)}&sort=downloads&page_size=$PAGE_SIZE&page_number=$page")
            .getJSONObject("data")
        val arr = d.getJSONArray("models")
        val repos = (0 until arr.length()).map { i ->
            val m = arr.getJSONObject(i)
            val id = m.getString("id")
            Repo(id, m.optString("display_name").ifBlank { id.substringAfter('/') }, m.optInt("downloads"), m.optLong("file_size"))
        }
        return Page(repos, d.optInt("total_count", repos.size))
    }

    /** G_/D_ training checkpoints and the f0G40k-style pretrained bases are not voice models. */
    fun isTrainingCheckpoint(fileName: String) =
        Regex("^(f0)?[GD](_\\d+|\\d{2}k.*)\\.pth$", RegexOption.IGNORE_CASE).matches(fileName)

    /** The models, archives and sub-folders in one folder ("" = repo root) of [repo]. */
    fun list(repo: Repo, folder: String): List<Item> {
        val root = if (folder.isEmpty()) "" else "&Root=${enc(folder)}"
        val files = get("$HOST/api/v1/models/${repo.id}/repo/files?Revision=master$root")
            .getJSONObject("Data").getJSONArray("Files")
        data class F(val name: String, val path: String, val dir: Boolean, val size: Long)
        val all = (0 until files.length()).map { i ->
            val f = files.getJSONObject(i)
            F(f.getString("Name"), f.getString("Path"), f.optString("Type") == "tree", f.optLong("Size"))
        }
        val folders = all.filter { it.dir && !it.name.startsWith(".") }.map { Item.Folder(it.name, it.path) }
        val pths = all.filter { !it.dir && it.name.endsWith(".pth", true) && !isTrainingCheckpoint(it.name) }
        val indexes = all.filter { !it.dir && it.name.endsWith(".index", true) }
        val archives = all.filter { !it.dir && it.name.endsWith(".zip", true) }.map { Item.Archive(it.name, it.path, it.size) }
        // a lone model in its folder is named after the folder ("七七/qiqi.pth" → 七七) or the repo
        val place = folder.substringAfterLast('/').ifEmpty { repo.title }
        val voices = pths.map { p ->
            val stem = p.name.substringBeforeLast('.')
            val idx = if (pths.size == 1) {
                indexes.sortedWith(compareByDescending<F> { it.name.contains(stem, true) }
                    .thenByDescending { it.name.startsWith("added") }.thenByDescending { it.size }).firstOrNull()
            } else {
                // several models in one folder: only an index carrying the model's name is its own
                indexes.filter { it.name.contains(stem, true) }.maxByOrNull { it.name.startsWith("added") }
            }
            val title = if (pths.size == 1 && (stem.equals("model", true) || folder.isNotEmpty())) place else stem
            Item.Voice(p.name, title, p.path, p.size, idx?.path, idx?.size ?: 0)
        }
        return voices + archives + folders.sortedBy { it.name }
    }

    fun fileUrl(repo: Repo, path: String) =
        "$HOST/models/${repo.id}/resolve/master/" + path.split('/').joinToString("/") { enc(it) }
}
