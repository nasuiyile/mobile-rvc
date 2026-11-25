package com.mobilervc.engine

import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLDecoder
import java.nio.charset.Charset
import java.util.zip.ZipEntry
import java.util.zip.ZipFile

/** HTTP downloads for the online model library: a .pth, an .index, or a .zip holding both. */
object ModelDownloader {
    class Cancelled : IOException("已取消")

    /** Opens [url] following redirects (CDNs chain several, across hosts); throws on non-200. */
    fun open(url: String, timeoutMs: Int = 20_000): HttpURLConnection {
        var target = URL(url)
        repeat(10) {
            val conn = (target.openConnection() as HttpURLConnection).apply {
                connectTimeout = timeoutMs
                readTimeout = 60_000
                instanceFollowRedirects = false
                setRequestProperty("User-Agent", "mobile-rvc")
            }
            val code = conn.responseCode
            if (code in 300..399) {
                val loc = conn.getHeaderField("Location") ?: throw IOException("HTTP $code 没有跳转地址")
                conn.disconnect()
                target = URL(target, loc)
                if (target.protocol != "https" && target.protocol != "http") throw IOException("不支持的地址: $target")
                return@repeat
            }
            if (code != 200) {
                conn.disconnect()
                throw IOException(
                    when (code) {
                        401, 403 -> "HTTP $code：没有访问权限"
                        404 -> "HTTP 404：文件不存在"
                        else -> "HTTP $code"
                    }
                )
            }
            return conn
        }
        throw IOException("跳转次数过多")
    }

    /**
     * Downloads [url] into [dir] as [name] (else the server's name for it).
     * [progress] gets (file name, bytes so far, total or -1); [cancelled] is polled while reading.
     */
    fun download(url: String, dir: File, name: String?, progress: (String, Long, Long) -> Unit, cancelled: () -> Boolean): File {
        val conn = open(url)
        try {
            val fileName = sanitize(name ?: fileName(conn.getHeaderField("Content-Disposition"), conn.url))
            val total = conn.contentLengthLong
            val f = File(dir, fileName)
            var done = 0L
            var lastReport = 0L
            conn.inputStream.use { input ->
                f.outputStream().use { out ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        if (cancelled()) throw Cancelled()
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        if (done - lastReport >= (1 shl 19)) {
                            lastReport = done
                            progress(fileName, done, total)
                        }
                    }
                }
            }
            if (total > 0 && done != total) throw IOException("下载不完整（$done / $total 字节）")
            progress(fileName, done, total)
            return f
        } finally {
            conn.disconnect()
        }
    }

    private fun fileName(disposition: String?, url: URL): String {
        val fromHeader = disposition?.let { d ->
            Regex("filename\\*=(?:UTF-8|utf-8)''([^;]+)").find(d)?.groupValues?.get(1)?.let { decode(it) }
                ?: Regex("filename=\"?([^\";]+)\"?").find(d)?.groupValues?.get(1)?.let { decode(it) }
        }
        return fromHeader?.takeIf { it.isNotBlank() } ?: decode(url.path.substringAfterLast('/'))
    }

    /** Percent-decodes until stable: some CDNs send names encoded twice ("%25E4%25B8…"). */
    private fun decode(s: String): String {
        var cur = s
        repeat(3) {
            if (!Regex("%[0-9A-Fa-f]{2}").containsMatchIn(cur)) return cur
            cur = runCatching { URLDecoder.decode(cur.replace("+", "%2B"), "UTF-8") }.getOrElse { return cur }
        }
        return cur
    }

    private fun sanitize(name: String) =
        name.substringAfterLast('/').substringAfterLast('\\').replace(Regex("[\\u0000-\\u001f:*?\"<>|]"), "_").ifBlank { "download" }

    /** True for a .zip that carries models; false for a torch checkpoint (also a zip: "<name>/data.pkl", …). */
    fun isModelArchive(f: File): Boolean {
        val names = openZip(f)?.use { z -> z.entries().toList().map { it.name } } ?: return false
        if (names.any { it.endsWith(".pth", true) || it.endsWith(".index", true) }) return true
        return f.name.endsWith(".zip", true) && names.none { it.endsWith("data.pkl") }
    }

    private fun openZip(f: File): ZipFile? {
        // archives made on Chinese Windows store file names in GBK
        for (cs in listOf(Charsets.UTF_8, Charset.forName("GBK"))) {
            val z = runCatching { ZipFile(f, cs).also { zf -> zf.entries().toList() } }.getOrNull()
            if (z != null) return z
        }
        return null
    }

    /** Extracts the model (largest .pth, not a G_/D_ training checkpoint) and its index (prefers added_*). */
    fun extract(zip: File, dir: File, cancelled: () -> Boolean): Pair<File?, File?> {
        val z = openZip(zip) ?: throw IOException("无法读取压缩包（只支持 .zip）")
        z.use {
            val entries = z.entries().toList().filter { !it.isDirectory && !it.name.contains("__MACOSX") }
            val pth = entries.filter { it.name.endsWith(".pth", true) && !ModelScope.isTrainingCheckpoint(it.name.substringAfterLast('/')) }
                .maxByOrNull { it.size }
            val idx = entries.filter { it.name.endsWith(".index", true) }
                .sortedWith(compareByDescending<ZipEntry> { it.name.substringAfterLast('/').startsWith("added") }.thenByDescending { it.size })
                .firstOrNull()
            fun out(e: ZipEntry?): File? = e?.let { entry ->
                val f = File(dir, sanitize(entry.name))
                z.getInputStream(entry).use { input ->
                    f.outputStream().use { o ->
                        val buf = ByteArray(1 shl 16)
                        while (true) {
                            if (cancelled()) throw Cancelled()
                            val n = input.read(buf)
                            if (n < 0) break
                            o.write(buf, 0, n)
                        }
                    }
                }
                f
            }
            return out(pth) to out(idx)
        }
    }
}
