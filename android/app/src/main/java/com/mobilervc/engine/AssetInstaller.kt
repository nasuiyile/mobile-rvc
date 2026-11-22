package com.mobilervc.engine

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import java.io.File
import java.io.InputStream
import java.util.zip.ZipInputStream

/**
 * Base models (HuBERT/ContentVec, FCPE, synthesizer templates) ship in the APK under
 * assets/rvc/ and are copied to filesDir/rvc once per app version, because ONNX Runtime
 * resolves external weight files by path. A "base pack" zip with the same layout can be
 * imported instead when the APK was built without assets.
 */
object AssetInstaller {
    const val DIR = "rvc"

    fun baseDir(ctx: Context) = File(ctx.filesDir, DIR)

    fun isReady(ctx: Context): Boolean {
        val d = baseDir(ctx)
        return File(d, "hubert/manifest.json").isFile && File(d, "fcpe/manifest.json").isFile &&
            (File(d, "synth").listFiles()?.isNotEmpty() == true)
    }

    private fun versionStamp(ctx: Context): String {
        val pi = if (Build.VERSION.SDK_INT >= 33) {
            ctx.packageManager.getPackageInfo(ctx.packageName, PackageManager.PackageInfoFlags.of(0))
        } else {
            @Suppress("DEPRECATION") ctx.packageManager.getPackageInfo(ctx.packageName, 0)
        }
        return "${pi.longVersionCode}-${pi.lastUpdateTime}"
    }

    fun apkHasAssets(ctx: Context): Boolean =
        runCatching { ctx.assets.list(DIR)?.contains("hubert") == true }.getOrDefault(false)

    /** Copies bundled assets if needed. progress(copiedBytes, totalBytes). */
    fun ensure(ctx: Context, progress: (Long, Long) -> Unit = { _, _ -> }): Boolean {
        val dest = baseDir(ctx)
        val stamp = File(dest, ".stamp")
        if (!apkHasAssets(ctx)) return isReady(ctx)
        val want = versionStamp(ctx)
        if (isReady(ctx) && stamp.isFile && stamp.readText() == want) return true
        val files = mutableListOf<String>()
        fun walk(path: String) {
            val children = ctx.assets.list(path) ?: emptyArray()
            if (children.isEmpty()) files += path else children.forEach { walk("$path/$it") }
        }
        walk(DIR)
        val total = files.sumOf { f -> runCatching { ctx.assets.openFd(f).use { it.length } }.getOrDefault(0L) }
        var done = 0L
        val tmp = File(ctx.filesDir, "$DIR.tmp")
        tmp.deleteRecursively()
        for (f in files) {
            val out = File(tmp, f.removePrefix("$DIR/"))
            out.parentFile?.mkdirs()
            ctx.assets.open(f).use { input -> done = copy(input, out, done, total, progress) }
        }
        File(tmp, ".stamp").writeText(want)
        dest.deleteRecursively()
        if (!tmp.renameTo(dest)) throw IllegalStateException("cannot install base models")
        return isReady(ctx)
    }

    /** Installs a base pack zip (entries hubert/..., fcpe/..., synth/... optionally under rvc/). */
    fun installZip(ctx: Context, input: InputStream) {
        val tmp = File(ctx.filesDir, "$DIR.tmp")
        tmp.deleteRecursively()
        ZipInputStream(input.buffered()).use { zip ->
            while (true) {
                val e = zip.nextEntry ?: break
                val name = e.name.removePrefix("$DIR/")
                if (e.isDirectory || name.contains("..")) continue
                val out = File(tmp, name)
                out.parentFile?.mkdirs()
                out.outputStream().use { zip.copyTo(it, 1 shl 20) }
            }
        }
        if (!File(tmp, "hubert/manifest.json").isFile) {
            tmp.deleteRecursively()
            throw IllegalArgumentException("not a base pack (missing hubert/manifest.json)")
        }
        baseDir(ctx).deleteRecursively()
        tmp.renameTo(baseDir(ctx))
    }

    private fun copy(input: InputStream, out: File, start: Long, total: Long, progress: (Long, Long) -> Unit): Long {
        var done = start
        val buf = ByteArray(1 shl 20)
        out.outputStream().use { o ->
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                o.write(buf, 0, n)
                done += n
                progress(done, total)
            }
        }
        return done
    }
}
