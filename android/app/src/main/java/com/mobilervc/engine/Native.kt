package com.mobilervc.engine

/** JNI surface of librvc.so. Every String result is JSON; failures carry an "error" key. */
object Native {
    init {
        System.loadLibrary("rvc")
    }

    @JvmStatic external fun create(assetsDir: String): Long
    @JvmStatic external fun destroy(handle: Long)
    @JvmStatic external fun importModel(assetsDir: String, pth: String, index: String, outDir: String, name: String): String
    @JvmStatic external fun inspectModel(assetsDir: String, pth: String): String
    @JvmStatic external fun attachIndex(modelDir: String, index: String): String
    @JvmStatic external fun load(handle: Long, modelDir: String, configJson: String): String
    @JvmStatic external fun start(handle: Long, optionsJson: String, captureFd: Int, playbackFd: Int): String
    @JvmStatic external fun stop(handle: Long)
    /** Virtual-mic mode (decided by the shell service): 0 idle, 1 capture here, 2 shell captures. */
    @JvmStatic external fun setCaptureMode(handle: Long, mode: Int)
    @JvmStatic external fun setParams(
        handle: Long, pitch: Float, formant: Float, indexRate: Float, rmsMix: Float, threshold: Float,
        inputNoiseReduce: Boolean, outputNoiseReduce: Boolean,
    )
    @JvmStatic external fun status(handle: Long): String
    @JvmStatic external fun bench(handle: Long, iters: Int): String
    @JvmStatic external fun deviceInfo(): String
    /** Returns [readFd, writeFd] of a pipe with a 1 MiB kernel buffer. */
    @JvmStatic external fun createPipe(): IntArray?
}
