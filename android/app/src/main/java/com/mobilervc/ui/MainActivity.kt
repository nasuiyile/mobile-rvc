package com.mobilervc.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import android.os.SystemClock
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.LocalMinimumInteractiveComponentSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.setProgress
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import com.mobilervc.BuildConfig
import com.mobilervc.engine.AppTab
import com.mobilervc.engine.AudioMode
import com.mobilervc.engine.Backend
import com.mobilervc.engine.Converter
import com.mobilervc.engine.RvcController
import com.mobilervc.service.VoiceService
import com.mobilervc.shizuku.RootBridge
import com.mobilervc.shizuku.ShellBridge
import com.mobilervc.shizuku.ShizukuBridge
import kotlinx.coroutines.delay
import kotlin.concurrent.thread
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        RvcController.init(this)
        ShellBridge.init()
        setContent { AppTheme { Screen() } }
        if (savedInstanceState == null) handleShare(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleShare(intent)
    }

    /** .pth / .index / audio opened ("用其他应用打开") or shared from QQ, WeChat, file managers... */
    private fun handleShare(intent: Intent?) {
        intent ?: return
        val uris = mutableListOf<Uri>()
        when (intent.action) {
            Intent.ACTION_VIEW -> intent.data?.let { uris += it }
            Intent.ACTION_SEND -> IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java)?.let { uris += it }
            Intent.ACTION_SEND_MULTIPLE ->
                IntentCompat.getParcelableArrayListExtra(intent, Intent.EXTRA_STREAM, Uri::class.java)?.let { uris += it }
            else -> return
        }
        intent.clipData?.let { clip -> for (i in 0 until clip.itemCount) clip.getItemAt(i).uri?.let { uris += it } }
        if (uris.isNotEmpty()) RvcController.receiveShared(uris.distinct())
    }

    override fun onResume() {
        super.onResume()
        ShellBridge.refresh()
        RvcController.refreshModels()
    }

    override fun onStart() {
        super.onStart()
        visible = true
    }

    override fun onStop() {
        visible = false
        AudioPreview.stop()
        super.onStop()
    }

    companion object {
        /** In front: a finished file conversion needs no notification. */
        @Volatile var visible = false
    }
}

/** System colours on Android 12+, and controls a little tighter than the 48 dp default. */
@Composable
private fun AppTheme(content: @Composable () -> Unit) {
    val ctx = LocalContext.current
    val dark = isSystemInDarkTheme()
    val colors = when {
        Build.VERSION.SDK_INT >= 31 -> if (dark) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        dark -> darkColorScheme()
        else -> lightColorScheme()
    }
    MaterialTheme(colorScheme = colors) {
        CompositionLocalProvider(LocalMinimumInteractiveComponentSize provides 40.dp, content = content)
    }
}

/**
 * Fixed top bar (title, progress of whatever is busy), scrolling settings in collapsible
 * sections, fixed bottom bar (start / stop and the live status): what is happening and the main
 * action stay on screen wherever the settings are scrolled to.
 */
@Composable
private fun Screen() {
    val c = RvcController
    val s = c.settings
    val ctx = LocalContext.current
    val micPerm = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { }
    LaunchedEffect(Unit) {
        val need = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= 33) need += Manifest.permission.POST_NOTIFICATIONS
        if (need.any { ContextCompat.checkSelfPermission(ctx, it) != PackageManager.PERMISSION_GRANTED }) micPerm.launch(need.toTypedArray())
    }
    LaunchedEffect(c.running) {
        while (true) {
            c.pollStatus()
            delay(if (c.running) 400 else 2000)
        }
    }
    var showImport by remember { mutableStateOf(false) }
    var showLibrary by remember { mutableStateOf(false) }
    val basePack = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { u -> c.importBasePack { ctx.contentResolver.openInputStream(u)!! } }
    }

    val tab = s.tab
    val scroll = remember(tab) { ScrollState(0) }  // every screen starts at its top
    Scaffold(
        topBar = { TopBar() },
        bottomBar = { if (tab == AppTab.CONVERT) ConvertBar() else BottomBar() },
        containerColor = MaterialTheme.colorScheme.surface,
    ) { pad ->
        Column(
            Modifier.fillMaxSize().padding(pad).verticalScroll(scroll).padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (!c.assetsReady && c.installProgress == null) {
                Panel {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("缺少基础模型", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                        Caption("用 tools/build_assets.sh 生成后重新打包 APK，或导入基础模型包 (.zip)。")
                        SmallButton("导入基础模型包", filled = true) { basePack.launch(arrayOf("application/zip", "*/*")) }
                    }
                }
            }
            ModelSection(onImport = { showImport = true })
            when (tab) {
                AppTab.LOCAL -> {
                    VoiceSection()
                    LocalAudioSection()
                    PerformanceSection()
                }
                AppTab.VIRTUAL_MIC -> {
                    VoiceSection()
                    VirtualMicSection()
                    PerformanceSection()
                }
                AppTab.CONVERT -> {
                    ConvertInputSection()
                    ConvertParamsSection()
                    ConvertResultsSection()
                }
            }
        }
    }
    if (AudioPreview.playing != null) LaunchedEffect(AudioPreview.playing) {
        while (true) {
            AudioPreview.tick()
            delay(200)
        }
    }

    if (showImport) ImportDialog(onDismiss = { showImport = false }, onLibrary = { showImport = false; showLibrary = true }) { pth, idx ->
        showImport = false
        c.importModel(pth, idx)
    }
    if (showLibrary) ModelLibraryDialog(onDismiss = { showLibrary = false })
    c.shared?.let { sh -> SharedFilesDialog(sh) }
    Converter.finished?.let { r -> FinishedDialog(r) }
    c.message?.let { msg -> MessageDialog(msg) { c.message = null } }
}

/**
 * Messages can be long (an accelerator's refusal comes with its log): the text scrolls, can be
 * selected, and "复制" puts all of it on the clipboard so it can be sent on as text.
 */
@Composable
private fun MessageDialog(msg: String, onClose: () -> Unit) {
    val ctx = LocalContext.current
    AlertDialog(
        onDismissRequest = onClose,
        confirmButton = { TextButton(onClose) { Text("确定") } },
        dismissButton = {
            if (msg.length > 120) TextButton({
                val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                cm.setPrimaryClip(android.content.ClipData.newPlainText("rvc", msg))
            }) { Text("复制") }
        },
        text = {
            SelectionContainer {
                Text(msg, Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState()),
                    style = if (msg.length > 300) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyMedium)
            }
        },
    )
}

// ---------------------------------------------------------------------------- fixed bars
@Composable
private fun TopBar() {
    val c = RvcController
    Surface(color = MaterialTheme.colorScheme.surface) {
        Column(Modifier.windowInsetsPadding(WindowInsets.statusBars)) {
            Row(Modifier.fillMaxWidth().height(48.dp).padding(start = 16.dp, end = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("rvc手机版", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(8.dp))
                Tag("FCPE")
                Spacer(Modifier.weight(1f))
                MoreInfoMenu()
            }
            TabBar()
            // whatever is in progress (asset install, import, model load, warm-up, benchmark)
            val install = c.installProgress
            val busy = c.busy ?: install?.let { "正在安装基础模型（HuBERT / FCPE）…" }
            var shown by remember { mutableStateOf("") }
            if (busy != null) shown = busy
            AnimatedVisibility(busy != null) {
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(shown, Modifier.weight(1f).padding(start = 16.dp, end = 16.dp, bottom = 6.dp), style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary, maxLines = 2, overflow = TextOverflow.Ellipsis)
                        if (c.downloading) TextButton(onClick = { c.cancelDownload() }) { Text("取消") }
                    }
                    if (install != null) LinearProgressIndicator(progress = { install }, modifier = Modifier.fillMaxWidth().height(3.dp))
                    else LinearProgressIndicator(Modifier.fillMaxWidth().height(3.dp))
                }
            }
            if (busy == null) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
        }
    }
}

@Composable
private fun BottomBar() {
    val c = RvcController
    val ctx = LocalContext.current
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, shadowElevation = 8.dp) {
        Column(
            Modifier.fillMaxWidth().windowInsetsPadding(WindowInsets.navigationBars).padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            if (c.running) StatusStrip()
            val busy = c.busy ?: if (Converter.converting) "正在转换文件…" else null
            val shown = if (c.settings.tab == AppTab.VIRTUAL_MIC) AudioMode.VIRTUAL_MIC else AudioMode.LOCAL
            if (c.running && c.runningMode != shown) {
                Caption("正在以「${modeName(c.runningMode)}」运行，停止后再开始才会换成「${modeName(shown)}」。")
            }
            if (c.running) {
                FilledTonalButton(
                    onClick = { VoiceService.stop(ctx) }, modifier = Modifier.fillMaxWidth().height(46.dp),
                    colors = ButtonDefaults.filledTonalButtonColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer, contentColor = MaterialTheme.colorScheme.onErrorContainer),
                ) { Text("停止音频转换", fontWeight = FontWeight.SemiBold) }
            } else {
                Button(
                    onClick = { VoiceService.start(ctx) }, modifier = Modifier.fillMaxWidth().height(46.dp),
                    enabled = busy == null && c.assetsReady && c.installProgress == null,
                ) {
                    if (busy != null) {
                        CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(8.dp))
                        Text(busy, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    } else {
                        Text("开始音频转换", fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }
    }
}

private fun modeName(m: AudioMode) = if (m == AudioMode.VIRTUAL_MIC) "虚拟麦克风" else "本地监听"

/** Live status above the stop button: one line, tap for the details. */
@Composable
private fun StatusStrip() {
    val c = RvcController
    val st = c.status?.takeIf { it.optBoolean("running") }
    val prefs = LocalContext.current.getSharedPreferences("ui", Context.MODE_PRIVATE)
    var details by remember { mutableStateOf(prefs.getBoolean("open_status", false)) }
    val virtual = c.runningMode == AudioMode.VIRTUAL_MIC
    val active = !virtual || c.captureMode != 0
    val small = MaterialTheme.typography.bodySmall
    val dim = MaterialTheme.colorScheme.onSurfaceVariant
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).clickable {
            details = !details
            prefs.edit().putBoolean("open_status", details).apply()
        }.padding(horizontal = 4.dp, vertical = 2.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(8.dp).clip(CircleShape).background(if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline))
            Spacer(Modifier.width(8.dp))
            Column(Modifier.weight(1f)) {
                if (st == null) {
                    Text("正在启动…", style = MaterialTheme.typography.bodyMedium)
                } else if (!active) {
                    Text("待命：其他应用开始录音时自动变声", style = MaterialTheme.typography.bodyMedium)
                } else {
                    val block = st.optDouble("block_ms")
                    val avg = st.optDouble("infer_avg_ms")
                    Row(verticalAlignment = Alignment.Bottom) {
                        Text("延迟 ", style = small, color = dim)
                        Text("%.0f ms".format(st.optDouble("latency_ms")), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        Text("　推理 %.0f / %.0f ms · 实时率 %.2f".format(avg, block, if (block > 0) avg / block else 0.0),
                            style = small, color = dim, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
                if (virtual && c.captureMode != 0) {
                    Text(if (c.captureMode == 1) "正在变声（语音消息模式）" else "正在变声（通话模式）", style = small, color = MaterialTheme.colorScheme.primary)
                }
            }
            Chevron(!details)  // the details open upwards
        }
        st?.optString("error")?.takeIf { it.isNotEmpty() }?.let { Text("错误：$it", style = small, color = MaterialTheme.colorScheme.error) }
        // a recording that converts nothing: say so instead of showing a healthy-looking delay
        if (st != null && active && st.optBoolean("capture_expected")) {
            val shell = st.optJSONObject("shell")
            val warn = when {
                st.optDouble("input_age_ms", -1.0) > 1500 -> "没有收到麦克风数据，正在重新连接采集…"
                st.optDouble("input_zero_ms", 0.0) > 1500 || shell?.optBoolean("app_silenced") == true && c.captureMode == 1 ||
                    shell?.optBoolean("shell_silenced") == true && c.captureMode == 2 ->
                    "采集到的是纯静音：麦克风被系统静音，或被别的应用占用"
                shell?.optBoolean("victim_silenced") == true -> "对方应用的录音被系统静音了（它不在前台，或麦克风被别的应用占用）"
                else -> null
            }
            if (warn != null) Text(warn, Modifier.padding(start = 16.dp), style = small, color = MaterialTheme.colorScheme.error)
        }
        AnimatedVisibility(details && st != null) {
            Column(Modifier.padding(start = 16.dp, top = 4.dp), verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Text("算法延迟 %.0f ms".format(st?.optJSONObject("engine")?.optDouble("latency_ms") ?: 0.0), style = small)
                st?.optJSONObject("stages")?.let {
                    Text("HuBERT %.0f · FCPE %.0f · 索引 %.0f · 编码 %.0f · 流 %.0f · 声码器 %.0f ms".format(
                        it.optDouble("hubert"), it.optDouble("f0"), it.optDouble("index"), it.optDouble("enc"), it.optDouble("flow"), it.optDouble("dec")),
                        style = small, color = dim)
                }
                Text("后端 ${backendSummary(st?.optJSONObject("engine")?.optJSONObject("backends"))}", style = small, color = dim)
                Text("超时块 ${st?.optLong("late_blocks")} · 欠载 ${st?.optLong("out_underruns")} · 丢弃 ${st?.optLong("dropped_samples")} 采样", style = small, color = dim)
                val reopens = (st?.optLong("input_reopens") ?: 0) + (st?.optLong("output_reopens") ?: 0)
                Text("输入电平 %.0f dB".format(st?.optDouble("input_level_db", -120.0) ?: -120.0) +
                    if (reopens > 0) " · 音频流重连 $reopens 次" else "", style = small, color = dim)
                st?.optJSONObject("shell")?.let { sh ->
                    Text("注入：采集 ${sh.optLong("captured")} / 注入 ${sh.optLong("injected")} 采样${sh.optString("error").let { e -> if (e.isNotEmpty()) " · $e" else "" }}",
                        style = small, color = dim)
                }
            }
        }
    }
}

/** {"hubert":"cpu","dec":"native_fp16",...} -> "HuBERT cpu · 声码器 native_fp16" */
private fun backendSummary(be: org.json.JSONObject?): String =
    if (be == null) "-" else "HuBERT ${be.optString("hubert", "-")} · 声码器 ${be.optString("dec", "-")}"

// ---------------------------------------------------------------------------- sections
@Composable
private fun ModelSection(onImport: () -> Unit) {
    val c = RvcController
    val s = c.settings
    val models = c.models
    val current = models.firstOrNull { it.id == s.modelId }
    val indexPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val m = c.models.firstOrNull { it.id == c.settings.modelId }
        if (uri != null && m != null) c.attachIndex(m, uri)
    }
    Section("model", "模型", summary = current?.name ?: "未选择") {
        SelectField(
            text = current?.let { "${it.name}（${it.version}，${it.sampleRate / 1000}k${if (it.hasIndex) "，含索引" else ""}）" } ?: "未选择模型",
            items = models.map { it.id to "${it.name}（${it.version}，${it.sampleRate / 1000}k${if (it.hasIndex) "，含索引" else ""}）" },
            placeholder = "还没有模型，请先导入 .pth",
        ) { id -> c.update { st -> st.copy(modelId = id) } }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            val idle = c.busy == null && !Converter.converting
            SmallButton("导入 .pth", filled = true, enabled = c.assetsReady && idle, onClick = onImport)
            SmallButton(if (current?.hasIndex == true) "更换 .index" else "添加 .index", enabled = current != null && idle) {
                indexPicker.launch(arrayOf("*/*"))
            }
            Spacer(Modifier.weight(1f))
            if (current != null) {
                TextButton(onClick = { c.deleteModel(current) }, enabled = !c.running && !Converter.converting,
                    colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("删除") }
            }
        }
    }
}

@Composable
private fun VoiceSection() {
    val c = RvcController
    val s = c.settings
    Section("voice", "变声参数", summary = "音调 %+.0f · 索引 %.2f".format(s.pitch, s.indexRate)) {
        ParamSlider("音调", s.pitch, -16f..16f, 31, "%+.0f".format(s.pitch)) { v -> c.update { it.copy(pitch = v) } }
        ParamSlider("性别因子", s.formant, -2f..2f, 79, "%+.2f".format(s.formant)) { v -> c.update { it.copy(formant = v) } }
        ParamSlider("Index Rate", s.indexRate, 0f..1f, 99, "%.2f".format(s.indexRate)) { v -> c.update { it.copy(indexRate = v) } }
        ParamSlider("响度因子", s.rmsMixRate, 0f..1f, 99, "%.2f".format(s.rmsMixRate)) { v -> c.update { it.copy(rmsMixRate = v) } }
        ParamSlider("响应阈值", s.threshold, -60f..0f, 59, if (s.threshold <= -60f) "关" else "%.0f dB".format(s.threshold)) { v -> c.update { it.copy(threshold = v) } }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Label("降噪")
            FilterChip(s.inputNoiseReduce, { c.update { it.copy(inputNoiseReduce = !it.inputNoiseReduce) } }, label = { Text("输入") })
            FilterChip(s.outputNoiseReduce, { c.update { it.copy(outputNoiseReduce = !it.outputNoiseReduce) } }, label = { Text("输出") })
        }
    }
}

@Composable
private fun PerformanceSection() {
    val c = RvcController
    val s = c.settings
    val providers = c.device?.optJSONArray("providers")?.let { a -> (0 until a.length()).map { a.getString(it) } } ?: emptyList()
    fun avail(b: Backend) = when (b) {
        Backend.CPU, Backend.NATIVE -> true
        Backend.XNNPACK -> "XnnpackExecutionProvider" in providers
        Backend.GPU -> "WebGpuExecutionProvider" in providers
        Backend.NNAPI -> "NnapiExecutionProvider" in providers
        Backend.QNN_HTP, Backend.QNN_GPU -> "QNNExecutionProvider" in providers
    }
    Section("perf", "延迟与性能", summary = "采样 %.2f s · %s + %s".format(s.blockTime, Backend.of(s.hubertBackend).short, Backend.of(s.decBackend).short),
        defaultOpen = false) {
        Caption("这一组在重新开始后生效。采样长度越小延迟越低，但必须大于每块的推理时间。")
        ParamSlider("采样长度", s.blockTime, 0.1f..1.0f, 89, "%.2f s".format(s.blockTime)) { v -> c.update { it.copy(blockTime = (v * 100).roundToInt() / 100f) } }
        ParamSlider("淡入淡出", s.crossfadeTime, 0.01f..0.15f, 13, "%.2f s".format(s.crossfadeTime)) { v -> c.update { it.copy(crossfadeTime = (v * 100).roundToInt() / 100f) } }
        ParamSlider("额外推理", s.extraTime, 0.05f..3.0f, 58, "%.2f s".format(s.extraTime)) { v -> c.update { it.copy(extraTime = (v * 100).roundToInt() / 100f) } }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Label("HuBERT")
            SelectField(Backend.of(s.hubertBackend).label, Backend.entries.filter { it != Backend.NATIVE && avail(it) }.map { it.id to it.label },
                Modifier.weight(1f)) { id -> c.update { it.copy(hubertBackend = id) } }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Label("声码器")
            SelectField(Backend.of(s.decBackend).label, Backend.entries.filter { avail(it) }.map { it.id to it.label },
                Modifier.weight(1f)) { id -> c.update { it.copy(decBackend = id) } }
        }
        val big = c.device?.optJSONArray("big_cores")?.length() ?: 4
        ParamSlider("CPU 线程", s.threads.toFloat(), 0f..8f, 7, if (s.threads == 0) "自动 $big" else "${s.threads}") { v -> c.update { it.copy(threads = v.roundToInt()) } }
        SwitchRow("FP16 半精度", "原生 NEON / GPU / NPU", s.fp16) { v -> c.update { it.copy(fp16 = v) } }
        SwitchRow("NSF 相位跨块连续", "实验功能", s.phaseLock) { v -> c.update { it.copy(phaseLock = v) } }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SmallButton("性能测试", enabled = !c.running && c.busy == null && !Converter.converting && s.modelId.isNotEmpty()) { c.bench() }
            if (c.benchResult == null) Caption("测出每块的推理耗时")
        }
        c.benchResult?.let {
            Text(it, Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(MaterialTheme.colorScheme.surfaceContainerHighest).padding(10.dp),
                style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun LocalAudioSection() {
    val c = RvcController
    val s = c.settings
    Section("audio", "本地监听", summary = "增益 %.1f×".format(s.outputGain) + if (s.passthrough) " · 输入监听" else "") {
        Caption("麦克风 → 变声 → 耳机。戴上耳机后点「开始音频转换」，避免啸叫。")
        ParamSlider("输出增益", s.outputGain, 0f..3f, 29, "%.1f×".format(s.outputGain)) { v -> c.update { it.copy(outputGain = v) } }
        Segmented(listOf("输出变声", "输入监听"), if (s.passthrough) 1 else 0) { i -> c.update { it.copy(passthrough = i == 1) } }
    }
}

@Composable
private fun VirtualMicSection() {
    val c = RvcController
    val s = c.settings
    val ctx = LocalContext.current
    val cs = MaterialTheme.colorScheme
    // rooted devices need no Shizuku: the service then runs as root through su
    val root = RootBridge.available
    val shizukuOk = ShellBridge.shizukuUsable
    Section("vmic", "虚拟麦克风", summary = ShellBridge.stateText + if (s.monitor) " · 本地监听" else "") {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.heightIn(min = 32.dp)) {
            Label("Shizuku")
            Text(ShellBridge.shizukuText + if (root && !shizukuOk) " · 有 Root 时不需要" else "", Modifier.weight(1f),
                style = MaterialTheme.typography.bodyMedium,
                color = when {
                    shizukuOk -> cs.onSurface
                    root -> cs.onSurfaceVariant
                    else -> cs.error
                })
            if (ShizukuBridge.state == ShizukuBridge.State.NO_PERMISSION) SmallButton("授权", filled = !root) { ShizukuBridge.requestPermission() }
        }
        if (root) Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.heightIn(min = 32.dp)) {
            Label("Root")
            val st = RootBridge.state
            Text(if (st == RootBridge.State.AVAILABLE && shizukuOk) "可用（优先使用 Shizuku）" else RootBridge.stateText, Modifier.weight(1f),
                style = MaterialTheme.typography.bodyMedium, color = if (st == RootBridge.State.DENIED) cs.error else cs.onSurface)
            if (!shizukuOk && (st == RootBridge.State.AVAILABLE || st == RootBridge.State.DENIED)) {
                SmallButton(if (st == RootBridge.State.DENIED) "重试" else "授权", filled = true, enabled = !c.running) {
                    thread(name = "rvc-root") { ShellBridge.connectRoot(ctx) }
                }
            }
        }
        if (root && RootBridge.state == RootBridge.State.DENIED) RootBridge.error?.let { Caption("Root 启动失败：$it") }
        SwitchRow("同时本地监听", "自己也能听到变声", s.monitor) { v -> c.update { it.copy(monitor = v) } }
        SwitchRow("防静音底噪", "没说话时不给对方纯静音，避免录音被判为无声而中断", s.comfortNoise) { v -> c.update { it.copy(comfortNoise = v) } }
        if (s.comfortNoise) {
            ParamSlider("底噪电平", s.comfortNoiseDb, -80f..-40f, 39, "%.0f dB".format(s.comfortNoiseDb)) { v -> c.update { it.copy(comfortNoiseDb = v) } }
        }
        Explain("虚拟麦克风说明",
            "微信、QQ 等应用录音时会自动收到变声后的声音；平时不占用麦克风。每次录音开头约 0.4–0.6 秒没有人声（不会漏出本音），采样长度越小这段越短。\n" +
                "开了降噪或响应阈值时，不说话的那段输出是纯静音，有些应用会据此认定麦克风异常并立刻结束录音；" +
                "防静音底噪会在整段录音里垫上一层和安静房间差不多的底噪（默认 −60 dB，仍被中断就调大），重新开始后生效。\n" +
                "需要 Shizuku 或 Root 二选一。已 root 的手机不用装 Shizuku：开始时会请求 Root 授权（KernelSU / APatch 要先在管理器里给本应用授予 Root）；Shizuku 已授权时优先用 Shizuku。\n" +
                "首次使用会通过 Shizuku / Root 开启「RVC 虚拟麦克风」无障碍服务，只用来让变声采集和语音消息同时进行，不读取屏幕内容。")
        ParamSlider("输出增益", s.outputGain, 0f..3f, 29, "%.1f×".format(s.outputGain)) { v -> c.update { it.copy(outputGain = v) } }
        Segmented(listOf("输出变声", "输入监听"), if (s.passthrough) 1 else 0) { i -> c.update { it.copy(passthrough = i == 1) } }
    }
}

// ---------------------------------------------------------------------------- tabs
/**
 * The three screens, always at the top: the selected tab is filled with the primary colour; a
 * dot marks the mode that is running, the percentage a file that is converting.
 */
@Composable
private fun TabBar() {
    val c = RvcController
    val cs = MaterialTheme.colorScheme
    val tabs = listOf(AppTab.LOCAL to "本地监听", AppTab.VIRTUAL_MIC to "虚拟麦克风", AppTab.CONVERT to "文件转换")
    Row(
        Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, bottom = 8.dp).clip(RoundedCornerShape(14.dp))
            .background(cs.surfaceContainerHighest).padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        tabs.forEach { (tab, label) ->
            val on = c.settings.tab == tab
            val bg by animateColorAsState(if (on) cs.primary else Color.Transparent, label = "tab")
            val fg = if (on) cs.onPrimary else cs.onSurfaceVariant
            val active = when (tab) {
                AppTab.LOCAL -> c.running && c.runningMode == AudioMode.LOCAL
                AppTab.VIRTUAL_MIC -> c.running && c.runningMode == AudioMode.VIRTUAL_MIC
                AppTab.CONVERT -> Converter.converting
            }
            Row(
                Modifier.weight(1f).height(44.dp).clip(RoundedCornerShape(10.dp)).background(bg)
                    .selectable(selected = on, role = Role.Tab) { c.selectTab(tab) },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (active) {
                    Box(Modifier.size(7.dp).clip(CircleShape).background(if (on) cs.onPrimary else cs.primary))
                    Spacer(Modifier.width(5.dp))
                }
                Text(label, style = MaterialTheme.typography.titleSmall, fontWeight = if (on) FontWeight.Bold else FontWeight.Medium,
                    color = fg, maxLines = 1)
                if (tab == AppTab.CONVERT && Converter.converting && Converter.progress > 0f) {
                    Text(" %d%%".format((Converter.progress * 100).roundToInt()), style = MaterialTheme.typography.labelSmall, color = fg)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------- file conversion
@Composable
private fun ConvertInputSection() {
    val conv = Converter
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let { conv.pick(it) } }
    val input = conv.input
    Section("cin", "音频文件", summary = input?.name ?: "未选择") {
        if (input != null) {
            Text(input.name, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.SemiBold, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Caption("${Converter.clock(input.durationS)} · ${input.info}")
            PlayProgress(input.file)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SmallButton(if (AudioPreview.playing == input.file) "停止试听" else "试听原音") { AudioPreview.toggle(input.file) }
                SmallButton("换一个文件", enabled = !conv.staging && !conv.converting) { pick.launch(arrayOf("audio/*", "video/*", "application/ogg")) }
            }
        } else {
            Caption("选择要变声的歌曲、录音或视频（mp3、m4a、wav、flac、ogg、amr，或视频里的声音）。")
            SmallButton("选择音频文件", filled = true, enabled = !conv.staging) { pick.launch(arrayOf("audio/*", "video/*", "application/ogg")) }
        }
        if (conv.staging) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Caption("读取文件…")
            }
        }
        Explain("从其他应用打开",
            "在微信、QQ、文件管理器、录音机、音乐播放器里，对音频选择「用其他应用打开」或「分享」，再选「用 rvc手机版 变声」，" +
                "就会来到这里并载入文件。微信 / QQ 的语音消息是 SILK 编码，系统解码不了，需要先转成 mp3 等格式。")
    }
}

@Composable
private fun ConvertParamsSection() {
    val c = RvcController
    val s = c.settings
    Section("cparams", "转换参数", summary = "变调 %+d · 检索 %.2f · %s".format(s.convPitch, s.convIndexRate, s.convFormat.uppercase())) {
        ParamSlider("变调", s.convPitch.toFloat(), -24f..24f, 47, "%+d".format(s.convPitch)) { v -> c.update { it.copy(convPitch = v.roundToInt()) } }
        ParamSlider("检索占比", s.convIndexRate, 0f..1f, 99, "%.2f".format(s.convIndexRate)) { v -> c.update { it.copy(convIndexRate = v) } }
        ParamSlider("音量融合", s.convRmsMix, 0f..1f, 99, "%.2f".format(s.convRmsMix)) { v -> c.update { it.copy(convRmsMix = v) } }
        ParamSlider("保护清辅音", s.convProtect, 0f..0.5f, 49, if (s.convProtect >= 0.5f) "关" else "%.2f".format(s.convProtect)) { v ->
            c.update { it.copy(convProtect = v) }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Label("输出格式")
            Box(Modifier.weight(1f)) {
                Segmented(listOf("WAV 无损", "M4A 体积小"), if (s.convFormat == "m4a") 1 else 0) { i ->
                    c.update { it.copy(convFormat = if (i == 1) "m4a" else "wav") }
                }
            }
        }
        Explain("参数说明",
            "和 RVC WebUI「模型推理」一样处理整段音频（音高算法 FCPE），默认值也相同。\n" +
                "变调：半音数，升八度 +12、降八度 −12；男声转女声常用 +12。\n" +
                "检索占比（检索特征占比）：用模型索引（.index）替换音色特征的比例，越高越像目标音色，太高可能吐字含糊；模型没有索引时不起作用。\n" +
                "音量融合（输入源音量包络替换输出音量包络融合比例）：越靠近 1 越保留模型输出的音量起伏，越靠近 0 越跟随原音频。\n" +
                "保护清辅音：保护清辅音和呼吸声，减少电音撕裂；0.5 为不开启。\n" +
                "WAV 和 RVC 一样按模型采样率输出；M4A（AAC）体积约为 WAV 的十分之一，适合发给别人。\n" +
                "「延迟与性能」里的 CPU 线程和 FP16 也用于这里；转换时手机可以锁屏或切到别的应用。")
    }
}

@Composable
private fun ConvertResultsSection() {
    val results = Converter.results
    Section("cresults", "转换结果", summary = if (results.isEmpty()) "无" else "${results.size} 个") {
        if (results.isEmpty()) {
            Caption("转换好的音频会出现在这里，可以试听、分享给微信 / QQ 好友，或保存到「音乐/RVC」。")
        }
        results.forEachIndexed { i, r ->
            if (i > 0) HorizontalDivider(Modifier.padding(vertical = 4.dp), color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
            ResultRow(r)
        }
    }
}

private fun sizeText(bytes: Long) = if (bytes >= 1 shl 20) "%.1f MB".format(bytes / 1048576.0) else "%d KB".format(bytes / 1024)

@Composable
private fun ResultRow(r: Converter.Result, showDelete: Boolean = true) {
    val ctx = LocalContext.current
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(r.file.name, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, maxLines = 2, overflow = TextOverflow.Ellipsis)
        Caption("${Converter.clock(r.durationS)} · ${sizeText(r.size)} · ${r.format.uppercase()} · ${r.model} 变调 %+d".format(r.pitch))
        PlayProgress(r.file)
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = { AudioPreview.toggle(r.file) }) { Text(if (AudioPreview.playing == r.file) "停止" else "播放") }
            TextButton(onClick = { runCatching { Converter.share(ctx, r) }.onFailure { RvcController.message = "无法分享：${it.message}" } }) { Text("分享") }
            TextButton(onClick = { Converter.saveToMusic(r) }) { Text("保存到音乐") }
            Spacer(Modifier.weight(1f))
            if (showDelete) {
                TextButton(onClick = {
                    if (AudioPreview.playing == r.file) AudioPreview.stop()
                    Converter.delete(r)
                }, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("删除") }
            }
        }
    }
}

/** Position of the file that is playing, when it is this one. */
@Composable
private fun PlayProgress(f: java.io.File) {
    if (AudioPreview.playing != f) return
    val d = AudioPreview.durationMs.coerceAtLeast(1)
    Row(verticalAlignment = Alignment.CenterVertically) {
        LinearProgressIndicator(progress = { AudioPreview.positionMs.toFloat() / d }, modifier = Modifier.weight(1f).height(3.dp))
        Spacer(Modifier.width(8.dp))
        Caption("${Converter.clock(AudioPreview.positionMs / 1000.0)} / ${Converter.clock(d / 1000.0)}")
    }
}

/** Bottom bar of the conversion screen: start, progress and cancel, or why it cannot start. */
@Composable
private fun ConvertBar() {
    val c = RvcController
    val conv = Converter
    val ctx = LocalContext.current
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, shadowElevation = 8.dp) {
        Column(
            Modifier.fillMaxWidth().windowInsetsPadding(WindowInsets.navigationBars).padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            when {
                c.running -> {
                    Caption("实时变声（${modeName(c.runningMode)}）正在运行，停止后才能转换文件。")
                    FilledTonalButton(
                        onClick = { VoiceService.stop(ctx) }, modifier = Modifier.fillMaxWidth().height(46.dp),
                        colors = ButtonDefaults.filledTonalButtonColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer, contentColor = MaterialTheme.colorScheme.onErrorContainer),
                    ) { Text("停止实时变声", fontWeight = FontWeight.SemiBold) }
                }
                conv.converting -> {
                    val p = conv.progress
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(if (p <= 0f) "加载模型…" else "转换中 %d%%".format((p * 100).roundToInt()),
                                style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                            if (p <= 0f) LinearProgressIndicator(Modifier.fillMaxWidth().height(4.dp))
                            else LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth().height(4.dp))
                            val elapsed = (SystemClock.elapsedRealtime() - conv.startedAt) / 1000.0
                            Caption(if (p > 0.1f) "已用 ${Converter.clock(elapsed)} · 剩余约 ${Converter.clock(elapsed * (1 - p) / p)}"
                                else "已用 ${Converter.clock(elapsed)}")
                        }
                        Spacer(Modifier.width(12.dp))
                        OutlinedButton(onClick = { conv.cancel() }) { Text("取消") }
                    }
                }
                else -> {
                    val model = c.models.firstOrNull { it.id == c.settings.modelId }
                    val why = when {
                        !c.assetsReady -> "基础模型未就绪"
                        model == null -> "先在上面选择或导入模型"
                        conv.input == null -> "先选择音频文件"
                        else -> null
                    }
                    Button(
                        onClick = { conv.start(ctx)?.let { c.message = it } }, modifier = Modifier.fillMaxWidth().height(46.dp),
                        enabled = why == null && c.busy == null && !conv.staging && c.installProgress == null,
                    ) { Text(why ?: "开始转换", fontWeight = FontWeight.SemiBold) }
                }
            }
        }
    }
}

@Composable
private fun FinishedDialog(r: Converter.Result) {
    val close = {
        if (AudioPreview.playing == r.file) AudioPreview.stop()
        Converter.finished = null
    }
    AlertDialog(
        onDismissRequest = close,
        title = { Text("转换完成") },
        text = { ResultRow(r, showDelete = false) },
        confirmButton = { TextButton(onClick = close) { Text("关闭") } },
    )
}

@Composable
private fun ImportDialog(
    initialPth: Uri? = null,
    initialIndex: Uri? = null,
    title: String = "导入 RVC 模型",
    onDismiss: () -> Unit,
    onLibrary: (() -> Unit)? = null,
    onImport: (Uri, Uri?) -> Unit,
) {
    var pth by remember { mutableStateOf(initialPth) }
    var idx by remember { mutableStateOf(initialIndex) }
    val store = RvcController.store
    val pickPth = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { pth = it }
    val pickIdx = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { idx = it }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("直接使用 RVC WebUI 导出的推理模型（weights/*.pth）和 added_*.index，无需转换。")
                OutlinedButton(onClick = { pickPth.launch(arrayOf("*/*")) }) { Text(pth?.let { store.displayName(it) } ?: "选择 .pth 文件") }
                OutlinedButton(onClick = { pickIdx.launch(arrayOf("*/*")) }) { Text(idx?.let { store.displayName(it) } ?: "选择 .index 文件（可选）") }
                if (onLibrary != null) {
                    HorizontalDivider(Modifier.padding(vertical = 4.dp))
                    Text("没有模型文件？从在线模型库挑选，点一下就能下载导入。", style = MaterialTheme.typography.bodySmall)
                    FilledTonalButton(onClick = onLibrary) { Text("打开在线模型库") }
                }
            }
        },
        confirmButton = { TextButton(onClick = { pth?.let { onImport(it, idx) } }, enabled = pth != null) { Text("导入") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

/** Files received from another app: a .pth opens the prefilled import dialog; a lone .index
 *  can be attached to an existing model or imported together with a .pth picked now. */
@Composable
private fun SharedFilesDialog(sh: RvcController.SharedFiles) {
    val c = RvcController
    val pthUri = sh.pth?.let { Uri.fromFile(it) }
    val idxUri = sh.index?.let { Uri.fromFile(it) }
    var asNewModel by remember(sh) { mutableStateOf(sh.pth != null) }
    if (asNewModel) {
        ImportDialog(initialPth = pthUri, initialIndex = idxUri, title = "导入收到的模型", onDismiss = { c.clearShared() }) { p, i ->
            c.importModel(p, i)
            c.clearShared()
        }
        return
    }
    val models = c.models
    AlertDialog(
        onDismissRequest = { c.clearShared() },
        title = { Text("收到索引文件") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("${sh.index?.name}\n选择要添加索引的模型（索引维度需与模型一致）：")
                if (models.isEmpty()) Text("还没有模型，请与 .pth 一起导入。", style = MaterialTheme.typography.bodySmall)
                models.forEach { m ->
                    OutlinedButton(onClick = {
                        idxUri?.let { c.attachIndex(m, it) }
                        c.clearShared()
                    }, modifier = Modifier.fillMaxWidth()) {
                        Text("${m.name} (${m.version}, ${m.sampleRate / 1000}k${if (m.hasIndex) ", 替换现有索引" else ""})")
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = { asNewModel = true }) { Text("选择 .pth 一起导入") } },
        dismissButton = { TextButton(onClick = { c.clearShared() }) { Text("取消") } },
    )
}

private const val REPO_URL = "https://github.com/nasuiyile/mobile-rvc"
private const val BILIBILI_UID = "79356601"
private const val BILIBILI_URL = "https://space.bilibili.com/$BILIBILI_UID"
/** Opens the profile page inside the Bilibili app; falls back to the web page when the app isn't installed. */
private const val BILIBILI_APP_URL = "bilibili://space/$BILIBILI_UID"
private const val RVC_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"

/** Top-right "⋮" menu: project links and version. */
@Composable
private fun MoreInfoMenu() {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    var open by remember { mutableStateOf(false) }
    fun openUrl(url: String, appUrl: String? = null) {
        open = false
        val view = { u: String -> ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(u))) }
        runCatching { view(appUrl ?: url) }
            .recoverCatching { if (appUrl != null) view(url) else throw it }
            .onFailure { RvcController.message = "无法打开链接：$url" }
    }
    @Composable
    fun LinkItem(title: String, subtitle: String, url: String, appUrl: String? = null) = DropdownMenuItem(
        text = {
            Column {
                Text(title)
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        },
        onClick = { openUrl(url, appUrl) },
    )
    Box {
        IconButton(onClick = { open = true }, modifier = Modifier.semantics { contentDescription = "更多信息" }) {
            Text("⋮", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            LinkItem("GitHub：给个 Star", "nasuiyile/mobile-rvc", REPO_URL)
            LinkItem("B 站：关注我", "space.bilibili.com/$BILIBILI_UID", BILIBILI_URL, BILIBILI_APP_URL)
            LinkItem("原版 RVC 项目", "RVC-Project/Retrieval-based-Voice-Conversion-WebUI", RVC_URL)
            HorizontalDivider()
            DropdownMenuItem(text = { Text("版本 ${BuildConfig.VERSION_NAME}") }, onClick = {}, enabled = false)
        }
    }
}

// ---------------------------------------------------------------------------- building blocks
@Composable
private fun Panel(content: @Composable () -> Unit) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceContainerLow, content = content)
}

/** A card that folds down to its title and a one-line summary; the state is remembered. */
@Composable
private fun Section(key: String, title: String, summary: String, defaultOpen: Boolean = true, content: @Composable ColumnScope.() -> Unit) {
    val prefs = LocalContext.current.getSharedPreferences("ui", Context.MODE_PRIVATE)
    var open by remember { mutableStateOf(prefs.getBoolean("open_$key", defaultOpen)) }
    Panel {
        Column {
            Row(
                Modifier.fillMaxWidth().clickable(role = Role.Button) {
                    open = !open
                    prefs.edit().putBoolean("open_$key", open).apply()
                }.padding(horizontal = 14.dp, vertical = 11.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text(if (open) "" else summary, Modifier.weight(1f).padding(horizontal = 10.dp), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis, textAlign = TextAlign.End)
                Chevron(open)
            }
            AnimatedVisibility(open) {
                Column(Modifier.padding(start = 14.dp, end = 14.dp, bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(4.dp), content = content)
            }
        }
    }
}

/** Points down, turns up when open. */
@Composable
private fun Chevron(open: Boolean) {
    val turn by animateFloatAsState(if (open) 180f else 0f, label = "chevron")
    val color = MaterialTheme.colorScheme.onSurfaceVariant
    Canvas(Modifier.size(18.dp).rotate(turn)) {
        val w = size.width
        val h = size.height
        val stroke = 1.8.dp.toPx()
        drawLine(color, Offset(w * 0.25f, h * 0.38f), Offset(w * 0.5f, h * 0.64f), stroke, StrokeCap.Round)
        drawLine(color, Offset(w * 0.5f, h * 0.64f), Offset(w * 0.75f, h * 0.38f), stroke, StrokeCap.Round)
    }
}

private val LabelWidth = 76.dp

/** Left column of a settings row. */
@Composable
private fun Label(text: String) {
    Text(text, Modifier.width(LabelWidth), style = MaterialTheme.typography.bodyMedium, maxLines = 2, lineHeight = 16.sp)
}

@Composable
private fun Caption(text: String, modifier: Modifier = Modifier) {
    Text(text, modifier, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun Tag(text: String) {
    Text(text, Modifier.clip(RoundedCornerShape(6.dp)).background(MaterialTheme.colorScheme.secondaryContainer).padding(horizontal = 6.dp, vertical = 2.dp),
        style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSecondaryContainer)
}

/** One row: label, slider, value. The slider snaps to `steps` intermediate positions. */
@Composable
private fun ParamSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, steps: Int, text: String, onChange: (Float) -> Unit) {
    Row(Modifier.fillMaxWidth().height(36.dp), verticalAlignment = Alignment.CenterVertically) {
        Label(label)
        val span = range.endInclusive - range.start
        SlimSlider(label, value, range, Modifier.weight(1f)) { v ->
            val n = (steps + 1).coerceAtLeast(1)
            onChange(range.start + ((v - range.start) / span * n).roundToInt() * span / n)
        }
        Text(text, Modifier.width(60.dp), style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.End, maxLines = 1)
    }
}

/**
 * A slider drawn slim (4 dp track, 18 dp round thumb) so that rows can sit close together;
 * Material's own thumb is 44 dp tall. Tap anywhere on the track or drag horizontally; vertical
 * drags still scroll the page.
 */
@Composable
private fun SlimSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, modifier: Modifier = Modifier, onChange: (Float) -> Unit) {
    val active = MaterialTheme.colorScheme.primary
    val inactive = MaterialTheme.colorScheme.surfaceContainerHighest
    val span = range.endInclusive - range.start
    val frac = if (span > 0) ((value - range.start) / span).coerceIn(0f, 1f) else 0f
    val thumb = with(LocalDensity.current) { 9.dp.toPx() }  // radius: the track is inset by it
    val set by rememberUpdatedState(onChange)
    var width by remember { mutableStateOf(1) }
    var dragX by remember { mutableStateOf(0f) }
    fun valueAt(x: Float) = range.start + ((x - thumb) / (width - 2 * thumb).coerceAtLeast(1f)).coerceIn(0f, 1f) * span
    Canvas(
        modifier.height(36.dp)
            .onSizeChanged { width = it.width }
            .semantics {
                contentDescription = label
                progressBarRangeInfo = ProgressBarRangeInfo(value, range)
                setProgress { v -> set(v.coerceIn(range)); true }
            }
            .pointerInput(range) { detectTapGestures { o -> set(valueAt(o.x)) } }
            .draggable(
                rememberDraggableState { d ->
                    dragX += d
                    set(valueAt(dragX))
                },
                Orientation.Horizontal, onDragStarted = { o -> dragX = o.x },
            ),
    ) {
        val y = size.height / 2
        val x0 = thumb
        val x1 = size.width - thumb
        val x = x0 + (x1 - x0) * frac
        val track = 4.dp.toPx()
        drawLine(inactive, Offset(x0, y), Offset(x1, y), track, StrokeCap.Round)
        drawLine(active, Offset(x0, y), Offset(x, y), track, StrokeCap.Round)
        drawCircle(active, thumb, Offset(x, y))
    }
}

/** The whole row toggles. */
@Composable
private fun SwitchRow(label: String, hint: String?, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 40.dp).clip(RoundedCornerShape(8.dp)).toggleable(checked, role = Role.Switch, onValueChange = onChange),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f).padding(vertical = 3.dp)) {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            if (hint != null) Caption(hint)
        }
        Switch(checked, null, Modifier.scale(0.8f))
    }
}

/** Two or three mutually exclusive choices in one pill. */
@Composable
private fun Segmented(options: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(50)).background(MaterialTheme.colorScheme.surfaceContainerHighest).padding(3.dp)) {
        options.forEachIndexed { i, text ->
            val on = i == selected
            Box(
                Modifier.weight(1f).clip(RoundedCornerShape(50)).background(if (on) MaterialTheme.colorScheme.primary else Color.Transparent)
                    .clickable(role = Role.RadioButton) { onSelect(i) }.padding(vertical = 7.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 1,
                    color = if (on) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun SmallButton(text: String, filled: Boolean = false, enabled: Boolean = true, onClick: () -> Unit) {
    val pad = androidx.compose.foundation.layout.PaddingValues(horizontal = 14.dp, vertical = 0.dp)
    if (filled) FilledTonalButton(onClick, Modifier.height(36.dp), enabled = enabled, contentPadding = pad) { Text(text) }
    else OutlinedButton(onClick, Modifier.height(36.dp), enabled = enabled, contentPadding = pad) { Text(text) }
}

/** Drop-down that looks like a field: current choice and a chevron. */
@Composable
private fun SelectField(text: String, items: List<Pair<String, String>>, modifier: Modifier = Modifier, placeholder: String? = null, onSelect: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    val shape = RoundedCornerShape(10.dp)
    Box(modifier) {
        Row(
            Modifier.fillMaxWidth().heightIn(min = 40.dp).clip(shape).border(1.dp, MaterialTheme.colorScheme.outlineVariant, shape)
                .clickable(enabled = items.isNotEmpty(), role = Role.DropdownList) { open = true }.padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(if (items.isEmpty() && placeholder != null) placeholder else text, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium,
                maxLines = 1, overflow = TextOverflow.Ellipsis,
                color = if (items.isEmpty()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurface)
            Chevron(open)
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            items.forEach { (id, label) ->
                DropdownMenuItem(text = { Text(label) }, onClick = {
                    open = false
                    onSelect(id)
                })
            }
        }
    }
}

/** A line of small print that opens into an explanation. */
@Composable
private fun Explain(title: String, text: String) {
    var open by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { open = !open }.padding(vertical = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(title, Modifier.weight(1f), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
            Chevron(open)
        }
        AnimatedVisibility(open) { Caption(text, Modifier.padding(top = 4.dp)) }
    }
}
