package com.mobilervc.ui

import android.Manifest
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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import com.mobilervc.BuildConfig
import com.mobilervc.engine.AudioMode
import com.mobilervc.engine.Backend
import com.mobilervc.engine.RvcController
import com.mobilervc.service.VoiceService
import com.mobilervc.shizuku.ShizukuBridge
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        RvcController.init(this)
        ShizukuBridge.init()
        setContent {
            val dark = isSystemInDarkTheme()
            MaterialTheme(colorScheme = if (dark) darkColorScheme() else lightColorScheme()) {
                Surface(Modifier.fillMaxSize()) { Screen() }
            }
        }
        if (savedInstanceState == null) handleShare(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleShare(intent)
    }

    /** .pth / .index opened ("用其他应用打开") or shared from QQ, WeChat, file managers... */
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
        ShizukuBridge.refresh()
        RvcController.refreshModels()
    }
}

@Composable
private fun Screen() {
    val c = RvcController
    val s = c.settings
    val ctx = androidx.compose.ui.platform.LocalContext.current
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
    val basePack = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { u -> c.importBasePack { ctx.contentResolver.openInputStream(u)!! } }
    }
    val indexPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val m = c.models.firstOrNull { it.id == s.modelId }
        if (uri != null && m != null) c.attachIndex(m, uri)
    }

    Column(
        Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("RVC 实时变声 · FCPE", Modifier.weight(1f), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            MoreInfoMenu()
        }
        c.installProgress?.let {
            Text("正在安装基础模型（HuBERT / FCPE）…")
            LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth())
        }
        if (!c.assetsReady && c.installProgress == null) {
            Card { Column(Modifier.padding(12.dp)) {
                Text("缺少基础模型。请用 tools/build_assets.sh 生成后重新打包 APK，或导入基础模型包 (.zip)。")
                OutlinedButton(onClick = { basePack.launch(arrayOf("application/zip", "*/*")) }) { Text("导入基础模型包") }
            } }
        }
        c.busy?.let { Row(verticalAlignment = Alignment.CenterVertically) { LinearProgressIndicator(Modifier.width(80.dp)); Spacer(Modifier.width(8.dp)); Text(it) } }

        Section("加载模型") {
            val models = c.models
            val current = models.firstOrNull { it.id == s.modelId }
            Dropdown(
                label = current?.let { "${it.name}  (${it.version}, ${it.sampleRate / 1000}k${if (it.hasIndex) ", 索引" else ""})" } ?: "未选择模型",
                items = models.map { it.id to "${it.name} (${it.version}, ${it.sampleRate / 1000}k)" },
                onSelect = { id -> c.update { st -> st.copy(modelId = id) } },
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { showImport = true }, enabled = c.assetsReady && c.busy == null) { Text("导入 .pth") }
                OutlinedButton(onClick = { indexPicker.launch(arrayOf("*/*")) }, enabled = current != null && c.busy == null) {
                    Text(if (current?.hasIndex == true) "更换 .index" else "添加 .index")
                }
                if (current != null) TextButton(onClick = { c.deleteModel(current) }, enabled = !c.running) { Text("删除") }
            }
        }

        Section("常规设置") {
            LabeledSlider("响应阈值 (dB)", s.threshold, -60f..0f, 59, "%.0f") { v -> c.update { it.copy(threshold = v) } }
            LabeledSlider("音调设置", s.pitch, -16f..16f, 31, "%+.0f") { v -> c.update { it.copy(pitch = v) } }
            LabeledSlider("性别因子/声线粗细", s.formant, -2f..2f, 79, "%+.2f") { v -> c.update { it.copy(formant = v) } }
            LabeledSlider("Index Rate", s.indexRate, 0f..1f, 99, "%.2f") { v -> c.update { it.copy(indexRate = v) } }
            LabeledSlider("响度因子", s.rmsMixRate, 0f..1f, 99, "%.2f") { v -> c.update { it.copy(rmsMixRate = v) } }
            Text("音高算法：FCPE", style = MaterialTheme.typography.bodySmall)
        }

        Section("性能设置（修改后重新开始生效）") {
            LabeledSlider("采样长度 (s)", s.blockTime, 0.1f..1.0f, 89, "%.2f") { v -> c.update { it.copy(blockTime = (v * 100).roundToInt() / 100f) } }
            LabeledSlider("淡入淡出长度 (s)", s.crossfadeTime, 0.01f..0.15f, 13, "%.2f") { v -> c.update { it.copy(crossfadeTime = (v * 100).roundToInt() / 100f) } }
            LabeledSlider("额外推理时长 (s)", s.extraTime, 0.05f..3.0f, 58, "%.2f") { v -> c.update { it.copy(extraTime = (v * 100).roundToInt() / 100f) } }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(s.inputNoiseReduce, { c.update { it.copy(inputNoiseReduce = !it.inputNoiseReduce) } }, label = { Text("输入降噪") })
                FilterChip(s.outputNoiseReduce, { c.update { it.copy(outputNoiseReduce = !it.outputNoiseReduce) } }, label = { Text("输出降噪") })
            }
            val providers = c.device?.optJSONArray("providers")?.let { a -> (0 until a.length()).map { a.getString(it) } } ?: emptyList()
            fun avail(b: Backend) = when (b) {
                Backend.CPU, Backend.NATIVE -> true
                Backend.XNNPACK -> "XnnpackExecutionProvider" in providers
                Backend.GPU -> "WebGpuExecutionProvider" in providers
                Backend.NNAPI -> "NnapiExecutionProvider" in providers
                Backend.QNN_HTP, Backend.QNN_GPU -> "QNNExecutionProvider" in providers
            }
            Text("HuBERT 特征提取后端", style = MaterialTheme.typography.labelLarge)
            Dropdown(Backend.of(s.hubertBackend).label,
                Backend.entries.filter { it != Backend.NATIVE && avail(it) }.map { it.id to it.label }) { id -> c.update { it.copy(hubertBackend = id) } }
            Text("声码器 (NSF-HiFiGAN) 后端", style = MaterialTheme.typography.labelLarge)
            Dropdown(Backend.of(s.decBackend).label, Backend.entries.filter { avail(it) }.map { it.id to it.label }) { id -> c.update { it.copy(decBackend = id) } }
            SwitchRow("FP16 半精度（原生 NEON / GPU / NPU）", s.fp16) { v -> c.update { it.copy(fp16 = v) } }
            val big = c.device?.optJSONArray("big_cores")?.length() ?: 4
            LabeledSlider("CPU 线程（0=自动, 大核 $big 个）", s.threads.toFloat(), 0f..8f, 7, "%.0f") { v -> c.update { it.copy(threads = v.roundToInt()) } }
            SwitchRow("NSF 相位跨块连续（实验）", s.phaseLock) { v -> c.update { it.copy(phaseLock = v) } }
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = { c.bench() }, enabled = !c.running && c.busy == null && s.modelId.isNotEmpty()) { Text("性能测试") }
            }
            c.benchResult?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        }

        Section("音频") {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(s.mode == AudioMode.LOCAL, { c.update { it.copy(mode = AudioMode.LOCAL) } }, label = { Text("本地监听") })
                FilterChip(s.mode == AudioMode.VIRTUAL_MIC, { c.update { it.copy(mode = AudioMode.VIRTUAL_MIC) } }, label = { Text("虚拟麦克风 (Shizuku)") })
            }
            if (s.mode == AudioMode.VIRTUAL_MIC) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Shizuku：${ShizukuBridge.stateText}", Modifier.weight(1f))
                    if (ShizukuBridge.state == ShizukuBridge.State.NO_PERMISSION) TextButton(onClick = { ShizukuBridge.requestPermission() }) { Text("授权") }
                }
                if (c.running) {
                    Text(
                        when (c.captureMode) {
                            1 -> "● 正在变声（语音消息模式）"
                            2 -> "● 正在变声（通话模式）"
                            else -> "○ 待命：其他应用开始录音时自动变声，平时不占用麦克风"
                        },
                        color = if (c.captureMode != 0) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                SwitchRow("同时本地监听", s.monitor) { v -> c.update { it.copy(monitor = v) } }
                Text(
                    "微信、QQ 等应用录音时会自动收到变声后的声音，开头约 0.4–0.6 秒为静音（不会漏出本音）；" +
                        "「采样长度」越小，唤醒越快、但 CPU 负担越大。" +
                        "首次使用会通过 Shizuku 自动开启「RVC 虚拟麦克风」无障碍服务，它只用于让变声采集和语音消息同时进行，不读取屏幕内容。",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            LabeledSlider("输出增益", s.outputGain, 0f..3f, 29, "%.1f") { v -> c.update { it.copy(outputGain = v) } }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(!s.passthrough, { c.update { it.copy(passthrough = false) } }, label = { Text("输出变声") })
                FilterChip(s.passthrough, { c.update { it.copy(passthrough = true) } }, label = { Text("输入监听") })
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = { VoiceService.start(ctx) }, enabled = !c.running && c.busy == null && c.assetsReady) { Text("开始音频转换") }
            OutlinedButton(onClick = { VoiceService.stop(ctx) }, enabled = c.running) { Text("停止音频转换") }
        }
        StatusPanel()
        Spacer(Modifier.height(24.dp))
    }

    if (showImport) ImportDialog(onDismiss = { showImport = false }) { pth, idx ->
        showImport = false
        c.importModel(pth, idx)
    }
    c.shared?.let { sh -> SharedFilesDialog(sh) }
    c.message?.let { msg ->
        AlertDialog(onDismissRequest = { c.message = null }, confirmButton = { TextButton({ c.message = null }) { Text("确定") } }, text = { Text(msg) })
    }
}

@Composable
private fun StatusPanel() {
    val st = RvcController.status ?: return
    if (!st.optBoolean("running")) return
    val stages = st.optJSONObject("stages")
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            val block = st.optDouble("block_ms")
            val avg = st.optDouble("infer_avg_ms")
            Text("推理时间：%.0f ms / 每块 %.0f ms（实时率 %.2f）".format(avg, block, if (block > 0) avg / block else 0.0), fontWeight = FontWeight.Bold)
            Text("估计总延迟：%.0f ms".format(st.optDouble("latency_ms")), fontWeight = FontWeight.Bold)
            st.optJSONObject("latency_parts")?.let {
                Text("　= 采样 %.0f + 推理 %.0f + 输出缓冲 %.0f + 拼接 %.0f + 声卡 %.0f ms".format(
                    it.optDouble("block"), it.optDouble("infer"), it.optDouble("queue"), it.optDouble("sola"), it.optDouble("io")),
                    style = MaterialTheme.typography.bodySmall)
            }
            stages?.let {
                Text("HuBERT %.0f · FCPE %.0f · 索引 %.0f · 编码 %.0f · 流 %.0f · 声码器 %.0f ms".format(
                    it.optDouble("hubert"), it.optDouble("f0"), it.optDouble("index"), it.optDouble("enc"), it.optDouble("flow"), it.optDouble("dec")),
                    style = MaterialTheme.typography.bodySmall)
            }
            Text("后端：${st.optJSONObject("engine")?.optJSONObject("backends") ?: "-"}" +
                if (st.optBoolean("pipeline")) " · HuBERT 与声码器并行" else "", style = MaterialTheme.typography.bodySmall)
            Text("超时块 ${st.optLong("late_blocks")} · 欠载 ${st.optLong("out_underruns")} · 丢弃 ${st.optLong("dropped_samples")} 采样" +
                " · 停顿处回收 %.0f ms".format(st.optDouble("trimmed_ms", 0.0)),
                style = MaterialTheme.typography.bodySmall)
            st.optJSONObject("shell")?.let { sh ->
                Text("注入：采集 ${sh.optLong("captured")} / 注入 ${sh.optLong("injected")} 采样${sh.optString("error").let { e -> if (e.isNotEmpty()) " · $e" else "" }}",
                    style = MaterialTheme.typography.bodySmall)
            }
            st.optString("error").takeIf { it.isNotEmpty() }?.let { Text("错误：$it", color = MaterialTheme.colorScheme.error) }
        }
    }
}

@Composable
private fun ImportDialog(
    initialPth: Uri? = null,
    initialIndex: Uri? = null,
    title: String = "导入 RVC 模型",
    onDismiss: () -> Unit,
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
private const val BILIBILI_URL = "https://space.bilibili.com/79356601"
private const val RVC_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"

/** Top-right "⋮" menu: project links and version. */
@Composable
private fun MoreInfoMenu() {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    var open by remember { mutableStateOf(false) }
    fun openUrl(url: String) {
        open = false
        runCatching { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
            .onFailure { RvcController.message = "无法打开链接：$url" }
    }
    @Composable
    fun LinkItem(title: String, subtitle: String, url: String) = DropdownMenuItem(
        text = {
            Column {
                Text(title)
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        },
        onClick = { openUrl(url) },
    )
    Box {
        IconButton(onClick = { open = true }, modifier = Modifier.semantics { contentDescription = "更多信息" }) {
            Text("⋮", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            LinkItem("GitHub：给个 Star", "nasuiyile/mobile-rvc", REPO_URL)
            LinkItem("B 站：关注我", "space.bilibili.com/79356601", BILIBILI_URL)
            LinkItem("原版 RVC 项目", "RVC-Project/Retrieval-based-Voice-Conversion-WebUI", RVC_URL)
            HorizontalDivider()
            DropdownMenuItem(text = { Text("版本 ${BuildConfig.VERSION_NAME}") }, onClick = {}, enabled = false)
        }
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            content()
        }
    }
}

@Composable
private fun LabeledSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, steps: Int, fmt: String, onChange: (Float) -> Unit) {
    Column {
        Row {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            Text(fmt.format(value), style = MaterialTheme.typography.bodyMedium)
        }
        val span = range.endInclusive - range.start
        Slider(
            value = value.coerceIn(range),
            onValueChange = { v ->
                // snap to the step grid without drawing tick marks
                val n = (steps + 1).coerceAtLeast(1)
                onChange(range.start + ((v - range.start) / span * n).roundToInt() * span / n)
            },
            valueRange = range,
        )
    }
}

@Composable
private fun SwitchRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.weight(1f))
        Switch(checked, onChange)
    }
}

@Composable
private fun Dropdown(label: String, items: List<Pair<String, String>>, onSelect: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Column {
        OutlinedButton(onClick = { open = true }, modifier = Modifier.fillMaxWidth()) { Text(label) }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            items.forEach { (id, text) ->
                DropdownMenuItem(text = { Text(text) }, onClick = {
                    open = false
                    onSelect(id)
                })
            }
        }
    }
}
