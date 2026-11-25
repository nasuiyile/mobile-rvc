package com.mobilervc.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.mobilervc.engine.ModelScope
import com.mobilervc.engine.RvcController
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private fun mb(bytes: Long) = when {
    bytes <= 0 -> ""
    bytes >= 1L shl 30 -> "%.1f GB".format(bytes / 1073741824.0)
    else -> "%.0f MB".format(bytes / 1048576.0).let { if (it == "0 MB") "<1 MB" else it }
}

private fun count(n: Int) = if (n >= 10000) "%.1f万".format(n / 10000.0) else n.toString()

/**
 * Browse RVC models on ModelScope and download one with a tap: search → repo → (folders) → model.
 * Back steps out of a folder, then the repo, then closes.
 */
@Composable
fun ModelLibraryDialog(onDismiss: () -> Unit) {
    val c = RvcController
    val keyboard = LocalSoftwareKeyboardController.current
    var query by remember { mutableStateOf("") }
    var searched by remember { mutableStateOf("") }
    var page by remember { mutableIntStateOf(1) }
    val repos = remember { mutableStateListOf<ModelScope.Repo>() }
    var total by remember { mutableIntStateOf(0) }
    var repo by remember { mutableStateOf<ModelScope.Repo?>(null) }
    val folders = remember { mutableStateListOf<String>() }  // path stack inside [repo]
    var items by remember { mutableStateOf<List<ModelScope.Item>?>(null) }
    var loading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var reload by remember { mutableIntStateOf(0) }

    LaunchedEffect(searched, page, reload) {
        if (repo != null && repos.isNotEmpty()) return@LaunchedEffect
        loading = true
        error = null
        runCatching { withContext(Dispatchers.IO) { ModelScope.search(searched, page) } }
            .onSuccess { p ->
                if (page == 1) repos.clear()
                repos += p.repos.filter { r -> repos.none { it.id == r.id } }
                total = p.total
            }
            .onFailure { error = "加载失败：${it.message ?: it.javaClass.simpleName}" }
        loading = false
    }
    val folder = folders.lastOrNull() ?: ""
    LaunchedEffect(repo, folder, reload) {
        val r = repo ?: return@LaunchedEffect
        items = null
        loading = true
        error = null
        runCatching { withContext(Dispatchers.IO) { ModelScope.list(r, folder) } }
            .onSuccess { items = it }
            .onFailure { error = "加载失败：${it.message ?: it.javaClass.simpleName}" }
        loading = false
    }

    fun back() {
        when {
            folders.isNotEmpty() -> folders.removeAt(folders.lastIndex)
            repo != null -> { repo = null; items = null; error = null }
            else -> onDismiss()
        }
    }
    fun download(title: String, pth: String, index: String?) {
        val r = repo ?: return
        c.downloadModel(title, ModelScope.fileUrl(r, pth), index?.let { ModelScope.fileUrl(r, it) })
        onDismiss()
    }

    Dialog(onDismissRequest = ::back, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.windowInsetsPadding(WindowInsets.safeDrawing)) {
                Row(Modifier.fillMaxWidth().padding(start = 4.dp, end = 4.dp, top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = ::back) { Text(if (repo == null) "关闭" else "‹ 返回") }
                    Text(
                        repo?.let { r -> (listOf(r.title) + folders.map { it.substringAfterLast('/') }).joinToString(" / ") } ?: "在线模型库",
                        Modifier.weight(1f), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold,
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                    )
                    Spacer(Modifier.width(8.dp))
                }
                if (repo == null) {
                    Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                        val submit: () -> Unit = {
                            keyboard?.hide()
                            searched = query.trim()
                            page = 1
                            reload += 1
                        }
                        OutlinedTextField(
                            query, { query = it }, Modifier.weight(1f), singleLine = true,
                            placeholder = { Text("搜索角色、歌手（如 miku、原神）") },
                            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                            keyboardActions = KeyboardActions(onSearch = { submit() }),
                        )
                        TextButton(onClick = submit) { Text("搜索") }
                    }
                    Text(
                        "来自魔搭社区（modelscope.cn），国内可直接访问。模型由社区上传，请遵守各模型的使用说明。",
                        Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                Status(loading && (repo != null || repos.isEmpty()), error, empty = when {
                    repo == null -> !loading && error == null && repos.isEmpty()
                    else -> !loading && error == null && items?.isEmpty() == true
                }, emptyText = if (repo == null) "没有找到相关模型" else "这里没有模型文件") { reload++ }
                LazyColumn(Modifier.fillMaxSize()) {
                    if (repo == null) {
                        items(repos, key = { it.id }) { r ->
                            Row2(r.title, "${r.id}  ·  下载 ${count(r.downloads)}" + mb(r.size).let { if (it.isEmpty()) "" else "  ·  $it" }) {
                                repo = r
                                folders.clear()
                            }
                        }
                        if (repos.isNotEmpty() && repos.size < total) item {
                            Row(Modifier.fillMaxWidth().padding(12.dp), horizontalArrangement = Arrangement.Center) {
                                if (loading) CircularProgressIndicator(Modifier.padding(4.dp))
                                else TextButton(onClick = { page++ }) { Text("加载更多（共 $total 个）") }
                            }
                        }
                    } else {
                        items(items.orEmpty(), key = { it.name }) { item ->
                            when (val it = item) {
                                is ModelScope.Item.Folder -> Row2("📁  ${it.name}", null) { folders += it.path }
                                is ModelScope.Item.Voice -> Row2(
                                    it.title,
                                    it.name + "  " + mb(it.pthSize) + if (it.index != null) "  ·  含索引 ${mb(it.indexSize)}" else "  ·  无索引",
                                    action = "下载", actionEnabled = c.busy == null,
                                ) { download(it.title, it.pth, it.index) }
                                is ModelScope.Item.Archive -> Row2(it.name.removeSuffix(".zip"), "压缩包  " + mb(it.size), action = "下载",
                                    actionEnabled = c.busy == null) { download(it.name.removeSuffix(".zip"), it.path, null) }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Status(loading: Boolean, error: String?, empty: Boolean, emptyText: String, onRetry: () -> Unit) {
    when {
        loading -> Row(Modifier.fillMaxWidth().padding(24.dp), horizontalArrangement = Arrangement.Center) { CircularProgressIndicator() }
        error != null -> Column(Modifier.fillMaxWidth().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(error, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
            TextButton(onClick = onRetry) { Text("重试") }
        }
        empty -> Text(emptyText, Modifier.fillMaxWidth().padding(24.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** One list row; with [action] the row shows a button for it, otherwise the whole row is tappable. */
@Composable
private fun Row2(title: String, subtitle: String?, action: String? = null, actionEnabled: Boolean = true, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().let { if (action == null) it.clickable(onClick = onClick) else it }.padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge, maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (subtitle != null) Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 2, overflow = TextOverflow.Ellipsis)
        }
        if (action != null) FilledTonalButton(onClick = onClick, enabled = actionEnabled, modifier = Modifier.padding(start = 8.dp)) { Text(action) }
        else Text("›", style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
