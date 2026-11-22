# Mobile RVC：Android 实时变声（FCPE）

在 Android 手机上运行 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的实时变声。功能对应原版 `realtime_gui.py`，音高算法固定为 FCPE：

- **直接导入原版 RVC 模型**：WebUI 导出的 `weights/*.pth`，加上可选的 `added_*.index`。手机上完成转换，不需要电脑。
- 支持 v1/v2、32k/40k/48k 全部标准结构，以及多说话人模型。
- 参数和原版界面一致：响应阈值、音调、性别因子（共振峰）、Index Rate、响度因子、采样长度、淡入淡出、额外推理时长、输入降噪、输出降噪、输入监听/输出变声。
- 两种工作模式
  - **本地监听**：麦克风 → 变声 → 耳机（AAudio 低延迟）。
  - **虚拟麦克风（Shizuku）**：拦截其他应用（微信、QQ、Discord、游戏等）的录音，换成变声后的声音。
- 计算后端：CPU（自研 NEON fp16 声码器 + ONNX Runtime）、GPU（ORT WebGPU/Vulkan）、NPU（骁龙 HTP / QNN，`qnn` 变体）、NNAPI。

## 工作原理

```
麦克风 48k ─► [阈值门限] ─► [输入降噪] ─► 重采样 16k ─┬─► HuBERT/ContentVec (v1: 9层+proj / v2: 12层)
                                                      │        └─► 索引检索 (IVF-Flat, 1/d² 加权)
                                                      └─► FCPE (STFT 原生, 网络 ONNX) ─► 音高缓存
  enc_p ─► z_p = m + e^logs·ε·0.66666 ─► flow⁻¹ ─► NSF 正弦源(原生) ─► NSF-HiFiGAN 声码器
  ─► [共振峰重采样] ─► 重采样 48k ─► [输出降噪] ─► 响度混合 ─► SOLA 拼接 ─► 输出
```

C++ 引擎（`android/app/src/main/cpp/engine/rvc_engine.cpp`）逐行移植自 RVC 的 `realtime_gui.py` 的 `audio_callback` 和 `infer/rtrvc.py`。分块方式、`skip_head` / `return_length`、音高缓存、SOLA、RMS 混合、TorchGate 降噪都和原版一致。

### 怎样直接使用原版 .pth

- `tools/rvc_mobile/export_synth.py` 为每种标准结构导出只含图、不含权重的 ONNX 模板（enc_p / flow / dec）。
- 图里的每个参数都以 `.pth` 中的原始名字和原始 fp16 数值作为外部初始化器。weight-norm 等变换写成图节点，由 ONNX Runtime 在加载时做常量折叠。
- 所以手机端导入时只需按清单把 `.pth` 里的张量字节拷进 `weights.bin`。负责这一步的是 `pth_reader.cpp`：用 C++ 实现了 zip 和 pickle 的解析，不依赖 Python。
- faiss 索引由 `ivf_index.cpp` 解析，转成紧凑格式（fp16 向量）。

## 手机端优化

| 优化 | 说明 |
|---|---|
| 自研 NEON fp16 声码器 | 声码器占整体约 80% 的计算量（40k 模型每秒音频约 90 GFLOP）。ORT 的 CPU 后端只有 fp32 卷积，所以自己实现了多抽头卷积内核：ARMv8.2 fp16 FMA，一次 8 路，每 64 项刷新到 fp32 累加以保证精度。转置卷积和带步长的噪声卷积都改写成多相形式，复用同一个内核。运行时检测 asimdhp，不支持时退回 fp32 NEON。 |
| GPU（WebGPU/Vulkan） | `onnxruntime-android` 1.32 自带 WebGPU EP。HuBERT 和声码器都能放到 GPU 上，加速器后端固定张量形状。 |
| NPU（QNN HTP） | `qnn` 变体内置 QNN 2.42 运行库（HTP V68–V81），使用 fp16 精度。编译好的上下文缓存到 `cache/accel_cache`，第二次启动不用重新编译。 |
| 并行与调度 | FCPE 和 HuBERT 并行执行；计算线程绑定大核（按 cpufreq 识别）；音频线程提到 -19 优先级。Android 13+ 通过 ADPF（PerformanceHint）上报每块耗时，帮助调度器提频。 |
| 实时驱动 | AAudio 独占低延迟流，配合无锁环形缓冲。积压超过上限时丢弃以控制延迟。 |
| 输出队列控制 | 变声结果和声卡之间排着的音频是纯延迟。以前它的长度听天由命：第一块推理慢（GPU 首次要编译着色器、NPU 要唤醒）或者中途卡一下，多出来的队列就永远留在那里，最多约一整块（虚拟麦克风模式两块）。现在：开始前先空跑 2 块预热；队列按实测的推理抖动维持在最小安全值；多出来的部分只在**停顿处**裁掉，快见底时在停顿处补齐，有声内容一个采样都不动。详见下文"延迟"。 |
| 两段流水线 | 每块拆成"分析"（HuBERT + FCPE + 索引）和"合成"（编码 + 流 + 声码器 + 拼接）两段。两段的重负载在不同处理器上时（如 HuBERT 用 GPU、声码器用 NPU 或 CPU），第 k+1 块的分析和第 k 块的合成重叠执行，实时条件从"两段之和 < 采样长度"放宽到"较慢的一段 < 采样长度"，采样长度可以设得更小。输出与顺序执行逐位相同。 |
| 降噪 | TorchGate 移植到 C++，非 2 的幂长度的 FFT 用 Bluestein 算法。 |

### 延迟

从说话到听到变声的总延迟由五部分相加（运行时状态栏会分别显示）：

```
总延迟 = 采样长度 + 推理时间 + 输出缓冲 + 拼接保留(淡入淡出 ≤40ms + 10ms) + 声卡缓冲
```

- **采样长度**是最大的一项，也是唯一需要自己权衡的一项：它必须大于推理时间（两段流水线开启时，大于较慢的一段即可）。先点"性能测试"看每块耗时，再把采样长度调到实时率 0.6–0.8 左右。
- **推理时间**取决于后端。HuBERT 和声码器分别放在 GPU 和 NPU（或 GPU 和 CPU）上时两段可以并行。
- **输出缓冲**由程序自动维持在刚好盖住推理抖动的水平，不需要设置。启动或卡顿后多出来的延迟会在说话的停顿处回收（状态栏"停顿处回收"）。
- 这些调度上的改动都不改变模型的计算：同样的输入、同样的设置，每一段有声内容都与改动前逐位相同。变的只有播出时间，以及停顿处被裁掉或补上的那几毫秒到几十毫秒。

### 虚拟麦克风原理（Shizuku）

`ShellAudioService` 通过 Shizuku 以 shell 身份（uid 2000）运行，有 `MODIFY_AUDIO_ROUTING`、`CAPTURE_AUDIO_OUTPUT`、`RECORD_AUDIO` 权限。

1. **注入**：注册 AudioPolicy 回环混音（`MIX_TYPE_RECORDERS` + `ROUTE_FLAG_LOOP_BACK`）。其他应用用 MIC、VOICE_COMMUNICATION、CAMCORDER、VOICE_RECOGNITION、UNPROCESSED 这几个录音源录音时，实际读到的是我们注入的变声音频。
2. **按需采集**：平时不录音。监测到有应用开始录音后，才开始采集真实麦克风。
   - 背景：Android 不允许两个普通录音同时进行，后启动的一方会把另一方静音（AOSP `AudioPolicyService::updateUidStates_l`）。
   - **语音消息类**（MIC 源，比如微信按住说话）：由 App 自己采集。App 开启了「RVC 虚拟麦克风」无障碍服务，无障碍应用的录音不受上面这条规则限制。
   - **通话类**（VOICE_COMMUNICATION 等隐私源）：由 shell 端采集，靠的是 `CAPTURE_AUDIO_OUTPUT` 权限。
   - 如果检测到我方采集被系统静音，会自动切换到另一条路径。
   - 无障碍服务由 shell 端通过 `settings put secure` 自动开启。它不读取屏幕内容，也不执行任何操作。
3. **唤醒和防漏音**：
   - 待命时持续给注入轨道写少量静音。否则注入通道会被系统判为不可用，新录音会先被接到真实麦克风上，漏出本音。启动时和每次录音结束后还要先把轨道的缓冲写满一次：流式 AudioTrack 在缓冲第一次填满之前不算开始播放，只靠几毫秒的补写永远填不满，开始后的第一次录音就会先接到真实麦克风约半秒。
   - **防静音底噪**：真实麦克风不会给出纯数字静音，很多应用把一串全零采样当成"麦克风被占用 / 没有权限"，直接结束录音。而变声流在两种情况下恰好是全零：每次录音开头第一块还没算出来的那段；以及开了响应阈值或降噪、人又没说话的时候（响度被压到零）。所以注入的音频里始终垫着一层相当于安静房间的底噪（粉噪，默认 −60 dBFS，可调可关），只进注入通道，不进本地监听。
   - 每次录音结束后，丢掉还没播出的音频，并用一次 250ms 的短录音把系统 Submix 管道里残留的旧音频读掉，避免下一次录音开头听到上一段。
   - 有应用开始录音时，通过录音状态回调立即响应；采集流事先建好，触发时只需启动。等第一块音频凑齐的这段时间里，先空跑一次推理来预热。
   - 从对方开始录音到收到变声，约等于采样长度 + 一次推理 + 约 100ms 系统缓冲（模拟器实测：采样长度 0.30 秒时约 600ms，0.15 秒时约 435ms）。

推理始终在 App 进程里进行。shell 端和 App 之间用两条管道传 float32 PCM。

## 验证结果

所有对比都在确定性模式下进行（噪声置零），参考是 RVC 原版代码（`infer/rtrvc.py` 加上 `realtime_gui.py` 的回调逻辑），测试模型为 v1 40k + 索引：

| 对比项 | 结果 |
|---|---|
| ONNX 合成器 vs RVC PyTorch（enc_p / flow / dec，多种长度） | 相对误差 ≤1.2e-5 |
| ONNX HuBERT vs Transformers（v1 / v2） | 相对误差 ≤4e-5 |
| ONNX FCPE vs torchfcpe | 最大 0.012 音分，清浊判断完全一致 |
| C++ 导入器 vs Python（weights.bin / index.bin） | 逐字节相同 |
| C++ 实时引擎 vs RVC 原版实时（默认参数） | SNR 80.6 dB |
| 同上，开启索引 0.6 + 升调 4 + 共振峰 2 + 门限 -45 + 响度 0.3 | SNR 78.3 dB |
| 同上，开启输入降噪 + 输出降噪 | SNR 64.1 dB |
| C++ TorchGate vs PyTorch（48k / 44.1k） | SNR 133 dB |
| 原生声码器 fp32 / fp16 vs ORT fp32 | 117 dB / 54.6 dB |
| 虚拟麦克风注入（Android 16 模拟器，uid 2000） | 用 MIC 源录音的应用收到 440 Hz 注入信号，真实麦克风同时正常采集 |

## 性能（实测）

> 目前**只在 arm64 模拟器上实测过**。模拟器跑在 Apple M6 主机上（2 个性能核 + 4 个能效核，模拟器配 6 个 vCPU）。**还没有在真实手机上跑过**。

测试条件：v1 40k 模型 + 索引，采样长度 0.3 s，额外推理时长 0.5 s。下表为每 300 ms 一块的平均处理时间（越低越好，必须低于 300 ms 才能实时）：

| 配置 | 每块耗时 | 实时率 |
|---|---|---|
| 全部用 ORT CPU（4 线程） | 224 ms | 0.75 |
| 原生 fp32 声码器 | 161 ms | 0.54 |
| **原生 fp16 声码器（默认）** | 119 ms | **0.40** |
| HuBERT + 声码器都用 WebGPU（主机 GPU 经模拟器转译） | 41 ms | 0.135 |

- App 内实时运行（本地模式和管道模式）：0 块超时、0 欠载。

端到端延迟（同一台模拟器，CPU 后端，采样长度 0.25 s，每块推理约 110 ms；用模拟声卡按实时速度播放，延迟由输入和播出波形的包络互相关测得，含采样、推理、输出缓冲和拼接，不含真实声卡）：

| 场景 | 改动前 | 改动后 |
|---|---|---|
| 一切顺利 | 434 ms（没有任何抖动余量） | 450 ms，0 欠载 |
| 第一块推理慢 900 ms（模拟 GPU / NPU 首次运行偏慢） | 660 ms，一直保持 | 10 秒内回到 455 ms，0 欠载 |
| 运行中卡顿一次 600 ms | 650 ms，一直保持 | 约 8 秒后 530 ms，之后继续回落 |
| 采样长度改为 0.15 s | — | 325–370 ms |

同一批测试里，停顿处的裁剪和补齐没有改动任何有声采样：实时输出中幅度大于 0.01 的 240 万个采样与离线处理的结果逐个相同。
- 手机能否实时，取决于大核的 NEON fp16 算力或 GPU/NPU。2024–2026 年的旗舰平台（骁龙 8 Gen 3 / 8 Elite / 8 Elite Gen 5、天玑 9300 / 9400 / 9500 等）按算力估计，默认 CPU 配置应该能实时，但这只是估计。
- 不够快时可以：
  - 在 App 里点“性能测试”看实际耗时；
  - 把声码器切到 GPU 或 NPU（骁龙机型用 `qnn` 包）；
  - 增大采样长度、减小额外推理时长。

## 构建

依赖：Android SDK（compileSdk 37）、NDK 29、CMake 3.31、JDK 17+、Python 3.10–3.12（只用于导出基础模型）。

```bash
cd tools
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

生成基础模型（会下载 RVC 的 hubert_base，产物在 `build/apk_assets/rvc`，约 231 MB）：

```bash
PYTHON=.venv/bin/python ./build_assets.sh
```

编译 APK。通用版（CPU / WebGPU / NNAPI / XNNPACK）：

```bash
cd ../android && ./gradlew :app:assembleStandardRelease
```

骁龙 NPU 版：

```bash
./gradlew :app:assembleQnnRelease
```

不打包模型的 APK 也能装。之后在 App 里导入基础模型包（`MAKE_ZIP=1 ./build_assets.sh` 会生成 `build/apk_assets/rvc_base_pack.zip`）。

### 命令行工具（开发调试）

`rvc_cli` 用同一份引擎代码，可以在电脑或 `adb shell` 上运行：

- `import` / `process`：导入模型、离线处理 wav；
- `bench`：基准测试；
- `pipetest`：管道模式实时测试；
- `dectest` / `hubtest`：对比加速器后端的精度与速度；
- `gatetest`：降噪对比。

Android 版的编译方式见 `android/app/src/main/cpp/CMakeLists.txt`。

不安装 Shizuku 也能检查虚拟麦克风链路：

```bash
adb shell 'CLASSPATH=$(pm path com.mobilervc | cut -d: -f2) app_process /system/bin com.mobilervc.shizuku.ShellSelfTest'
```

## 使用

1. 导入模型，两种方式：
   - 在 App 里点“导入 .pth”，选择 RVC 导出的推理模型，再选一个 `added_*.index`（可选）。
   - 在 QQ、微信或文件管理器里，对 `.pth` / `.index` 选“用其他应用打开”或“分享”，再选“导入到 RVC 实时变声”。`.pth` 和 `.index` 可以一起分享。只收到 `.index` 时，App 会让你选择把它添加到哪个已有模型。
2. 调整参数。常规设置可以实时调；性能设置要停止后重新开始才生效。
3. 选择模式：
   - **本地监听**：戴上耳机后点“开始音频转换”。
   - **虚拟麦克风**：先安装并启动 Shizuku，在 App 里授权，再点“开始音频转换”。之后 App 处于待命状态，微信、QQ 等应用一开始录音就会自动变声（开头约 0.5 秒没有人声，只有底噪）。如果对方应用一录音就自己停掉，把"防静音底噪"的电平调大一些再试。
4. 开始 / 停止按钮和运行状态固定在屏幕底部，加载、预热的进度固定在顶部，设置滚到哪里都看得到。底部状态一行显示总延迟、推理时间和实时率，点一下展开延迟构成（采样 + 推理 + 缓冲 + 拼接 + 声卡）、各阶段耗时、超时块数。想降低延迟时，先在"延迟与性能"里点"性能测试"，按它给出的建议把采样长度调小。
5. 各组设置点标题可以折叠，折叠后标题右侧显示当前值的摘要；折叠状态会记住。

## 目录结构

```
android/                 Android 工程（Kotlin + C++）
  app/src/main/cpp/engine/   平台无关的 C++ 引擎：pth/faiss 解析、DSP、ORT、原生声码器、降噪
  app/src/main/cpp/android/  AAudio 实时驱动、JNI
  app/src/main/java/...      Compose 界面、前台服务、Shizuku 桥接、shell 端音频服务
tools/                   模型导出和桌面参考实现（Python）
  rvc_mobile/rt_pipeline.py  引擎的可执行规格（numpy + ORT），C++ 按它移植
build/                   生成物（已忽略）
```

## 已知限制

- 只支持带音高（f0=1）的模型，不支持 nono 模型。非标准结构需要在电脑上运行 `export_synth.py --pth` 生成模板。
- 不同机型的虚拟麦克风行为可能不同（厂商可能改过音频策略）。部分系统会限制侧载应用的无障碍服务，这时需要在系统设置里手动开启「RVC 虚拟麦克风」。
- 我方开始采集后，个别应用可能通过系统录音回调发现“另一路录音”。按需采集已经把这段时间压缩到对方录音期间。
- 不 root 就无法直接改写其他应用进程里的音频数据。AudioPolicy 重定向是不 root 时唯一可行的替代办法。
- QNN / NNAPI / WebGPU 都没有在真机上验证过。加速器初始化失败时自动退回 CPU，界面会给出提示。
- ORT 的 fp16 计算图在 WebGPU 上精度不足，所以所有加速器都用 fp32 图，由 EP 自己选择精度。

## 许可

本项目代码参考并移植自以下 MIT 许可项目：RVC-Project/Retrieval-based-Voice-Conversion-WebUI、CNChTu/FCPE（torchfcpe）、TorchGate。依赖：ONNX Runtime（MIT）、Shizuku API（Apache-2.0）、HiddenApiBypass（Apache-2.0）。
