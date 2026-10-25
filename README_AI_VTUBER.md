# AI VTuber 语音服务适配

## 版本与范围

本适配为 AI 杂谈直播控制台提供固定声线的流式合成、完整性校验和请求取消。独立入口为 `api_vtuber.py`，原有 `api_v2.py` 和 `/tts` 继续保留。

| 项目 | 值 |
| --- | --- |
| 适配版本 | `0.1.0` |
| 流协议 | `1`，帧标识 `AVT1` |
| 上游基线 | `b4570b876f5c086819ba84e3215472733c6a7b70` |
| 用户 fork | [lyts825/GPT-SoVITS](https://github.com/lyts825/GPT-SoVITS/tree/codex/ai-vtuber)，分支 `codex/ai-vtuber` |
| 目标版本标签 | `ai-vtuber-v0.1.0`，发布状态以实际仓库为准 |
| 版本日期 | 2026-10-25 |

本轮只完成本地实现和离线检查。真实推理环境尚未安装，也未运行真实模型、音频设备或直播。当前没有首音延迟、实时率、读音准确率或性能提升数据。

## 1. 准备独立推理环境

按本仓库保留的[上游安装说明](README.md)准备 GPT-SoVITS 环境、对应的 PyTorch 依赖和模型资源。推理服务使用独立环境，直播控制台继续使用自己的环境。

新增入口使用 Pydantic 2 的接口。本轮离线检查使用 `pydantic==2.10.6`；复用已有环境时，确认没有继续使用 Pydantic 1。真实推理依赖组合仍需单独验证。

以下 PowerShell 命令均从本 fork 根目录执行。先将 `$TtsPython` 改为已经准备好的推理环境解释器路径，再安装 `requirements-vtuber.txt` 中的接口依赖约束。离线验证用的小环境不包含完整模型依赖，不能用来启动真实推理。

```powershell
$TtsPython = "D:\你的推理环境\python.exe"

# 上游完整环境准备好后，安装本入口的 FastAPI 和 Pydantic 约束。
& $TtsPython -m pip install -r requirements-vtuber.txt

# 首次创建本地声线配置。已有此文件时直接编辑。
Copy-Item -LiteralPath vtuber.example.json -Destination vtuber.local.json
New-Item -ItemType Directory -Path .local -Force | Out-Null
```

`vtuber.local.json` 和 `.local/` 已加入 Git 忽略规则。模型、参考素材、凭据、测试录音和实验报告保留在本地。

## 2. 固定模型与声线

### 模型 YAML

在 `.local/tts_infer.yaml` 中提供显式 `custom` 节。以下示例使用上游 v2 的路径名称；你必须先准备这些资源，或改成实际使用的权重和目录。

```yaml
custom:
  version: v2
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
```

四个模型路径都必须显式填写。相对模型路径按仓库根目录解析；`version` 必须与实际加载的权重一致。`device` 和 `is_half` 按运行环境设置。示例中的 CPU 使用 `is_half: false`；改用 CUDA 时，先确认对应依赖和权重支持所选精度。

中文配置使用非 v1 模型时，还需提前准备 `GPT_SoVITS/text/G2PWModel/` 本地资源。入口会检查 ONNX 模型、配置和所需词表，不自动下载。LoRA 权重还需要对应版本的底模。相关说话人编码器和声码器也要按上游版本要求备齐。启动时会将这些实际依赖纳入指纹检查。

### 声线 JSON

编辑 `vtuber.local.json` 中的四个字段：

| 字段 | 用途 |
| --- | --- |
| `reference_audio` | 本地参考音频；相对路径按 JSON 所在目录解析 |
| `prompt_text` | 参考音频实际说出的原文 |
| `prompt_lang` | 参考文本语言 |
| `text_lang` | 合成文本语言 |

参考文件上限为 32 MiB，还须满足所选模型的参考音频要求。服务启动时会生成内容快照，提取参考音频和提示文本特征。这个过程不合成预热样例，也不打开播放设备。首次真实合成的耗时仍需实测。

默认 `options` 使用 `streaming_mode: 1`、`gpt_streaming: false` 和 `fragment_interval: 0.0`。服务固定 `batch_size: 1`、`text_split_method: "cut0"`，关闭 `parallel_infer`、`split_bucket` 和 `super_sampling`。一个请求对应一段已经确定的完整文本，请求开始后不能追加文本。

模式 2、3 必须显式设置 `gpt_streaming: true`。当前入口只允许 `v2ProPlus`、`v5turbo` 和 `v5dev` 使用语义流式路径，且仍需受支持的加速后端。要试验 hybrid，再显式设置 `streaming_window_mode: "hybrid"`；默认值为 `fixed`。这些选项的实际速度、连续性和读音效果尚未验证。

模型、参考音频、提示文本和合成参数在单个服务实例内固定。修改后需要重启服务，并重新绑定指纹。HTTP 请求不能切换这些配置。

## 3. 检查配置并启动服务

先检查声线 JSON 的结构和参数范围：

```powershell
& $TtsPython api_vtuber.py --profile vtuber.local.json --tts-config .local/tts_infer.yaml --check-config
```

`--check-config` 不读取 TTS YAML，也不检查模型、参考音频和 G2PW 文件是否存在。它不会加载模型或启动服务。真实资源、模型版本和后端兼容性在正式启动时检查。

服务和控制台都需要环境变量 `VTUBER_TTS_TOKEN`。令牌至少包含 24 个非空白 ASCII 字符。下面的命令生成一次随机值，保存在当前终端环境中；请把同一值提供给控制台进程。

```powershell
$env:VTUBER_TTS_TOKEN = & $TtsPython -c "import secrets; print(secrets.token_urlsafe(32))"

& $TtsPython api_vtuber.py --profile vtuber.local.json --tts-config .local/tts_infer.yaml --host 127.0.0.1 --port 9881
```

入口只允许回环地址 `127.0.0.1` 或 `::1`，默认端口为 `9881`，使用单个服务进程。启动成功后会打印以 `profile_fingerprint=` 开头的一行。保留后面的 64 位十六进制指纹，用于下一步绑定。

指纹包含模型资源、参考素材、前端代码和固定参数。服务启动会核对实际加载的配置，防止缺失资源触发静默替换。资源或配置有变化时，以本次启动打印的指纹为准。

指纹范围限于实现中列出的资源和前端文件，包括相应词典、现存缓存和所需 LoRA 底模。Python 环境和全部外部语言依赖仍需单独管理，不能由声线指纹证明完整锁定。

## 4. 接入直播控制台

在直播控制台的 `config.local.toml` 中编辑已有的 `[voices.gpt_sovits_framed]` 配置节。将占位值替换为服务打印的指纹。

```toml
[voices.gpt_sovits_framed]
label = "GPT-SoVITS 项目 fork / 完整流校验"
adapter = "gpt_sovits_framed"
base_url = "http://127.0.0.1:9881"
api_key_env = "VTUBER_TTS_TOKEN"
expected_profile_fingerprint = "替换为服务打印的 64 位小写十六进制指纹"
timeout_seconds = 40
```

在控制台终端设置相同令牌，再按控制台项目的启动说明运行。令牌值只放环境变量，不写入 TOML、JSON 或 Git。控制台中选择这项 framed 语音服务；空指纹或不匹配的指纹会阻止合成。

客户端首次合成前会读取能力信息，绑定服务实例和固定声线指纹。服务重启后，即使声线指纹相同，实例 ID 也会改变。停止控制台当前会话后再启动，以创建新的语音适配器。更新声线时，还须同步修改 `expected_profile_fingerprint`。

原版 `gpt_sovits` 适配器继续连接旧 `/tts` 接口。新增的 AVT1 帧流只供 `gpt_sovits_framed` 使用。完整字段和错误约定见[流协议说明](docs/ai_vtuber_protocol.md)。

## 5. 运行边界与故障处理

服务只有一个请求槽，忙时返回 `409`，不建立等待队列。待发队列最多保留 2 帧，满时暂停拉取推理输出。只有生成器退出、推理线程已 join 且 HTTP 响应收尾，服务才释放请求槽。慢客户端也会占用这个槽位。

取消接口只标记指定请求。断连在等待首音期间也能触发取消。推理通过检查点协作退出，无法中断正在执行的 GPU kernel。`cancellation_requested` 只表示登记成功；`worker_stopped` 单独表示推理线程是否已经退出。

| 情况 | 处理 |
| --- | --- |
| `401` | 检查服务和控制台是否使用同一个令牌 |
| `400`、`422` | 检查请求字段、文本长度和哈希 |
| `409` | 检查任务是否尚未收尾，或服务实例、声线指纹是否已变化 |
| `503` | 引擎或生成器清理异常导致实例隔离；排查原因后重启服务 |
| ERROR 帧、缺失 END、序号或校验不符 | 将该次合成记为失败；客户端不自动重试 |

服务只保留最近 128 个结束请求 ID，客户端应为每次合成生成新的唯一 ID。旧任务的取消标记不会转移给其他活动任务。

END 帧携带 PCM 字节数、采样帧数、块数和 SHA-256。客户端同时验证 END 和干净 EOF，才能认定生成流完整。**完整性校验不证明读音正确，也不证明音频已经播放。** 播放记忆仍由控制台的实际播放回执驱动。

## 6. 验证状态

本轮检查使用假引擎、人工构造的协议数据和进程内 ASGI（Asynchronous Server Gateway Interface，异步服务器网关接口）。检查覆盖鉴权、配置绑定、分帧、完整性、背压、取消、单槽生命周期和异常隔离。没有监听测试端口，也没有调用真实模型或设备。

本轮相关检查共 103 项：fork 内 47 项，控制台新旧协议适配 53 项，跨仓库互通 3 项。服务及互通检查已使用 FastAPI 0.115.14、Pydantic 2.10.6 和 NumPy 1.26.4 通过。互通检查使用缓冲的内存传输，不能据此判断真实网络流式时序。

离线测试材料放在 `.local/tests/`，不纳入 Git。真实部署仍需验证权重与前端兼容性、参考音频、首音等待、连续播放、取消退出和实际读音。任何速度或准确率结论都应来自对应硬件、模型和文本的实测。
