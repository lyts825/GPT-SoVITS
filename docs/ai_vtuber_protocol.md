# AI VTuber 流协议 1

## 范围与版本

适配版本为 0.1.0，上游基线为 `b4570b876f5c086819ba84e3215472733c6a7b70`。独立入口为 `api_vtuber.py`。旧 `api_v2.py` 和 `/tts` 保持兼容。

服务仅加载一组固定模型、参考音频和合成参数。请求不能切换权重、参考音频或采样参数。运行时仅允许一个推理任务；忙时返回 409，不建立等待队列。接口全部要求 `Authorization: Bearer <token>`，默认只监听本机。

## HTTP 接口

- `GET /ai-vtuber/v1/capabilities`：返回 `protocol=1`、`service_version`、`instance_id`、`profile_fingerprint`、`format="pcm_s16le"`、`channels=1`、`sample_width=2`、`max_text_chars=512`、`max_pcm_bytes=16777216` 和固定的 `streaming_options`。读操作不合成或预热。
- `POST /ai-vtuber/v1/tts`：请求包含 `request_id`、`speech_unit_hash`、`text_hash`、`text`、`expected_instance_id` 和 `expected_profile_fingerprint`。哈希均为小写 SHA-256 十六进制串；`text_hash` 是 text 的 UTF-8 字节哈希。request_id 允许 1～128 个 ASCII 字母、数字及 `_.:-`。身份、指纹或文本哈希不匹配时拒绝，不能静默降级。
- `POST /ai-vtuber/v1/cancel/{request_id}`：返回 `state="cancellation_requested" / "already_finished" / "unknown"` 和布尔 `worker_stopped`。只标记该请求，不等待推理锁。未知请求返回 unknown，不改变活动任务。仅保留最多 128 个已结束 ID；ID 在保留期内不能重用。

无效请求返回 400/422，鉴权失败返回 401，忙或身份不匹配返回 409。引擎或生成器清理异常会隔离实例，后续合成返回 503，需重启服务。错误响应不回传文件路径、原始异常、凭据或文本。客户端不自动重试合成。

## 二进制帧

成功响应类型为 `application/x-ai-vtuber-pcm`。正文以 4 字节 ASCII `AVT1` 开始。每帧是 `kind:uint8 + payload_length:uint32（大端）+ payload`。网络块与帧边界无关。

| kind | 类型 | payload |
| --- | --- | --- |
| 1 | START | UTF-8 JSON，最多 8192 字节 |
| 2 | PCM | `sequence:uint32（大端，从 0 开始）+ PCM16LE`；单帧 PCM 最多 16384 字节，非空且完整对齐 |
| 3 | END | UTF-8 JSON，最多 8192 字节 |
| 4 | ERROR | UTF-8 JSON，最多 8192 字节 |

START 恰好一次，包含 `protocol`、`request_id`、`speech_unit_hash`、`text_hash`、`instance_id`、`profile_fingerprint`、`sample_rate`、`channels`、`sample_width` 和 `format`。采样率必须在 8000～192000 Hz，格式为 `pcm_s16le`、单声道、2 字节。

至少一帧 PCM 后才允许 END。END 包含 `request_id`、`speech_unit_hash`、`text_hash`、`pcm_bytes`、`pcm_frames`、`chunks` 和 `pcm_sha256`。字节数、帧数、块数和哈希均按已发送 PCM 累计。只有推理生成器自然结束、没有取消或异常时，才能发送 END。END 后必须干净 EOF，任何多余帧或字节都无效。

ERROR 包含 `request_id` 和固定错误代码 `code`，如 `generation_failed`、`cancelled`、`invalid_audio`。它不是成功终态，不能附加 END。空输出、异常、取消、缺少 END、序号缺失、格式变化和校验不符都必须让客户端合成迭代器抛错。

客户端只在合法 END 和 EOF 后正常结束迭代。END 证明生成与传输完整，不代表读音正确或音频已播出；主播端仍以播放器回执入账。

## 取消与背压

服务使用请求独立的取消标记，断开连接同样触发取消。工作线程在进入推理前、每次取下一块前、提交输出前检查标记。推理内部在支持的检查点观察同一标记；不能中断正在执行的 GPU kernel。

待发队列最多 2 帧。队列满时停止继续拉取推理输出；取消可解除该背压。旧任务的线程已 join、生成器已退出且 HTTP 响应已收尾之后才能释放单任务槽。推理结束但响应阻塞时仍占用槽位；即使 HTTP 客户端已断开，也不能提前把共享模型交给下一请求。

客户端取消或异常关闭流时，尽力调用对应 request_id 的取消接口，使用独立且有界的连接超时。返回 cancellation_requested 只证明请求已登记，不能解释为 GPU 已停止。
