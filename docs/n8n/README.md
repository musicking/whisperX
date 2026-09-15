# n8n：音频生成 SRT 字幕

导入文件：[whisperx-subtitles.json](whisperx-subtitles.json)。仅使用 n8n 内置节点，无需安装插件或配置凭据。

流程：`Audio Webhook → Has Document → 自动字幕或原稿字幕 → Respond SRT JSON`。
两个 HTTP Request 节点均采用原生逐项 Form-Data 字段，不对整组表单字段使用表达式。

## 导入和运行

1. 在 n8n 新建工作流，菜单选择 **Import from File**，导入 JSON。
2. 打开 **Generate Subtitles** 和 **Generate Document Subtitles**，确认 API 地址为 `http://192.168.1.33:7865/v1/audio/subtitles`。地址必须能从 n8n 容器或服务器访问。
3. 打开 **Audio Webhook**，点击 **Listen for test event**，复制节点显示的 **Test URL**。
4. 使用下方命令发送音频；响应为 JSON，`srtContent` 为完整字幕文本。
5. 替换旧工作流时先取消发布旧版本，再发布新版，避免相同 Webhook 路径冲突。生产地址为 `https://n8n.singlion.cn/webhook/whisperx/subtitles`。

Webhook 的 **Binary Data** 和 **Raw Body** 保持关闭，使用默认 multipart 解析。一次提交一个文件，对外字段名为 `data`；HTTP Request 节点将 n8n 的 `data` 二进制映射为 API 的 `file` 字段。

## 调用示例

将下方示例 URL 替换为节点显示的 Test URL，或者使用已发布的生产地址。

Windows PowerShell：

```powershell
curl.exe "https://n8n.singlion.cn/webhook-test/whisperx/subtitles" `
  -F "data=@D:\King\SingLionVideo\Source\如果有两亿 通常会装配什么资产.mp3" `
  -o result.json
```

根据原稿生成字幕（将 TXT 内容作为 `document_text` 文本字段发送，不上传 document 文件）：

```powershell
curl.exe "https://n8n.singlion.cn/webhook-test/whisperx/subtitles" `
  -F "data=@D:\King\SingLionVideo\Source\如果有两亿 通常会装配什么资产.mp3" `
  -F "document_text=<D:\King\SingLionVideo\Source\如果有两亿 通常会装配什么资产.txt" `
  -o result.json
```

Linux：

```bash
curl 'https://n8n.singlion.cn/webhook-test/whisperx/subtitles' \
  -F 'data=@/path/to/narration.mp3' \
  -F 'document_text=</path/to/document.txt' \
  -o result.json
```

不使用原稿时，删除 `document_text` 那一行即可。原稿文件使用 UTF-8，内容应与音频一致。

## 输入与输出

| 输入字段 | 用途 |
| --- | --- |
| `data` | 必填，音频文件 |
| `document_text` | 可选，原稿文字；未填写或空白时不传给 API |

成功：HTTP 200，普通 JSON 响应，与附件 ComfyUI 工作流的字段保持一致：

```json
{"success":true,"srtContent":"1\n00:00:00,031 --> 00:00:00,552\n大家好\n\n"}
```

HTTP Request 使用 Text 响应格式，字幕位于 JSON 的 `srtContent` 字段，不生成二进制字幕文件。
API 错误：保留 API 状态码，返回 `{"success":false,"error":"API 错误响应文本"}`。
WhisperX 没有任务 ID，不添加 ComfyUI 的 `promptId`。
连接失败或超时：由 n8n 记录节点错误，不自动重试。

调用超时为 15 分钟。工作流不再固定语言，由 Whisper 自动检测；无原稿时使用 Whisper 原生时间戳分段，有原稿时按对齐后的标点切句。`max_line_width` 和 `max_line_count` 已移除。字幕末尾标点由 WhisperX API 删除，n8n 不再次加工字幕。
如果 n8n 前面有反向代理，其等待超时也需覆盖音频推理耗时。

## 验证范围

旧版使用整组数组表达式，实测导致 file 未传入 API；截图同时显示表单字段区未正常显示。
新版改为原生字段数组，并通过 IF 选择原稿分支；已验证配置结构、分支连接和单字段表达式。
新版仍需重新导入发布后执行端到端验证；仅在 Node.js 中执行表达式不能替代 n8n 的配置结构验证。

参考：[Webhook](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/)、
[HTTP Request](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/)、
[Respond to Webhook](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.respondtowebhook/)。
