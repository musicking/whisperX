# n8n：音频生成 SRT 字幕

导入文件：[whisperx-subtitles.json](whisperx-subtitles.json)。仅使用三个 n8n 内置节点，无需安装插件或配置凭据。

流程：`Audio Webhook → Generate Subtitles → Return SRT`。

## 导入和运行

1. 在 n8n 新建工作流，菜单选择 **Import from File**，导入 JSON。
2. 打开 **Generate Subtitles**，确认 API 地址为 `http://192.168.1.33:7865/v1/audio/subtitles`。地址必须能从 n8n 容器或服务器访问。
3. 打开 **Audio Webhook**，点击 **Listen for test event**，复制节点显示的 **Test URL**。
4. 使用下方命令发送音频；响应即 SRT 文件。
5. 保存并激活/发布工作流后，使用节点显示的 **Production URL**，无需手动监听。

Webhook 的 **Binary Data** 和 **Raw Body** 保持关闭，使用默认 multipart 解析。一次提交一个文件，字段名为 `file`。

## 调用示例

将下方 `http://你的n8n地址:5678/webhook-test/whisperx-subtitles` 替换为节点显示的 Test URL。
生产调用通常使用 `/webhook/whisperx-subtitles`，以节点显示的 URL 为准。

Windows PowerShell：

```powershell
curl.exe "http://你的n8n地址:5678/webhook-test/whisperx-subtitles" `
  -F "file=@D:\King\SingLionVideo\Source\如果有两亿 通常会装配什么资产.mp3" `
  -F "language=zh" `
  -o subtitles.srt
```

根据原稿生成字幕（将 TXT 内容作为 `document_text` 文本字段发送，不上传 document 文件）：

```powershell
curl.exe "http://你的n8n地址:5678/webhook-test/whisperx-subtitles" `
  -F "file=@D:\King\SingLionVideo\Source\如果有两亿 通常会装配什么资产.mp3" `
  -F "document_text=<D:\King\SingLionVideo\Source\如果有两亿 通常会装配什么资产.txt" `
  -F "language=zh" `
  -o subtitles.srt
```

Linux：

```bash
curl 'http://你的n8n地址:5678/webhook-test/whisperx-subtitles' \
  -F 'file=@/path/to/narration.mp3' \
  -F 'document_text=</path/to/document.txt' \
  -F 'language=zh' \
  -o subtitles.srt
```

不使用原稿时，删除 `document_text` 那一行即可。原稿文件使用 UTF-8，内容应与音频一致。

## 输入与输出

| 输入字段 | 用途 |
| --- | --- |
| `file` | 必填，音频文件 |
| `language` | 可选，默认 `zh` |
| `document_text` | 可选，原稿文字；未填写或空白时不传给 API |

成功：HTTP 200，下载 `subtitles.srt`。在 n8n 执行结果中也可查看 **Generate Subtitles → Binary → subtitles**。
API 错误：原样返回 API 状态码及 JSON 错误；不要将错误响应当作字幕。
连接失败或超时：由 n8n 记录节点错误，不自动重试。

调用超时为 15 分钟。中文按标点切句、删除字幕末尾标点的行为由 WhisperX API 实现，n8n 不再次加工字幕。
如果 n8n 前面有反向代理，其等待超时也需覆盖音频推理耗时。

## 验证范围

已核对 JSON、节点连接和 n8n 官方参数定义，并验证有原稿/无原稿时的请求参数表达式。
未连接你的 n8n 实例，尚未执行真实的工作流导入和端到端运行。

参考：[Webhook](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/)、
[HTTP Request](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/)、
[Respond to Webhook](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.respondtowebhook/)。
