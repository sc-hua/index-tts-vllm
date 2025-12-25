# IndexTTS-vLLM API 接入文档

本文档介绍 `api_server_modified.py` 提供的 TTS API 接口使用方法。

## 服务启动

### 启动命令

```bash
python api_server_modified.py --model_dir <模型路径> [选项]
```

### 启动参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--host` | string | `0.0.0.0` | 服务监听地址 |
| `--port` | int | `7880` | 服务监听端口 |
| `--model_dir` | string | `ckpts/indextts-1_5-vllm` | 模型权重路径 |
| `--gpu_memory_utilization` | float | `0.02` | vLLM GPU 显存占用率 |

### 启动示例

```bash
# 使用默认配置
python api_server_modified.py

# 自定义配置
python api_server_modified.py \
    --host 0.0.0.0 \
    --port 7880 \
    --model_dir ./checkpoints/Index-TTS-1.5-vLLM \
    --gpu_memory_utilization 0.25
```

---

## API 接口

### 1. 健康检查

检查服务运行状态。

**请求**

```
GET /health
```

**响应**

成功 (200):
```json
{
    "status": "healthy",
    "message": "Service is running",
    "timestamp": 1703500000.123
}
```

失败 (503):
```json
{
    "status": "unhealthy",
    "message": "TTS model not initialized"
}
```

**示例**

```bash
curl http://localhost:7880/health
```

---

### 2. TTS 合成 (音频路径/Base64)

使用参考音频文件路径或 Base64 编码的音频进行语音合成。

**请求**

```
POST /tts_url
Content-Type: application/json
```

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 待合成的文本内容 |
| `audio_paths` | string[] | 否* | 参考音频文件路径列表（服务器本地路径） |
| `spk_audio_base64` | string | 否* | Base64 编码的参考音频（WAV 格式） |
| `speaker_audio_base64` | string | 否* | 同 `spk_audio_base64`，别名 |
| `seed` | int | 否 | 随机种子，默认 8 |

> *注：`audio_paths` 和 `spk_audio_base64`/`speaker_audio_base64` 至少提供一种

**响应**

- 成功：返回 WAV 格式音频二进制数据 (`audio/wav`)
- 失败：返回 JSON 错误信息

**示例**

使用音频路径：

```python
import requests

url = "http://localhost:7880/tts_url"
data = {
    "text": "你好，这是一段测试语音",
    "audio_paths": [
        "assets/jay_promptvn.wav",
        "assets/vo_card_klee_endOfGame_fail_01.wav"
    ],
    "seed": 8
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

使用 Base64 音频：

```python
import requests
import base64

# 读取本地音频并编码为 base64
with open("reference.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

url = "http://localhost:7880/tts_url"
data = {
    "text": "你好，这是一段测试语音",
    "spk_audio_base64": audio_base64
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

cURL 示例：

```bash
curl -X POST http://localhost:7880/tts_url \
    -H "Content-Type: application/json" \
    -d '{"text": "你好世界", "audio_paths": ["assets/jay_promptvn.wav"]}' \
    --output output.wav
```

---

### 3. TTS 合成 (预注册角色)

使用预注册的角色名称进行语音合成。

**请求**

```
POST /tts
Content-Type: application/json
```

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 待合成的文本内容 |
| `character` | string | 是 | 预注册的角色名称 |

**响应**

- 成功：返回 WAV 格式音频二进制数据 (`audio/wav`)
- 失败：返回 JSON 错误信息

**示例**

```python
import requests

url = "http://localhost:7880/tts"
data = {
    "text": "你好，这是一段测试语音",
    "character": "jay_klee"
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

cURL 示例：

```bash
curl -X POST http://localhost:7880/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "你好世界", "character": "jay_klee"}' \
    --output output.wav
```

---

### 4. 获取可用角色列表

获取所有预注册的角色及其参考音频配置。

**请求**

```
GET /audio/voices
```

**响应**

返回 JSON 格式的角色配置：

```json
{
    "jay_klee": [
        "assets/jay_promptvn.wav",
        "assets/vo_card_klee_endOfGame_fail_01.wav"
    ]
}
```

**示例**

```bash
curl http://localhost:7880/audio/voices
```

```python
import requests

response = requests.get("http://localhost:7880/audio/voices")
voices = response.json()
print(voices)
```

---

### 5. OpenAI 兼容接口

兼容 OpenAI TTS API 格式，便于与现有系统集成。

**请求**

```
POST /audio/speech
Content-Type: application/json
```

**请求参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input` | string | 是 | 待合成的文本内容 |
| `voice` | string | 是 | 预注册的角色名称 |
| `model` | string | 是 | 模型名称（当前忽略，可填任意值） |

**响应**

- 成功：返回 WAV 格式音频二进制数据 (`audio/wav`)
- 失败：返回 JSON 错误信息

**示例**

```python
import requests

url = "http://localhost:7880/audio/speech"
data = {
    "input": "你好，这是一段测试语音",
    "voice": "jay_klee",
    "model": "index-tts-1.5"
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

cURL 示例：

```bash
curl -X POST http://localhost:7880/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "你好世界", "voice": "jay_klee", "model": "index-tts"}' \
    --output output.wav
```

---

## 角色注册

角色通过 `assets/speaker.json` 文件进行注册。服务启动时会自动加载该文件。

### 配置格式

```json
{
    "角色名称": [
        "参考音频路径1",
        "参考音频路径2"
    ]
}
```

### 配置示例

```json
{
    "jay_klee": [
        "assets/jay_promptvn.wav",
        "assets/vo_card_klee_endOfGame_fail_01.wav"
    ],
    "another_voice": [
        "assets/another_reference.wav"
    ]
}
```

> 支持多个参考音频，输出的声线为多个参考音频的混合版本。

---

## 错误处理

所有接口在发生错误时返回 JSON 格式的错误信息：

```json
{
    "status": "error",
    "error": "错误详情..."
}
```

常见错误：

| HTTP 状态码 | 说明 |
|-------------|------|
| 400 | 请求参数错误（如未提供参考音频） |
| 500 | 服务器内部错误（如模型推理失败） |
| 503 | 服务不可用（如模型未初始化） |

---

## 注意事项

1. **参考音频格式**：建议使用 WAV 格式，采样率与训练数据一致
2. **文本长度**：过长的文本可能影响合成质量，建议分段合成
3. **并发请求**：服务支持并发，但受 GPU 显存限制
4. **CORS**：服务默认开启 CORS，允许所有来源访问（生产环境建议限制）
