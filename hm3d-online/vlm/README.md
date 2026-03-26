# VLM 调用模块使用文档

基于阿里云百炼平台的 **Qwen3-VL-Flash** 视觉语言模型调用模块。

## 文件结构

```
vlm/
├── config.py       # API Key 及模型配置
├── vlm_client.py   # 核心客户端
├── __init__.py     # 包入口，提供模块级快捷调用
└── README.md       # 本文档
```

## 依赖安装

```bash
pip install openai
```

## 快速开始

### 方式一：模块级快捷调用（最简单）

```python
import sys
sys.path.insert(0, "D:/DeepLearning/vln")  # 根据实际路径调整

import vlm

# 纯文本
result = vlm.chat("用一句话解释视觉导航是什么")
print(result)

# 单张图片（URL）
result = vlm.chat("描述这张图片", images="https://example.com/image.jpg")
print(result)

# 单张图片（本地路径）
result = vlm.chat("这张图里有什么？", images="./my_image.png")
print(result)
```

### 方式二：实例化 VLMClient

```python
from vlm import VLMClient

vlm_client = VLMClient()

# 单图问答
answer = vlm_client.chat("图中的场景是室内还是室外？", images="./scene.jpg")
print(answer)

# 多图对比
answer = vlm_client.chat(
    "这两张图有什么区别？",
    images=["./before.jpg", "./after.jpg"]
)
print(answer)

# 带系统提示词
answer = vlm_client.chat(
    "请分析图中的导航路径",
    images="./nav_map.png",
    system="你是一个室内视觉导航专家，请从导航角度分析图像。"
)
print(answer)

# 流式输出（实时打印）
vlm_client.chat("详细描述这张图", images="./image.jpg", stream=True)
```

### 方式三：快捷方法

```python
from vlm import VLMClient

vlm_client = VLMClient()

# 描述单张图片
desc = vlm_client.describe_image("./scene.jpg")

# 对比多张图片
comparison = vlm_client.compare_images(["./img1.jpg", "./img2.jpg"])

# 分析文档/表格
analysis = vlm_client.analyze_document("./document.png", "提取表格中的数据")
```

## 参数说明

### `VLMClient.__init__`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | str | config 中的值 | 阿里云 API Key |
| `base_url` | str | 百炼北京节点 | API 端点 |
| `model` | str | `qwen3-vl-flash` | 模型名称 |

### `VLMClient.chat`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | 必填 | 用户问题/指令 |
| `images` | str / list / None | None | 图片路径或 URL，单张传字符串，多张传列表 |
| `system` | str | None | 系统提示词 |
| `max_tokens` | int | 2048 | 最大输出 token 数 |
| `temperature` | float | 0.7 | 采样温度（0~1） |
| `stream` | bool | False | 是否流式输出 |

## 切换模型

```python
from vlm import VLMClient

# 使用更强的 Plus 模型
client = VLMClient(model="qwen3-vl-plus")

# 查看可用模型
from vlm import AVAILABLE_MODELS
for name, desc in AVAILABLE_MODELS.items():
    print(f"{name}: {desc}")
```

## 图片输入格式

| 格式 | 示例 |
|------|------|
| HTTP URL | `"https://example.com/image.jpg"` |
| HTTPS URL | `"https://example.com/image.png"` |
| 本地绝对路径 | `"D:/images/scene.jpg"` |
| 本地相对路径 | `"./data/image.png"` |
| 多图列表 | `["./img1.jpg", "https://example.com/img2.png"]` |

支持格式：`.jpg` `.jpeg` `.png` `.gif` `.webp` `.bmp`

## 配置说明

编辑 `vlm/config.py` 可修改默认参数：

```python
API_KEY = "sk-d95ce20c2bcc475a8eb4054bd183307d"   # API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 端点
DEFAULT_MODEL = "qwen3-vl-flash"                  # 默认模型
DEFAULT_MAX_TOKENS = 2048                          # 默认最大输出
DEFAULT_TEMPERATURE = 0.7                          # 默认温度
```

> **注意**：API Key 以明文存储在 `config.py` 中，请勿将此文件提交到公开代码仓库。
