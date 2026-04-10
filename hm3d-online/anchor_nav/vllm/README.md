# vLLM API 调用模块

本目录提供针对你当前部署参数的本地客户端封装：

- 服务地址：`http://127.0.0.1:8000/v1`
- 模型名：`Qwen2.5-VL-32B-Instruct`
- 模块文件：`qwen_vllm_api.py`

## 为什么不命名成 `vllm.api_client`？

因为环境里已经安装了 pip 包 `vllm`。  
若在本仓库再做同名 Python 包，容易与官方包冲突。

因此建议按下面方式引用本地模块：

```python
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]   # 按实际层级调整
sys.path.insert(0, str(repo_root / "vllm"))

from qwen_vllm_api import VLLMClientConfig, VLLMOpenAIClient
```

## 快速示例

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm"))
from qwen_vllm_api import VLLMClientConfig, VLLMOpenAIClient

client = VLLMOpenAIClient(
    VLLMClientConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="Qwen2.5-VL-32B-Instruct",
        api_key="EMPTY",
    )
)

print(client.chat_text("你好"))
print(client.chat_with_image_file("test-cases/images/f5.png", "图片的答案是什么？"))
```
