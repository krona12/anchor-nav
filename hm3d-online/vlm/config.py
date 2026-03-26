# VLM 配置文件
# 阿里云百炼 API 配置

API_KEY = "sk-d95ce20c2bcc475a8eb4054bd183307d"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 默认模型
DEFAULT_MODEL = "qwen3-vl-flash"

# 可用模型列表
AVAILABLE_MODELS = {
    "qwen3-vl-flash": "Qwen3-VL Flash - 速度快、成本低，适合响应速度敏感场景",
    "qwen3-vl-plus": "Qwen3-VL Plus - 性能最强",
    "qwen3.5-flash": "Qwen3.5 Flash - 高性价比视觉理解",
    "qwen3.5-plus": "Qwen3.5 Plus - 最新一代视觉理解最强模型",
}

# 请求默认参数
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
