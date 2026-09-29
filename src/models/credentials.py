"""按环境变量→项目本地文件的顺序读取密钥，不修改环境或记录密钥。

read_api_key 决定优先级；load_local_keys 校验配置格式。
不在模块导入时读取密钥，使离线测试和 dry-run 能在无凭据环境运行。
"""

import json
import os

from project_paths import load_paths

# 保留可注入位置用于测试；实际运行延迟读取路径配置，导入时不访问密钥。
LOCAL_KEYS_PATH = None
KEY_NAMES = {"glm": "ZHIPU_API_KEY", "openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


def load_local_keys():
    """只解析本地JSON；错误消息不包含文件内容、字段值或原始异常。"""

    def unique_fields(pairs):
        """拒绝同名密钥字段，防止 JSON 后值静默覆盖前值导致鉴权来源不明确。"""
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    try:
        path = LOCAL_KEYS_PATH if LOCAL_KEYS_PATH is not None else load_paths()["api_keys"]
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        raise ValueError("Cannot read local API key file; check access and UTF-8") from None
    try:
        data = json.loads(raw, object_pairs_hook=unique_fields)
        if not isinstance(data, dict):
            raise ValueError("Expected object")
    except (ValueError, TypeError):
        raise ValueError("Invalid local API key file; use a JSON object with unique keys") from None
    return data


def read_api_key(provider):
    """非空环境变量优先；文件位置由路径配置指定，不受启动目录影响。"""
    if provider not in KEY_NAMES:
        raise ValueError("Unknown credential provider")
    name = KEY_NAMES[provider]
    key = os.environ.get(name, "").strip()
    if not key:
        key = load_local_keys().get(name, "")
        if not isinstance(key, str):
            raise ValueError(f"{name} must be a string in local API key file")
        key = key.strip()
    if not key:
        raise ValueError(f"Set {name} in local API key file or the environment")
    # 防止无效头字段把密钥带入底层HTTP异常文本；不对服务商密钥前缀做假设。
    if any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError(
            f"Invalid {name} format; remove internal whitespace or non-ASCII characters"
        )
    return key
