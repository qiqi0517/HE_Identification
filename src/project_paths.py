"""集中解析可迁移的资源路径；导入模块时不读配置、数据或密钥。

load_paths 读取 configs/paths.json；ProjectPaths 保存解析结果和版本指纹。
配置里的相对路径以项目根目录为基准，命令行显式路径仍遵循调用目录语义。
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs/paths.json"
# 这里只定义字段契约；具体目录仅在 JSON 中维护，避免两套默认值漂移。
PATH_KEYS = frozenset(
    {
        "dataset",
        "default_split",
        "runs",
        "threshold_sweeps",
        "baseline_prompt",
        "medical_prompt",
        "medical_sources",
        "review_prompt",
        "api_keys",
        "base_split",
        "split_output",
        "bioengineering_source",
        "bioengineering_extracted",
        "processed_he",
        "processed_bio",
        "web_source",
        "processed_web",
        "web_staging",
    }
)


@dataclass(frozen=True)
class ProjectPaths:
    """保存一次读取的路径配置，避免同次推理的提示词与数据使用不同默认值。"""

    source: Path
    sha256: str
    values: dict[str, Path | None]

    def __getitem__(self, key):
        """按已校验的字段名取得绝对路径；只有 default_split 允许为空。"""
        return self.values[key]

    def metadata(self):
        """返回可保存的配置指纹与解析路径；只包含位置，不读取资源或密钥内容。"""
        return {
            "file": str(self.source),
            "sha256": self.sha256,
            "resolved_paths": {
                k: str(v) if v is not None else None for k, v in self.values.items()
            },
        }


def _unique_fields(pairs):
    """拒绝 JSON 重复字段，避免手工改配置时被后面的同名字段悄悄覆盖。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field in path configuration")
        result[key] = value
    return result


def load_paths(config_path=None, *, project=None):
    """加载独立路径配置，并相对项目根目录解析路径。

    配置选择优先级：显式文件参数 > HE_PATHS_CONFIG > 项目 configs/paths.json。
    project 仅供历史数据脚本的 --project 覆盖相对路径基准，不改变配置文件位置。
    不要求资源已存在：输出目录本来就应在运行时新建。
    """
    root = Path(project).expanduser().resolve() if project is not None else PROJECT
    selected = config_path if config_path is not None else os.environ.get("HE_PATHS_CONFIG")
    source = Path(selected).expanduser() if selected else DEFAULT_CONFIG
    # 环境变量指定的相对配置文件名也相对代码项目根目录，不受启动目录影响。
    source = (PROJECT / source).resolve() if not source.is_absolute() else source.resolve()
    try:
        raw = source.read_bytes()
        data = json.loads(raw, object_pairs_hook=_unique_fields)
    except OSError:
        raise ValueError(f"Cannot read path configuration: {source}") from None
    except (ValueError, UnicodeError):
        raise ValueError(f"Invalid JSON in path configuration: {source}") from None
    if not isinstance(data, dict) or set(data) != {"version", "paths"}:
        raise ValueError("Path configuration must contain only version and paths")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Unsupported path configuration version; expected 1")
    entries = data["paths"]
    if not isinstance(entries, dict) or set(entries) != PATH_KEYS:
        raise ValueError(
            "Path configuration fields must match configs/paths.json (missing or unknown field)"
        )
    resolved = {}
    for name, value in entries.items():
        if name == "default_split" and value is None:
            resolved[name] = None
            continue
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"Path configuration field {name} must be a non-empty path string")
        path = Path(value).expanduser()
        resolved[name] = (root / path).resolve() if not path.is_absolute() else path.resolve()
    return ProjectPaths(source, hashlib.sha256(raw).hexdigest(), resolved)


def display_path(path, project):
    """溯源路径优先写项目内相对路径；外置数据源保留绝对位置，避免 relative_to 报错。"""
    path, root = Path(path).resolve(), Path(project).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
