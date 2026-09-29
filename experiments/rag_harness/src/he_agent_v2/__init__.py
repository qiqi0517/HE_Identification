"""隔离的固定参考原图 + Harness 实验；只读复用主项目模型与概率评估。"""

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src"))
