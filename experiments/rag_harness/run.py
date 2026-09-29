"""Agent 2.0 独立入口：不改变主项目 Agent 1.x 的运行方式。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from he_agent_v2.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
