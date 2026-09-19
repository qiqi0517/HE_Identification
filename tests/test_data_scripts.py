"""确保数据脚本导入时不会解析命令行或改动数据。"""

import argparse
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


class DataScriptImportTests(unittest.TestCase):
    def test_import_only_defines_helpers_and_main(self):
        for name in (
            "merge_normal_labels.py",
            "merge_processed_uuid.py",
            "prepare_web_us_filtered.py",
        ):
            with (
                self.subTest(script=name),
                (
                    patch.object(
                        argparse.ArgumentParser,
                        "parse_args",
                        side_effect=AssertionError("CLI executed on import"),
                    )
                ),
                patch.object(
                    Path, "mkdir", side_effect=AssertionError("Directory created on import")
                ),
                patch.object(
                    Path, "write_text", side_effect=AssertionError("File written on import")
                ),
                patch("shutil.copy2", side_effect=AssertionError("File copied on import")),
                patch("shutil.rmtree", side_effect=AssertionError("Directory deleted on import")),
            ):
                namespace = runpy.run_path(str(PROJECT / "scripts/data" / name))
                self.assertTrue(callable(namespace["main"]))


if __name__ == "__main__":
    unittest.main()
