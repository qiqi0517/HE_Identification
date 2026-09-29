"""验证资源搬迁、默认路径覆盖及旧划分保护；所有写入均在临时目录，无 API 请求。"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import test_experiment_matrix as matrix

import he_agent
import project_paths
from models import credentials
from prompt_profiles import load_prompt


class PathConfigTests(unittest.TestCase):
    def write_config(self, root, **overrides):
        """从当前配置复制完整字段，覆盖到隔离资源，避免测试随真实路径变化而改写数据。"""
        data = json.loads(project_paths.DEFAULT_CONFIG.read_text())
        data["paths"].update(overrides)
        path = root / "paths.json"
        path.write_text(json.dumps(data))
        return path

    def test_resolution_ignores_working_directory_and_config_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cfg = self.write_config(
                root,
                dataset="other data",
                runs=str(root / "external"),
                api_keys="~/he-test-key.json",
            )
            with patch("pathlib.Path.cwd", side_effect=AssertionError("CWD-dependent path")):
                paths = project_paths.load_paths(cfg, project=root / "project")
            self.assertEqual(paths["dataset"], root / "project/other data")
            self.assertEqual(paths["runs"], root / "external")
            self.assertEqual(paths["api_keys"], Path.home() / "he-test-key.json")
            self.assertIsNone(paths["default_split"])
            self.assertEqual(paths.sha256, hashlib.sha256(cfg.read_bytes()).hexdigest())
            self.assertFalse((root / "external").exists())

    def test_explicit_config_overrides_environment_and_changes_are_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.write_config(root, dataset="first")
            with patch.dict(os.environ, {"HE_PATHS_CONFIG": str(root / "missing.json")}):
                self.assertEqual(project_paths.load_paths(cfg)["dataset"].name, "first")
            with patch.dict(os.environ, {"HE_PATHS_CONFIG": str(cfg)}):
                self.assertEqual(project_paths.load_paths()["dataset"].name, "first")
                self.write_config(root, dataset="second")
                self.assertEqual(project_paths.load_paths()["dataset"].name, "second")

    def test_bad_config_fails_before_resource_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.write_config(root)
            good = json.loads(cfg.read_text())
            invalid = ["not json", '{"version":1,"version":1,"paths":{}}', "[]"]
            for key, value in (("version", True), ("version", 9), ("paths", [])):
                invalid.append(json.dumps({**good, key: value}))
            for field, value in (
                ("dataset", None),
                ("dataset", " "),
                ("runs", 3),
                ("runs", "a\x00b"),
                ("typo", "folder"),
            ):
                invalid.append(json.dumps({**good, "paths": {**good["paths"], field: value}}))
            for raw in invalid:
                with self.subTest(raw=raw[:50]):
                    cfg.write_text(raw)
                    with self.assertRaises(ValueError):
                        project_paths.load_paths(cfg)
            with self.assertRaisesRegex(ValueError, "Cannot read path configuration"):
                project_paths.load_paths(root / "absent.json")

    def test_relocated_prompts_keep_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = project_paths.load_paths()
            overrides = {}
            for key in ("medical_prompt", "baseline_prompt", "medical_sources"):
                dest = root / (key + ".renamed")
                shutil.copy2(original[key], dest)
                overrides[key] = str(dest)
            paths = project_paths.load_paths(self.write_config(root, **overrides))
            text, meta, evidence = load_prompt("knowledge", paths=paths)
            self.assertEqual(text, original["medical_prompt"].read_text())
            self.assertEqual(meta["prompt_file_sha256"], evidence["prompt_sha256"])
            paths["medical_prompt"].write_text(text + "tamper")
            with self.assertRaisesRegex(ValueError, "Frozen medical prompt changed"):
                load_prompt("knowledge", paths=paths)

    def test_relocated_key_file_and_environment_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = root / "fake-keys.json"
            keys.write_text('{"ZHIPU_API_KEY":"fake-local-test-key"}')
            cfg = self.write_config(root, api_keys=str(keys))
            with patch.dict(os.environ, {"HE_PATHS_CONFIG": str(cfg), "ZHIPU_API_KEY": ""}):
                self.assertEqual(credentials.read_api_key("glm"), "fake-local-test-key")
                with patch.dict(os.environ, {"ZHIPU_API_KEY": "fake-env-test-key"}):
                    with patch.object(credentials, "load_local_keys", side_effect=AssertionError):
                        self.assertEqual(credentials.read_api_key("glm"), "fake-env-test-key")
                self.assertNotIn(
                    "fake-local-test-key", json.dumps(project_paths.load_paths().metadata())
                )

    def test_cli_outside_project_uses_config_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dataset = root / "external dataset"
            dataset.mkdir()
            _, split = matrix.MatrixTests().fixture(dataset)
            review = root / "review.txt"
            review.write_text("RELOCATED_REVIEW_SENTINEL")
            cfg = self.write_config(
                root,
                dataset=str(dataset),
                default_split=str(split),
                runs=str(root / "configured-runs"),
                review_prompt=str(review),
                api_keys=str(root / "missing-key.json"),
            )
            env = {**os.environ, "HE_PATHS_CONFIG": str(cfg), "PYTHONDONTWRITEBYTECODE": "1"}
            command = [
                sys.executable,
                str(project_paths.PROJECT / "src/he_agent.py"),
                "--mode",
                "reference_knowledge",
                "--review",
                "--limit",
                "0",
                "--dry-run",
            ]
            result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = next((root / "configured-runs").iterdir())
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["selected"], 2)
            self.assertEqual(run["dataset"], str(dataset))
            self.assertEqual(run["path_config"]["file"], str(cfg))
            self.assertIn("RELOCATED_REVIEW_SENTINEL", (output / "review_prompt.txt").read_text())
            self.assertEqual(
                json.loads((output / "validation.json").read_text())["network_calls"], 0
            )
            self.assertFalse((root / "missing-key.json").exists())
            # 配置指向不存在的数据/划分；显式 CLI 覆盖仍可使用有效资源。
            self.write_config(root, dataset="missing-dataset", default_split="missing-split")
            explicit_output = root / "explicit-output"
            result = subprocess.run(
                command
                + [
                    "--dataset",
                    str(dataset),
                    "--split",
                    str(split),
                    "--output",
                    str(explicit_output),
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((explicit_output / "run.json").is_file())

    def test_default_split_null_preserves_baseline_and_split_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dataset = root / "dataset"
            dataset.mkdir()
            _, split = matrix.MatrixTests().fixture(dataset)
            cfg = self.write_config(
                root, dataset=str(dataset), default_split=None, runs=str(root / "runs")
            )
            args = ["he_agent.py", "--dry-run", "--limit", "0"]
            with patch.dict(os.environ, {"HE_PATHS_CONFIG": str(cfg)}):
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(he_agent.main(), 0)
                output = next((root / "runs").iterdir())
                self.assertEqual(json.loads((output / "run.json").read_text())["selected"], 6)
                broken = json.loads(split.read_text())
                broken["samples"][broken["test"][0]]["sha256"] = "0" * 64
                split.write_text(json.dumps(broken))
                self.write_config(root, dataset=str(dataset), default_split=str(split))
                with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exc:
                        he_agent.main()
                    self.assertEqual(exc.exception.code, 2)

    def test_external_provenance_does_not_require_source_inside_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(
                project_paths.display_path(root / "images/a.png", root), "images/a.png"
            )
            external = root.parent / "another-source/image.png"
            self.assertEqual(project_paths.display_path(external, root), str(external))


if __name__ == "__main__":
    unittest.main()
