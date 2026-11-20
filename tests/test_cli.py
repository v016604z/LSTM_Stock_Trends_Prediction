import builtins
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from lstm_comparison.cli import main


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "research.json"


class CliTests(unittest.TestCase):
    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(arguments)
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_help_and_validation_need_no_research_imports_or_network(self):
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in {"tensorflow", "keras", "numpy", "pandas", "yfinance", "sklearn", "matplotlib"}:
                raise AssertionError(f"Unexpected research import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked_import), patch.object(
            socket, "socket", side_effect=AssertionError("Unexpected network access")
        ):
            for command in ([], ["validate-config"], ["download"], ["load-snapshot"], ["run"], ["report"], ["evaluate"]):
                with self.subTest(command=command):
                    code, output, _ = self.invoke(command + ["--help"])
                    self.assertEqual(code, 0)
                    self.assertIn("usage:", output)
            code, output, error = self.invoke(["validate-config", "--config", str(CONFIG)])
        self.assertEqual(code, 0)
        self.assertIn("設定檢查通過", output)
        self.assertIn(str(CONFIG.parent.parent / "res" / "research").replace("\\", "\\\\"), output)
        self.assertEqual(error, "")

    def test_invalid_config_returns_error_without_traceback(self):
        code, output, error = self.invoke(["validate-config", "--config", str(CONFIG.parent / "missing.json")])
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("設定錯誤", error)
        self.assertNotIn("Traceback", error)

    def test_missing_report_input_is_not_successful(self):
        code, output, error = self.invoke(["report", "--run-dir", "does-not-exist"])
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("報告錯誤", error)
        self.assertNotIn("Traceback", error)

    def test_missing_required_arguments_fail(self):
        for command in ([], ["validate-config"], ["download"], ["load-snapshot"], ["run"], ["report"], ["evaluate"]):
            with self.subTest(command=command):
                code, _, error = self.invoke(command)
                self.assertEqual(code, 2)
                self.assertIn("required", error)

    def test_download_and_offline_load_validate_configuration(self):
        for command in ("download", "load-snapshot", "run"):
            with self.subTest(command=command):
                code, _, error = self.invoke([command, "--config", str(CONFIG.parent / "missing.json")])
                self.assertEqual(code, 2)
                self.assertIn("設定錯誤", error)

    def test_snapshot_errors_do_not_report_success(self):
        from lstm_comparison.data import SnapshotError

        for command, operation in (("download", "download_snapshot"), ("load-snapshot", "load_snapshot")):
            with self.subTest(command=command), patch(
                f"lstm_comparison.data.{operation}", side_effect=SnapshotError("invalid snapshot")
            ):
                code, output, error = self.invoke([command, "--config", str(CONFIG)])
                self.assertEqual(code, 1)
                self.assertEqual(output, "")
                self.assertIn("快照錯誤", error)

    def test_run_dispatches_only_training_stage_and_reports_directory(self):
        with patch("lstm_comparison.training.run_experiment", return_value=Path("offline-run")) as run:
            code, output, error = self.invoke(["run", "--config", str(CONFIG)])
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertIn("訓練階段完成", output)
        self.assertIn("offline-run", output)
        self.assertIn("後續階段", output)
        self.assertEqual(run.call_args.args[0]["training"]["seeds"], [42, 43, 44])

    def test_run_snapshot_dataset_and_training_errors_return_one(self):
        from lstm_comparison.data import SnapshotError
        from lstm_comparison.dataset import DatasetError
        from lstm_comparison.training import TrainingError

        for error in (SnapshotError("missing snapshot"), DatasetError("no training samples"), TrainingError("fit failed")):
            with self.subTest(error=type(error).__name__), patch(
                "lstm_comparison.training.run_experiment", side_effect=error
            ):
                code, output, message = self.invoke(["run", "--config", str(CONFIG)])
                self.assertEqual(code, 1)
                self.assertEqual(output, "")
                self.assertIn("訓練錯誤", message)
                self.assertNotIn("Traceback", message)

    def test_evaluate_dispatch_and_failure_code(self):
        with patch("lstm_comparison.evaluation.evaluate_experiment", return_value=Path("run/evaluation")) as evaluate:
            code, output, error = self.invoke(["evaluate", "--run-dir", "run"])
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertIn("測試集評估完成", output)
        evaluate.assert_called_once_with("run")
        from lstm_comparison.evaluation import EvaluationError

        with patch("lstm_comparison.evaluation.evaluate_experiment", side_effect=EvaluationError("bad artifact")):
            code, output, error = self.invoke(["evaluate", "--run-dir", "run"])
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("評估錯誤", error)
        self.assertNotIn("Traceback", error)

    def test_report_dispatches_saved_artifacts_and_reports_only_generation(self):
        with patch("lstm_comparison.report.generate_report", return_value=Path("run/report")) as report:
            code, output, error = self.invoke(["report", "--run-dir", "run"])
        report.assert_called_once_with("run")
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertIn("圖表與實驗報告已產生", output)
        self.assertIn("不代表 Commit 8", output)


if __name__ == "__main__":
    unittest.main()
