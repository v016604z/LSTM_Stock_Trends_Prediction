import builtins
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))

from matplotlib import image as mpimg
import numpy as np

import test_evaluation as evaluation_fixture
from lstm_comparison.report import generate_report


class ReportIntegrationTests(unittest.TestCase):
    @staticmethod
    def read_csv(path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_saved_evaluation_generates_real_report_without_source_models_or_pickle(self):
        # Reuse the saved-artifact fixture, not its TestCase class as a module
        # alias: unittest must discover these integration tests only once.
        fixture = evaluation_fixture.EvaluationTests(
            "test_nine_predictions_naive_and_metrics_are_independently_recomputable"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        evaluated = fixture.evaluate()
        run_dir = fixture.run_dir
        per_seed = self.read_csv(evaluated / "metrics_per_seed.csv")
        summaries = self.read_csv(evaluated / "metrics_summary.csv")
        evaluation_manifest = json.loads((evaluated / "evaluation.json").read_text(encoding="utf-8"))

        unavailable = [fixture.root / f"prices{suffix}" for suffix in (".csv", ".metadata.json", ".sessions.csv")]
        unavailable += list(run_dir.glob("scalers/*.pkl"))
        unavailable += list(run_dir.glob("*/seed_*/best.weights.h5"))
        for path in unavailable:
            path.unlink()
            self.assertFalse(path.exists())
        source_hashes = {
            path.relative_to(run_dir).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in run_dir.rglob("*") if path.is_file()
        }
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in {"tensorflow", "keras", "h5py"} or name in {
                "lstm_comparison.model", "lstm_comparison.training", "lstm_comparison.evaluation",
                "lstm_comparison.dataset", "lstm_comparison.data",
            }:
                raise AssertionError(f"Report attempted model or raw data import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked_import), patch(
            "pickle.load", side_effect=AssertionError("Unexpected pickle load")
        ), patch("pickle.loads", side_effect=AssertionError("Unexpected pickle loads")), patch(
            "sklearn.preprocessing.MinMaxScaler.fit", side_effect=AssertionError("Unexpected fit")
        ), patch("lstm_comparison.model.build_model", side_effect=AssertionError("Unexpected model")), patch(
            "lstm_comparison.data.load_snapshot", side_effect=AssertionError("Unexpected source read")
        ), patch("yfinance.download", side_effect=AssertionError("Unexpected download")), patch(
            "socket.socket", side_effect=AssertionError("Unexpected network")
        ):
            output = generate_report(run_dir)

        self.assertEqual(output, run_dir / "report")
        for relative, digest in source_hashes.items():
            self.assertEqual(sha256((run_dir / relative).read_bytes()).hexdigest(), digest)
        self.assertTrue(all(not path.exists() for path in unavailable))
        manifest = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["stage"], "report")
        self.assertEqual(manifest["plot_seed"], 42)
        self.assertIn("completed_at_utc", manifest)
        self.assertIn("matplotlib", manifest["environment"]["packages"])
        self.assertEqual(manifest["prediction_source_sha256"], evaluation_manifest["artifact_sha256"])
        required_inputs = {
            "config.json", "run.json", "snapshot.metadata.json", "environment.json",
            "evaluation/evaluation.json", "evaluation/environment.json",
            "evaluation/metrics_per_seed.csv", "evaluation/metrics_summary.csv",
            *(f"evaluation/{name}" for name in evaluation_manifest["prediction_files"]),
        }
        for group in ("A", "B", "C"):
            for seed in (42, 43, 44):
                required_inputs.update({f"{group}/seed_{seed}/training.json", f"{group}/seed_{seed}/history.csv"})
        self.assertEqual(set(manifest["input_sha256"]), required_inputs)
        for relative, digest in manifest["input_sha256"].items():
            self.assertEqual(digest, source_hashes[relative])

        markdown = (output / "report.md").read_text(encoding="utf-8")
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
        self.assertEqual(set(images), {f"figures/{key}.png" for key in ("prices", "errors", "direction", "loss")})
        for relative in images:
            path = output / relative
            self.assertFalse(Path(relative).is_absolute())
            self.assertTrue(path.resolve().is_relative_to(output.resolve()))
            pixels = mpimg.imread(path)
            self.assertGreater(pixels.shape[0], 500)
            self.assertGreater(pixels.shape[1], 800)
            self.assertTrue(np.isfinite(pixels).all())
        self.assertEqual(set(manifest["output_sha256"]), {"report.md", *images})
        for relative, digest in manifest["output_sha256"].items():
            self.assertEqual(sha256((output / relative).read_bytes()).hexdigest(), digest)

        for phrase in ("資料與研究方法", "研究限制", "上漲、下跌、持平", "Naive", "seed 42",
                       "ddof=1", "調整後價格（TWD）", "單一股票", "不代表統計顯著", "不能視為正式研究結果"):
            self.assertIn(phrase, markdown)
        for digest in (evaluation_manifest["data_sha256"], evaluation_manifest["calendar_sha256"],
                       evaluation_manifest["config_sha256"]):
            self.assertIn(digest, markdown)
        table_rows = [tuple(cell.strip() for cell in line.strip("|").split("|"))
                      for line in markdown.splitlines() if line.startswith("| ")]

        def shown(value, *, percent=False):
            return "N/A" if value == "N/A" else f"{float(value) * (100 if percent else 1):.6g}"

        for row in per_seed:
            expected = (row["group"], row["seed"], shown(row["mae"]), shown(row["rmse"]),
                        shown(row["directional_accuracy"], percent=True),
                        shown(row["mae_improvement_pct"]), shown(row["rmse_improvement_pct"]))
            self.assertIn(expected, table_rows)
        for row in summaries:
            aggregate = []
            for key in ("mae", "rmse", "directional_accuracy"):
                percent = key == "directional_accuracy"
                mean, std = shown(row[f"{key}_mean"], percent=percent), row[f"{key}_std"]
                aggregate.append(mean if mean == "N/A" else f"{mean}（std N/A）" if std == "N/A"
                                 else f"{mean} ± {shown(std, percent=percent)}")
            self.assertIn((row["group"], row["n_seeds"], *aggregate,
                           shown(row["mae_improvement_pct"]), shown(row["rmse_improvement_pct"])), table_rows)


if __name__ == "__main__":
    unittest.main()
