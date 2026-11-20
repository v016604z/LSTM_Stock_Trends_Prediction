import base64
import builtins
import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from lstm_comparison.metrics import calculate_metrics, error_improvement_pct, summarize_metrics
from lstm_comparison.report import METRIC_COLUMNS, ReportError, _load_bundle, generate_report
import test_evaluation as fixture_module


class ReportTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.EvaluationTests("test_nine_predictions_naive_and_metrics_are_independently_recomputable")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.evaluate()
        self.run_dir, self.root = fixture.run_dir, fixture.root

    def json(self, relative):
        return json.loads((self.run_dir / relative).read_text(encoding="utf-8"))

    def write_json(self, relative, value):
        (self.run_dir / relative).write_text(json.dumps(value), encoding="utf-8")

    def rows(self, relative):
        with (self.run_dir / relative).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def write_rows(self, relative, rows):
        with (self.run_dir / relative).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: "N/A" if value is None else value for key, value in row.items()} for row in rows)

    @staticmethod
    def fake_figures(bundle, output_dir):
        output_dir.mkdir()
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=")
        paths = {}
        for key in ("prices", "errors", "direction", "loss"):
            paths[key] = output_dir / f"{key}.png"
            paths[key].write_bytes(png)
        return paths

    def generate(self):
        with patch("lstm_comparison.plots.render_figures", side_effect=self.fake_figures):
            return generate_report(self.run_dir)

    def replace_errors(self, errors, *, zero_baseline=False):
        manifest = self.json("evaluation/evaluation.json")
        metrics = []
        naive = None
        files = ["predictions_naive.csv", *[f"predictions_{name}_seed_{seed}.csv" for name in ("A", "B", "C") for seed in (42, 43, 44)]]
        for filename in files:
            relative = f"evaluation/{filename}"
            rows = self.rows(relative)
            group = rows[0]["group"]
            seed = None if group == "Naive" else int(rows[0]["seed"])
            if zero_baseline:
                # A coherent constant-price test series has a zero-error Naive.
                for row in rows:
                    row["actual_close"] = row["previous_close"] = "100.0"
            actual = np.array([float(row["actual_close"]) for row in rows])
            previous = np.array([float(row["previous_close"]) for row in rows])
            prediction = previous if group == "Naive" else actual + errors[group, seed]
            result = calculate_metrics(actual, prediction, previous, naive=group == "Naive")
            if group == "Naive":
                naive = result
            for i, row in enumerate(rows):
                row.update(predicted_close=float(prediction[i]), absolute_error=float(abs(prediction[i] - actual[i])),
                           squared_error=float((prediction[i] - actual[i]) ** 2),
                           actual_direction=int(np.sign(actual[i] - previous[i])),
                           predicted_direction=int(np.sign(prediction[i] - previous[i])),
                           direction_correct=None if group == "Naive" else int(np.sign(actual[i] - previous[i]) == np.sign(prediction[i] - previous[i])))
            self.write_rows(relative, rows)
            if group != "Naive":
                metrics.append({"group": group, "seed": seed, "n_samples": len(rows), **result,
                                "mae_improvement_pct": error_improvement_pct(result["mae"], naive["mae"]),
                                "rmse_improvement_pct": error_improvement_pct(result["rmse"], naive["rmse"])})
        summary = summarize_metrics(metrics, naive)
        metrics.append({"group": "Naive", "seed": None, "n_samples": manifest["n_samples"], **naive,
                        "mae_improvement_pct": None, "rmse_improvement_pct": None})
        self.write_rows("evaluation/metrics_per_seed.csv", metrics)
        self.write_rows("evaluation/metrics_summary.csv", summary)
        manifest.update(metrics_per_seed=metrics, metrics_summary=summary)
        self.write_json("evaluation/evaluation.json", manifest)

    def test_saved_only_report_preserves_sources_and_records_hashes(self):
        # The original data, weights and pickle locations no longer exist.
        for suffix in (".csv", ".metadata.json", ".sessions.csv"):
            path = self.root / f"prices{suffix}"
            path.rename(path.with_name(path.name + ".archived"))
        (self.run_dir / "scalers").rename(self.run_dir / "scalers.archived")
        for path in self.run_dir.glob("*/seed_*/best.weights.h5"):
            path.rename(path.with_name("weights.archived"))
        original = {path: sha256(path.read_bytes()).hexdigest() for path in self.run_dir.rglob("*") if path.is_file()}
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in {"tensorflow", "keras", "yfinance"} or name in {
                "lstm_comparison.model", "lstm_comparison.training", "lstm_comparison.evaluation"}:
                raise AssertionError(f"Unexpected inference/training import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("lstm_comparison.plots.render_figures", side_effect=self.fake_figures), patch(
            "builtins.__import__", side_effect=checked_import
        ), patch("pickle.load", side_effect=AssertionError("Unexpected unpickle")), patch(
            "sklearn.preprocessing.MinMaxScaler.fit", side_effect=AssertionError("Unexpected fit")
        ):
            output = generate_report(self.run_dir)
        manifest = self.json("report/report.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["stage"], "report")
        self.assertEqual(manifest["plot_seed"], 42)
        text = (output / "report.md").read_text(encoding="utf-8")
        for phrase in ("本次為短測試", "最大 epochs：1", "seed 42", "Naive", "N/A", "ddof=1", "不自動宣告 Commit 8", "不等同投資報酬"):
            self.assertIn(phrase, text)
        self.assertEqual(text.count("\n## "), 8)
        for relative, digest in manifest["output_sha256"].items():
            self.assertEqual(sha256((output / relative).read_bytes()).hexdigest(), digest)
        for relative, digest in manifest["input_sha256"].items():
            self.assertEqual(sha256((self.run_dir / relative).read_bytes()).hexdigest(), digest)
        for path, digest in original.items():
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest)

    def test_result_text_handles_improvement_regression_and_negative_baseline_improvement(self):
        self.replace_errors({(group, seed): {"A": 4., "B": 1., "C": 2.}[group] for group in ("A", "B", "C") for seed in (42, 43, 44)})
        text = (self.generate() / "report.md").read_text(encoding="utf-8")
        self.assertIn("B 平均 MAE 1 低於 A 的 4", text)
        self.assertIn("C 平均 MAE 2 高於 B 的 1", text)
        self.assertIn("平均 MAE 劣於 Naive", text)
        self.assertIn("3 次改善", text)
        self.assertIn("3 次變差", text)

    def test_result_text_preserves_ties(self):
        self.replace_errors({(group, seed): 1. for group in ("A", "B", "C") for seed in (42, 43, 44)})
        text = (self.generate() / "report.md").read_text(encoding="utf-8")
        self.assertIn("B 平均 MAE 1 等於 A 的 1", text)
        self.assertIn("3 次平手", text)

    def test_result_text_reports_inconsistent_seeds(self):
        errors = {(group, seed): 1. for group in ("A", "B", "C") for seed in (42, 43, 44)}
        errors.update({("B", 42): 0.5, ("B", 43): 1., ("B", 44): 2.})
        self.replace_errors(errors)
        text = (self.generate() / "report.md").read_text(encoding="utf-8")
        self.assertIn("1 次改善、1 次平手、1 次變差", text)
        self.assertIn("不同種子的比較方向不一致", text)

    def test_zero_naive_baseline_keeps_undefined_improvements_as_na(self):
        self.replace_errors({(group, seed): 1. for group in ("A", "B", "C") for seed in (42, 43, 44)}, zero_baseline=True)
        bundle = _load_bundle(self.run_dir)
        self.assertTrue(all(row["mae_improvement_pct"] is None for row in bundle["summary"]))
        self.assertIn("N/A", (self.generate() / "report.md").read_text(encoding="utf-8"))

    def test_single_seed_keeps_sample_standard_deviations_as_na(self):
        config = self.json("config.json")
        config["training"]["seeds"] = [42]
        self.write_json("config.json", config)
        digest = sha256((self.run_dir / "config.json").read_bytes()).hexdigest()
        training = self.json("run.json")
        training.update(seeds=[42], expected_runs=3, runs=[row for row in training["runs"] if row["seed"] == 42])
        self.write_json("run.json", training)
        evaluation = self.json("evaluation/evaluation.json")
        rows = [row for row in evaluation["metrics_per_seed"] if row["seed"] in (42, None)]
        baseline = next(row for row in rows if row["group"] == "Naive")
        summary = summarize_metrics([row for row in rows if row["group"] != "Naive"], baseline)
        artifacts = {path: value for path, value in evaluation["artifact_sha256"].items()
                     if "/seed_" not in path or "/seed_42/" in path}
        artifacts["config.json"] = digest
        evaluation.update(seeds=[42], config_sha256=digest, artifact_sha256=artifacts,
                          metrics_per_seed=rows, metrics_summary=summary,
                          prediction_files=[name for name in evaluation["prediction_files"]
                                            if name == "predictions_naive.csv" or "seed_42.csv" in name])
        self.write_json("evaluation/evaluation.json", evaluation)
        self.write_rows("evaluation/metrics_per_seed.csv", [{key: row[key] for key in METRIC_COLUMNS} for row in rows])
        self.write_rows("evaluation/metrics_summary.csv", summary)
        text = (self.generate() / "report.md").read_text(encoding="utf-8")
        self.assertIn("std N/A", text)
        self.assertIn("僅有 1 個固定種子", text)

    def test_other_ticker_uses_saved_currency_or_neutral_unit(self):
        config = self.json("config.json")
        config["data"]["ticker"] = "OTHER"
        self.write_json("config.json", config)
        digest = sha256((self.run_dir / "config.json").read_bytes()).hexdigest()
        evaluation = self.json("evaluation/evaluation.json")
        evaluation["config_sha256"] = evaluation["artifact_sha256"]["config.json"] = digest
        self.write_json("evaluation/evaluation.json", evaluation)
        metadata = self.json("snapshot.metadata.json")
        metadata["source"]["parameters"]["tickers"] = "OTHER"
        self.write_json("snapshot.metadata.json", metadata)
        self.assertEqual(_load_bundle(self.run_dir)["price_unit"], "source price unit")
        metadata["currency"] = "JPY"
        self.write_json("snapshot.metadata.json", metadata)
        self.assertEqual(_load_bundle(self.run_dir)["price_unit"], "JPY")

    def test_tampered_manifest_table_history_and_seed_selection_fail_before_output(self):
        cases = [
            ("evaluation/evaluation.json", lambda value: value.update(status="failed")),
            ("evaluation/evaluation.json", lambda value: value.update(config_sha256="0" * 64)),
            ("evaluation/evaluation.json", lambda value: value["metrics_summary"][0].update(mae_mean=999)),
            ("evaluation/evaluation.json", lambda value: value["artifact_sha256"].pop("scalers/y.pkl")),
            ("run.json", lambda value: value["runs"][0].update(best_val_loss=999)),
            ("A/seed_42/training.json", lambda value: value.update(best_epoch=2)),
        ]
        for relative, change in cases:
            path = self.run_dir / relative
            before = path.read_bytes()
            try:
                value = self.json(relative)
                change(value)
                self.write_json(relative, value)
                with self.subTest(relative=relative), self.assertRaises(ReportError):
                    self.generate()
                self.assertFalse((self.run_dir / "report").exists())
            finally:
                path.write_bytes(before)
        path = self.run_dir / "config.json"
        value = self.json("config.json")
        value["evaluation"]["plot_seed"] = 43
        self.write_json("config.json", value)
        with self.assertRaisesRegex(ReportError, "seed 42"):
            self.generate()
        self.assertFalse((self.run_dir / "report").exists())

    def test_missing_history_and_metrics_tampering_fail_before_output(self):
        path = self.run_dir / "A/seed_42/history.csv"
        saved = path.with_suffix(".archived")
        path.rename(saved)
        try:
            with self.assertRaises(ReportError):
                self.generate()
            self.assertFalse((self.run_dir / "report").exists())
        finally:
            saved.rename(path)
        rows = self.rows("evaluation/metrics_per_seed.csv")
        rows[0]["mae"] = "999"
        self.write_rows("evaluation/metrics_per_seed.csv", rows)
        with self.assertRaises(ReportError):
            self.generate()
        self.assertFalse((self.run_dir / "report").exists())

    def test_consistently_stale_input_dates_are_rejected(self):
        manifest = self.json("evaluation/evaluation.json")
        for filename in manifest["prediction_files"]:
            rows = self.rows(f"evaluation/{filename}")
            old = rows[0]["input_end_date"]
            for row in rows:
                row["input_end_date"] = old
            self.write_rows(f"evaluation/{filename}", rows)
        with self.assertRaisesRegex(ReportError, "輸入結束日"):
            self.generate()
        self.assertFalse((self.run_dir / "report").exists())

    def test_invalid_history_epochs_and_losses_fail_before_output(self):
        relative = "A/seed_42/history.csv"
        path = self.run_dir / relative
        original = path.read_bytes()
        for column, value in (("epoch", "0"), ("loss", "-0.1"), ("val_loss", "NaN")):
            try:
                rows = self.rows(relative)
                rows[0][column] = value
                self.write_rows(relative, rows)
                with self.subTest(column=column, value=value), self.assertRaises(ReportError):
                    self.generate()
                self.assertFalse((self.run_dir / "report").exists())
            finally:
                path.write_bytes(original)

    def test_invalid_adjacent_previous_price_is_rejected(self):
        relative = "evaluation/predictions_naive.csv"
        rows = self.rows(relative)
        rows[1]["previous_close"] = str(float(rows[0]["actual_close"]) + 10)
        self.write_rows(relative, rows)
        with self.assertRaisesRegex(ReportError, "相鄰交易日"):
            self.generate()
        self.assertFalse((self.run_dir / "report").exists())

    def test_existing_report_and_plotting_failure_preserve_prior_stages(self):
        original = {relative: (self.run_dir / relative).read_bytes() for relative in ("run.json", "evaluation/evaluation.json")}
        with patch("lstm_comparison.plots.render_figures", side_effect=OSError("controlled plot failure")), self.assertRaises(ReportError):
            generate_report(self.run_dir)
        manifest = self.json("report/report.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertNotIn("completed_at_utc", manifest)
        before = (self.run_dir / "report/report.json").read_bytes()
        with self.assertRaisesRegex(ReportError, "不覆寫"):
            self.generate()
        self.assertEqual((self.run_dir / "report/report.json").read_bytes(), before)
        for relative, expected in original.items():
            self.assertEqual((self.run_dir / relative).read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
