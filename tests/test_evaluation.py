import copy
import csv
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lstm_comparison.config import load_config
from lstm_comparison.data import download_snapshot
from lstm_comparison.dataset import build_dataset
from lstm_comparison.evaluation import EvaluationError, evaluate_experiment
from lstm_comparison import evaluation
from lstm_comparison.training import run_experiment


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        settings = json.loads((ROOT / "configs" / "research.json").read_text(encoding="utf-8"))
        settings["data"].update({"start": "2022-08-01", "end": "2023-02-28", "end_exclusive": "2023-03-01",
                                 "snapshot_path": "prices.csv", "output_dir": "runs"})
        settings["split"] = {"train": {"start": "2022-08-01", "end": "2022-12-31"},
                             "validation": {"start": "2023-01-01", "end": "2023-01-31"},
                             "test": {"start": "2023-02-01", "end": "2023-02-28"}}
        settings["training"]["max_epochs"] = 1
        path = self.root / "config.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        self.config = load_config(path)
        sessions = pd.bdate_range("2022-08-01", "2023-02-28", name="Date").as_unit("ns")
        positions = np.arange(len(sessions))
        close = 100 + positions / 10 + np.sin(positions / 4)
        self.frame = pd.DataFrame({"Open": close - 0.25, "High": close + 1, "Low": close - 1,
                                   "Close": close, "Volume": 1000 + positions}, index=sessions)
        info = {"name": "XTAI", "version": "synthetic-test", "source": "https://example.test/calendar",
                "additional_closures": [], "additional_sessions": [],
                "requested_start": "2022-08-01", "requested_end": "2023-02-28"}
        with patch("yfinance.download", return_value=self.frame), patch("yfinance.set_tz_cache_location"), patch(
            "lstm_comparison.data._calendar_sessions", return_value=(sessions, info)
        ):
            download_snapshot(self.config)
        self.dataset = build_dataset(self.frame, self.config)

        def fake_train(group, config, seed, output_dir):
            output_dir.mkdir(parents=True)
            units, width = config["model"]["lstm_units"], len(group.columns)
            summary = {"stage": "training", "status": "completed", "seed": seed,
                       "columns": list(group.columns), "parameter_count": 4 * units * (units + width + 1) + units + 1,
                       "best_epoch": 1, "epochs_trained": 1, "best_val_loss": 0.2,
                       "weights": "best.weights.h5", "history": "history.csv"}
            (output_dir / "best.weights.h5").write_bytes(b"synthetic weights for mocked predictor")
            (output_dir / "history.csv").write_text("epoch,loss,val_loss\n1,0.1,0.2\n", encoding="utf-8")
            self.write_json(output_dir / "training.json", summary)
            return summary

        with patch("lstm_comparison.training.train_one", side_effect=fake_train):
            self.run_dir = run_experiment(self.config)
        self.calls = []

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def rows(self, path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def predictor(self, shape, config, seed):
        name = {1: "A", 5: "B", 10: "C"}[shape[1]]
        self.calls.append((name, seed))
        reference = self.dataset.groups[name].splits["test"]
        raw = reference.y_original + shape[1] / 100 + (seed - 42) / 10
        predictions = self.dataset.y_scaler.transform(raw)
        test = self

        class Predictor:
            optimizer = object()
            position = 0

            def load_weights(self, path, *, objects_to_skip):
                test.assertEqual(objects_to_skip, [self.optimizer])
                if path.read_bytes() == b"corrupt":
                    raise ValueError("invalid weights")

            def __call__(self, X, *, training):
                test.assertFalse(training)
                values = predictions[self.position:self.position + len(X)]
                self.position += len(X)
                return values

        return Predictor()

    def evaluate(self):
        with patch("lstm_comparison.evaluation.build_model", side_effect=self.predictor), patch(
            "sklearn.preprocessing.MinMaxScaler.fit", side_effect=AssertionError("Unexpected scaler fit")
        ), patch("yfinance.download", side_effect=AssertionError("Unexpected network")):
            return evaluate_experiment(self.run_dir)

    def test_nine_predictions_naive_and_metrics_are_independently_recomputable(self):
        training_manifest = (self.run_dir / "run.json").read_bytes()
        output = self.evaluate()
        self.assertEqual(self.calls, [(name, seed) for name in ("A", "B", "C") for seed in (42, 43, 44)])
        self.assertEqual((self.run_dir / "run.json").read_bytes(), training_manifest)
        manifest = self.read_json(output / "evaluation.json")
        self.assertEqual(manifest["stage"], "evaluation")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["split"], "test")
        self.assertEqual(len(manifest["prediction_files"]), 10)
        self.assertEqual(manifest["data_sha256"], sha256((self.root / "prices.csv").read_bytes()).hexdigest())
        self.assertIn("packages", self.read_json(output / "environment.json"))
        for relative, expected_hash in manifest["artifact_sha256"].items():
            self.assertEqual(sha256((self.run_dir / relative).read_bytes()).hexdigest(), expected_hash)
        reference = self.dataset.groups["A"].splits["test"]
        naive_rows = self.rows(output / "predictions_naive.csv")
        self.assertTrue(all(row["direction_correct"] == "N/A" and row["predicted_direction"] == "0" for row in naive_rows))
        np.testing.assert_array_equal([float(row["predicted_close"]) for row in naive_rows], reference.previous_close[:, 0])
        metric_rows = self.rows(output / "metrics_per_seed.csv")
        self.assertEqual(len(metric_rows), 10)
        errors_by_group = {}
        for metric in metric_rows:
            name = metric["group"]
            path = output / ("predictions_naive.csv" if name == "Naive" else f"predictions_{name}_seed_{metric['seed']}.csv")
            rows = self.rows(path)
            self.assertEqual([row["target_date"] for row in rows], reference.target_dates.strftime("%Y-%m-%d").tolist())
            actual = np.array([float(row["actual_close"]) for row in rows])
            previous = np.array([float(row["previous_close"]) for row in rows])
            predicted = np.array([float(row["predicted_close"]) for row in rows])
            np.testing.assert_array_equal(actual, reference.y_original[:, 0])
            np.testing.assert_array_equal(previous, reference.previous_close[:, 0])
            self.assertAlmostEqual(float(metric["mae"]), np.mean(abs(actual - predicted)))
            self.assertAlmostEqual(float(metric["rmse"]), np.sqrt(np.mean((actual - predicted) ** 2)))
            np.testing.assert_allclose([float(row["absolute_error"]) for row in rows], abs(actual - predicted))
            if name != "Naive":
                accuracy = np.mean(np.sign(predicted - previous) == np.sign(actual - previous))
                self.assertAlmostEqual(float(metric["directional_accuracy"]), accuracy)
                errors_by_group.setdefault(name, []).append(float(metric["mae"]))
            else:
                self.assertEqual(metric["directional_accuracy"], "N/A")
        for row in self.rows(output / "metrics_summary.csv"):
            if row["group"] != "Naive":
                errors = errors_by_group[row["group"]]
                self.assertAlmostEqual(float(row["mae_mean"]), np.mean(errors))
                self.assertAlmostEqual(float(row["mae_std"]), np.std(errors, ddof=1))
        self.assertIsNone(manifest["metrics_per_seed"][-1]["directional_accuracy"])
        self.assertNotIn("NaN", (output / "evaluation.json").read_text(encoding="utf-8"))

    def test_existing_evaluation_is_not_overwritten(self):
        output = self.evaluate()
        before = (output / "evaluation.json").read_bytes()
        with self.assertRaisesRegex(EvaluationError, "不覆寫"):
            self.evaluate()
        self.assertEqual((output / "evaluation.json").read_bytes(), before)

    def test_invalid_training_manifests_and_columns_fail_before_output(self):
        cases = [
            ("run.json", lambda data: data.update(status="failed")),
            ("run.json", lambda data: data.update(seeds=[42])),
            ("run.json", lambda data: data["runs"].__setitem__(1, copy.deepcopy(data["runs"][0]))),
            ("A/seed_42/training.json", lambda data: data.update(status="failed")),
            ("A/seed_42/training.json", lambda data: data.update(seed=43)),
            ("B/seed_42/training.json", lambda data: data.update(columns=list(reversed(data["columns"])))),
            ("scalers/description.json", lambda data: data["X"]["B"].update(columns=["Close"])),
        ]
        for relative, change in cases:
            path = self.run_dir / relative
            before = path.read_bytes()
            try:
                data = self.read_json(path)
                change(data)
                self.write_json(path, data)
                with self.subTest(relative=relative), self.assertRaises(EvaluationError):
                    self.evaluate()
                self.assertFalse((self.run_dir / "evaluation").exists())
            finally:
                path.write_bytes(before)

    def test_missing_weights_scalers_and_corrupt_saved_scaler_fail_before_output(self):
        for relative in ("A/seed_42/best.weights.h5", "scalers/X_A.pkl"):
            path = self.run_dir / relative
            moved = path.with_name(path.name + ".missing")
            path.rename(moved)
            try:
                with self.subTest(relative=relative), self.assertRaises(EvaluationError):
                    self.evaluate()
                self.assertFalse((self.run_dir / "evaluation").exists())
            finally:
                moved.rename(path)
        path = self.run_dir / "scalers" / "y.pkl"
        with path.open("rb") as handle:
            scaler = pickle.load(handle)
        scaler.clip = True
        with path.open("wb") as handle:
            pickle.dump(scaler, handle)
        with self.assertRaises(EvaluationError):
            self.evaluate()
        self.assertFalse((self.run_dir / "evaluation").exists())

    def test_valid_but_different_snapshot_hash_is_rejected(self):
        # Update BOTH current snapshot data and its checksum: load_snapshot is
        # still valid, but it no longer identifies the frozen training input.
        changed = self.frame.copy()
        changed.loc[:, ["Open", "High", "Low", "Close"]] *= 2
        path = self.root / "prices.csv"
        path.write_text(changed.to_csv(date_format="%Y-%m-%d", lineterminator="\n"), encoding="utf-8", newline="")
        metadata_path = path.with_suffix(".metadata.json")
        metadata = self.read_json(metadata_path)
        metadata["data"]["sha256"] = sha256(path.read_bytes()).hexdigest()
        self.write_json(metadata_path, metadata)
        with self.assertRaisesRegex(EvaluationError, "hash"):
            self.evaluate()
        self.assertFalse((self.run_dir / "evaluation").exists())

    def test_date_actual_and_previous_misalignment_is_rejected(self):
        original = self.dataset.groups["B"].splits["test"]
        cases = [replace(original, target_dates=original.target_dates[::-1]),
                 replace(original, y_original=original.y_original + 1),
                 replace(original, previous_close=original.previous_close + 1)]
        for bad_split in cases:
            groups = self.dataset.groups.copy()
            splits = groups["B"].splits.copy()
            splits["test"] = bad_split
            groups["B"] = replace(groups["B"], splits=splits)
            dataset = replace(self.dataset, groups=groups)
            with patch("lstm_comparison.evaluation.build_dataset", return_value=dataset), self.assertRaises(EvaluationError):
                self.evaluate()
            self.assertFalse((self.run_dir / "evaluation").exists())

    def test_nonfinite_prediction_retains_failed_evaluation_without_changing_training(self):
        before = (self.run_dir / "run.json").read_bytes()
        with patch("lstm_comparison.evaluation._predict", return_value=np.full_like(self.dataset.groups["A"].splits["test"].y, np.nan)), self.assertRaises(EvaluationError):
            self.evaluate()
        manifest = self.read_json(self.run_dir / "evaluation" / "evaluation.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertNotIn("completed_at_utc", manifest)
        self.assertEqual((self.run_dir / "run.json").read_bytes(), before)

    def test_invalid_weights_retains_failed_evaluation(self):
        (self.run_dir / "A" / "seed_42" / "best.weights.h5").write_bytes(b"corrupt")
        with self.assertRaises(EvaluationError):
            self.evaluate()
        self.assertEqual(self.read_json(self.run_dir / "evaluation" / "evaluation.json")["status"], "failed")

    def test_config_hash_io_failure_is_wrapped_before_output_creation(self):
        real_hash = evaluation._hash

        def fail_config(path):
            if path.name == "config.json":
                raise PermissionError("controlled config read failure")
            return real_hash(path)

        with patch("lstm_comparison.evaluation._hash", side_effect=fail_config), self.assertRaises(EvaluationError):
            self.evaluate()
        self.assertFalse((self.run_dir / "evaluation").exists())

    def test_daily_squared_error_overflow_marks_evaluation_failed(self):
        prediction = np.full_like(self.dataset.groups["A"].splits["test"].y, 1e200)
        with patch("lstm_comparison.evaluation._predict", return_value=prediction), self.assertRaisesRegex(EvaluationError, "CSV"):
            self.evaluate()
        manifest = self.read_json(self.run_dir / "evaluation" / "evaluation.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertNotIn("completed_at_utc", manifest)


if __name__ == "__main__":
    unittest.main()
