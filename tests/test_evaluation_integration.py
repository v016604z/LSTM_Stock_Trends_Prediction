import csv
from hashlib import sha256
import json
import math
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
from lstm_comparison.evaluation import evaluate_experiment
from lstm_comparison.features import build_feature_groups
from lstm_comparison.model import build_model
from lstm_comparison.training import run_experiment


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))


class EvaluationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        settings = json.loads((ROOT / "configs" / "research.json").read_text(encoding="utf-8"))
        settings["data"].update({
            "start": "2022-08-01", "end": "2023-02-28", "end_exclusive": "2023-03-01",
            "snapshot_path": "prices.csv", "output_dir": "runs",
        })
        settings["split"] = {
            "train": {"start": "2022-08-01", "end": "2022-12-23"},
            "validation": {"start": "2022-12-24", "end": "2022-12-31"},
            "test": {"start": "2023-01-01", "end": "2023-02-28"},
        }
        settings["training"].update({"seeds": [42], "max_epochs": 1})
        path = self.root / "config.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        self.config = load_config(path)
        # Deliberately synthetic business dates, including 2023-01-02. These
        # are test sessions, not an assertion about the real TWSE calendar.
        self.sessions = pd.bdate_range("2022-08-01", "2023-02-28", name="Date").as_unit("ns")
        positions = np.arange(len(self.sessions))
        close = 100 + (positions // 7) * 0.25 + np.take([0, 2, 2, -1, 1, -1, 0], positions % 7)
        self.frame = pd.DataFrame({
            "Open": close - 0.25, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": 1000 + positions,
        }, index=self.sessions)

    def freeze(self):
        info = {
            "name": "XTAI", "version": "synthetic-test", "source": "https://example.test/calendar",
            "additional_closures": [], "additional_sessions": [],
            "requested_start": self.config["data"]["start"], "requested_end": self.config["data"]["end"],
        }
        with patch("yfinance.download", return_value=self.frame), patch(
            "yfinance.set_tz_cache_location"
        ), patch("lstm_comparison.data._calendar_sessions", return_value=(self.sessions, info)):
            download_snapshot(self.config)

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def read_csv(path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_real_saved_models_evaluate_without_refitting_and_match_independent_csv_math(self):
        self.freeze()
        source_files = [self.root / f"prices{suffix}" for suffix in (".csv", ".metadata.json", ".sessions.csv")]
        source_hashes = {path: sha256(path.read_bytes()).hexdigest() for path in source_files}
        with patch("yfinance.download", side_effect=AssertionError("Unexpected network access")):
            run_dir = run_experiment(self.config)
            original_artifacts = {
                path.relative_to(run_dir): sha256(path.read_bytes()).hexdigest()
                for path in run_dir.rglob("*") if path.is_file()
            }
            with patch("sklearn.preprocessing.MinMaxScaler.fit", side_effect=AssertionError("Evaluation must not fit")) as fit:
                output = evaluate_experiment(run_dir)
            fit.assert_not_called()

        self.assertEqual(output, run_dir / "evaluation")
        for relative, digest in original_artifacts.items():
            self.assertEqual(sha256((run_dir / relative).read_bytes()).hexdigest(), digest)
        for path, digest in source_hashes.items():
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(self.read_json(run_dir / "run.json")["stage"], "training")

        manifest = self.read_json(output / "evaluation.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["stage"], "evaluation")
        self.assertEqual(manifest["split"], "test")
        self.assertEqual(manifest["groups"], ["A", "B", "C"])
        self.assertEqual(manifest["seeds"], [42])
        self.assertIn("completed_at_utc", manifest)
        self.assertEqual(manifest["data_sha256"], source_hashes[self.root / "prices.csv"])
        self.assertEqual(manifest["config_sha256"], original_artifacts[Path("config.json")])
        # Trace every configuration, fitted mapping and trained weight file
        # needed to reconstruct these predictions, rather than only a count.
        prediction_sources = {"config.json", "scalers/y.pkl"}
        for group in self.config["features"]["groups"]:
            prediction_sources.add(f"scalers/X_{group}.pkl")
            prediction_sources.update(
                f"{group}/seed_{seed}/best.weights.h5" for seed in self.config["training"]["seeds"]
            )
        self.assertEqual(set(manifest["artifact_sha256"]), prediction_sources)
        for relative, digest in manifest["artifact_sha256"].items():
            self.assertEqual(digest, original_artifacts[Path(relative)])
        self.assertTrue((output / "environment.json").is_file())
        self.assertTrue((output / "metrics_per_seed.csv").is_file())
        self.assertTrue((output / "metrics_summary.csv").is_file())
        for filename in manifest["prediction_files"]:
            self.assertTrue((output / filename).is_file())
        self.assertEqual(len(manifest["prediction_files"]), 4)

        targets = self.frame.index[self.frame.index >= "2023-01-01"]
        target_positions = self.frame.index.get_indexer(targets)
        expected_dates = targets.strftime("%Y-%m-%d").tolist()
        expected_previous_dates = self.frame.index[target_positions - 1].strftime("%Y-%m-%d").tolist()
        self.assertEqual(expected_dates[0], "2023-01-02")
        self.assertEqual(expected_previous_dates[0], "2022-12-30")
        self.assertEqual(manifest["n_samples"], len(targets))
        self.assertEqual(manifest["first_target"], expected_dates[0])
        self.assertEqual(manifest["last_target"], expected_dates[-1])

        measured = {}
        for group in ("A", "B", "C", "Naive"):
            filename = "predictions_naive.csv" if group == "Naive" else f"predictions_{group}_seed_42.csv"
            rows = self.read_csv(output / filename)
            self.assertEqual([row["target_date"] for row in rows], expected_dates)
            self.assertEqual([row["input_end_date"] for row in rows], expected_previous_dates)
            self.assertEqual({row["group"] for row in rows}, {group})
            self.assertEqual({row["seed"] for row in rows}, {"N/A" if group == "Naive" else "42"})
            errors, correct = [], []
            for row, position in zip(rows, target_positions):
                actual, previous, predicted = (float(row[key]) for key in ("actual_close", "previous_close", "predicted_close"))
                self.assertEqual(actual, self.frame["Close"].iloc[position])
                self.assertEqual(previous, self.frame["Close"].iloc[position - 1])
                self.assertTrue(math.isfinite(predicted))
                error = predicted - actual
                errors.append(error)
                actual_direction = (actual > previous) - (actual < previous)
                predicted_direction = (predicted > previous) - (predicted < previous)
                correct.append(actual_direction == predicted_direction)
                self.assertAlmostEqual(float(row["absolute_error"]), abs(error))
                self.assertAlmostEqual(float(row["squared_error"]), error**2)
                self.assertEqual(int(row["actual_direction"]), actual_direction)
                self.assertEqual(int(row["predicted_direction"]), predicted_direction)
                if group == "Naive":
                    self.assertEqual(predicted, previous)
                    self.assertEqual(row["direction_correct"], "N/A")
                else:
                    self.assertEqual(int(row["direction_correct"]), int(correct[-1]))
            measured[group] = {
                "mae": math.fsum(abs(error) for error in errors) / len(errors),
                "rmse": math.sqrt(math.fsum(error**2 for error in errors) / len(errors)),
                "directional_accuracy": None if group == "Naive" else sum(correct) / len(correct),
            }

        per_seed = self.read_csv(output / "metrics_per_seed.csv")
        summary = self.read_csv(output / "metrics_summary.csv")
        self.assertEqual(len(per_seed), 4)
        self.assertEqual(len(summary), 4)
        json_summary = {row["group"]: row for row in manifest["metrics_summary"]}
        for row in per_seed:
            group = row["group"]
            self.assertEqual(int(row["n_samples"]), len(targets))
            for key, value in measured[group].items():
                if value is None:
                    self.assertEqual(row[key], "N/A")
                else:
                    self.assertAlmostEqual(float(row[key]), value)
            for key in ("mae", "rmse"):
                if group == "Naive":
                    self.assertEqual(row[f"{key}_improvement_pct"], "N/A")
                else:
                    expected = 100 * (measured["Naive"][key] - measured[group][key]) / measured["Naive"][key]
                    self.assertAlmostEqual(float(row[f"{key}_improvement_pct"]), expected)
        for row in summary:
            group = row["group"]
            self.assertEqual(int(row["n_seeds"]), 0 if group == "Naive" else 1)
            for key, value in measured[group].items():
                self.assertEqual(row[f"{key}_std"], "N/A")
                self.assertIsNone(json_summary[group][f"{key}_std"])
                if value is None:
                    self.assertEqual(row[f"{key}_mean"], "N/A")
                    self.assertIsNone(json_summary[group][f"{key}_mean"])
                else:
                    self.assertAlmostEqual(float(row[f"{key}_mean"]), value)
            for key in ("mae", "rmse"):
                if group == "Naive":
                    self.assertEqual(row[f"{key}_improvement_pct"], "N/A")
                else:
                    expected = 100 * (measured["Naive"][key] - measured[group][key]) / measured["Naive"][key]
                    self.assertAlmostEqual(float(row[f"{key}_improvement_pct"]), expected)

        # Independently reconstruct inputs with saved affine scaler parameters,
        # build different random initial weights, then load the real HDF5 weights.
        # This catches evaluation using fresh weights or skipping price inversion.
        with (run_dir / "scalers" / "y.pkl").open("rb") as handle:
            y_scaler = pickle.load(handle)
        features = build_feature_groups(self.frame, self.config)
        lookback = self.config["sequence"]["lookback"]
        window_positions = target_positions[:, None] - lookback + np.arange(lookback)
        for group in ("A", "B", "C"):
            with (run_dir / "scalers" / f"X_{group}.pkl").open("rb") as handle:
                x_scaler = pickle.load(handle)
            X = features[group].to_numpy(dtype=float)[window_positions] * x_scaler.scale_ + x_scaler.min_
            model = build_model(X.shape[1:], self.config, 999)

            def predict_scaled():
                return np.concatenate([
                    np.asarray(model(X[start:start + 32], training=False))
                    for start in range(0, len(X), 32)
                ])

            initial = predict_scaled()
            model.load_weights(run_dir / group / "seed_42" / "best.weights.h5", objects_to_skip=[model.optimizer])
            restored = predict_scaled()
            self.assertFalse(np.allclose(initial, restored))
            expected = (restored[:, 0] - y_scaler.min_[0]) / y_scaler.scale_[0]
            actual = [float(row["predicted_close"]) for row in self.read_csv(output / f"predictions_{group}_seed_42.csv")]
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
