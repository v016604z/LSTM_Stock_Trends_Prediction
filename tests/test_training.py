import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from lstm_comparison.config import load_config
from lstm_comparison.data import SnapshotError, download_snapshot
from lstm_comparison.dataset import build_dataset
from lstm_comparison.model import build_model
from lstm_comparison import training
from lstm_comparison.training import TrainingError, run_experiment, train_one


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        settings = json.loads((ROOT / "configs" / "research.json").read_text(encoding="utf-8"))
        settings["data"].update({"start": "2022-08-01", "end": "2023-02-28",
                                 "end_exclusive": "2023-03-01", "snapshot_path": "prices.csv",
                                 "output_dir": "runs"})
        settings["split"] = {
            "train": {"start": "2022-08-01", "end": "2022-12-31"},
            "validation": {"start": "2023-01-01", "end": "2023-01-31"},
            "test": {"start": "2023-02-01", "end": "2023-02-28"},
        }
        settings["training"]["max_epochs"] = 1
        path = self.root / "config.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        self.config = load_config(path)
        self.sessions = pd.bdate_range("2022-08-01", "2023-02-28", name="Date").as_unit("ns")
        positions = np.arange(len(self.sessions))
        close = 100 + positions / 10 + np.sin(positions / 4)
        self.frame = pd.DataFrame({"Open": close - 0.25, "High": close + 1,
                                   "Low": close - 1, "Close": close, "Volume": 1000 + positions},
                                  index=self.sessions)
        self.dataset = build_dataset(self.frame, self.config)

    def freeze(self):
        # A frozen SYNTHETIC calendar, isolated from Yahoo and real research data.
        info = {"name": "XTAI", "version": "synthetic-test", "source": "https://example.test/calendar",
                "additional_closures": [], "additional_sessions": [],
                "requested_start": self.config["data"]["start"], "requested_end": self.config["data"]["end"]}
        with patch("yfinance.download", return_value=self.frame), patch(
            "yfinance.set_tz_cache_location"
        ), patch("lstm_comparison.data._calendar_sessions", return_value=(self.sessions, info)):
            return download_snapshot(self.config)

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_real_offline_run_saves_three_models_and_traceable_artifacts(self):
        _, metadata = self.freeze()
        config = copy.deepcopy(self.config)
        config["training"]["seeds"] = [42]
        with patch("yfinance.download", side_effect=AssertionError("Unexpected network")):
            run_dir = run_experiment(config)
        manifest = self.read(run_dir / "run.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["stage"], "training")
        self.assertEqual(manifest["expected_runs"], 3)
        self.assertEqual(len(manifest["runs"]), 3)
        self.assertEqual(manifest["data_sha256"], sha256((self.root / "prices.csv").read_bytes()).hexdigest())
        self.assertEqual(self.read(run_dir / "snapshot.metadata.json"), metadata)
        self.assertEqual(self.read(run_dir / "config.json"), config)
        environment = self.read(run_dir / "environment.json")
        self.assertEqual(environment["device"], "cpu")
        self.assertTrue(environment["deterministic_ops"])
        self.assertIn("tensorflow-cpu", environment["packages"])
        with (run_dir / "scalers" / "y.pkl").open("rb") as handle:
            y_scaler = pickle.load(handle)
        for name, expected_count in (("A", 16961), ("B", 17985), ("C", 19265)):
            directory = run_dir / name / "seed_42"
            summary = self.read(directory / "training.json")
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["seed"], 42)
            self.assertEqual(summary["epochs_trained"], 1)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["parameter_count"], expected_count)
            history = pd.read_csv(directory / "history.csv")
            self.assertTrue(np.isfinite(history.to_numpy()).all())
            self.assertEqual(history.columns.tolist(), ["epoch", "loss", "val_loss"])
            self.assertTrue((directory / "best.weights.h5").is_file())
            with (run_dir / "scalers" / f"X_{name}.pkl").open("rb") as handle:
                x_scaler = pickle.load(handle)
            np.testing.assert_array_equal(x_scaler.data_min_, self.dataset.groups[name].x_scaler.data_min_)
            validation = self.dataset.groups[name].splits["validation"]
            model = build_model(validation.X.shape[1:], config, 999)
            model.load_weights(directory / "best.weights.h5", objects_to_skip=[model.optimizer])
            self.assertTrue(np.isfinite(np.asarray(model(validation.X, training=False))).all())
            np.testing.assert_allclose(y_scaler.inverse_transform(validation.y), validation.y_original)
        self.assertFalse(list(run_dir.rglob("*predictions*")))

    def test_restores_best_weights_both_early_stop_and_max_epochs(self):
        import tensorflow as tf

        real_callbacks = training._callbacks

        class ForcedValidationLoss(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.weights_by_epoch = []

            def on_epoch_end(self, epoch, logs=None):
                weights = [np.full_like(value, (epoch + 1) / 1000) for value in self.model.get_weights()]
                self.model.set_weights(weights)
                self.weights_by_epoch.append(weights)
                logs["val_loss"] = [0.5, 0.1, 0.3, 0.4, 0.5, 0.6][epoch]

        for stop_early in (True, False):
            with self.subTest(stop_early=stop_early):
                config = copy.deepcopy(self.config)
                config["training"]["max_epochs"] = 6 if stop_early else 3
                config["training"]["early_stopping"]["patience"] = 1 if stop_early else 10
                forced = ForcedValidationLoss()
                directory = self.root / f"controlled_{stop_early}"
                group = self.dataset.groups["A"]
                # Prove fit/prediction does not access the held-out test split.
                group = type(group)(group.columns, group.x_scaler, {
                    "train": group.splits["train"], "validation": group.splits["validation"]})
                with patch("lstm_comparison.training._callbacks", side_effect=lambda cfg: [forced, *real_callbacks(cfg)]):
                    result = train_one(group, config, 42, directory)
                self.assertEqual(result["best_epoch"], 2)
                self.assertEqual(result["epochs_trained"], 3)
                self.assertEqual(result["stopped_early"], stop_early)
                for actual, expected in zip(forced.model.get_weights(), forced.weights_by_epoch[1]):
                    np.testing.assert_array_equal(actual, expected)
                validation = group.splits["validation"]
                expected_prediction = np.asarray(forced.model(validation.X, training=False))
                loaded = build_model(validation.X.shape[1:], config, 43)
                loaded.load_weights(directory / "best.weights.h5", objects_to_skip=[loaded.optimizer])
                np.testing.assert_array_equal(np.asarray(loaded(validation.X, training=False)), expected_prediction)
                self.assertEqual(self.read(directory / "training.json")["best_epoch"], 2)

    def test_nine_run_schedule_and_repeat_never_overwrite_existing_run(self):
        self.freeze()
        calls = []

        def fake_train(group, config, seed, directory):
            calls.append((len(group.columns), seed))
            Path(directory).mkdir(parents=True)
            return {"status": "completed", "seed": seed, "parameter_count": 1,
                    "best_epoch": 1, "epochs_trained": 1, "elapsed_seconds": 0.0}

        with patch("lstm_comparison.training.train_one", side_effect=fake_train):
            first = run_experiment(self.config)
            (first / "preserve.txt").write_text("keep", encoding="utf-8")
            second = run_experiment(self.config)
        expected = [(width, seed) for width in (1, 5, 10) for seed in (42, 43, 44)]
        self.assertEqual(calls, expected * 2)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "preserve.txt").read_text(encoding="utf-8"), "keep")
        for directory in (first, second):
            manifest = self.read(directory / "run.json")
            self.assertEqual(manifest["expected_runs"], 9)
            self.assertEqual(len(manifest["runs"]), 9)
            self.assertTrue(all(run["status"] == "completed" for run in manifest["runs"]))

    def test_missing_or_corrupt_snapshot_fails_before_creating_run_directory(self):
        with self.assertRaises(SnapshotError):
            run_experiment(self.config)
        self.assertFalse(Path(self.config["data"]["output_dir"]).exists())
        self.freeze()
        (self.root / "prices.csv").write_bytes(b"corrupted")
        with self.assertRaises(SnapshotError):
            run_experiment(self.config)
        self.assertFalse(Path(self.config["data"]["output_dir"]).exists())

    def test_failed_run_retains_partial_artifacts_and_failed_manifest(self):
        self.freeze()

        def fail(group, config, seed, directory):
            Path(directory).mkdir(parents=True)
            (Path(directory) / "partial.txt").write_text("diagnostic", encoding="utf-8")
            raise TrainingError("controlled failure")

        with patch("lstm_comparison.training.train_one", side_effect=fail), self.assertRaises(TrainingError):
            run_experiment(self.config)
        run_dir, = Path(self.config["data"]["output_dir"]).iterdir()
        manifest = self.read(run_dir / "run.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["runs"][0]["status"], "failed")
        self.assertTrue((run_dir / "A" / "seed_42" / "partial.txt").exists())
        self.assertNotIn("completed_at_utc", manifest)

    def test_nonfinite_history_prediction_or_save_failure_never_succeeds(self):
        group = self.dataset.groups["A"]
        for failure in ("history", "prediction", "save"):
            with self.subTest(failure=failure):
                fake_model = Mock()
                fake_model.count_params.return_value = 16961
                fake_model.fit.return_value = SimpleNamespace(history={
                    "loss": [0.2], "val_loss": [np.nan if failure == "history" else 0.3]})
                fake_model.side_effect = lambda X, training: np.full((len(X), 1), np.nan if failure == "prediction" else 0.1)
                if failure == "save":
                    fake_model.save_weights.side_effect = OSError("controlled save failure")
                directory = self.root / failure
                with patch("lstm_comparison.training.build_model", return_value=fake_model), self.assertRaises(TrainingError):
                    train_one(group, self.config, 42, directory)
                summary = self.read(directory / "training.json")
                self.assertEqual(summary["status"], "failed")
                self.assertNotIn("completed_at_utc", summary)
                self.assertFalse((directory / "best.weights.h5").exists())

    def test_batch_nonfinite_loss_is_rejected_and_existing_training_not_overwritten(self):
        callback = training._callbacks(self.config)[0]
        with self.assertRaises(TrainingError):
            callback.on_train_batch_end(0, {"loss": np.inf})
        directory = self.root / "existing"
        directory.mkdir()
        marker = directory / "training.json"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(TrainingError):
            train_one(self.dataset.groups["A"], self.config, 42, directory)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
