import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from lstm_comparison.config import ConfigError, load_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "research.json"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.json"

    def save(self, settings):
        self.path.write_text(json.dumps(settings), encoding="utf-8")
        return self.path

    def test_default_research_contract(self):
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["training"]["seeds"], [42, 43, 44])
        self.assertEqual(config["sequence"], {"lookback": 60, "horizon": 1})
        self.assertEqual([len(group) for group in config["features"]["groups"].values()], [1, 5, 10])
        self.assertEqual(config["split"]["validation"]["start"], "2023-01-01")
        self.assertEqual(config["split"]["test"]["start"], "2024-01-01")
        self.assertEqual(config["data"]["end_exclusive"], "2026-01-01")
        self.assertEqual(config["training"]["max_epochs"], 100)

    def test_paths_use_config_directory_independent_of_cwd(self):
        self.settings["data"]["snapshot_path"] = "data/prices.csv"
        path = self.save(self.settings)
        expected = str(path.parent / "data" / "prices.csv")
        original_cwd = Path.cwd()
        try:
            os.chdir(path.parent)
            first = load_config(path)
            os.chdir(path.parent.parent)
            second = load_config(path)
        finally:
            os.chdir(original_cwd)
        self.assertEqual(first["data"]["snapshot_path"], expected)
        self.assertEqual(first, second)
        self.assertFalse(Path(expected).exists())

    def test_smoke_training_settings_can_be_shortened(self):
        self.settings["training"]["max_epochs"] = 1
        self.settings["training"]["seeds"] = [42]
        config = load_config(self.save(self.settings))
        self.assertEqual(config["training"]["max_epochs"], 1)

    def test_invalid_values_have_field_errors(self):
        cases = [
            (("sequence", "lookback"), True, "sequence.lookback"),
            (("sequence", "horizon"), 2, "sequence.horizon"),
            (("training", "batch_size"), 0, "training.batch_size"),
            (("training", "max_epochs"), 1.5, "training.max_epochs"),
            (("training", "learning_rate"), float("nan"), "training.learning_rate"),
            (("training", "learning_rate"), float("inf"), "training.learning_rate"),
            (("training", "learning_rate"), 10**400, "training.learning_rate"),
            (("training", "seeds"), [42, True], "training.seeds"),
            (("training", "seeds"), [42, 42], "training.seeds"),
            (("training", "seeds"), [2**32], "training.seeds"),
            (("training", "seeds"), [], "training.seeds"),
            (("training", "shuffle"), 0, "training.shuffle"),
            (("data", "start"), "2015-02-30", "data.start"),
            (("data", "end_exclusive"), "2025-12-31", "end_exclusive"),
            (("data", "snapshot_path"), "", "data.snapshot_path"),
            (("data", "snapshot_path"), "invalid\u0000path", "data.snapshot_path"),
            (("features", "groups", "B"), ["Close"], "features.groups.B"),
            (("features", "indicators", "rsi_period"), 10, "features.indicators.rsi_period"),
            (("features", "indicators", "macd_fast"), 10, "features.indicators.macd_fast"),
            (("features", "indicators", "macd_slow"), 24, "features.indicators.macd_slow"),
            (("features", "indicators", "macd_signal"), 8, "features.indicators.macd_signal"),
            (("evaluation", "plot_seed"), 99, "evaluation.plot_seed"),
            (("split", "validation", "start"), "2022-12-31", "split.validation"),
            (("split", "validation", "start"), "2023-01-02", "split.validation"),
            (("split", "test", "end"), "2026-01-01", "split.test"),
        ]
        for keys, value, message in cases:
            with self.subTest(keys=keys, value=value):
                settings = copy.deepcopy(self.settings)
                target = settings
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ConfigError, message):
                    load_config(self.save(settings))

    def test_missing_and_unknown_keys_fail(self):
        for mutate in (
            lambda settings: settings.pop("model"),
            lambda settings: settings["training"].update({"max_epoch": 1}),
        ):
            settings = copy.deepcopy(self.settings)
            mutate(settings)
            with self.assertRaisesRegex(ConfigError, "欄位錯誤"):
                load_config(self.save(settings))

    def test_unreadable_and_invalid_json_fail(self):
        with self.assertRaises(ConfigError):
            load_config(self.path)
        with self.assertRaises(ConfigError):
            load_config("invalid\u0000path")
        for content in ("{", "[]", "null"):
            self.path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ConfigError):
                load_config(self.path)


if __name__ == "__main__":
    unittest.main()
