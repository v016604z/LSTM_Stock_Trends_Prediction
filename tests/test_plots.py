import builtins
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from lstm_comparison.metrics import calculate_metrics, summarize_metrics
from lstm_comparison.plots import render_figures


def synthetic_bundle(seeds=(42, 43, 44), days=24):
    """Synthetic preview/test inputs, never official research results."""
    dates = pd.bdate_range("2024-01-02", periods=days)
    position = np.arange(days)
    actual = 100 + position * 0.3 + 4 * np.sin(position / 4)
    previous = np.r_[actual[0] - 0.5, actual[:-1]]
    base = pd.DataFrame({"target_date": dates, "actual_close": actual,
                         "previous_close": previous, "predicted_close": previous})
    predictions, histories, rows = {}, {}, []
    for number, group in enumerate(("A", "B", "C"), 1):
        for seed in seeds:
            frame = base.copy(deep=True)
            # Seed 44 is deliberately perfect: the price figure must still
            # select seed 42 rather than searching for the best test result.
            offset = 0 if seed == 44 else (1.8 if seed == 42 else 0.8)
            frame["predicted_close"] = actual + offset * number * (1 + 0.2 * np.cos(position / 3))
            predictions[group, seed] = frame
            rows.append({"group": group, "seed": seed,
                         **calculate_metrics(actual, frame["predicted_close"], previous)})
            epoch = np.arange(1, 4 + seed % 3)
            histories[group, seed] = pd.DataFrame({
                "epoch": epoch, "loss": (0.8 + number / 10) / (epoch + (seed - 42) / 3),
                "val_loss": (1 + number / 10) / (epoch + (seed - 42) / 3) + 0.02 * epoch,
            })
    baseline = calculate_metrics(actual, previous, previous, naive=True)
    return {"ticker": "SYNTHETIC PREVIEW | DEMO", "price_unit": "JPY", "seeds": list(seeds),
            "plot_seed": 42, "predictions": predictions, "naive_predictions": base,
            "summary": summarize_metrics(rows, baseline), "histories": histories}


class PlotsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bundle = synthetic_bundle()

    def capture(self, bundle=None):
        snapshots = {}
        original = Figure.savefig

        def save(figure, target, *args, **kwargs):
            snapshots[Path(target.name).stem] = {
                "texts": [text.get_text() for text in figure.texts],
                "legends": [[text.get_text() for text in legend.get_texts()] for legend in figure.legends],
                "axes": [{
                    "lines": {line.get_label(): {"x": np.asarray(line.get_xdata()).copy(),
                               "y": np.asarray(line.get_ydata()).copy(), "style": line.get_linestyle(),
                               "marker": line.get_marker(), "marker_size": line.get_markersize(),
                               "marker_face": line.get_markerfacecolor()}
                              for line in axis.lines},
                    "bars": [(bar.get_x() + bar.get_width() / 2, bar.get_height()) for bar in axis.patches],
                    "errorbars": [collection.get_segments() for container in axis.containers
                                  if isinstance(container, ErrorbarContainer) for collection in container.lines[2]],
                    "texts": [text.get_text() for text in axis.texts],
                    "legend": [] if axis.get_legend() is None else [text.get_text() for text in axis.get_legend().get_texts()],
                    "ylabel": axis.get_ylabel(), "xlabel": axis.get_xlabel(), "title": axis.get_title(loc="left"),
                    "ticks": axis.get_xticks().copy(),
                } for axis in figure.axes],
            }
            original(figure, target, *args, **kwargs)

        with patch.object(Figure, "savefig", autospec=True, side_effect=save):
            paths = render_figures(self.bundle if bundle is None else bundle, self.root / "figures")
        return paths, snapshots

    def test_artist_values_labels_fixed_seed_and_all_seed_losses(self):
        _, figures = self.capture()
        price = figures["prices"]["axes"][0]
        self.assertEqual(price["legend"], ["Actual", "A (seed 42)", "B (seed 42)", "C (seed 42)", "Naive (previous close)"])
        self.assertIn("JPY", price["ylabel"])
        self.assertIn("2024-01-02", price["title"])
        self.assertIn("Fixed LSTM seed 42", price["title"])
        for key in figures:
            self.assertIn(self.bundle["ticker"], " ".join(figures[key]["texts"]))
        for group in ("A", "B", "C"):
            line = price["lines"][f"{group} (seed 42)"]
            expected = self.bundle["predictions"][group, 42]
            pd.testing.assert_index_equal(pd.DatetimeIndex(line["x"]), pd.DatetimeIndex(expected["target_date"]), check_names=False)
            np.testing.assert_array_equal(line["y"], expected["predicted_close"])
            self.assertFalse(np.array_equal(line["y"], self.bundle["predictions"][group, 44]["predicted_close"]))
        np.testing.assert_array_equal(price["lines"]["Actual"]["y"], self.bundle["naive_predictions"]["actual_close"])
        np.testing.assert_array_equal(price["lines"]["Naive (previous close)"]["y"], self.bundle["naive_predictions"]["previous_close"])

        summary = {row["group"]: row for row in self.bundle["summary"]}
        for axis, metric in zip(figures["errors"]["axes"], ("mae", "rmse")):
            self.assertEqual(len(axis["bars"]), 4)
            self.assertEqual(len(axis["errorbars"]), 3)
            self.assertIn("JPY", axis["ylabel"])
            for index, group in enumerate(("A", "B", "C", "Naive")):
                self.assertAlmostEqual(axis["bars"][index][1], summary[group][f"{metric}_mean"])
                if group != "Naive":
                    mean, std = summary[group][f"{metric}_mean"], summary[group][f"{metric}_std"]
                    np.testing.assert_allclose(axis["errorbars"][index][0][:, 1], [mean - std, mean + std])
        direction = figures["direction"]["axes"][0]
        self.assertEqual(len(direction["bars"]), 3)
        self.assertEqual([bar[0] for bar in direction["bars"]], [0, 1, 2])
        self.assertIn("N/A", direction["texts"])
        self.assertIn("%", direction["ylabel"])
        for index, group in enumerate(("A", "B", "C")):
            mean, std = summary[group]["directional_accuracy_mean"], summary[group]["directional_accuracy_std"]
            self.assertAlmostEqual(direction["bars"][index][1], mean * 100)
            np.testing.assert_allclose(direction["errorbars"][index][0][:, 1], [(mean - std) * 100, (mean + std) * 100])
        loss = figures["loss"]
        self.assertEqual(len(loss["axes"]), 3)
        self.assertEqual(len(loss["legends"][0]), 6)
        for axis, group in zip(loss["axes"], ("A", "B", "C")):
            self.assertEqual(len(axis["lines"]), 6)
            self.assertEqual(axis["ylabel"], "MSE (scaled Close)")
            self.assertEqual(axis["ticks"][0], 1)
            for seed in self.bundle["seeds"]:
                history = self.bundle["histories"][group, seed]
                for label, column, style in (("train", "loss", "-"), ("validation", "val_loss", "--")):
                    line = axis["lines"][f"seed {seed}: {label}"]
                    np.testing.assert_array_equal(line["x"], history["epoch"])
                    np.testing.assert_array_equal(line["y"], history[column])
                    self.assertEqual(line["style"], style)
                    self.assertIn(line["marker"], (None, "None", "none", ""))

    def test_single_epoch_losses_have_distinguishable_visible_markers_even_when_equal(self):
        bundle = synthetic_bundle(seeds=(42,))
        for pair, history in bundle["histories"].items():
            point = history.iloc[:1].copy()
            point["val_loss"] = point["loss"]
            bundle["histories"][pair] = point
        _, figures = self.capture(bundle)
        for axis in figures["loss"]["axes"]:
            train = axis["lines"]["seed 42: train"]
            validation = axis["lines"]["seed 42: validation"]
            np.testing.assert_array_equal(train["x"], [1])
            np.testing.assert_array_equal(validation["x"], [1])
            np.testing.assert_array_equal(train["y"], validation["y"])
            self.assertEqual(train["marker"], "o")
            self.assertEqual(validation["marker"], "D")
            self.assertGreater(train["marker_size"], 0)
            self.assertGreater(validation["marker_size"], train["marker_size"])
            self.assertNotEqual(train["marker_face"], "none")
            self.assertEqual(validation["marker_face"], "none")
            self.assertEqual(train["style"], "-")
            self.assertEqual(validation["style"], "--")

    def test_single_target_price_series_have_visible_points(self):
        bundle = synthetic_bundle(days=1)
        _, figures = self.capture(bundle)
        price = figures["prices"]["axes"][0]
        self.assertEqual(len(price["lines"]), 5)
        for label, line in price["lines"].items():
            with self.subTest(label=label):
                self.assertEqual(len(line["x"]), 1)
                self.assertEqual(len(line["y"]), 1)
                self.assertNotIn(line["marker"], (None, "None", "none", ""))
                self.assertGreater(line["marker_size"], 0)
        self.assertEqual(len(price["ticks"]), 1)

    def test_pngs_decode_and_render_preserves_inputs_rc_and_unrelated_figures(self):
        original = copy.deepcopy(self.bundle)
        existing = plt.figure()
        self.addCleanup(plt.close, existing)
        numbers = plt.get_fignums()
        settings = dict(matplotlib.rcParams)
        import_function = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in {"tensorflow", "keras"} or name in {
                "lstm_comparison.model", "lstm_comparison.training", "lstm_comparison.data"
            }:
                raise AssertionError(f"Unexpected model or source import: {name}")
            return import_function(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked_import):
            paths = render_figures(self.bundle, self.root / "figures")
        self.assertEqual(set(paths), {"prices", "errors", "direction", "loss"})
        for path in paths.values():
            self.assertEqual(path.parent, self.root / "figures")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            pixels = mpimg.imread(path)
            self.assertGreater(pixels.shape[0], 500)
            self.assertGreater(pixels.shape[1], 800)
            self.assertTrue(np.isfinite(pixels).all())
        self.assertEqual(plt.get_fignums(), numbers)
        self.assertEqual(dict(matplotlib.rcParams), settings)
        for key in ("ticker", "price_unit", "seeds", "plot_seed", "summary"):
            self.assertEqual(self.bundle[key], original[key])
        pd.testing.assert_frame_equal(self.bundle["naive_predictions"], original["naive_predictions"])
        for category in ("predictions", "histories"):
            for key in self.bundle[category]:
                pd.testing.assert_frame_equal(self.bundle[category][key], original[category][key])

    def test_single_seed_omits_error_bars_instead_of_inventing_zero_sd(self):
        _, figures = self.capture(synthetic_bundle(seeds=(42,)))
        for axis in figures["errors"]["axes"] + figures["direction"]["axes"]:
            self.assertEqual(axis["errorbars"], [])
        self.assertEqual(len(figures["direction"]["axes"][0]["bars"]), 3)
        self.assertIn("N/A", figures["direction"]["axes"][0]["texts"])
        for axis in figures["loss"]["axes"]:
            self.assertEqual(len(axis["lines"]), 2)

    def test_save_failure_releases_figure_and_restores_rc_without_closing_other_figures(self):
        existing = plt.figure()
        self.addCleanup(plt.close, existing)
        numbers = plt.get_fignums()
        settings = dict(matplotlib.rcParams)
        failed = []

        def fail(figure, *args, **kwargs):
            failed.append(figure)
            raise OSError("controlled write failure")

        with patch.object(Figure, "savefig", autospec=True, side_effect=fail), self.assertRaisesRegex(OSError, "controlled"):
            render_figures(self.bundle, self.root / "figures")
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].axes, [])
        self.assertEqual(plt.get_fignums(), numbers)
        self.assertEqual(dict(matplotlib.rcParams), settings)

    def test_existing_outputs_and_nonfixed_seed_are_rejected(self):
        directory = self.root / "figures"
        directory.mkdir()
        marker = directory / "prices.png"
        marker.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            render_figures(self.bundle, directory)
        self.assertEqual(marker.read_bytes(), b"preserve")
        self.bundle["plot_seed"] = 44
        with self.assertRaisesRegex(ValueError, "42"):
            render_figures(self.bundle, self.root / "wrong_seed")
        self.assertFalse((self.root / "wrong_seed").exists())


if __name__ == "__main__":
    unittest.main()
