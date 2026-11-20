import copy
import json
import math
import unittest

import numpy as np

from lstm_comparison.metrics import (
    MetricsError, calculate_metrics, error_improvement_pct, summarize_metrics,
)


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_price_errors_and_three_way_direction(self):
        actual = [110, 90, 100, 110, 90, 100]
        predicted = [105, 95, 100, 100, 110, 99]
        result = calculate_metrics(actual, predicted, [100] * 6)
        self.assertAlmostEqual(result["mae"], 41 / 6)
        self.assertAlmostEqual(result["rmse"], math.sqrt(551 / 6))
        self.assertEqual(result["directional_accuracy"], 0.5)

    def test_direction_uses_each_previous_close_and_no_epsilon(self):
        result = calculate_metrics([101, 90, 50], [102, 99, 50], [100, 100, 50])
        self.assertEqual(result["directional_accuracy"], 1)
        higher = np.nextafter(1.0, np.inf)
        self.assertEqual(calculate_metrics([higher], [1.0], [1.0])["directional_accuracy"], 0)
        self.assertEqual(calculate_metrics([higher], [higher], [1.0])["directional_accuracy"], 1)
        self.assertEqual(calculate_metrics([1.0], [higher], [1.0])["directional_accuracy"], 0)

    def test_naive_is_previous_close_and_direction_is_not_applicable(self):
        result = calculate_metrics([110, 90, 100], [100, 100, 100], [100, 100, 100], naive=True)
        self.assertAlmostEqual(result["mae"], 20 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(200 / 3))
        self.assertIsNone(result["directional_accuracy"])
        with self.assertRaisesRegex(MetricsError, "previous_close"):
            calculate_metrics([110], [100.0000001], [100], naive=True)

    def test_column_vectors_are_supported_without_broadcasting_or_mutation(self):
        actual, predicted = np.array([[1.0], [3.0]]), np.array([2.0, 2.0])
        original = actual.copy()
        result = calculate_metrics(actual, predicted, [[0.0], [4.0]])
        self.assertEqual(result, {"mae": 1.0, "rmse": 1.0, "directional_accuracy": 1.0})
        np.testing.assert_array_equal(actual, original)
        with self.assertRaisesRegex(MetricsError, "equal lengths"):
            calculate_metrics(actual, [2.0], [0.0, 4.0])

    def test_invalid_vectors_fail_with_clear_value_errors(self):
        invalid = ([], [[]], 3, None, [[1, 2]], [[1], [2, 3]],
                   np.zeros((2, 1, 1)), [float("nan")], [float("inf")],
                   [float("-inf")], ["1"], [True], [1 + 2j], [object()])
        for name in ("actual", "predicted", "previous_close"):
            for value in invalid:
                arguments = {"actual": [1], "predicted": [1], "previous_close": [1]}
                arguments[name] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(MetricsError, name):
                    calculate_metrics(**arguments)
        with self.assertRaisesRegex(MetricsError, "naive"):
            calculate_metrics([1], [1], [1], naive=1)

    def test_large_finite_errors_do_not_overflow_squaring_or_averaging(self):
        result = calculate_metrics([1e200, 1e200], [0, 0], [-1e200, -1e200])
        self.assertEqual(result, {"mae": 1e200, "rmse": 1e200, "directional_accuracy": 1.0})
        json.dumps(result, allow_nan=False)
        with self.assertRaisesRegex(MetricsError, "finite"):
            calculate_metrics([1e308], [-1e308], [0])

    def test_improvement_retains_negative_values_and_zero_baseline_is_none(self):
        self.assertEqual(error_improvement_pct(3, 4), 25)
        self.assertEqual(error_improvement_pct(6, 4), -50)
        self.assertEqual(error_improvement_pct(0, 4), 100)
        self.assertEqual(error_improvement_pct(4, 4), 0)
        self.assertIsNone(error_improvement_pct(0, 0))
        self.assertIsNone(error_improvement_pct(3, 0))
        for invalid in (-1, float("nan"), float("inf"), None, True, "1"):
            for arguments in ((invalid, 1), (1, invalid)):
                with self.subTest(arguments=arguments), self.assertRaises(MetricsError):
                    error_improvement_pct(*arguments)
        with self.assertRaisesRegex(MetricsError, "finite"):
            error_improvement_pct(1e308, 1e-308)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {"mae": 4, "rmse": 5, "directional_accuracy": None}
        self.rows = [
            {"group": "A", "seed": 7, "mae": 2, "rmse": 3, "directional_accuracy": 0.25},
            {"group": "A", "seed": 19, "mae": 4, "rmse": 5, "directional_accuracy": 0.75},
            {"group": "B", "seed": 7, "mae": 6, "rmse": 8, "directional_accuracy": 0},
        ]

    def test_group_means_sample_standard_deviations_and_baseline_improvements(self):
        original = copy.deepcopy((self.rows, self.baseline))
        summary = summarize_metrics(self.rows, self.baseline)
        self.assertEqual([row["group"] for row in summary], ["A", "B", "Naive"])
        row = summary[0]
        self.assertEqual(row["n_seeds"], 2)
        self.assertEqual(row["mae_mean"], 3)
        self.assertEqual(row["rmse_mean"], 4)
        self.assertAlmostEqual(row["mae_std"], math.sqrt(2))
        self.assertAlmostEqual(row["rmse_std"], math.sqrt(2))
        self.assertEqual(row["directional_accuracy_mean"], 0.5)
        self.assertAlmostEqual(row["directional_accuracy_std"], math.sqrt(0.125))
        self.assertEqual(row["mae_improvement_pct"], 25)
        self.assertEqual(row["rmse_improvement_pct"], 20)
        self.assertEqual(summary[1]["mae_improvement_pct"], -50)
        self.assertEqual(summary[1]["rmse_improvement_pct"], -60)
        self.assertEqual((self.rows, self.baseline), original)
        json.dumps(summary, allow_nan=False)

    def test_single_seed_and_naive_summary_have_explicit_none_values(self):
        row, baseline = summarize_metrics(self.rows[2:], self.baseline)
        self.assertEqual(row["n_seeds"], 1)
        for key in ("mae_std", "rmse_std", "directional_accuracy_std"):
            self.assertIsNone(row[key])
        self.assertEqual(baseline, {
            "group": "Naive", "n_seeds": 0, "mae_mean": 4.0, "mae_std": None,
            "rmse_mean": 5.0, "rmse_std": None,
            "directional_accuracy_mean": None, "directional_accuracy_std": None,
            "mae_improvement_pct": None, "rmse_improvement_pct": None,
        })

    def test_zero_baseline_returns_none_for_each_improvement(self):
        baseline = {"mae": 0, "rmse": 0, "directional_accuracy": None}
        summary = summarize_metrics(self.rows, baseline)
        for row in summary:
            self.assertIsNone(row["mae_improvement_pct"])
            self.assertIsNone(row["rmse_improvement_pct"])

    def test_duplicate_group_seed_is_rejected_but_same_seed_across_groups_is_valid(self):
        self.assertEqual(len(summarize_metrics(self.rows, self.baseline)), 3)
        with self.assertRaisesRegex(MetricsError, "duplicate group-seed"):
            summarize_metrics(self.rows + [self.rows[0].copy()], self.baseline)

    def test_invalid_model_metric_rows_are_rejected(self):
        for field, values in {
            "group": (None, "", " ", "Naive", 1),
            "seed": (None, -1, True, 1.2, "7"),
            "mae": (-1, float("inf"), float("nan"), None, True, "1"),
            "rmse": (-1, float("inf"), float("nan"), None, True),
            "directional_accuracy": (-0.01, 1.01, None, float("nan"), True),
        }.items():
            for value in values:
                row = dict(self.rows[0], **{field: value})
                with self.subTest(field=field, value=value), self.assertRaisesRegex(MetricsError, field):
                    summarize_metrics([row], self.baseline)
        for field in ("mae", "rmse", "directional_accuracy"):
            row = self.rows[0].copy()
            del row[field]
            with self.subTest(missing=field), self.assertRaisesRegex(MetricsError, field):
                summarize_metrics([row], self.baseline)
        for value in (None, [], {}, "rows", [None]):
            with self.subTest(rows=value), self.assertRaises(MetricsError):
                summarize_metrics(value, self.baseline)

    def test_invalid_naive_metrics_are_rejected(self):
        for baseline in (None, {}, {"mae": -1, "rmse": 5, "directional_accuracy": None},
                         dict(self.baseline, directional_accuracy=0)):
            with self.subTest(baseline=baseline), self.assertRaises(MetricsError):
                summarize_metrics(self.rows, baseline)

    def test_large_finite_means_and_standard_deviations_do_not_overflow(self):
        rows = [dict(self.rows[0], seed=seed, mae=1e200, rmse=1e200) for seed in (1, 2)]
        summary = summarize_metrics(rows, {"mae": 1e200, "rmse": 1e200, "directional_accuracy": None})
        self.assertEqual(summary[0]["mae_mean"], 1e200)
        self.assertEqual(summary[0]["mae_std"], 0)
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
