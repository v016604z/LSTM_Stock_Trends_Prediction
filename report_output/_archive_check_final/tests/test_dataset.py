import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lstm_comparison.config import load_config
from lstm_comparison.dataset import DatasetError, build_dataset
from lstm_comparison.features import build_feature_groups


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "research.json"


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        # An explicit synthetic calendar, not a claim about TWSE sessions.
        dates = pd.bdate_range("2022-06-01", "2024-04-30", name="Date")
        positions = np.arange(len(dates))
        close = 100 + positions / 4 + np.take([0, 2, -1, 3, 1, -2, 4], positions % 7)
        self.frame = pd.DataFrame({
            "Open": close - 0.25, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": 1000 + 3 * positions,
        }, index=dates)

    def build(self, frame=None, config=None):
        return build_dataset(self.frame if frame is None else frame,
                             self.config if config is None else config)

    def test_common_dates_shapes_and_source_are_preserved(self):
        original = self.frame.copy(deep=True)
        dataset = self.build()
        expected_targets = self.frame.index[93:]
        for name, width in (("A", 1), ("B", 5), ("C", 10)):
            group = dataset.groups[name]
            self.assertEqual(group.columns, tuple(self.config["features"]["groups"][name]))
            self.assertEqual(list(group.splits), ["train", "validation", "test"])
            for split_name, split in group.splits.items():
                bounds = self.config["split"][split_name]
                dates = expected_targets[(expected_targets >= bounds["start"]) & (expected_targets <= bounds["end"])]
                pd.testing.assert_index_equal(split.target_dates, dates, check_freq=False)
                self.assertEqual(split.X.shape, (len(dates), 60, width))
                for prices in (split.y, split.y_original, split.previous_close):
                    self.assertEqual(prices.shape, (len(dates), 1))
                    self.assertTrue(np.isfinite(prices).all())
                self.assertTrue(np.isfinite(split.X).all())
                reference = dataset.groups["A"].splits[split_name]
                pd.testing.assert_index_equal(split.target_dates, reference.target_dates)
                pd.testing.assert_index_equal(split.input_end_dates, reference.input_end_dates)
                np.testing.assert_array_equal(split.y, reference.y)
                np.testing.assert_array_equal(split.y_original, reference.y_original)
                np.testing.assert_array_equal(split.previous_close, reference.previous_close)
        pd.testing.assert_frame_equal(self.frame, original)

    def test_first_last_and_every_label_match_raw_source_positions(self):
        dataset = self.build()
        for name in ("A", "B"):
            group = dataset.groups[name]
            for split in group.splits.values():
                targets = self.frame.index.get_indexer(split.target_dates)
                positions = targets[:, None] - 60 + np.arange(60)
                columns = list(group.columns)
                expected_X = self.frame[columns].to_numpy()[positions]
                actual_X = group.x_scaler.inverse_transform(split.X.reshape(-1, len(columns))).reshape(split.X.shape)
                np.testing.assert_allclose(actual_X, expected_X, rtol=0, atol=1e-10)
                np.testing.assert_array_equal(split.y_original[:, 0], self.frame["Close"].to_numpy()[targets])
                np.testing.assert_array_equal(split.previous_close[:, 0], self.frame["Close"].to_numpy()[targets - 1])
                pd.testing.assert_index_equal(split.input_end_dates, self.frame.index[targets - 1])
        first = dataset.groups["A"].splits["train"]
        self.assertEqual(first.target_dates[0], self.frame.index[93])
        first_X = dataset.groups["A"].x_scaler.inverse_transform(first.X[0])[:, 0]
        np.testing.assert_allclose(first_X, self.frame["Close"].iloc[33:93])
        self.assertEqual(first.y_original[0, 0], self.frame["Close"].iloc[93])
        self.assertEqual(dataset.groups["A"].splits["test"].target_dates[-1], self.frame.index[-1])

    def test_year_boundaries_use_target_date_and_prior_year_history(self):
        dataset = self.build()
        group = dataset.groups["A"]
        for split_name, year in (("validation", 2023), ("test", 2024)):
            split = group.splits[split_name]
            expected_first = self.frame.index[self.frame.index.year == year][0]
            self.assertEqual(split.target_dates[0], expected_first)
            target = self.frame.index.get_loc(expected_first)
            self.assertEqual(split.input_end_dates[0], self.frame.index[target - 1])
            self.assertEqual(split.input_end_dates[0].year, year - 1)
            actual = group.x_scaler.inverse_transform(split.X[0])[:, 0]
            np.testing.assert_allclose(actual, self.frame["Close"].iloc[target - 60:target])
        train = group.splits["train"]
        self.assertEqual(train.target_dates[-1], self.frame.index[self.frame.index.year == 2022][-1])
        self.assertEqual(group.splits["validation"].target_dates[-1].year, 2023)
        # The fixture's first validation target is Monday: input ends Friday,
        # with no weekend rows invented by the builder.
        self.assertEqual((group.splits["validation"].target_dates[0]
                          - group.splits["validation"].input_end_dates[0]).days, 3)

    def test_scaler_fit_uses_unique_training_inputs_and_separate_targets(self):
        dataset = self.build()
        features = build_feature_groups(self.frame, self.config)
        train_targets = self.frame.index.get_indexer(dataset.groups["A"].splits["train"].target_dates)
        # With contiguous training samples, the union starts at warm-up 33
        # and ends BEFORE the last training target (which belongs only to y).
        fit_positions = np.arange(33, train_targets[-1])
        for name, group in dataset.groups.items():
            expected = features[name].iloc[fit_positions].to_numpy()
            np.testing.assert_allclose(group.x_scaler.data_min_, expected.min(axis=0))
            np.testing.assert_allclose(group.x_scaler.data_max_, expected.max(axis=0))
            self.assertEqual(group.x_scaler.n_samples_seen_, len(fit_positions))
            self.assertFalse(group.x_scaler.clip)
        expected_y = self.frame["Close"].iloc[train_targets].to_numpy()
        np.testing.assert_allclose(dataset.y_scaler.data_min_, [expected_y.min()])
        np.testing.assert_allclose(dataset.y_scaler.data_max_, [expected_y.max()])
        self.assertEqual(dataset.y_scaler.n_samples_seen_, len(train_targets))
        self.assertFalse(dataset.y_scaler.clip)
        self.assertNotEqual(dataset.groups["A"].x_scaler.data_min_[0], self.frame["Close"].iloc[0])

    def test_last_training_target_extreme_only_changes_y_fit(self):
        baseline = self.build()
        last_target = baseline.groups["A"].splits["train"].target_dates[-1]
        changed = self.frame.copy(deep=True)
        changed.loc[last_target, ["Open", "High", "Low", "Close"]] *= 100
        dataset = self.build(changed)
        self.assertEqual(dataset.y_scaler.data_max_[0], changed.loc[last_target, "Close"])
        self.assertNotEqual(dataset.y_scaler.data_max_[0], baseline.y_scaler.data_max_[0])
        for name, group in dataset.groups.items():
            original = baseline.groups[name]
            np.testing.assert_array_equal(group.x_scaler.data_min_, original.x_scaler.data_min_)
            np.testing.assert_array_equal(group.x_scaler.data_max_, original.x_scaler.data_max_)
            np.testing.assert_array_equal(group.splits["train"].X, original.splits["train"].X)
            np.testing.assert_array_equal(group.splits["train"].y_original[:-1], original.splits["train"].y_original[:-1])
            self.assertEqual(group.splits["train"].y_original[-1, 0], changed.loc[last_target, "Close"])
        first_validation = dataset.groups["A"].splits["validation"]
        self.assertEqual(first_validation.input_end_dates[0], last_target)
        self.assertGreater(first_validation.X[0, -1, 0], 1)
        self.assertEqual(first_validation.previous_close[0, 0], changed.loc[last_target, "Close"])

    def test_unseen_extremes_do_not_refit_or_clip_or_change_training(self):
        baseline = self.build()
        for multiplier in (100.0, 0.001):
            with self.subTest(multiplier=multiplier):
                changed = self.frame.copy(deep=True)
                future = changed.index.year >= 2023
                changed.loc[future, ["Open", "High", "Low", "Close"]] *= multiplier
                changed.loc[future, "Volume"] = 1 if multiplier < 1 else 10_000_000
                dataset = self.build(changed)
                for name, group in dataset.groups.items():
                    original = baseline.groups[name]
                    for attribute in ("data_min_", "data_max_", "scale_", "min_"):
                        np.testing.assert_array_equal(getattr(group.x_scaler, attribute), getattr(original.x_scaler, attribute))
                    np.testing.assert_array_equal(group.splits["train"].X, original.splits["train"].X)
                    np.testing.assert_array_equal(group.splits["train"].y, original.splits["train"].y)
                for attribute in ("data_min_", "data_max_", "scale_", "min_"):
                    np.testing.assert_array_equal(getattr(dataset.y_scaler, attribute), getattr(baseline.y_scaler, attribute))
                validation = dataset.groups["A"].splits["validation"]
                if multiplier > 1:
                    self.assertTrue((validation.X > 1).any())
                    self.assertTrue((validation.y > 1).any())
                else:
                    self.assertTrue((validation.X < 0).any())
                    self.assertTrue((validation.y < 0).any())
                np.testing.assert_allclose(dataset.y_scaler.inverse_transform(validation.y), validation.y_original)

    def test_inverse_scaling_all_groups_splits_and_constant_columns(self):
        constant = self.frame.copy(deep=True)
        constant.loc[:, ["Open", "High", "Low", "Close"]] = [99.75, 101.0, 99.0, 100.0]
        constant.loc[:, "Volume"] = 1000
        for frame in (self.frame, constant):
            dataset = self.build(frame)
            features = build_feature_groups(frame, self.config)
            for name, group in dataset.groups.items():
                for split in group.splits.values():
                    targets = frame.index.get_indexer(split.target_dates)
                    positions = targets[:, None] - 60 + np.arange(60)
                    raw_X = features[name].to_numpy()[positions]
                    restored_X = group.x_scaler.inverse_transform(split.X.reshape(-1, len(group.columns))).reshape(split.X.shape)
                    np.testing.assert_allclose(restored_X, raw_X, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(dataset.y_scaler.inverse_transform(split.y), split.y_original, rtol=0, atol=1e-10)

    def test_internal_invalid_feature_removes_whole_windows_without_stitching(self):
        groups = build_feature_groups(self.frame, self.config)
        missing = self.frame.index.get_loc(pd.Timestamp("2023-06-15"))
        groups["C"].iloc[missing, groups["C"].columns.get_loc("MA5")] = np.nan
        with patch("lstm_comparison.dataset.build_feature_groups", return_value=groups):
            dataset = self.build()
        expected_positions = [j for j in range(93, len(self.frame)) if not j - 60 <= missing < j]
        for group in dataset.groups.values():
            actual = pd.DatetimeIndex(np.concatenate([part.target_dates.to_numpy() for part in group.splits.values()]), name="Date")
            pd.testing.assert_index_equal(actual, self.frame.index[expected_positions])
            self.assertIn(self.frame.index[missing], actual)
            self.assertNotIn(self.frame.index[missing + 1], actual)
            self.assertNotIn(self.frame.index[missing + 60], actual)
            self.assertIn(self.frame.index[missing + 61], actual)

    def test_configurable_lookback_keeps_same_next_session_contract(self):
        config = copy.deepcopy(self.config)
        config["sequence"]["lookback"] = 3
        dataset = self.build(config=config)
        group = dataset.groups["A"]
        first = group.splits["train"]
        self.assertEqual(first.X.shape[1:], (3, 1))
        self.assertEqual(first.target_dates[0], self.frame.index[36])
        np.testing.assert_allclose(group.x_scaler.inverse_transform(first.X[0])[:, 0], self.frame["Close"].iloc[33:36])
        self.assertEqual(first.y_original[0, 0], self.frame["Close"].iloc[36])

    def test_no_common_samples_and_empty_partitions_have_clear_errors(self):
        for frame in (self.frame.iloc[:0], self.frame.iloc[:33], self.frame.iloc[:93]):
            with self.subTest(length=len(frame)), self.assertRaises(DatasetError):
                self.build(frame)
        for split_name, frame in (
            ("train", self.frame[self.frame.index.year >= 2023]),
            ("validation", self.frame[self.frame.index.year != 2023]),
            ("test", self.frame[self.frame.index.year < 2024]),
        ):
            with self.subTest(split=split_name), self.assertRaisesRegex(DatasetError, split_name):
                self.build(frame)

    def test_invalid_source_is_rejected_before_any_sample_filtering(self):
        for position in (0, 40, len(self.frame) - 1):
            invalid = self.frame.copy(deep=True)
            invalid.iloc[position, invalid.columns.get_loc("Close")] = np.nan
            with self.subTest(position=position), self.assertRaisesRegex(DatasetError, "缺值"):
                self.build(invalid)
        for lookback, horizon in ((0, 1), (True, 1), (1.5, 1), (60, 2)):
            invalid = copy.deepcopy(self.config)
            invalid["sequence"] = {"lookback": lookback, "horizon": horizon}
            with self.subTest(sequence=invalid["sequence"]), self.assertRaises(DatasetError):
                self.build(config=invalid)

    def test_persisted_scalers_are_transform_only_and_match_default_dataset(self):
        original = self.build()
        x_scalers = {name: group.x_scaler for name, group in original.groups.items()}
        with patch("sklearn.preprocessing.MinMaxScaler.fit", side_effect=AssertionError("Unexpected fit")):
            restored = build_dataset(self.frame, self.config, x_scalers=x_scalers, y_scaler=original.y_scaler)
        self.assertIs(restored.y_scaler, original.y_scaler)
        for name, group in restored.groups.items():
            self.assertIs(group.x_scaler, original.groups[name].x_scaler)
            for split_name, split in group.splits.items():
                expected = original.groups[name].splits[split_name]
                np.testing.assert_array_equal(split.X, expected.X)
                np.testing.assert_array_equal(split.y, expected.y)
                pd.testing.assert_index_equal(split.target_dates, expected.target_dates)

    def test_persisted_scalers_reject_missing_groups_wrong_ranges_and_settings(self):
        dataset = self.build()
        scalers = {name: group.x_scaler for name, group in dataset.groups.items()}
        cases = [(None, dataset.y_scaler), (scalers, None), ({"A": scalers["A"]}, dataset.y_scaler)]
        for attribute, value in (("clip", True), ("n_features_in_", 99),
                                 ("data_min_", np.array([-999.])), ("scale_", np.array([999.]))):
            changed = copy.deepcopy(scalers)
            setattr(changed["A"], attribute, value)
            cases.append((changed, dataset.y_scaler))
        invalid_y = copy.deepcopy(dataset.y_scaler)
        invalid_y.data_max_ = np.array([9999.])
        cases.append((scalers, invalid_y))
        for x_scalers, y_scaler in cases:
            with self.subTest(x_type=type(x_scalers).__name__), self.assertRaises(DatasetError):
                build_dataset(self.frame, self.config, x_scalers=x_scalers, y_scaler=y_scaler)

    def test_persisted_copy_false_scalers_preserve_original_prices_and_windows(self):
        expected = self.build()
        original = self.frame.copy(deep=True)
        x_scalers = {name: copy.deepcopy(group.x_scaler) for name, group in expected.groups.items()}
        y_scaler = copy.deepcopy(expected.y_scaler)
        for scaler in (*x_scalers.values(), y_scaler):
            scaler.copy = False
        with patch("sklearn.preprocessing.MinMaxScaler.fit", side_effect=AssertionError("Unexpected fit")):
            actual = build_dataset(self.frame, self.config, x_scalers=x_scalers, y_scaler=y_scaler)
        pd.testing.assert_frame_equal(self.frame, original)
        for name, group in actual.groups.items():
            for split_name, split in group.splits.items():
                wanted = expected.groups[name].splits[split_name]
                np.testing.assert_array_equal(split.X, wanted.X)
                np.testing.assert_array_equal(split.y, wanted.y)
                np.testing.assert_array_equal(split.y_original, wanted.y_original)
                np.testing.assert_array_equal(split.previous_close, wanted.previous_close)


if __name__ == "__main__":
    unittest.main()
