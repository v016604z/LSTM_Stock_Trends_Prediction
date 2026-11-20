from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from lstm_comparison.config import load_config
from lstm_comparison.data import SnapshotError
from lstm_comparison.features import build_feature_groups


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "research.json"


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def frame(self, closes=None):
        if closes is None:
            pattern = [0, 2, -1, 3, 1, -2, 4]
            closes = [100 + i / 4 + pattern[i % 7] for i in range(60)]
        closes = np.asarray(closes, dtype=float)
        return pd.DataFrame({
            "Open": closes - 0.25, "High": closes + 1, "Low": closes - 1,
            "Close": closes, "Volume": np.arange(1000, 1000 + len(closes)),
        }, index=pd.bdate_range("2020-01-02", periods=len(closes), name="Date"))

    def features(self, frame):
        return build_feature_groups(frame, self.config)

    def test_group_columns_indices_values_and_input_are_preserved(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        groups = self.features(frame)
        self.assertEqual(list(groups), ["A", "B", "C"])
        self.assertEqual([len(group.columns) for group in groups.values()], [1, 5, 10])
        for name, columns in self.config["features"]["groups"].items():
            self.assertEqual(groups[name].columns.tolist(), columns)
            pd.testing.assert_index_equal(groups[name].index, original.index)
        pd.testing.assert_frame_equal(groups["A"], original[["Close"]])
        pd.testing.assert_frame_equal(groups["B"], original)
        pd.testing.assert_frame_equal(groups["C"].iloc[:, :5], original)
        pd.testing.assert_frame_equal(frame, original)
        # Returned groups must not share mutable data with the input or each other.
        groups["B"].iloc[0, 3] = 9999.0
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(groups["A"].iloc[0, 0], original.iloc[0, 3])
        self.assertEqual(groups["C"].iloc[0, 3], original.iloc[0, 3])

    def test_ma_uses_full_trailing_window_including_today(self):
        features = self.features(self.frame(range(100, 160)))["C"]
        self.assertAlmostEqual(features["MA5"].iloc[4], 102.0)
        self.assertAlmostEqual(features["MA5"].iloc[5], 103.0)
        self.assertAlmostEqual(features["MA20"].iloc[19], 109.5)
        self.assertAlmostEqual(features["MA20"].iloc[20], 110.5)

    def test_rsi_matches_independent_decimal_reference(self):
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
                  45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
                  46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21]
        # Golden values calculated with 50-digit Decimal arithmetic, using the
        # first 14 changes' arithmetic means and Wilder's 13/14 recurrence.
        expected = [70.46413502109705, 66.24961855355508, 66.48094183471267,
                    69.34685316290870, 66.29471265892625, 57.91502067008556,
                    62.88071830996239]
        actual = self.features(self.frame(closes))["C"]["RSI14"]
        np.testing.assert_allclose(actual.iloc[14:], expected, rtol=0, atol=1e-10)

    def test_macd_matches_independent_decimal_reference(self):
        # Golden values use 50-digit Decimal arithmetic on the exact quarter-
        # price fixture, SMA-seeded EMA12/26 and SMA-seeded signal EMA9.
        features = self.features(self.frame())["C"]
        positions = [25, 26, 33, 34, 45, 59]
        expected_macd = [1.8616505598558092, 1.6051573113309838,
                         1.5882952296287411, 1.8683165121745595,
                         1.8421711853902422, 1.8338144553491245]
        expected_signal = [1.7751390814134576, 1.7937745675656780,
                           1.7760727242682261, 1.7638749715071185]
        np.testing.assert_allclose(features["MACD"].iloc[positions], expected_macd, rtol=0, atol=1e-12)
        np.testing.assert_allclose(features["MACD_signal"].iloc[positions[2:]], expected_signal, rtol=0, atol=1e-12)

    def test_exact_warmup_and_short_histories(self):
        warmups = {"MA5": 4, "MA20": 19, "RSI14": 14, "MACD": 25, "MACD_signal": 33}
        source = self.frame()
        for length in (1, 4, 5, 14, 15, 19, 20, 25, 26, 33, 34, 60):
            with self.subTest(length=length):
                groups = self.features(source.iloc[:length])
                self.assertEqual(len(groups["C"]), length)
                self.assertFalse(groups["A"].isna().any().any())
                self.assertFalse(groups["B"].isna().any().any())
                for column, warmup in warmups.items():
                    values = groups["C"][column]
                    self.assertTrue(values.iloc[:warmup].isna().all())
                    self.assertTrue(np.isfinite(values.iloc[warmup:]).all())
                self.assertEqual(groups["C"].notna().all(axis=1).sum(), max(0, length - 33))

    def test_constant_rising_and_falling_prices(self):
        for closes, expected_rsi, expected_macd in (
            ([100.0] * 60, 50.0, 0.0),
            (list(range(100, 160)), 100.0, 7.0),
            (list(range(300, 240, -1)), 0.0, -7.0),
        ):
            with self.subTest(expected_rsi=expected_rsi):
                features = self.features(self.frame(closes))["C"]
                np.testing.assert_allclose(features["RSI14"].iloc[14:], expected_rsi)
                np.testing.assert_allclose(features["MACD"].iloc[25:], expected_macd, atol=1e-12)
                np.testing.assert_allclose(features["MACD_signal"].iloc[33:], expected_macd, atol=1e-12)

    def test_changing_future_values_does_not_change_historical_features(self):
        source = self.frame()
        changed = source.copy(deep=True)
        changed.iloc[40:, :4] *= 11
        changed.iloc[40:, 4] *= 7
        original_groups, changed_groups = self.features(source), self.features(changed)
        for name in original_groups:
            pd.testing.assert_frame_equal(original_groups[name].iloc[:40], changed_groups[name].iloc[:40])
        self.assertNotEqual(original_groups["C"]["MACD"].iloc[-1], changed_groups["C"]["MACD"].iloc[-1])

    def test_appending_future_rows_does_not_change_historical_features(self):
        source = self.frame()
        for length in (14, 26, 34, 40):
            with self.subTest(length=length):
                short_groups, full_groups = self.features(source.iloc[:length]), self.features(source)
                for name in short_groups:
                    pd.testing.assert_frame_equal(short_groups[name], full_groups[name].iloc[:length])

    def test_invalid_ohlcv_is_rejected_without_filling_or_sorting(self):
        source = self.frame()
        duplicate_columns = source.copy()
        duplicate_columns.columns = ["Open", "Open", "Low", "Close", "Volume"]
        invalid = [None, pd.DataFrame(), source.drop(columns="Volume"),
                   source.iloc[::-1], pd.concat([source, source.iloc[:1]]), duplicate_columns]
        for column, value in (("Close", np.nan), ("Open", np.inf), ("Low", 0.0),
                              ("High", 1.0), ("Volume", 0), ("Volume", 1.5)):
            damaged = source.copy()
            damaged[column] = damaged[column].astype(float)
            damaged.iloc[10, damaged.columns.get_loc(column)] = value
            invalid.append(damaged)
        for frame in invalid:
            with self.subTest(frame_type=type(frame).__name__), self.assertRaises(SnapshotError):
                self.features(frame)

    def test_invalid_date_indices_are_rejected(self):
        source = self.frame()
        for index in (pd.RangeIndex(len(source)), source.index.tz_localize("Asia/Taipei"),
                      source.index + pd.Timedelta(hours=1),
                      pd.bdate_range("2030-01-01", periods=len(source))):
            invalid = source.copy()
            invalid.index = index
            with self.subTest(index_type=type(index).__name__), self.assertRaises(SnapshotError):
                self.features(invalid)


if __name__ == "__main__":
    unittest.main()
