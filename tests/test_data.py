import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lstm_comparison.data import (
    COLUMNS, SnapshotError, _calendar_sessions, download_snapshot,
    load_snapshot, normalize_ohlcv,
)
from lstm_comparison.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "prices.csv"
        self.metadata_path = self.path.with_suffix(".metadata.json")
        self.sessions_path = self.path.with_suffix(".sessions.csv")
        self.config = {"data": {
            "ticker": "2330.TW", "start": "2024-01-01", "end": "2024-01-07",
            "end_exclusive": "2024-01-08", "interval": "1d", "auto_adjust": True,
            "snapshot_path": str(self.path),
        }}
        # New Year's Day and the final weekend are deliberately absent.
        self.sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], name="Date").as_unit("ns")
        self.frame = pd.DataFrame({
            "Open": [100.12345678912345 + i for i in range(4)],
            "High": [105.0 + i for i in range(4)],
            "Low": [99.0 + i for i in range(4)],
            "Close": [103.2 + i for i in range(4)],
            "Volume": [1000 + i for i in range(4)],
        }, index=self.sessions)
        self.calendar_info = {
            "name": "XTAI", "version": "4.13.3", "source": "https://example.test/calendar",
            "additional_closures": [], "additional_sessions": [],
            "requested_start": "2024-01-01", "requested_end": "2024-01-07",
        }

    def freeze(self, raw=None):
        with patch("yfinance.download", return_value=self.frame if raw is None else raw) as fetch, patch(
            "lstm_comparison.data._calendar_sessions", return_value=(self.sessions, self.calendar_info)
        ), patch("yfinance.set_tz_cache_location"):
            frame, metadata = download_snapshot(self.config)
        return frame, metadata, fetch

    def edit_metadata(self, change):
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        change(metadata)
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_roundtrip_preserves_source_values_metadata_and_lf(self):
        frame, metadata, fetch = self.freeze()
        pd.testing.assert_frame_equal(frame, self.frame)
        pd.testing.assert_frame_equal(load_snapshot(self.config)[0], self.frame)
        self.assertEqual(metadata["data"]["row_count"], 4)
        self.assertEqual(metadata["data"]["first_date"], "2024-01-02")
        self.assertEqual(metadata["data"]["last_date"], "2024-01-05")
        self.assertEqual(sha256(self.path.read_bytes()).hexdigest(), metadata["data"]["sha256"])
        self.assertEqual(sha256(self.sessions_path.read_bytes()).hexdigest(), metadata["calendar"]["sha256"])
        self.assertNotIn(b"\r\n", self.path.read_bytes())
        self.assertNotIn(b"\r\n", self.sessions_path.read_bytes())
        self.assertEqual(self.path.read_text().splitlines()[0], "Date,Open,High,Low,Close,Volume")
        parameters = fetch.call_args.kwargs
        self.assertIs(parameters["auto_adjust"], True)
        self.assertIs(parameters["keepna"], True)
        self.assertIs(parameters["repair"], False)
        self.assertEqual(parameters["end"], "2024-01-08")

    def test_multiindex_layouts_and_unsorted_dates(self):
        for ticker_level in (0, 1):
            with self.subTest(ticker_level=ticker_level):
                raw = self.frame.iloc[::-1].copy()
                raw.columns = pd.MultiIndex.from_tuples([(name, "2330.TW") for name in COLUMNS])
                if ticker_level == 0:
                    raw.columns = raw.columns.swaplevel()
                pd.testing.assert_frame_equal(normalize_ohlcv(raw, "2330.TW"), self.frame)

    def test_aware_dates_use_taipei_trading_day(self):
        raw = self.frame.copy()
        raw.index = raw.index.tz_localize("Asia/Taipei").tz_convert("UTC")
        pd.testing.assert_frame_equal(normalize_ohlcv(raw, "2330.TW"), self.frame)

    def test_source_schema_and_duplicate_dates_fail(self):
        multiple = self.frame.copy()
        multiple.columns = pd.MultiIndex.from_tuples([(name, "2330.TW" if i else "2317.TW") for i, name in enumerate(COLUMNS)])
        duplicate_columns = self.frame.copy()
        duplicate_columns.columns = ["Open", "Open", "Low", "Close", "Volume"]
        for raw in (None, pd.DataFrame(), self.frame.drop(columns="Volume"),
                    duplicate_columns, multiple, pd.concat([self.frame, self.frame.iloc[:1]])):
            with self.subTest(raw_type=type(raw).__name__), self.assertRaises(SnapshotError):
                normalize_ohlcv(raw, "2330.TW")

    def test_invalid_values_fail_before_publication(self):
        for column, value in (("Open", np.nan), ("Close", np.inf), ("Low", 0),
                              ("High", 1.0), ("Low", 1000.0), ("Volume", -1), ("Volume", 0), ("Volume", 1.5)):
            with self.subTest(column=column, value=value):
                raw = self.frame.copy()
                raw[column] = raw[column].astype(float)
                raw.loc[raw.index[0], column] = value
                with self.assertRaises(SnapshotError):
                    self.freeze(raw)
                self.assertFalse(any(item.exists() for item in (self.path, self.sessions_path, self.metadata_path)))

    def test_missing_internal_boundary_and_extra_dates_fail(self):
        for missing_date in self.sessions:
            with self.subTest(missing_date=missing_date), self.assertRaisesRegex(SnapshotError, "交易日不完整"):
                self.freeze(self.frame.drop(missing_date))
        extra = self.frame.iloc[:1].copy()
        extra.index = pd.DatetimeIndex(["2024-01-01"], name="Date")
        with self.assertRaisesRegex(SnapshotError, "多出 1 日"):
            self.freeze(pd.concat([extra, self.frame]))
        outside = self.frame.copy()
        outside.index = pd.date_range("2023-12-28", periods=4, name="Date")
        with self.assertRaisesRegex(SnapshotError, "超出"):
            self.freeze(outside)

    def test_incomplete_source_reports_missing_extra_and_nan_without_freezing(self):
        raw = self.frame.drop(self.sessions[1]).copy()
        raw.loc[self.sessions[2], "Close"] = np.nan
        placeholder = self.frame.iloc[:1].copy()
        placeholder.index = pd.DatetimeIndex(["2024-01-01"], name="Date")
        raw = pd.concat([placeholder, raw])
        with self.assertRaises(SnapshotError) as caught:
            self.freeze(raw)
        for message in ("缺少 1 日", "2024-01-03", "多出 1 日", "2024-01-01", "缺值或非有限", "2024-01-04"):
            self.assertIn(message, str(caught.exception))
        self.assertFalse(any(item.exists() for item in (self.path, self.sessions_path, self.metadata_path)))

    def test_existing_snapshot_is_offline_immutable_and_movable(self):
        _, metadata, _ = self.freeze()
        original = {path.name: path.read_bytes() for path in (self.path, self.metadata_path, self.sessions_path)}
        moved = Path(self.directory.name) / "moved"
        moved.mkdir()
        for name, content in original.items():
            (moved / name).write_bytes(content)
        config = copy.deepcopy(self.config)
        config["data"]["snapshot_path"] = str(moved / self.path.name)
        config["training"] = {"seeds": [1], "max_epochs": 1}
        with patch("yfinance.download", side_effect=AssertionError("network")), patch(
            "lstm_comparison.data._calendar_sessions", side_effect=AssertionError("regenerated calendar")
        ):
            loaded, saved_metadata = download_snapshot(config)
        pd.testing.assert_frame_equal(loaded, self.frame)
        self.assertEqual(saved_metadata, metadata)
        for name, content in original.items():
            self.assertEqual((moved / name).read_bytes(), content)

    def test_partial_bundle_is_not_overwritten_or_downloaded(self):
        self.path.write_bytes(b"user owned partial file")
        with patch("yfinance.download", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(SnapshotError, "不完整"):
                download_snapshot(self.config)
        self.assertEqual(self.path.read_bytes(), b"user owned partial file")

    def test_checksum_metadata_and_source_mismatches_fail(self):
        self.freeze()
        original = self.path.read_bytes()
        self.path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(SnapshotError, "SHA256"):
            load_snapshot(self.config)
        self.path.write_bytes(original)
        sessions = self.sessions_path.read_bytes()
        self.sessions_path.write_bytes(sessions + b"\n")
        with self.assertRaisesRegex(SnapshotError, "SHA256"):
            load_snapshot(self.config)
        self.sessions_path.write_bytes(sessions)
        metadata_bytes = self.metadata_path.read_bytes()
        for change in (
            lambda metadata: metadata["data"].update(row_count=99),
            lambda metadata: metadata["calendar"].update(last_date="2024-01-04"),
            lambda metadata: metadata["calendar"].update(additional_closures=[{}]),
            lambda metadata: metadata["source"]["parameters"].update(auto_adjust=False),
            lambda metadata: metadata.pop("versions"),
        ):
            with self.subTest(change=change):
                self.metadata_path.write_bytes(metadata_bytes)
                self.edit_metadata(change)
                with self.assertRaises(SnapshotError):
                    load_snapshot(self.config)
        self.metadata_path.write_bytes(b"{broken json")
        with self.assertRaises(SnapshotError):
            load_snapshot(self.config)

    def test_write_failure_cleans_only_this_publication(self):
        real_link = os.link
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PermissionError("simulated disk/publication failure")
            return real_link(source, destination)

        with patch("lstm_comparison.data.os.link", side_effect=fail_second):
            with self.assertRaisesRegex(SnapshotError, "simulated"):
                self.freeze()
        self.assertFalse(any(item.exists() for item in (self.path, self.sessions_path, self.metadata_path)))
        self.assertEqual(list(self.path.parent.glob(".snapshot-*")), [])

    def test_source_exception_and_empty_data_do_not_create_snapshot(self):
        with patch("yfinance.download", side_effect=TimeoutError("source unavailable")), patch(
            "lstm_comparison.data._calendar_sessions", return_value=(self.sessions, self.calendar_info)
        ), patch("yfinance.set_tz_cache_location"):
            with self.assertRaisesRegex(SnapshotError, "source unavailable"):
                download_snapshot(self.config)
        with self.assertRaises(SnapshotError):
            self.freeze(pd.DataFrame())
        self.assertFalse(self.path.exists())

    def test_non_csv_destination_is_rejected(self):
        self.config["data"]["snapshot_path"] = str(self.path.with_suffix(".json"))
        with self.assertRaisesRegex(SnapshotError, ".csv"):
            download_snapshot(self.config)


class CalendarTests(unittest.TestCase):
    def test_xtai_sessions_include_documented_typhoon_patch(self):
        sessions, metadata = _calendar_sessions("2015-01-01", "2025-12-31")
        self.assertEqual(len(sessions), 2682)
        self.assertEqual(sessions[0], pd.Timestamp("2015-01-05"))
        self.assertEqual(sessions[-1], pd.Timestamp("2025-12-31"))
        for day in ("2022-02-04", "2023-01-18", "2024-07-24", "2024-07-25", "2024-10-02", "2024-10-03", "2024-10-31"):
            self.assertNotIn(pd.Timestamp(day), sessions)
        for day in ("2016-01-30", "2016-06-04", "2016-09-10", "2017-02-18", "2017-06-03",
                    "2017-09-30", "2018-03-31", "2018-12-22"):
            self.assertIn(pd.Timestamp(day), sessions)
        self.assertIn(pd.Timestamp("2025-11-12"), sessions)
        self.assertEqual(metadata["version"], "4.13.3")
        self.assertEqual([item["date"] for item in metadata["additional_closures"]], ["2022-02-04", "2023-01-18", "2024-10-31"])
        self.assertEqual(len(metadata["additional_sessions"]), 8)
        self.assertIn("twse.com.tw", metadata["additional_closures"][0]["source"])


class TwseSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs" / "research_twse.json")
        self.frame, self.metadata = load_snapshot(self.config)

    def test_frozen_twse_bundle_loads_offline(self):
        self.assertEqual(len(self.frame), 2682)
        self.assertEqual(self.frame.index[0], pd.Timestamp("2015-01-05"))
        self.assertEqual(self.frame.index[-1], pd.Timestamp("2025-12-31"))
        self.assertEqual(self.metadata["source"]["name"], "TWSE_STOCK_DAY")
        self.assertEqual(self.metadata["source"]["price_basis"], "unadjusted")
        self.assertEqual(self.metadata["source"]["volume_unit"], "shares")

    def test_twse_source_and_hash_errors_fail_without_touching_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            original_path = Path(self.config["data"]["snapshot_path"])
            paths = (original_path, original_path.with_suffix(".metadata.json"), original_path.with_suffix(".sessions.csv"))
            snapshot_path = Path(directory) / "prices.csv"
            copied = (snapshot_path, snapshot_path.with_suffix(".metadata.json"), snapshot_path.with_suffix(".sessions.csv"))
            for source, target in zip(paths, copied):
                target.write_bytes(source.read_bytes())
            config = copy.deepcopy(self.config)
            config["data"]["snapshot_path"] = str(snapshot_path)

            for change in (
                lambda metadata: metadata["source"].update(price_basis="adjusted"),
                lambda metadata: metadata["source"]["parameters"].update(stockNo="9999"),
                lambda metadata: metadata["source"].update(agency="wrong"),
            ):
                metadata_path = copied[1]
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                change(metadata)
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.subTest(change=change), self.assertRaises(SnapshotError):
                    load_snapshot(config)
                metadata_path.write_bytes(paths[1].read_bytes())

            copied[0].write_bytes(copied[0].read_bytes() + b"\\n")
            with self.assertRaisesRegex(SnapshotError, "SHA256"):
                load_snapshot(config)


if __name__ == "__main__":
    unittest.main()
