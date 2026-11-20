"""Validate immutable OHLCV snapshots from Yahoo or the TWSE official API."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd


COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
ADDITIONAL_CLOSURES = [{
    "date": "2022-02-04",
    "reason": "Lunar New Year closure omitted by XTAI 4.13.3",
    "source": "https://www.twse.com.tw/staticFiles/news/news/tsecnews/ff8080817d22b9cb017e3336c4e203e3.pdf",
}, {
    "date": "2023-01-18",
    "reason": "Pre-Lunar-New-Year settlement day; no TWSE trading after January 17 until January 30",
    "source": "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=20230101&response=json",
    "stock_source": "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20230101&response=json&stockNo=2330",
}, {
    "date": "2024-10-31",
    "reason": "Typhoon Kong-rey; omitted by exchange_calendars 4.13.3 XTAI",
    "source": "https://wwwc.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d69236c2e30192dd523eb10329.pdf",
}]
ADDITIONAL_SESSIONS = [{
    "date": day, "reason": "Saturday session with positive TWSE 2330 volume and valid OHLC",
    "source": f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={month}&response=json&stockNo=2330",
} for day, month in (
    ("2016-01-30", "20160101"), ("2016-06-04", "20160601"),
    ("2016-09-10", "20160901"), ("2017-02-18", "20170201"),
    ("2017-06-03", "20170601"),
    ("2017-09-30", "20170901"), ("2018-03-31", "20180301"),
    ("2018-12-22", "20181201"),
)]


class SnapshotError(ValueError):
    """A download or a frozen snapshot fails the research data contract."""


def _paths(config):
    path = Path(config["data"]["snapshot_path"])
    if path.suffix.lower() != ".csv":
        raise SnapshotError("data.snapshot_path 必須使用 .csv 副檔名")
    return path, path.with_suffix(".metadata.json"), path.with_suffix(".sessions.csv")


def _parameters(config):
    data = config["data"]
    return {
        "tickers": data["ticker"], "start": data["start"],
        "end": data["end_exclusive"], "interval": data["interval"],
        "auto_adjust": data["auto_adjust"], "back_adjust": False,
        "repair": False, "actions": False, "keepna": True,
        "ignore_tz": True, "threads": False, "progress": False,
        "rounding": False, "timeout": 30, "multi_level_index": True,
    }



def _source_mode(config):
    """A missing source retains compatibility with existing Yahoo configs."""
    return config["data"].get("source", "yfinance")


def _twse_parameters(config):
    data = config["data"]
    ticker = data["ticker"]
    if not ticker.endswith(".TW") or not ticker[:-3].isdigit() or len(ticker[:-3]) != 4:
        raise SnapshotError("TWSE 股票代號必須為數字加 .TW，例如 2330.TW")
    return {
        "tickers": ticker,
        "start": data["start"],
        "end": data["end_exclusive"],
        "interval": data["interval"],
        "auto_adjust": False,
        "stockNo": ticker[:-3],
    }


def _check_twse_source(metadata, config):
    """Check official-source parameters and per-month provenance."""
    from urllib.parse import parse_qs, urlparse

    source = metadata["source"]
    expected_params = _twse_parameters(config)
    base_url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
    if (source.get("name") != "TWSE_STOCK_DAY"
            or source.get("agency") != "臺灣證券交易所"
            or source.get("base_url") != base_url
            or source.get("price_basis") != "unadjusted"
            or source.get("volume_unit") != "shares"
            or source.get("parameters") != expected_params):
        raise SnapshotError("TWSE 來源、原始股價／成交股數單位或下載參數與設定不符")
    flags = metadata.get("validation", {})
    if (not isinstance(flags, dict) or any(flags.get(flag) is not True for flag in (
            "all_expected_sessions_present", "no_unexpected_sessions",
            "unique_sorted_dates", "all_ohlcv_valid"))):
        raise SnapshotError("TWSE 快照 metadata 缺少完整資料品質驗證紀錄")

    # Every month must have the raw official JSON's SHA256 and original URL.
    start_month = pd.Period(config["data"]["start"], freq="M")
    end_month = pd.Period(config["data"]["end"], freq="M")
    expected_months = [str(period).replace("-", "") for period in pd.period_range(start_month, end_month, freq="M")]
    requests = source.get("requests")
    if not isinstance(requests, list) or len(requests) != len(expected_months):
        raise SnapshotError("TWSE 每月行情的來源稽核紀錄數量不符")
    for requested, item in zip(expected_months, requests):
        if not isinstance(item, dict) or item.get("month") != requested:
            raise SnapshotError("TWSE 月份來源紀錄不連續或順序不正確")
        digest = item.get("cache_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdefABCDEF" for ch in digest)):
            raise SnapshotError("TWSE 原始 JSON 快取 SHA256 格式不正確")
        if item.get("source") not in ("download", "cache"):
            raise SnapshotError("TWSE 原始月份來源類型不正確")
        url = item.get("url")
        if not isinstance(url, str) or urlparse(url).scheme + "://" + urlparse(url).netloc + urlparse(url).path != base_url:
            raise SnapshotError("TWSE 來源 URL 非官方 STOCK_DAY API")
        query = parse_qs(urlparse(url).query)
        if (query.get("date") != [requested + "01"]
                or query.get("stockNo") != [expected_params["stockNo"]]
                or query.get("response") != ["json"]):
            raise SnapshotError("TWSE 月度 API 查詢參數與快照設定不符")

def _dates(values):
    dates = pd.DatetimeIndex(values)
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Taipei").tz_localize(None)
    dates = dates.normalize().as_unit("ns")
    dates.freq = None
    return dates.rename("Date")


def normalize_ohlcv(raw, ticker):
    """Flatten one ticker and keep source OHLCV values without filling rows."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise SnapshotError("來源未回傳行情資料，可能為連線、限流或來源錯誤")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels != 2:
            raise SnapshotError("來源 MultiIndex 欄位必須只有價格及單一股票兩層")
        ticker_levels = [level for level in (0, 1)
                         if set(frame.columns.get_level_values(level)) == {ticker}]
        if len(ticker_levels) != 1:
            raise SnapshotError("來源必須只包含指定的單一股票，不可混入多股或錯誤代號")
        frame.columns = frame.columns.droplevel(ticker_levels[0])
    if frame.columns.has_duplicates or not set(COLUMNS).issubset(frame.columns):
        raise SnapshotError(f"來源欄位缺漏或重複，必須包含 {COLUMNS}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise SnapshotError("來源行情必須使用日期索引")
    frame = frame[COLUMNS].copy()
    frame.index = _dates(frame.index)
    if frame.index.isna().any() or frame.index.has_duplicates:
        raise SnapshotError("來源日期無效或重複")
    return frame.sort_index()


def _calendar_sessions(start, end):
    import exchange_calendars as calendars

    # Padding lets the requested boundaries be holidays, not necessarily sessions.
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    calendar = calendars.get_calendar(
        "XTAI", start=first - timedelta(days=31), end=last + timedelta(days=31)
    )
    sessions = _dates(calendar.sessions)
    sessions = sessions[(sessions >= first) & (sessions <= last)]
    patches = [item for item in ADDITIONAL_CLOSURES if start <= item["date"] <= end]
    additions = [item for item in ADDITIONAL_SESSIONS if start <= item["date"] <= end]
    sessions = sessions.difference(pd.DatetimeIndex([item["date"] for item in patches]))
    sessions = sessions.union(pd.DatetimeIndex([item["date"] for item in additions])).sort_values()
    return sessions.rename("Date"), {
        "name": "XTAI", "version": version("exchange-calendars"),
        "source": "https://pypi.org/project/exchange_calendars/4.13.3/",
        "additional_closures": patches, "additional_sessions": additions,
        "requested_start": start, "requested_end": end,
    }


def validate_ohlcv(frame, sessions, start, end):
    """Reject damaged values and every missing or extra expected trading day."""
    if frame.empty or list(frame.columns) != COLUMNS:
        raise SnapshotError("行情不可為空且 OHLCV 欄位必須符合固定順序")
    if frame.index.isna().any() or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise SnapshotError("行情日期必須有效、唯一並已排序")
    if (frame.index < pd.Timestamp(start)).any() or (frame.index > pd.Timestamp(end)).any():
        raise SnapshotError("行情日期超出指定資料期間")
    if (sessions.empty or sessions.isna().any() or sessions.has_duplicates
            or not sessions.is_monotonic_increasing
            or (sessions < pd.Timestamp(start)).any() or (sessions > pd.Timestamp(end)).any()):
        raise SnapshotError("預期交易日檔案的日期、順序或期間無效")
    missing, extra = sessions.difference(frame.index), frame.index.difference(sessions)
    problems = []
    if len(missing) or len(extra):
        problems.append(
            f"交易日不完整：缺少 {len(missing)} 日 {missing.strftime('%Y-%m-%d').tolist()[:10]}；"
            f"多出 {len(extra)} 日 {extra.strftime('%Y-%m-%d').tolist()[:10]}"
        )
    for column in COLUMNS:
        if (not pd.api.types.is_numeric_dtype(frame[column])
                or pd.api.types.is_bool_dtype(frame[column])):
            raise SnapshotError(f"{column} 必須是數值")
    values = frame.to_numpy(dtype=float)
    invalid = frame.index[~np.isfinite(values).all(axis=1)]
    if len(invalid):
        problems.append(f"缺值或非有限數值 {len(invalid)} 日 {invalid.strftime('%Y-%m-%d').tolist()[:10]}")
    nonpositive_volume = frame.index[frame["Volume"].le(0)]
    if len(nonpositive_volume):
        problems.append(f"Volume 必須大於零，異常日期 {nonpositive_volume.strftime('%Y-%m-%d').tolist()[:15]}")
    if problems:
        raise SnapshotError("；".join(problems))
    prices = frame[COLUMNS[:4]]
    if (prices <= 0).any().any():
        raise SnapshotError("OHLC 價格必須大於零")
    if ((frame["High"] < prices.max(axis=1)).any()
            or (frame["Low"] > prices.min(axis=1)).any()):
        raise SnapshotError("High/Low 不符合當日 OHLC 價格範圍")
    volume = frame["Volume"].to_numpy(dtype=float)
    if not np.equal(volume, np.floor(volume)).all():
        raise SnapshotError("Volume 必須是來源的正整數，不得套用價格調整因子")


def _summary(index):
    return {"row_count": len(index), "first_date": index[0].strftime("%Y-%m-%d"),
            "last_date": index[-1].strftime("%Y-%m-%d")}


def _read_dates(frame):
    if "Date" not in frame.columns or frame["Date"].isna().any():
        raise SnapshotError("CSV 缺少有效的 Date 欄位")
    original = frame.pop("Date")
    dates = pd.DatetimeIndex(pd.to_datetime(original, format="%Y-%m-%d", errors="raise"))
    if dates.strftime("%Y-%m-%d").tolist() != original.tolist():
        raise SnapshotError("CSV 日期必須使用 ISO YYYY-MM-DD")
    return dates.as_unit("ns").rename("Date")


def load_snapshot(config):
    """Load one frozen CSV/sessions/metadata bundle offline; never repair values."""
    path, metadata_path, sessions_path = _paths(config)
    mode = _source_mode(config)
    if mode not in ("yfinance", "twse"):
        raise SnapshotError(f"不支援的來源：{mode}")
    try:
        if not all(item.is_file() for item in (path, metadata_path, sessions_path)):
            raise SnapshotError("快照三檔不完整；不自動修復或覆寫，請檢查殘留或使用新 snapshot_path")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        csv_bytes, session_bytes = path.read_bytes(), sessions_path.read_bytes()
        if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
            raise SnapshotError("快照 schema_version 無效")
        if mode == "yfinance":
            if (metadata["source"]["name"] != "yfinance"
                    or metadata["source"]["parameters"] != _parameters(config)):
                raise SnapshotError("Yahoo 快照來源參數或 metadata 版本與設定不符")
        else:
            _check_twse_source(metadata, config)
        if (sha256(csv_bytes).hexdigest() != metadata["data"]["sha256"]
                or sha256(session_bytes).hexdigest() != metadata["calendar"]["sha256"]):
            raise SnapshotError("快照 CSV 或預期交易日的 SHA256 不符")
        downloaded = datetime.fromisoformat(metadata["downloaded_at_utc"])
        if downloaded.tzinfo is None or downloaded.utcoffset() != timedelta(0):
            raise SnapshotError("metadata 下載時間必須包含 UTC 時區")
        required_versions = ("python", "yfinance", "pandas", "numpy", "exchange_calendars") if mode == "yfinance" else (
            "python", "requests", "pandas", "numpy", "exchange_calendars")
        if (not isinstance(metadata["versions"], dict) or not all(
                isinstance(metadata["versions"].get(name), str) and metadata["versions"][name]
                for name in required_versions)):
            raise SnapshotError(f"metadata 缺少 {mode} 所需的套件版本")
        calendar_info = metadata["calendar"]
        if (calendar_info["name"] != "XTAI" or not isinstance(calendar_info["version"], str)
                or not calendar_info["version"] or not isinstance(calendar_info["source"], str)
                or not isinstance(calendar_info["additional_closures"], list)
                or not isinstance(calendar_info["additional_sessions"], list)
                or calendar_info["requested_start"] != config["data"]["start"]
                or calendar_info["requested_end"] != config["data"]["end"]):
            raise SnapshotError("metadata 日曆來源或期間不符")
        for adjustment in calendar_info["additional_closures"] + calendar_info["additional_sessions"]:
            day = datetime.strptime(adjustment["date"], "%Y-%m-%d").strftime("%Y-%m-%d")
            if (day != adjustment["date"] or not config["data"]["start"] <= day <= config["data"]["end"]
                    or not isinstance(adjustment["source"], str) or not adjustment["source"]):
                raise SnapshotError("metadata 開休市調整紀錄無效")
        frame = pd.read_csv(BytesIO(csv_bytes), float_precision="round_trip")
        if list(frame.columns) != ["Date", *COLUMNS]:
            raise SnapshotError("快照 CSV 欄位必須是 Date,Open,High,Low,Close,Volume")
        frame.index = _read_dates(frame)
        calendar_frame = pd.read_csv(BytesIO(session_bytes))
        if list(calendar_frame.columns) != ["Date"]:
            raise SnapshotError("預期交易日 CSV 必須只有 Date 欄位")
        sessions = _read_dates(calendar_frame)
        validate_ohlcv(frame, sessions, config["data"]["start"], config["data"]["end"])
        for section, index in (("data", frame.index), ("calendar", sessions)):
            if any(metadata[section][key] != value for key, value in _summary(index).items()):
                raise SnapshotError(f"metadata.{section} 筆數或日期邊界與 CSV 不符")
        if metadata["data"]["columns"] != ["Date", *COLUMNS]:
            raise SnapshotError("metadata.data.columns 不符")
        if mode == "twse":
            # Compare against the independent calendar; do not trust a self-consistent
            # but altered CSV + session list + hash merely because they match each other.
            expected, expected_calendar = _calendar_sessions(config["data"]["start"], config["data"]["end"])
            missing, extra = expected.difference(sessions), sessions.difference(expected)
            if len(missing) or len(extra):
                raise SnapshotError(
                    "TWSE 快照與 XTAI 修正後日曆不一致："
                    f"缺 {len(missing)} 天 {missing.strftime('%Y-%m-%d').tolist()[:10]}，"
                    f"多 {len(extra)} 天 {extra.strftime('%Y-%m-%d').tolist()[:10]}")
            for kind in ("additional_sessions", "additional_closures"):
                stored_days = [item["date"] for item in calendar_info[kind]]
                official_days = [item["date"] for item in expected_calendar[kind]]
                if sorted(stored_days) != sorted(official_days) or len(stored_days) != len(set(stored_days)):
                    raise SnapshotError(f"TWSE 日曆 {kind} 的歷史修正紀錄與預期不符")
            if (config["data"]["ticker"] == "2330.TW"
                    and config["data"]["start"] == "2015-01-01"
                    and config["data"]["end"] == "2025-12-31"
                    and len(sessions) != 2682):
                raise SnapshotError("2015–2025 台積電研究預期有 2682 個交易日")
        return frame, metadata
    except SnapshotError:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        raise SnapshotError(f"無法讀取或驗證快照：{error}") from error


def _publish(paths, contents):
    """Publish immutable files with metadata last; refuse existing destinations."""
    parent = paths[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=parent) as directory:
        staged, published = [], []
        for destination, content in zip(paths, contents):
            temporary = Path(directory) / destination.name
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append(temporary)
        try:
            for temporary, destination in zip(staged, paths):
                os.link(temporary, destination)
                published.append((temporary, destination))
        except OSError:
            for temporary, destination in reversed(published):
                try:
                    if destination.samefile(temporary):
                        destination.unlink()
                except OSError:
                    pass  # An interrupted/failed cleanup is rejected as an incomplete bundle.
            raise


def download_snapshot(config):
    """Reuse a complete snapshot offline, or download and freeze a new one."""
    path, metadata_path, sessions_path = _paths(config)
    if any(item.exists() for item in (path, metadata_path, sessions_path)):
        return load_snapshot(config)
    if _source_mode(config) == "twse":
        raise SnapshotError(
            "TWSE 快照不存在；請先在專案根目錄執行 "
            "python script/download_twse_2330.py，再使用 load-snapshot 驗證。"
        )
    if _source_mode(config) != "yfinance":
        raise SnapshotError(f"不支援的來源：{_source_mode(config)}")
    try:
        import yfinance as yf

        yf.set_tz_cache_location(str(path.parent / ".cache" / "yfinance"))
        sessions, calendar_info = _calendar_sessions(config["data"]["start"], config["data"]["end"])
        frame = normalize_ohlcv(yf.download(**_parameters(config)), config["data"]["ticker"])
        validate_ohlcv(frame, sessions, config["data"]["start"], config["data"]["end"])
        csv_bytes = frame.to_csv(date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
        session_bytes = pd.DataFrame(index=sessions).to_csv(date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
        metadata = {
            "schema_version": 1, "source": {"name": "yfinance", "parameters": _parameters(config)},
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"python": platform.python_version(), "yfinance": version("yfinance"),
                         "pandas": version("pandas"), "numpy": version("numpy"),
                         "exchange_calendars": version("exchange-calendars")},
            "data": {**_summary(frame.index), "columns": ["Date", *COLUMNS], "sha256": sha256(csv_bytes).hexdigest()},
            "calendar": {**calendar_info, **_summary(sessions), "sha256": sha256(session_bytes).hexdigest()},
        }
        metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        # CSV and sessions become visible first; metadata is the final commit marker.
        _publish((path, sessions_path, metadata_path), (csv_bytes, session_bytes, metadata_bytes))
        return load_snapshot(config)
    except SnapshotError:
        raise
    except Exception as error:
        raise SnapshotError(f"下載或保存行情失敗：{error}") from error
