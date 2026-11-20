#!/usr/bin/env python3
"""Download TSMC (2330) official, UNADJUSTED TWSE monthly OHLCV.

Requires: requests, pandas, numpy, exchange-calendars, and the existing
`lstm_comparison` project package (for its independent XTAI session calendar).

Run from project root on Windows:
    python download_twse_2330.py

Outputs only when all 132 months and every independent trading-day check pass:
    res/research/2330.TW_2015-2025_twse.csv
    res/research/2330.TW_2015-2025_twse.sessions.csv
    res/research/2330.TW_2015-2025_twse.metadata.json

It NEVER silently interpolates data, and it does NOT claim to reconstruct
Yahoo dividend/split-adjusted prices. Existing data.py must be updated to
accept TWSE source metadata before this snapshot can be used with `run`.
"""
from __future__ import annotations

import argparse
import calendar as month_calendar
import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib.metadata import version as package_version
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
import time
import requests
from urllib.parse import urlencode

BASE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
DEFAULT_TICKER = "2330"
DEFAULT_START = "2015-01"
DEFAULT_END = "2025-12"
SCRIPT_VERSION = "2026-11-20-calendar-hotfix-2"
FIELDS = ("日期", "成交股數", "開盤價", "最高價", "最低價", "收盤價")
OUTPUT_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")

# These are real TWSE Saturday trading sessions absent from the pinned
# exchange_calendars XTAI calendar used by the original LSTM project.
# Their OHLCV rows must be retained, not dropped to satisfy the calendar.
# Source: TWSE STOCK_DAY official per-stock monthly records.
ADDITIONAL_TWSE_SESSIONS = (
    ("2017-09-30", "20170901"),
    ("2018-03-31", "20180301"),
    ("2018-12-22", "20181201"),
)


class DownloadError(RuntimeError):
    pass


def months(start: str, end: str):
    try:
        s, e = date.fromisoformat(start + "-01"), date.fromisoformat(end + "-01")
    except ValueError as exc:
        raise DownloadError("日期必須為 YYYY-MM") from exc
    if s > e:
        raise DownloadError("起始月份不能晚於結束月份")
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        yield f"{y:04d}{m:02d}01"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _decimal(value, label: str, day: str) -> Decimal:
    try:
        n = Decimal(str(value).strip().replace(",", ""))
    except InvalidOperation as exc:
        raise DownloadError(f"{day} {label} 非有效數字：{value!r}") from exc
    if not n.is_finite():
        raise DownloadError(f"{day} {label} 非有限數字")
    return n


def parse_month(payload: dict, requested_yyyymm01: str) -> list[dict]:
    """Fail closed if TWSE structure, month, prices, or volume are wrong."""
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        raise DownloadError(f"{requested_yyyymm01}: TWSE stat 非 OK：{payload.get('stat') if isinstance(payload,dict) else payload!r}")
    fields = [str(x).strip() for x in payload.get("fields", [])]
    if len(fields) != len(set(fields)) or not set(FIELDS).issubset(fields):
        raise DownloadError(f"{requested_yyyymm01}: TWSE 欄位不符合預期：{fields}")
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise DownloadError(f"{requested_yyyymm01}: 未取得任何交易資料")
    positions = {name: fields.index(name) for name in FIELDS}
    result = []
    for row in records:
        if not isinstance(row, list) or len(row) < len(fields):
            raise DownloadError(f"{requested_yyyymm01}: 回傳欄位數不正確：{row!r}")
        raw_day = str(row[positions["日期"]]).strip()
        m = re.fullmatch(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", raw_day)
        if not m:
            raise DownloadError(f"{requested_yyyymm01}: 民國日期格式錯誤：{raw_day!r}")
        try:
            day = date(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError as exc:
            raise DownloadError(f"{requested_yyyymm01}: 日期無效：{raw_day}") from exc
        if day[:7].replace("-", "") != requested_yyyymm01[:6]:
            raise DownloadError(f"{requested_yyyymm01}: 跨月資料 {day}")
        o, h, l, c = (_decimal(row[positions[k]], k, day) for k in ("開盤價", "最高價", "最低價", "收盤價"))
        v = _decimal(row[positions["成交股數"]], "成交股數", day)
        if min(o, h, l, c) <= 0 or h < max(o, l, c) or l > min(o, h, c):
            raise DownloadError(f"{day}: OHLC 價格或高低價關係異常")
        if v <= 0 or v != v.to_integral_value():
            raise DownloadError(f"{day}: 成交股數必須為正整數")
        result.append(dict(zip(OUTPUT_COLUMNS, (day, str(o), str(h), str(l), str(c), str(int(v))))))
    dates = [r["Date"] for r in result]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise DownloadError(f"{requested_yyyymm01}: 重複或未排序的日期")
    return result


def _cache_path(cache_dir: Path, month: str) -> Path:
    return cache_dir / f"{month[:6]}.json"


def fetch_one_month(session, month: str, ticker: str, cache_dir: Path, retries: int, pause: float):
    path = _cache_path(cache_dir, month)
    params = {"date": month, "stockNo": ticker, "response": "json"}
    url = BASE_URL + "?" + urlencode(params)
    if path.exists():
        data = path.read_bytes()
        try:
            payload = json.loads(data)
            rows = parse_month(payload, month)
        except (ValueError, DownloadError) as exc:
            raise DownloadError(f"快取 {path} 無效，請先人工檢查，不自動覆寫：{exc}") from exc
        return rows, {"month": month[:6], "url": url, "cache_sha256": sha256(data).hexdigest(), "source": "cache"}

    for attempt in range(retries):
        try:
            response = session.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            rows = parse_month(payload, month)
            data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            cache_dir.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as f:
                f.write(data)
            if pause:
                time.sleep(pause)
            return rows, {"month": month[:6], "url": url, "cache_sha256": sha256(data).hexdigest(), "source": "download"}
        except (requests.RequestException, ValueError, DownloadError) as exc:
            if attempt + 1 == retries:
                raise DownloadError(f"{month[:6]} 下載失敗，重試 {retries} 次：{exc}") from exc
            wait = pause + min(30, 2 ** (attempt + 1))
            print(f"{month[:6]} 重試 {attempt+1}/{retries}，原因：{exc}，暫停 {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _expected_sessions(start: str, end: str, ticker: str = DEFAULT_TICKER):
    # This project already has a previously validated Taiwan exchange calendar
    # with the historical Saturday openings and typhoon closures added.
    try:
        from lstm_comparison.data import _calendar_sessions
    except ImportError:
        # Work when script is copied to project root but package isn't pip-installed.
        sys.path.insert(0, str(Path.cwd() / "src"))
        try:
            from lstm_comparison.data import _calendar_sessions
        except ImportError as exc:
            raise DownloadError("無法載入 lstm_comparison.data；請從專案根目錄、正確 .venv 執行") from exc
    try:
        sessions, info = _calendar_sessions(start, end)
    except Exception as exc:
        raise DownloadError(f"獨立交易日曆驗證失敗：{exc}") from exc

    # Preserve the project's existing closures and Saturday openings, while
    # adding the three additional, independently documented TWSE sessions.
    # Keep the correction in the metadata for reproducibility and provenance.
    dates = set(sessions.strftime("%Y-%m-%d").tolist())
    additions = list(info.get("additional_sessions", []))
    known_additions = {item["date"] for item in additions}
    for day, month in ADDITIONAL_TWSE_SESSIONS:
        if start <= day <= end:
            dates.add(day)
            if day not in known_additions:
                additions.append({
                    "date": day,
                    "reason": "Historical Saturday session present in TWSE STOCK_DAY but omitted by XTAI",
                    "source": BASE_URL + "?" + urlencode({
                        "date": month, "stockNo": ticker, "response": "json"
                    }),
                })
    info = {**info, "additional_sessions": sorted(additions, key=lambda item: item["date"])}
    return sorted(dates), info


def csv_bytes(rows: list[dict], fieldnames) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_outputs(out_dir: Path, ticker: str, start: str, end: str, rows: list[dict], sources: list[dict], sessions: list[str], calendar_info: dict):
    base = f"{ticker}.TW_{start[:4]}-{end[:4]}_twse"
    out_csv = out_dir / (base + ".csv")
    out_sessions = out_dir / (base + ".sessions.csv")
    out_metadata = out_dir / (base + ".metadata.json")
    outputs = [out_csv, out_sessions, out_metadata]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise DownloadError("輸出已存在，不覆寫：" + ", ".join(existing))
    data = csv_bytes(rows, OUTPUT_COLUMNS)
    calendar = csv_bytes([{"Date": day} for day in sessions], ["Date"])
    yyyy1, mm1 = map(int, end.split("-"))
    end_exclusive = f"{yyyy1+1:04d}-01-01" if mm1 == 12 else f"{yyyy1:04d}-{mm1+1:02d}-01"
    def pkg_version(name):
        try:
            return package_version(name)
        except Exception:
            return "unknown"

    # Shape mirrors the project's existing snapshot schema; source identifiers
    # and parameters remain truthfully TWSE-specific. Loader changes are needed.
    metadata = {
        "schema_version": 1,
        "source": {"name": "TWSE_STOCK_DAY", "agency": "臺灣證券交易所", "base_url": BASE_URL,
                   "price_basis": "unadjusted", "volume_unit": "shares",
                   "parameters": {"tickers": f"{ticker}.TW", "start": start+"-01", "end": end_exclusive,
                                  "interval": "1d", "auto_adjust": False, "stockNo": ticker},
                   "requests": sources},
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": sys.version.split()[0], "requests": pkg_version("requests"),
                     "pandas": pkg_version("pandas"), "numpy": pkg_version("numpy"),
                     "exchange_calendars": pkg_version("exchange-calendars")},
        "data": {"row_count": len(rows), "first_date": rows[0]["Date"], "last_date": rows[-1]["Date"],
                 "columns": list(OUTPUT_COLUMNS), "sha256": sha256(data).hexdigest()},
        "calendar": {**calendar_info, "row_count": len(sessions),
                     "first_date": sessions[0], "last_date": sessions[-1], "sha256": sha256(calendar).hexdigest()},
        "validation": {"all_expected_sessions_present": True, "no_unexpected_sessions": True,
                       "unique_sorted_dates": True, "all_ohlcv_valid": True},
        "important": "Raw TWSE OHLCV, not Yahoo adjusted price; update config and data/report source validation before training.",
    }
    meta_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, blob in zip(outputs, [data, calendar, meta_bytes]):
            with path.open("xb") as f:
                f.write(blob)
            created.append(path)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return outputs


def download(ticker: str, start: str, end: str, out_dir: Path, cache_dir: Path, pause: float, retries: int):
    if not re.fullmatch(r"\d{4}", ticker):
        raise DownloadError("股票代碼應為四位數字，如 2330")
    end_year, end_month = map(int, end.split("-"))
    period_start = start + "-01"
    period_end = f"{end_year:04d}-{end_month:02d}-{month_calendar.monthrange(end_year, end_month)[1]:02d}"
    expected, calendar_info = _expected_sessions(period_start, period_end, ticker)
    fixed_days = [d for d, _ in ADDITIONAL_TWSE_SESSIONS if period_start <= d <= period_end]
    missing_fixes = sorted(set(fixed_days) - set(expected))
    print(f"[calendar] XTAI 修正後預期交易日：{len(expected)} 筆", flush=True)
    print(f"[calendar] 三個歷史補行交易日：{fixed_days}", flush=True)
    if missing_fixes:
        raise DownloadError(f"交易日曆修正未生效，缺少：{missing_fixes}")
    if ticker == "2330" and start == "2015-01" and end == "2025-12" and len(expected) != 2682:
        raise DownloadError(
            f"本研究預期 2682 個交易日，實際只有 {len(expected)}；"
            "請檢查 lstm_comparison.data 版本和交易日曆來源，勿略過檢核"
        )
    all_rows, sources = [], []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TWSE-research-downloader/1.0)",
                                "Accept": "application/json, text/javascript, */*", "Referer": "https://www.twse.com.tw/"})
        for i, month in enumerate(months(start, end), 1):
            rows, source = fetch_one_month(session, month, ticker, cache_dir, retries, pause)
            all_rows.extend(rows)
            sources.append(source)
            print(f"[{i:03d}] {month[:6]}: {len(rows):2d}  ({source['source']})", flush=True)
    all_rows.sort(key=lambda r: r["Date"])
    actual = [row["Date"] for row in all_rows]
    if len(actual) != len(set(actual)):
        raise DownloadError("合併月份後有重複日期，不建立正式資料")
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        report = out_dir / f"{ticker}.TW_calendar_mismatch.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"missing": missing, "extra": extra, "actual_count": len(actual),
                                      "expected_count": len(expected),
                                      "script_version": SCRIPT_VERSION,
                                      "calendar_patches": fixed_days}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise DownloadError(f"TWSE 與 XTAI 交易日不一致：缺 {len(missing)} 天、多 {len(extra)} 天；詳見 {report}")
    print(f"交易日完整：{len(actual)} 筆，與 XTAI 獨立日曆完全一致")
    paths = write_outputs(out_dir, ticker, start, end, all_rows, sources, expected, calendar_info)
    for path in paths:
        print("已產生：", path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="下載 TWSE 官方 2330 原始行情並檢查交易日完整性，不合成或修補資料")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--start", default=DEFAULT_START, help="起始 YYYY-MM")
    parser.add_argument("--end", default=DEFAULT_END, help="截止 YYYY-MM")
    parser.add_argument("--out-dir", type=Path, default=Path("res/research"))
    parser.add_argument("--cache-dir", type=Path, default=Path("res/research/_twse_cache"))
    parser.add_argument("--pause", type=float, default=1.2, help="成功下載後請求間隔秒數")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--check-calendar", action="store_true", help="只檢查修正後 XTAI 日期，不下載行情")
    args = parser.parse_args(argv)
    try:
        if args.pause < 0 or args.retries < 1:
            raise DownloadError("pause 不得為負數，retries 必須 >=1")
        print(f"[script] 版本：{SCRIPT_VERSION}", flush=True)
        print(f"[script] 正在執行：{Path(__file__).resolve()}", flush=True)
        if args.check_calendar:
            ey, em = map(int, args.end.split("-"))
            end_day = date(ey, em, month_calendar.monthrange(ey, em)[1]).isoformat()
            expected, _ = _expected_sessions(args.start + "-01", end_day, args.ticker)
            print(f"[calendar] 預期交易日：{len(expected)}", flush=True)
            for day, _ in ADDITIONAL_TWSE_SESSIONS:
                print(f"[calendar] {day}：{'包含' if day in expected else '缺少'}", flush=True)
            if args.ticker == "2330" and args.start == "2015-01" and args.end == "2025-12" and len(expected) != 2682:
                raise DownloadError(f"預期 2682 個交易日，實際 {len(expected)}；請檢查 XTAI 版本及修正設定")
        else:
            download(args.ticker, args.start, args.end, args.out_dir, args.cache_dir, args.pause, args.retries)
    except (DownloadError, OSError) as exc:
        print(f"TWSE 下載/驗證失敗：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
