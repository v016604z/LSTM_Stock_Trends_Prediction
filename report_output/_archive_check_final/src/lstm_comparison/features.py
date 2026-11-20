"""Causal research features; retain warm-up NaNs for the dataset stage."""

import numpy as np
import pandas as pd

from .data import SnapshotError, validate_ohlcv


def _ema(series, period):
    """Seed with the first full-window SMA, then use alpha=2/(period+1)."""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return pd.Series(result, index=series.index)
    first = finite[0]
    seed = first + period - 1
    if seed < len(values):
        result[seed] = values[first:seed + 1].mean()
        alpha = 2.0 / (period + 1)
        for position in range(seed + 1, len(values)):
            result[position] = result[position - 1] + alpha * (values[position] - result[position - 1])
    return pd.Series(result, index=series.index)


def _rsi(close, period):
    """Wilder RSI: seed from period price changes, including flat changes."""
    changes = np.diff(close.to_numpy(dtype=float))
    gains, losses = np.maximum(changes, 0), np.maximum(-changes, 0)
    result = np.full(len(close), np.nan)
    if len(changes) < period:
        return pd.Series(result, index=close.index)
    average_gain, average_loss = gains[:period].mean(), losses[:period].mean()
    for position in range(period, len(close)):
        if position > period:
            average_gain += (gains[position - 1] - average_gain) / period
            average_loss += (losses[position - 1] - average_loss) / period
        if average_loss == 0:
            result[position] = 50.0 if average_gain == 0 else 100.0
        else:
            result[position] = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    return pd.Series(result, index=close.index)


def build_feature_groups(frame, config):
    """Return A/B/C frames from validated settings and adjusted OHLCV.

    Preserve every source date and leave indicator warm-up rows as NaN. The
    caller must use load_snapshot first to verify the frozen trading calendar;
    this function validates the provided values but does not infer missing days.
    It neither fills warm-up values nor selects the common 60-day samples.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SnapshotError("特徵輸入必須是非空 OHLCV DataFrame")
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None
            or not frame.index.equals(frame.index.normalize())):
        raise SnapshotError("特徵輸入必須使用無時區的交易日日期索引")
    validate_ohlcv(frame, frame.index, config["data"]["start"], config["data"]["end"])
    indicators = config["features"]["indicators"]
    features = frame.copy(deep=True)
    close = frame["Close"].astype(float)
    for period in indicators["ma_periods"]:
        features[f"MA{period}"] = close.rolling(period, min_periods=period).mean()
    features["RSI14"] = _rsi(close, indicators["rsi_period"])
    features["MACD"] = _ema(close, indicators["macd_fast"]) - _ema(close, indicators["macd_slow"])
    features["MACD_signal"] = _ema(features["MACD"], indicators["macd_signal"])
    return {
        name: features.loc[:, columns].copy(deep=True)
        for name, columns in config["features"]["groups"].items()
    }
