"""Aligned next-session datasets with scalers fitted on training samples only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .data import SnapshotError
from .features import build_feature_groups


class DatasetError(ValueError):
    """The source or common samples cannot satisfy the experiment splits."""


@dataclass(frozen=True)
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    target_dates: pd.DatetimeIndex
    input_end_dates: pd.DatetimeIndex
    y_original: np.ndarray
    previous_close: np.ndarray


@dataclass(frozen=True)
class FeatureDataset:
    columns: tuple[str, ...]
    x_scaler: MinMaxScaler
    splits: dict[str, DatasetSplit]


@dataclass(frozen=True)
class ResearchDataset:
    groups: dict[str, FeatureDataset]
    y_scaler: MinMaxScaler


def _validate_scaler(scaler, values, name):
    """Check a persisted [0,1] mapping against its actual training inputs."""
    try:
        if (not isinstance(scaler, MinMaxScaler) or tuple(scaler.feature_range) != (0, 1)
                or scaler.clip is not False or scaler.n_features_in_ != values.shape[1]
                or scaler.n_samples_seen_ != len(values)):
            raise ValueError("型別、欄數、樣本數或縮放設定不符")
        lower, upper = values.min(axis=0), values.max(axis=0)
        span = upper - lower
        # MinMaxScaler treats near-constant float64 training ranges as one.
        denominator = np.where(span < 10 * np.finfo(values.dtype).eps, 1.0, span)
        scale = 1.0 / denominator
        expected = {"data_min_": lower, "data_max_": upper, "data_range_": span,
                    "scale_": scale, "min_": -lower * scale}
        for attribute, wanted in expected.items():
            actual = np.asarray(getattr(scaler, attribute))
            if actual.shape != wanted.shape or not np.isfinite(actual).all() or not np.array_equal(actual, wanted):
                raise ValueError(f"{attribute} 與訓練資料不符")
    except (AttributeError, TypeError, ValueError) as error:
        raise DatasetError(f"保存的 {name} scaler 無效：{error}") from error


def build_dataset(frame, config, *, x_scalers=None, y_scaler=None):
    """Build common windows, split by target date, and fit training-only scalers.

    Settings must come from load_config and the adjusted source from
    load_snapshot. All source validation precedes sample selection. A target at
    position j uses exactly [j-lookback:j], never the target day's features.
    X scalers fit the union of rows in common training windows; the shared y
    scaler fits only common training targets. No validation/test refit or clip.
    Supplying BOTH persisted x_scalers (group-name dict) and y_scaler validates
    their training ranges and uses transform only, without fitting any scaler.
    """
    lookback = config["sequence"]["lookback"]
    if type(lookback) is not int or lookback < 1 or config["sequence"]["horizon"] != 1:
        raise DatasetError("lookback 必須是正整數，horizon 必須是 1")
    try:
        features = build_feature_groups(frame, config)
    except SnapshotError as error:
        raise DatasetError(f"無法建立資料集：{error}") from error
    persisted = x_scalers is not None or y_scaler is not None
    if persisted and (not isinstance(x_scalers, dict) or y_scaler is None
                      or set(x_scalers) != set(features)):
        raise DatasetError("必須一起提供所有組別的 x_scalers 及 y_scaler")

    # Count invalid rows in each ORIGINAL contiguous window; never drop rows
    # and concatenate observations on either side of a missing indicator.
    valid_rows = np.logical_and.reduce([
        np.isfinite(group.to_numpy(dtype=float)).all(axis=1)
        for group in features.values()
    ])
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid_rows)))
    targets = np.arange(lookback, len(frame))
    targets = targets[invalid_prefix[targets] == invalid_prefix[targets - lookback]]
    if not len(targets):
        raise DatasetError("暖機後沒有完整輸入視窗與次日標籤的共同樣本")

    dates = frame.index[targets]
    masks = {}
    for name in ("train", "validation", "test"):
        bounds = config["split"][name]
        masks[name] = (dates >= pd.Timestamp(bounds["start"])) & (dates <= pd.Timestamp(bounds["end"]))
        if not masks[name].any():
            raise DatasetError(f"{name} 分區沒有共同有效樣本，請檢查期間與歷史資料長度")

    positions = targets[:, None] - lookback + np.arange(lookback)
    fit_positions = np.unique(positions[masks["train"]])
    close = frame["Close"].to_numpy(dtype=float)
    raw_y = close[targets].reshape(-1, 1)
    previous_close = close[targets - 1].reshape(-1, 1)
    if persisted:
        _validate_scaler(y_scaler, raw_y[masks["train"]], "y")
    else:
        y_scaler = MinMaxScaler(feature_range=(0, 1), clip=False).fit(raw_y[masks["train"]])
    scaled_y = y_scaler.transform(raw_y.copy())
    groups = {}
    for name, features_frame in features.items():
        values = features_frame.to_numpy(dtype=float)
        width = values.shape[1]
        if persisted:
            x_scaler = x_scalers[name]
            _validate_scaler(x_scaler, values[fit_positions], f"X_{name}")
        else:
            x_scaler = MinMaxScaler(feature_range=(0, 1), clip=False).fit(values[fit_positions])
        raw_X = values[positions]
        scaled_X = x_scaler.transform(raw_X.reshape(-1, width).copy()).reshape(raw_X.shape)
        groups[name] = FeatureDataset(
            columns=tuple(features_frame.columns), x_scaler=x_scaler,
            splits={
                split: DatasetSplit(
                    X=scaled_X[mask].copy(), y=scaled_y[mask].copy(),
                    target_dates=dates[mask].copy(), input_end_dates=frame.index[targets[mask] - 1].copy(),
                    y_original=raw_y[mask].copy(), previous_close=previous_close[mask].copy(),
                ) for split, mask in masks.items()
            },
        )
    return ResearchDataset(groups=groups, y_scaler=y_scaler)
