"""Price errors, three-way direction accuracy and repeated-seed summaries."""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np


class MetricsError(ValueError):
    """Metrics inputs are invalid or their result cannot be represented finitely."""


def _vector(values, name):
    try:
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise MetricsError(f"{name} must contain real numbers")
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1 or not array.size:
            raise MetricsError(f"{name} must be a nonempty vector with shape (n,) or (n, 1)")
        with np.errstate(over="raise", invalid="raise"):
            array = array.astype(np.float64)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        if isinstance(error, MetricsError):
            raise
        raise MetricsError(f"{name} must be a nonempty finite numeric vector") from error
    if not np.isfinite(array).all():
        raise MetricsError(f"{name} must contain only finite numbers")
    return array


def _number(value, name, *, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MetricsError(f"{name} must be a finite nonnegative number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise MetricsError(f"{name} must be a finite nonnegative number") from error
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        bounds = f"between 0 and {maximum} inclusive" if maximum is not None else "nonnegative"
        raise MetricsError(f"{name} must be finite and {bounds}")
    return value


def calculate_metrics(actual, predicted, previous_close, *, naive=False):
    """Calculate MAE, RMSE and direction accuracy on original price units.

    Vectors may have shape (n,) or (n, 1); they must have equal lengths.
    Direction is the exact sign relative to each target's previous close,
    including a distinct flat class, with no epsilon or price rounding.
    Naive predictions must equal previous close and have no direction metric.
    """
    if type(naive) is not bool:
        raise MetricsError("naive must be a boolean")
    actual = _vector(actual, "actual")
    predicted = _vector(predicted, "predicted")
    previous_close = _vector(previous_close, "previous_close")
    if not (actual.shape == predicted.shape == previous_close.shape):
        raise MetricsError("actual, predicted and previous_close must have equal lengths")
    if naive and not np.array_equal(predicted, previous_close):
        raise MetricsError("Naive predicted values must equal previous_close exactly")
    try:
        with np.errstate(over="raise", invalid="raise"):
            errors = np.abs(actual - predicted)
            scale = float(np.max(errors))
            # Scaling avoids overflow from summing errors or squaring prices.
            normalized = errors / scale if scale else errors
            mae = float(np.mean(normalized) * scale)
            rmse = float(np.sqrt(np.mean(normalized**2)) * scale)
    except FloatingPointError as error:
        raise MetricsError("price errors cannot be represented as finite numbers") from error
    if not math.isfinite(mae) or not math.isfinite(rmse):
        raise MetricsError("price errors cannot be represented as finite numbers")
    # Comparisons implement sign(a - previous) without subtraction overflow.
    direction = None if naive else float(np.mean(
        ((actual > previous_close) == (predicted > previous_close))
        & ((actual < previous_close) == (predicted < previous_close))
    ))
    return {"mae": mae, "rmse": rmse, "directional_accuracy": direction}


def error_improvement_pct(error, baseline):
    """Return 100 * (baseline - error) / baseline, or None for zero baseline."""
    error = _number(error, "error")
    baseline = _number(baseline, "baseline")
    if baseline == 0:
        return None
    improvement = 100 * ((baseline - error) / baseline)
    if not math.isfinite(improvement):
        raise MetricsError("error improvement cannot be represented as a finite percentage")
    return improvement


def _validated_metrics(metrics, *, naive):
    if not isinstance(metrics, Mapping):
        raise MetricsError("metrics must be a mapping")
    try:
        mae = _number(metrics["mae"], "mae")
        rmse = _number(metrics["rmse"], "rmse")
        direction = metrics["directional_accuracy"]
    except KeyError as error:
        raise MetricsError(f"missing metric: {error.args[0]}") from error
    if naive:
        if direction is not None:
            raise MetricsError("Naive directional_accuracy must be None")
    else:
        direction = _number(direction, "directional_accuracy", maximum=1)
    return {"mae": mae, "rmse": rmse, "directional_accuracy": direction}


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.max(array))
    normalized = array / scale if scale else array
    mean = float(np.mean(normalized) * scale)
    std = None if len(array) == 1 else float(np.std(normalized, ddof=1) * scale)
    if not math.isfinite(mean) or (std is not None and not math.isfinite(std)):
        raise MetricsError("metric mean or sample standard deviation is not finite")
    return mean, std


def summarize_metrics(per_seed_rows, naive_metrics):
    """Summarize model rows by group, followed by one Naive baseline row.

    Each input row needs group, seed, mae, rmse and directional_accuracy.
    Extra fields are ignored. Seeds are distinct nonnegative integers within
    each group; neither group names nor seed values are fixed to research
    defaults. Standard deviation uses ddof=1 and is None for a single seed.
    The separate Naive metrics have directional_accuracy=None; its summary
    has n_seeds=0 and no standard deviations or improvement percentages.
    """
    baseline = _validated_metrics(naive_metrics, naive=True)
    if isinstance(per_seed_rows, (str, bytes, Mapping)):
        raise MetricsError("per_seed_rows must be a nonempty iterable of model metric rows")
    try:
        rows = list(per_seed_rows)
    except TypeError as error:
        raise MetricsError("per_seed_rows must be a nonempty iterable of model metric rows") from error
    if not rows:
        raise MetricsError("per_seed_rows must not be empty")
    groups, seen = {}, set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise MetricsError("each model metric row must be a mapping")
        group, seed = row.get("group"), row.get("seed")
        if not isinstance(group, str) or not group.strip() or group == "Naive":
            raise MetricsError("group must be a nonempty model name other than Naive")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise MetricsError("seed must be a nonnegative integer")
        identity = (group, int(seed))
        if identity in seen:
            raise MetricsError(f"duplicate group-seed: {group}/{seed}")
        seen.add(identity)
        groups.setdefault(group, []).append(_validated_metrics(row, naive=False))

    summary = []
    for group in sorted(groups):
        metrics = groups[group]
        row = {"group": group, "n_seeds": len(metrics)}
        for key in ("mae", "rmse", "directional_accuracy"):
            row[f"{key}_mean"], row[f"{key}_std"] = _mean_std([item[key] for item in metrics])
        for key in ("mae", "rmse"):
            row[f"{key}_improvement_pct"] = error_improvement_pct(row[f"{key}_mean"], baseline[key])
        summary.append(row)
    summary.append({
        "group": "Naive", "n_seeds": 0,
        "mae_mean": baseline["mae"], "mae_std": None,
        "rmse_mean": baseline["rmse"], "rmse_std": None,
        "directional_accuracy_mean": None, "directional_accuracy_std": None,
        "mae_improvement_pct": None, "rmse_improvement_pct": None,
    })
    return summary
