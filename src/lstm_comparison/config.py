"""Load the research settings without importing numerical or network packages."""

from datetime import date, timedelta
import json
import math
from pathlib import Path


class ConfigError(ValueError):
    """A configuration file is unreadable or contains invalid research settings."""


def _object(value, name, fields):
    if not isinstance(value, dict):
        raise ConfigError(f"{name} 必須是物件")
    missing = set(fields) - value.keys()
    unknown = value.keys() - set(fields)
    if missing or unknown:
        raise ConfigError(
            f"{name} 欄位錯誤：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    return value


def _number(value, name, *, integer=False, minimum=0, inclusive=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    try:
        finite = valid_type and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ConfigError(f"{name} 必須是有限{'整數' if integer else '數值'}")
    if value < minimum or (not inclusive and value == minimum):
        raise ConfigError(f"{name} 必須{'大於或等於' if inclusive else '大於'} {minimum}")


def _choice(value, name, options):
    if value not in options or type(value) not in (str, bool):
        raise ConfigError(f"{name} 必須是 {options!r} 其中之一")


def _day(value, name):
    try:
        parsed = date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.isoformat() != value:
        raise ConfigError(f"{name} 必須是 YYYY-MM-DD 日期")
    return parsed


def load_config(path):
    """Validate settings and resolve data paths relative to the JSON file."""
    try:
        path = Path(path).expanduser().resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        raise ConfigError(f"無法讀取 {path}：{error}") from error

    _object(config, "config", (
        "data", "split", "sequence", "features", "model", "training", "evaluation"
    ))
    data = _object(config["data"], "data", (
        "ticker", "start", "end", "end_exclusive", "interval", "auto_adjust",
        "snapshot_path", "output_dir"
    ))
    if not isinstance(data["ticker"], str) or not data["ticker"].strip():
        raise ConfigError("data.ticker 必須是非空字串")
    start, end = (_day(data[key], f"data.{key}") for key in ("start", "end"))
    if start >= end or _day(data["end_exclusive"], "data.end_exclusive") - end != timedelta(days=1):
        raise ConfigError("data 日期順序錯誤，end_exclusive 必須是 end 的下一日")
    _choice(data["interval"], "data.interval", ("1d",))
    _choice(data["auto_adjust"], "data.auto_adjust", (True,))
    for key in ("snapshot_path", "output_dir"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ConfigError(f"data.{key} 必須是非空路徑字串")
        try:
            data[key] = str((path.parent / Path(data[key]).expanduser()).resolve())
        except (OSError, ValueError) as error:
            raise ConfigError(f"data.{key} 路徑無效：{error}") from error

    split = _object(config["split"], "split", ("train", "validation", "test"))
    previous_end = None
    for name in ("train", "validation", "test"):
        part = _object(split[name], f"split.{name}", ("start", "end"))
        first, last = (_day(part[key], f"split.{name}.{key}") for key in ("start", "end"))
        if not start <= first <= last <= end:
            raise ConfigError(f"split.{name} 日期必須依序且位於 data 期間內")
        if previous_end is not None and first - previous_end != timedelta(days=1):
            raise ConfigError(f"split.{name} 必須緊接前一區間，不可重疊或留下日期缺口")
        previous_end = last
    if split["train"]["start"] != data["start"] or split["test"]["end"] != data["end"]:
        raise ConfigError("split 必須涵蓋完整 data 期間")

    sequence = _object(config["sequence"], "sequence", ("lookback", "horizon"))
    for key in sequence:
        _number(sequence[key], f"sequence.{key}", integer=True)
    if sequence["horizon"] != 1:
        raise ConfigError("sequence.horizon 必須是 1，研究目標為下一交易日")

    features = _object(config["features"], "features", ("groups", "indicators"))
    groups = _object(features["groups"], "features.groups", ("A", "B", "C"))
    expected_groups = {
        "A": ["Close"],
        "B": ["Open", "High", "Low", "Close", "Volume"],
        "C": ["Open", "High", "Low", "Close", "Volume", "MA5", "MA20", "RSI14", "MACD", "MACD_signal"],
    }
    for name, columns in expected_groups.items():
        if groups[name] != columns:
            raise ConfigError(f"features.groups.{name} 必須依序包含 {columns!r}")
    indicators = _object(features["indicators"], "features.indicators", (
        "ma_periods", "rsi_period", "rsi_smoothing", "macd_fast", "macd_slow", "macd_signal"
    ))
    if indicators["ma_periods"] != [5, 20]:
        raise ConfigError("features.indicators.ma_periods 必須是 [5, 20]")
    for key in ("rsi_period", "macd_fast", "macd_slow", "macd_signal"):
        _number(indicators[key], f"features.indicators.{key}", integer=True)
    if indicators["rsi_period"] != 14:
        raise ConfigError("features.indicators.rsi_period 必須是 14")
    for key, expected in (("macd_fast", 12), ("macd_slow", 26), ("macd_signal", 9)):
        if indicators[key] != expected:
            raise ConfigError(f"features.indicators.{key} 必須是 {expected}")
    _choice(indicators["rsi_smoothing"], "features.indicators.rsi_smoothing", ("wilder",))

    model = _object(config["model"], "model", ("lstm_units", "dense_units", "bidirectional"))
    for key in ("lstm_units", "dense_units"):
        _number(model[key], f"model.{key}", integer=True)
    if model["dense_units"] != 1:
        raise ConfigError("model.dense_units 必須是 1，輸出單一收盤價")
    _choice(model["bidirectional"], "model.bidirectional", (False,))

    training = _object(config["training"], "training", (
        "seeds", "optimizer", "learning_rate", "loss", "batch_size", "max_epochs",
        "early_stopping", "shuffle", "device"
    ))
    seeds = training["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("training.seeds 必須是非空整數陣列")
    for seed in seeds:
        _number(seed, "training.seeds", integer=True, inclusive=True)
        if seed >= 2**32:
            raise ConfigError("training.seeds 必須小於 2**32")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("training.seeds 不可重複")
    for key in ("batch_size", "max_epochs"):
        _number(training[key], f"training.{key}", integer=True)
    _number(training["learning_rate"], "training.learning_rate")
    for key, allowed in (("optimizer", ("adam",)), ("loss", ("mse",)),
                         ("shuffle", (False,)), ("device", ("cpu",))):
        _choice(training[key], f"training.{key}", allowed)
    stopping = _object(training["early_stopping"], "training.early_stopping", (
        "monitor", "patience", "restore_best_weights"
    ))
    _number(stopping["patience"], "training.early_stopping.patience", integer=True, inclusive=True)
    _choice(stopping["monitor"], "training.early_stopping.monitor", ("val_loss",))
    _choice(stopping["restore_best_weights"], "training.early_stopping.restore_best_weights", (True,))

    evaluation = _object(config["evaluation"], "evaluation", (
        "metrics", "primary_metric", "direction", "baseline", "naive_direction_accuracy", "plot_seed"
    ))
    if evaluation["metrics"] != ["mae", "rmse", "directional_accuracy"]:
        raise ConfigError("evaluation.metrics 必須是 mae、rmse、directional_accuracy")
    for key, allowed in (("primary_metric", ("mae",)), ("direction", ("three_way",)),
                         ("baseline", ("previous_close",)), ("naive_direction_accuracy", ("N/A",))):
        _choice(evaluation[key], f"evaluation.{key}", allowed)
    _number(evaluation["plot_seed"], "evaluation.plot_seed", integer=True, inclusive=True)
    if evaluation["plot_seed"] not in seeds:
        raise ConfigError("evaluation.plot_seed 必須包含在 training.seeds")
    return config
