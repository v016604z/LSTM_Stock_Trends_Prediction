"""Reload completed training artifacts and evaluate aligned test predictions."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform

import numpy as np

from .config import load_config
from .data import load_snapshot
from .dataset import build_dataset
from .metrics import calculate_metrics, error_improvement_pct, summarize_metrics
from .model import build_model


class EvaluationError(RuntimeError):
    """Evaluation inputs or predictions fail the frozen experiment contract."""


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _same_targets(dataset):
    reference = dataset.groups["A"].splits["test"]
    for name, group in dataset.groups.items():
        current = group.splits["test"]
        if (not current.target_dates.equals(reference.target_dates)
                or not current.input_end_dates.equals(reference.input_end_dates)
                or not np.array_equal(current.y_original, reference.y_original)
                or not np.array_equal(current.previous_close, reference.previous_close)):
            raise EvaluationError(f"{name} 的目標日期、真值或前收盤未對齊")
    return reference


def _load_inputs(run_dir):
    manifest = _read_json(run_dir / "run.json")
    config = load_config(run_dir / "config.json")
    groups = config["features"]["groups"]
    seeds = config["training"]["seeds"]
    expected = {(group, seed) for group in groups for seed in seeds}
    if (manifest.get("schema_version") != 1 or manifest.get("stage") != "training"
            or manifest.get("status") != "completed" or manifest.get("groups") != list(groups)
            or manifest.get("seeds") != seeds or manifest.get("expected_runs") != len(expected)
            or not isinstance(manifest.get("runs"), list) or len(manifest["runs"]) != len(expected)):
        raise EvaluationError("run.json 必須是設定所列完整組別與種子的已完成訓練")
    entries = {}
    for entry in manifest["runs"]:
        key = (entry.get("group"), entry.get("seed"))
        if (type(key[1]) is not int or key not in expected or key in entries
                or entry.get("status") != "completed"
                or entry.get("directory") != f"{key[0]}/seed_{key[1]}"):
            raise EvaluationError("訓練清單有缺漏、重複、未完成或不符設定的組別／種子")
        entries[key] = entry
    if set(entries) != expected:
        raise EvaluationError("訓練清單與設定的組別／種子不一致")

    frozen = _read_json(run_dir / "snapshot.metadata.json")
    frame, metadata = load_snapshot(config)
    if (manifest.get("data_sha256") != frozen["data"]["sha256"]
            or metadata["data"] != frozen["data"] or metadata["calendar"] != frozen["calendar"]
            or metadata["source"] != frozen["source"]):
        raise EvaluationError("目前快照與訓練保存的資料、交易日 hash 或來源不一致")

    scaler_dir = run_dir / "scalers"
    description = _read_json(scaler_dir / "description.json")
    x_scalers = {}
    artifact_hashes = {"config.json": _hash(run_dir / "config.json")}
    for name in (*groups, "y"):
        path = scaler_dir / ("y.pkl" if name == "y" else f"X_{name}.pkl")
        with path.open("rb") as handle:
            scaler = pickle.load(handle)
        if name == "y":
            y_scaler = scaler
            details = description["y"]
            columns = ["Close"]
        else:
            x_scalers[name] = scaler
            details = description["X"][name]
            columns = groups[name]
        if (description["clip"] is not False or details["columns"] != columns
                or not np.array_equal(details["data_min"], scaler.data_min_)
                or not np.array_equal(details["data_max"], scaler.data_max_)):
            raise EvaluationError(f"{name} 的保存縮放器描述、欄序或最值不一致")
        artifact_hashes[str(path.relative_to(run_dir)).replace("\\", "/")] = _hash(path)
    dataset = build_dataset(frame, config, x_scalers=x_scalers, y_scaler=y_scaler)
    reference = _same_targets(dataset)

    weights = {}
    for (name, seed), entry in entries.items():
        directory = run_dir / name / f"seed_{seed}"
        training = _read_json(directory / "training.json")
        units, width = config["model"]["lstm_units"], len(groups[name])
        parameter_count = 4 * units * (units + width + 1) + units + 1
        if (training.get("stage") != "training" or training.get("status") != "completed"
                or training.get("seed") != seed or training.get("columns") != groups[name]
                or training.get("parameter_count") != parameter_count
                or entry.get("parameter_count") != parameter_count
                or training.get("weights") != "best.weights.h5"
                or training.get("history") != "history.csv"
                or type(training.get("best_epoch")) is not int
                or type(training.get("epochs_trained")) is not int
                or not 1 <= training["best_epoch"] <= training["epochs_trained"]
                or training.get("best_epoch") != entry.get("best_epoch")
                or training.get("epochs_trained") != entry.get("epochs_trained")):
            raise EvaluationError(f"{name}/seed_{seed} 的訓練狀態、模型、欄序或最佳權重紀錄不符")
        path = directory / "best.weights.h5"
        if not path.is_file() or not (directory / "history.csv").is_file():
            raise EvaluationError(f"缺少 {name}/seed_{seed} 的最佳權重或歷程")
        weights[name, seed] = path
        artifact_hashes[f"{name}/seed_{seed}/best.weights.h5"] = _hash(path)
    return config, dataset, reference, weights, metadata, artifact_hashes


def _predict(model, split, batch_size):
    prediction = np.concatenate([
        np.asarray(model(split.X[first:first + batch_size], training=False))
        for first in range(0, len(split.X), batch_size)
    ])
    if prediction.shape != split.y.shape or not np.isfinite(prediction).all():
        raise EvaluationError("測試預測形狀錯誤或不是有限數值")
    return prediction


def _write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "N/A" if row[key] is None else row[key] for key in columns})


def _write_predictions(path, split, group, seed, prediction):
    actual = split.y_original[:, 0]
    previous = split.previous_close[:, 0]
    predicted = prediction[:, 0]
    try:
        with np.errstate(over="raise", invalid="raise"):
            error = predicted - actual
            squared_error = np.square(error)
            actual_change = actual - previous
            predicted_change = predicted - previous
        if not all(np.isfinite(values).all() for values in (error, squared_error, actual_change, predicted_change)):
            raise FloatingPointError("nonfinite daily values")
    except FloatingPointError as error:
        raise EvaluationError("逐日 CSV 的誤差、平方誤差或方向變動無法表示為有限數值") from error
    actual_direction = np.sign(actual_change).astype(int)
    predicted_direction = np.sign(predicted_change).astype(int)
    columns = ["target_date", "input_end_date", "group", "seed", "actual_close", "previous_close",
               "predicted_close", "absolute_error", "squared_error", "actual_direction",
               "predicted_direction", "direction_correct"]
    rows = ({"target_date": day.strftime("%Y-%m-%d"),
             "input_end_date": split.input_end_dates[i].strftime("%Y-%m-%d"),
             "group": group, "seed": seed, "actual_close": float(actual[i]),
             "previous_close": float(previous[i]), "predicted_close": float(predicted[i]),
             "absolute_error": float(abs(error[i])), "squared_error": float(squared_error[i]),
             "actual_direction": int(actual_direction[i]), "predicted_direction": int(predicted_direction[i]),
             "direction_correct": None if group == "Naive" else int(actual_direction[i] == predicted_direction[i])}
            for i, day in enumerate(split.target_dates))
    _write_csv(path, columns, rows)


def evaluate_experiment(run_dir):
    """Evaluate only held-out test samples; never fit or overwrite an evaluation."""
    run_dir = Path(run_dir).expanduser().resolve()
    output = run_dir / "evaluation"
    if output.exists():
        raise EvaluationError(f"評估目錄已存在，不覆寫：{output}")
    try:
        config, dataset, reference, weights, metadata, artifact_hashes = _load_inputs(run_dir)
    except Exception as error:
        raise EvaluationError(f"無法驗證評估輸入：{error}") from error
    try:
        output.mkdir(exist_ok=False)
    except OSError as error:
        raise EvaluationError(f"無法建立新的評估目錄：{error}") from error
    manifest = {"schema_version": 1, "stage": "evaluation", "status": "running", "split": "test",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_sha256": metadata["data"]["sha256"], "calendar_sha256": metadata["calendar"]["sha256"],
                "config_sha256": artifact_hashes["config.json"], "artifact_sha256": artifact_hashes,
                "groups": list(dataset.groups), "seeds": list(config["training"]["seeds"]),
                "n_samples": len(reference.target_dates), "first_target": reference.target_dates[0].strftime("%Y-%m-%d"),
                "last_target": reference.target_dates[-1].strftime("%Y-%m-%d"), "prediction_files": []}
    try:
        _write_json(output / "evaluation.json", manifest)
        _write_json(output / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: version(name) for name in (
                        "tensorflow-cpu", "keras", "numpy", "pandas", "scikit-learn", "lstm-comparison")}})
        naive = calculate_metrics(reference.y_original, reference.previous_close, reference.previous_close, naive=True)
        _write_predictions(output / "predictions_naive.csv", reference, "Naive", None, reference.previous_close)
        manifest["prediction_files"].append("predictions_naive.csv")
        metric_rows, csv_rows = [], []
        for name, group in dataset.groups.items():
            split = group.splits["test"]
            for seed in config["training"]["seeds"]:
                model = build_model(split.X.shape[1:], config, seed)
                model.load_weights(weights[name, seed], objects_to_skip=[model.optimizer])
                prediction = dataset.y_scaler.inverse_transform(_predict(model, split, config["training"]["batch_size"]))
                metrics = calculate_metrics(split.y_original, prediction, split.previous_close)
                row = {"group": name, "seed": seed, **metrics}
                metric_rows.append(row)
                csv_rows.append({**row, "n_samples": len(split.X),
                                 "mae_improvement_pct": error_improvement_pct(metrics["mae"], naive["mae"]),
                                 "rmse_improvement_pct": error_improvement_pct(metrics["rmse"], naive["rmse"])})
                filename = f"predictions_{name}_seed_{seed}.csv"
                _write_predictions(output / filename, split, name, seed, prediction)
                manifest["prediction_files"].append(filename)
        csv_rows.append({"group": "Naive", "seed": None, "n_samples": len(reference.X), **naive,
                         "mae_improvement_pct": None, "rmse_improvement_pct": None})
        columns = ["group", "seed", "n_samples", "mae", "rmse", "directional_accuracy",
                   "mae_improvement_pct", "rmse_improvement_pct"]
        _write_csv(output / "metrics_per_seed.csv", columns, csv_rows)
        summary = summarize_metrics(metric_rows, naive)
        _write_csv(output / "metrics_summary.csv", list(summary[0]), summary)
        manifest.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "metrics_per_seed": csv_rows, "metrics_summary": summary})
        _write_json(output / "evaluation.json", manifest)
        return output
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error),
                         "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        manifest.pop("completed_at_utc", None)
        try:
            _write_json(output / "evaluation.json", manifest)
        except OSError:
            pass
        raise EvaluationError(f"評估失敗，保留目錄 {output}：{error}") from error
