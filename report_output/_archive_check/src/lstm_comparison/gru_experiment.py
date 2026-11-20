"""Run a GRU against the frozen LSTM experiment without altering its artifacts."""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import pickle
import shutil
from statistics import mean, stdev
import time
from uuid import uuid4

import numpy as np

from .config import load_config
from .data import load_snapshot
from .dataset import build_dataset
from .evaluation import _predict, _write_predictions
from .metrics import calculate_metrics
from .model import build_gru_model
from .training import _environment, _now, _write_json, train_one


DEFAULT_BASELINE = (Path(__file__).resolve().parents[2] / "results"
                    / "20261120T134010119780Z_82eaad21")
GROUPS = ("A", "B", "C")
SEEDS = (42, 43, 44)
class GRUExperimentError(RuntimeError):
    """The frozen LSTM inputs or GRU experiment artifacts are inconsistent."""


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_baseline(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    training_manifest = _read_json(run_dir / "run.json")
    config_path = run_dir / "config.json"
    config = load_config(config_path)
    evaluation = _read_json(run_dir / "evaluation" / "evaluation.json")
    frozen = _read_json(run_dir / "snapshot.metadata.json")
    if (training_manifest.get("status") != "completed"
            or training_manifest.get("stage") != "training"
            or training_manifest.get("groups") != list(GROUPS)
            or training_manifest.get("seeds") != list(SEEDS)
            or training_manifest.get("expected_runs") != 9
            or len(training_manifest.get("runs", [])) != 9):
        raise GRUExperimentError("The baseline LSTM run must be a completed A/B/C x 3-seed experiment")
    if (evaluation.get("status") != "completed" or evaluation.get("split") != "test"
            or evaluation.get("groups") != list(GROUPS)
            or evaluation.get("seeds") != list(SEEDS)):
        raise GRUExperimentError("The baseline LSTM test evaluation is missing or incomplete")
    if (config["training"]["seeds"] != list(SEEDS)
            or config["sequence"] != {"lookback": 60, "horizon": 1}
            or config["split"] != {
                "train": {"start": "2015-01-01", "end": "2022-12-31"},
                "validation": {"start": "2023-01-01", "end": "2023-12-31"},
                "test": {"start": "2024-01-01", "end": "2025-12-31"},
            }
            or config["training"]["device"] != "cpu"
            or config["model"]["bidirectional"] is not False
            or config["model"]["lstm_units"] != 64
            or config["model"]["dense_units"] != 1):
        raise GRUExperimentError("Baseline config does not match the approved GRU experiment design")
    expected_training = {"optimizer": "adam", "learning_rate": 0.001, "loss": "mse",
                         "batch_size": 32, "max_epochs": 100, "shuffle": False,
                         "device": "cpu", "early_stopping": {
                             "monitor": "val_loss", "patience": 10, "restore_best_weights": True}}
    if any(config["training"].get(key) != value for key, value in expected_training.items()):
        raise GRUExperimentError("Baseline training settings do not match the approved LSTM experiment")
    if (evaluation.get("config_sha256") != _hash(config_path)
            or evaluation.get("data_sha256") != frozen["data"]["sha256"]
            or training_manifest.get("data_sha256") != frozen["data"]["sha256"]):
        raise GRUExperimentError("Baseline config or snapshot hashes do not match its manifests")
    snapshot_sessions = Path(config["data"]["snapshot_path"]).with_suffix(".sessions.csv")
    if evaluation.get("calendar_sha256") != _hash(snapshot_sessions):
        raise GRUExperimentError("Baseline trading-calendar hash does not match the frozen calendar")

    eval_hashes = evaluation.get("artifact_sha256", {})
    scaler_files = ["scalers/description.json", "scalers/X_A.pkl", "scalers/X_B.pkl",
                    "scalers/X_C.pkl", "scalers/y.pkl"]
    # The evaluation manifest covers all pickled scalers and all trained weights.
    for relative in ["config.json", "scalers/X_A.pkl", "scalers/X_B.pkl", "scalers/X_C.pkl",
                     "scalers/y.pkl", *(f"{group}/seed_{seed}/best.weights.h5"
                                           for group in GROUPS for seed in SEEDS)]:
        if eval_hashes.get(relative) != _hash(run_dir / relative):
            raise GRUExperimentError(f"Baseline artifact hash mismatch: {relative}")
    if evaluation.get("n_samples") != 485:
        raise GRUExperimentError("The baseline evaluation does not contain the expected 485 test targets")
    return run_dir, config_path, config, training_manifest, evaluation, frozen, scaler_files


def _load_baseline_scalers(run_dir, config):
    scaler_dir = run_dir / "scalers"
    description = _read_json(scaler_dir / "description.json")
    scalers = {}
    for name in GROUPS:
        with (scaler_dir / f"X_{name}.pkl").open("rb") as handle:
            scalers[name] = pickle.load(handle)
    with (scaler_dir / "y.pkl").open("rb") as handle:
        y_scaler = pickle.load(handle)
    for name in GROUPS:
        details = description["X"][name]
        scaler = scalers[name]
        if (details["columns"] != config["features"]["groups"][name]
                or not np.array_equal(details["data_min"], scaler.data_min_)
                or not np.array_equal(details["data_max"], scaler.data_max_)):
            raise GRUExperimentError(f"Frozen X_{name} scaler description does not match its pickle")
    if (description["y"]["columns"] != ["Close"]
            or not np.array_equal(description["y"]["data_min"], y_scaler.data_min_)
            or not np.array_equal(description["y"]["data_max"], y_scaler.data_max_)):
        raise GRUExperimentError("Frozen y scaler description does not match its pickle")
    return scalers, y_scaler


def _baseline_metrics(run_dir, dataset, group, seed):
    path = run_dir / "evaluation" / f"predictions_{group}_seed_{seed}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split = dataset.groups[group].splits["test"]
    if len(rows) != len(split.X):
        raise GRUExperimentError(f"LSTM {group}/seed_{seed} prediction count differs from the test split")
    dates = [day.strftime("%Y-%m-%d") for day in split.target_dates]
    input_dates = [day.strftime("%Y-%m-%d") for day in split.input_end_dates]
    if ([row["target_date"] for row in rows] != dates
            or [row["input_end_date"] for row in rows] != input_dates
            or any(row["group"] != group or row["seed"] != str(seed) for row in rows)):
        raise GRUExperimentError(f"LSTM {group}/seed_{seed} dates or identity differ from the shared dataset")
    actual = np.array([float(row["actual_close"]) for row in rows])
    previous = np.array([float(row["previous_close"]) for row in rows])
    prediction = np.array([float(row["predicted_close"]) for row in rows])
    if (not np.array_equal(actual, split.y_original[:, 0])
            or not np.array_equal(previous, split.previous_close[:, 0])):
        raise GRUExperimentError(f"LSTM {group}/seed_{seed} actual/previous_close differs from shared data")
    metrics = calculate_metrics(actual, prediction, previous)
    with (run_dir / "evaluation" / "metrics_per_seed.csv").open(encoding="utf-8", newline="") as handle:
        metric_row = next(row for row in csv.DictReader(handle)
                          if row["group"] == group and row["seed"] == str(seed))
    for key, value in metrics.items():
        if not math.isclose(value, float(metric_row[key]), rel_tol=1e-12, abs_tol=1e-12):
            raise GRUExperimentError(f"Recomputed LSTM {group}/seed_{seed} {key} differs from its CSV")
    training = _read_json(run_dir / group / f"seed_{seed}" / "training.json")
    return metrics, int(training["parameter_count"]), int(training["epochs_trained"]), float(training["elapsed_seconds"])


def _predict_gru(run_dir, dataset, config, group, seed):
    split = dataset.groups[group].splits["test"]
    model = build_gru_model(split.X.shape[1:], config, seed)
    model.load_weights(run_dir / group / f"seed_{seed}" / "best.weights.h5",
                       objects_to_skip=[model.optimizer])
    scaled = _predict(model, split, config["training"]["batch_size"])
    predicted = dataset.y_scaler.inverse_transform(scaled)
    if predicted.shape != split.y_original.shape or not np.isfinite(predicted).all():
        raise GRUExperimentError(f"GRU {group}/seed_{seed} predictions are not finite or aligned")
    _write_predictions(run_dir / "evaluation" / f"predictions_{group}_seed_{seed}.csv",
                       split, group, seed, predicted)
    metrics = calculate_metrics(split.y_original, predicted, split.previous_close)
    return metrics, predicted


def _summary(rows):
    results = []
    for group in GROUPS:
        for model in ("LSTM", "GRU"):
            selected = [row for row in rows if row["group"] == group and row["model"] == model]
            result = {"group": group, "model": model, "n_seeds": len(selected),
                      "parameter_count": selected[0]["parameter_count"],
                      "train_seconds_mean": mean(row["train_seconds"] for row in selected)}
            for key in ("mae", "rmse", "directional_accuracy"):
                values = [row[key] for row in selected]
                result[f"{key}_mean"] = mean(values)
                result[f"{key}_std"] = stdev(values)
            results.append(result)
    return results


def _write_report(path, baseline_dir, config, source_metadata, environment, rows, summary,
                  source_scaler_hashes, data_hash, calendar_hash):
    len_rows = {name: count for name, count in source_metadata["sample_counts"].items()}
    date_first, date_last = source_metadata["test_dates"]
    summaries = {(row["group"], row["model"]): row for row in summary}
    blocks = []
    for group in GROUPS:
        lstm, gru = summaries[group, "LSTM"], summaries[group, "GRU"]
        blocks.append(
            f"### {group}（{', '.join(config['features']['groups'][group])}）\n\n"
            f"| Model | Parameters | train seconds mean | MAE mean ± sample SD (TWD) | "
            f"RMSE mean ± sample SD (TWD) | Directional accuracy mean ± sample SD |\n"
            f"|---|---:|---:|---:|---:|---:|\n"
            f"| LSTM | {lstm['parameter_count']} | {lstm['train_seconds_mean']:.2f} | "
            f"{lstm['mae_mean']:.4f} ± {lstm['mae_std']:.4f} | "
            f"{lstm['rmse_mean']:.4f} ± {lstm['rmse_std']:.4f} | "
            f"{lstm['directional_accuracy_mean'] * 100:.2f}% ± {lstm['directional_accuracy_std'] * 100:.2f}% |\n"
            f"| GRU | {gru['parameter_count']} | {gru['train_seconds_mean']:.2f} | "
            f"{gru['mae_mean']:.4f} ± {gru['mae_std']:.4f} | "
            f"{gru['rmse_mean']:.4f} ± {gru['rmse_std']:.4f} | "
            f"{gru['directional_accuracy_mean'] * 100:.2f}% ± {gru['directional_accuracy_std'] * 100:.2f}% |"
        )
    versions = environment["packages"]
    report = f"""# LSTM 與 GRU 股票收盤價比較

## 實驗設定

- 原 LSTM 實驗：`{baseline_dir}`
- 模型：單向 LSTM(64) 與 GRU(64)，各接 Dense(1)；相同 units 不代表相同參數量，依各模型實際參數量列示。
- 資料：TWSE {config['data']['ticker']} 原始 OHLCV；資料 SHA256 `{data_hash}`，交易日曆 SHA256 `{calendar_hash}`。
- 特徵組 A/B/C、lookback {config['sequence']['lookback']} 個交易日、horizon 1；train {config['split']['train']['start']}–{config['split']['train']['end']}、validation {config['split']['validation']['start']}–{config['split']['validation']['end']}、test {config['split']['test']['start']}–{config['split']['test']['end']}。
- 每組樣本數：train {len_rows['train']}、validation {len_rows['validation']}、test {len_rows['test']}；測試目標日期 {date_first} 至 {date_last}。
- GRU 重用原 LSTM run 的 X_A/X_B/X_C/y scalers（只 transform，不 refit）；scaler SHA256：`{source_scaler_hashes}`。
- 相同 optimizer={config['training']['optimizer']}、learning rate={config['training']['learning_rate']}、loss={config['training']['loss']}、batch size={config['training']['batch_size']}、shuffle={config['training']['shuffle']}、max epochs={config['training']['max_epochs']}、early stopping monitor={config['training']['early_stopping']['monitor']} patience={config['training']['early_stopping']['patience']} restore_best_weights={config['training']['early_stopping']['restore_best_weights']}；seeds={config['training']['seeds']}。
- 評估指標由共同測試資料上的逐日預測重算；MAE/RMSE 單位為 TWD，方向正確率是上漲/下跌/持平三分類比例。表格方向指標以百分比表示。
- GRU 執行環境：Python {environment['python']}，TensorFlow CPU {versions.get('tensorflow-cpu')}，Keras {versions.get('keras')}，device={environment['device']}。

## 三組結果

{chr(10).join(blocks)}

每種子結果見 `evaluation/comparison_per_seed.csv`；彙總見 `evaluation/comparison_summary.csv`。標準差為三個固定種子的樣本標準差。

訓練秒數來自各 run 的訓練紀錄。若兩次實驗套件或硬體環境不同，時間只能作為各自紀錄，不能視為公平速度比較；環境版本見本 run 的 `environment.json` 及原 run 的 `environment.json`。

此比較固定資料、日期切分、縮放器、特徵組與訓練規則，未根據 test 調整超參數。結果不代表統計顯著差異或未來預測表現。
"""
    path.write_text(report, encoding="utf-8")


def run_gru_experiment(baseline_dir=DEFAULT_BASELINE):
    baseline_dir, config_path, config, original_manifest, original_evaluation, frozen, scaler_files = (
        _validate_baseline(baseline_dir)
    )
    frame, metadata = load_snapshot(config)
    if (metadata["data"] != frozen["data"] or metadata["calendar"] != frozen["calendar"]
            or metadata["source"] != frozen["source"]):
        raise GRUExperimentError("Current snapshot metadata differs from the completed LSTM experiment")
    x_scalers, y_scaler = _load_baseline_scalers(baseline_dir, config)
    dataset = build_dataset(frame, config, x_scalers=x_scalers, y_scaler=y_scaler)
    if list(dataset.groups) != list(GROUPS):
        raise GRUExperimentError("Loaded feature groups differ from the frozen LSTM experiment")
    test = dataset.groups["A"].splits["test"]
    if (len(test.X) != original_evaluation["n_samples"]
            or test.target_dates[0].strftime("%Y-%m-%d") != original_evaluation["first_target"]
            or test.target_dates[-1].strftime("%Y-%m-%d") != original_evaluation["last_target"]):
        raise GRUExperimentError("Regenerated test targets differ from the LSTM evaluation manifest")
    for group in GROUPS:
        for split_name in ("train", "validation", "test"):
            current = dataset.groups[group].splits[split_name]
            baseline_counts = original_manifest["runs"][0]
            if not len(current.X) or not np.isfinite(current.X).all() or not np.isfinite(current.y).all():
                raise GRUExperimentError(f"{group}/{split_name} data is empty or non-finite")
            if split_name == "train" and len(current.X) != baseline_counts["train_samples"]:
                raise GRUExperimentError(f"{group} train sample count differs from LSTM artifacts")
            if split_name == "validation" and len(current.X) != baseline_counts["validation_samples"]:
                raise GRUExperimentError(f"{group} validation sample count differs from LSTM artifacts")
    baseline_metrics = {(group, seed): _baseline_metrics(baseline_dir, dataset, group, seed)
                        for group in GROUPS for seed in SEEDS}

    output_root = Path(config["data"]["output_dir"])
    run_dir = output_root / ("gru_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                             + "_" + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter()
    for filename in ("config.json", "snapshot.metadata.json"):
        _copy(baseline_dir / filename, run_dir / filename)
    for relative in [*scaler_files]:
        _copy(baseline_dir / relative, run_dir / relative)
    _write_json(run_dir / "environment.json", _environment())
    data_hash = metadata["data"]["sha256"]
    calendar_hash = _hash(Path(config["data"]["snapshot_path"]).with_suffix(".sessions.csv"))
    scaler_hashes = {relative: _hash(baseline_dir / relative) for relative in scaler_files}
    sample_counts = {name: len(dataset.groups["A"].splits[name].X)
                     for name in ("train", "validation", "test")}
    manifest = {"schema_version": 1, "stage": "gru_comparison", "model_type": "GRU",
                "status": "running", "started_at_utc": _now(), "baseline_run": str(baseline_dir),
                "data_sha256": data_hash, "calendar_sha256": calendar_hash,
                "config_sha256": _hash(run_dir / "config.json"), "scaler_sha256": scaler_hashes,
                "groups": list(GROUPS), "seeds": list(SEEDS), "lookback": config["sequence"]["lookback"],
                "sample_counts": sample_counts, "test_first": test.target_dates[0].strftime("%Y-%m-%d"),
                "test_last": test.target_dates[-1].strftime("%Y-%m-%d"), "runs": []}
    _write_json(run_dir / "run.json", manifest)
    try:
        for group in GROUPS:
            feature_group = dataset.groups[group]
            for seed in SEEDS:
                relative = f"{group}/seed_{seed}"
                try:
                    result = train_one(feature_group, config, seed, run_dir / relative,
                                       model_builder=build_gru_model, model_type="GRU")
                except Exception as error:
                    manifest["runs"].append({"group": group, "seed": seed, "status": "failed",
                                             "directory": relative, "error": str(error)})
                    _write_json(run_dir / "run.json", manifest)
                    raise
                manifest["runs"].append({"group": group, "seed": seed,
                                         "directory": relative, **result})
                _write_json(run_dir / "run.json", manifest)
                print(f"GRU completed {group}/seed_{seed}: {result['epochs_trained']} epochs, "
                      f"{result['elapsed_seconds']:.1f}s", flush=True)

        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(exist_ok=False)
        per_seed_rows = []
        for group in GROUPS:
            for seed in SEEDS:
                lstm_metrics, lstm_parameters, lstm_epochs, lstm_seconds = baseline_metrics[group, seed]
                gru_metrics, _ = _predict_gru(run_dir, dataset, config, group, seed)
                for model, metrics, parameters, epochs, seconds in (
                    ("LSTM", lstm_metrics, lstm_parameters, lstm_epochs, lstm_seconds),
                    ("GRU", gru_metrics,
                     _read_json(run_dir / group / f"seed_{seed}" / "training.json")["parameter_count"],
                     _read_json(run_dir / group / f"seed_{seed}" / "training.json")["epochs_trained"],
                     _read_json(run_dir / group / f"seed_{seed}" / "training.json")["elapsed_seconds"]),
                ):
                    per_seed_rows.append({"group": group, "model": model, "seed": seed,
                                          "n_samples": len(dataset.groups[group].splits["test"].X),
                                          "parameter_count": parameters, "epochs_trained": epochs,
                                          "train_seconds": seconds, **metrics})
        summary = _summary(per_seed_rows)
        _write_csv(evaluation_dir / "comparison_per_seed.csv",
                   ["group", "model", "seed", "n_samples", "parameter_count", "epochs_trained",
                    "train_seconds", "mae", "rmse", "directional_accuracy"], per_seed_rows)
        _write_csv(evaluation_dir / "comparison_summary.csv",
                   ["group", "model", "n_seeds", "parameter_count", "train_seconds_mean",
                    "mae_mean", "mae_std", "rmse_mean", "rmse_std",
                    "directional_accuracy_mean", "directional_accuracy_std"], summary)
        _write_json(evaluation_dir / "evaluation.json", {
            "schema_version": 1, "stage": "gru_comparison_evaluation", "status": "completed",
            "completed_at_utc": _now(), "baseline_run": str(baseline_dir),
            "sample_counts": sample_counts, "metrics_per_seed": per_seed_rows,
            "metrics_summary": summary,
        })
        _write_report(run_dir / "comparison.md", baseline_dir, config,
                      {"sample_counts": sample_counts,
                       "test_dates": [manifest["test_first"], manifest["test_last"]]},
                      _read_json(run_dir / "environment.json"), per_seed_rows, summary,
                      scaler_hashes, data_hash, calendar_hash)
        manifest.update({"status": "completed", "completed_at_utc": _now(),
                         "elapsed_seconds": sum(row["elapsed_seconds"] for row in manifest["runs"]),
                         "total_elapsed_seconds": time.perf_counter() - total_started,
                         "comparison": "comparison.md",
                         "prediction_files": [f"evaluation/predictions_{group}_seed_{seed}.csv"
                                              for group in GROUPS for seed in SEEDS]})
        _write_json(run_dir / "run.json", manifest)
        print(f"GRU comparison completed: {run_dir}", flush=True)
        return run_dir
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error), "failed_at_utc": _now()})
        _write_json(run_dir / "run.json", manifest)
        raise GRUExperimentError(f"GRU comparison failed; partial artifacts retained at {run_dir}: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and compare GRU against the frozen LSTM run.")
    parser.add_argument("--baseline-run-dir", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)
    run_gru_experiment(args.baseline_run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
