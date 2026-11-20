"""CPU training and traceable artifacts; price evaluation is a later stage."""

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import time
from uuid import uuid4

import numpy as np

from .data import load_snapshot
from .dataset import build_dataset
from .model import build_model


class TrainingError(RuntimeError):
    """Training or saving an experiment failed; partial artifacts are retained."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _callbacks(config):
    import tensorflow as tf

    class FiniteLoss(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if not np.isfinite((logs or {}).get("loss", np.nan)):
                raise TrainingError("訓練 loss 不是有限數值")

        def on_epoch_end(self, epoch, logs=None):
            for name in ("loss", "val_loss"):
                if not np.isfinite((logs or {}).get(name, np.nan)):
                    raise TrainingError(f"{name} 缺漏或不是有限數值")

    settings = config["training"]["early_stopping"]
    return [FiniteLoss(), tf.keras.callbacks.EarlyStopping(
        monitor=settings["monitor"], patience=settings["patience"],
        restore_best_weights=settings["restore_best_weights"], mode="min",
    )]


def _batches(split, batch_size):
    import tensorflow as tf

    # Bounded private pools avoid creating a large new pool for each fit while
    # keeping the original chronological order. No shuffle operation is used.
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return tf.data.Dataset.from_tensor_slices((split.X, split.y)).batch(batch_size).with_options(options)


def _validation_predictions(model, split, batch_size):
    predictions = np.concatenate([
        np.asarray(model(split.X[first:first + batch_size], training=False))
        for first in range(0, len(split.X), batch_size)
    ])
    if predictions.shape != split.y.shape or not np.isfinite(predictions).all():
        raise TrainingError("最佳權重的驗證預測形狀錯誤或不是有限數值")
    return predictions


def train_one(group, config, seed, output_dir):
    """Fit one independent model on train/validation and save restored weights.

    No test inputs or labels are used. The output directory must be new;
    training.json records failures as well as successful runs.
    """
    output_dir = Path(output_dir)
    started = time.perf_counter()
    summary = {"stage": "training", "status": "running", "seed": seed,
               "columns": list(group.columns), "started_at_utc": _now()}
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise TrainingError(f"無法建立新的訓練目錄 {output_dir}：{error}") from error
    try:
        _write_json(output_dir / "training.json", summary)
        train, validation = group.splits["train"], group.splits["validation"]
        for split in (train, validation):
            if not len(split.X) or not np.isfinite(split.X).all() or not np.isfinite(split.y).all():
                raise TrainingError("訓練與驗證資料必須非空且為有限數值")
        model = build_model(train.X.shape[1:], config, seed)
        summary["parameter_count"] = int(model.count_params())
        settings = config["training"]
        history = model.fit(
            _batches(train, settings["batch_size"]),
            validation_data=_batches(validation, settings["batch_size"]),
            epochs=settings["max_epochs"], shuffle=settings["shuffle"],
            callbacks=_callbacks(config), verbose=0,
        ).history
        losses = np.asarray(history.get("loss", []), dtype=float)
        val_losses = np.asarray(history.get("val_loss", []), dtype=float)
        if (not len(losses) or losses.shape != val_losses.shape
                or not np.isfinite(losses).all() or not np.isfinite(val_losses).all()):
            raise TrainingError("訓練歷程的 loss／val_loss 缺漏或不是有限數值")
        _validation_predictions(model, validation, settings["batch_size"])
        # EarlyStopping.on_train_end restores the best validation weights,
        # including when training reaches max_epochs without stopping early.
        model.save_weights(output_dir / "best.weights.h5")
        with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["epoch", "loss", "val_loss"])
            writer.writerows((epoch + 1, float(loss), float(val_loss))
                             for epoch, (loss, val_loss) in enumerate(zip(losses, val_losses)))
        summary.update({"status": "completed", "epochs_trained": len(losses),
                        "stopped_early": len(losses) < settings["max_epochs"],
                        "best_epoch": int(np.argmin(val_losses)) + 1,
                        "best_val_loss": float(val_losses.min()),
                        "train_samples": len(train.X), "validation_samples": len(validation.X),
                        "weights": "best.weights.h5", "history": "history.csv",
                        "completed_at_utc": _now(), "elapsed_seconds": time.perf_counter() - started})
        _write_json(output_dir / "training.json", summary)
        return summary
    except Exception as error:
        summary.update({"status": "failed", "error": str(error), "failed_at_utc": _now(),
                        "elapsed_seconds": time.perf_counter() - started})
        try:
            _write_json(output_dir / "training.json", summary)
        except OSError:
            pass
        raise TrainingError(f"seed {seed} 訓練失敗：{error}") from error


def _environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": "cpu", "deterministic_ops": True,
            "packages": {name: version(name) for name in (
                "tensorflow-cpu", "keras", "numpy", "pandas", "scikit-learn",
                "matplotlib", "yfinance", "exchange-calendars", "lstm-comparison")}}


def run_experiment(config):
    """Validate a frozen snapshot offline before creating a unique training run."""
    # A missing/invalid snapshot or dataset must not create an experiment dir.
    frame, metadata = load_snapshot(config)
    dataset = build_dataset(frame, config)
    run_dir = Path(config["data"]["output_dir"]) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise TrainingError(f"無法建立新的實驗目錄：{error}") from error
    started = time.perf_counter()
    manifest = {"schema_version": 1, "stage": "training", "status": "running",
                "started_at_utc": _now(), "data_sha256": metadata["data"]["sha256"],
                "groups": list(dataset.groups), "seeds": list(config["training"]["seeds"]),
                "expected_runs": len(dataset.groups) * len(config["training"]["seeds"]), "runs": []}
    try:
        _write_json(run_dir / "run.json", manifest)
        _write_json(run_dir / "config.json", config)
        _write_json(run_dir / "snapshot.metadata.json", metadata)
        _write_json(run_dir / "environment.json", _environment())
        scalers = run_dir / "scalers"
        scalers.mkdir()
        with (scalers / "y.pkl").open("xb") as handle:
            pickle.dump(dataset.y_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for name, group in dataset.groups.items():
            with (scalers / f"X_{name}.pkl").open("xb") as handle:
                pickle.dump(group.x_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _write_json(scalers / "description.json", {
            "X": {name: {"columns": list(group.columns), "data_min": group.x_scaler.data_min_.tolist(),
                         "data_max": group.x_scaler.data_max_.tolist()} for name, group in dataset.groups.items()},
            "y": {"columns": ["Close"], "data_min": dataset.y_scaler.data_min_.tolist(),
                  "data_max": dataset.y_scaler.data_max_.tolist()}, "clip": False,
        })
        for name, group in dataset.groups.items():
            for seed in config["training"]["seeds"]:
                relative = f"{name}/seed_{seed}"
                try:
                    result = train_one(group, config, seed, run_dir / relative)
                except TrainingError as error:
                    manifest["runs"].append({"group": name, "seed": seed, "directory": relative,
                                             "status": "failed", "error": str(error)})
                    raise
                if result.get("status") != "completed":
                    raise TrainingError(f"{relative} 未完成，不能將實驗標記成功")
                manifest["runs"].append({"group": name, "directory": relative, **result})
                _write_json(run_dir / "run.json", manifest)
        manifest.update({"status": "completed", "completed_at_utc": _now(),
                         "elapsed_seconds": time.perf_counter() - started})
        _write_json(run_dir / "run.json", manifest)
        return run_dir
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error), "failed_at_utc": _now(),
                         "elapsed_seconds": time.perf_counter() - started})
        try:
            _write_json(run_dir / "run.json", manifest)
        except OSError:
            pass
        raise TrainingError(f"實驗訓練失敗，保留目錄 {run_dir}：{error}") from error
