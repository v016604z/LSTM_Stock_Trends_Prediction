"""Validate saved results and generate an offline Chinese research report."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import calculate_metrics, error_improvement_pct, summarize_metrics


class ReportError(RuntimeError):
    """Saved results are inconsistent, or report generation failed."""


PREDICTION_COLUMNS = ["target_date", "input_end_date", "group", "seed", "actual_close", "previous_close",
                      "predicted_close", "absolute_error", "squared_error", "actual_direction",
                      "predicted_direction", "direction_correct"]
METRIC_COLUMNS = ["group", "seed", "n_samples", "mae", "rmse", "directional_accuracy",
                  "mae_improvement_pct", "rmse_improvement_pct"]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _number(value):
    if isinstance(value, bool):
        raise ReportError("數值不可為布林值")
    result = float(value)
    if not np.isfinite(result):
        raise ReportError("報告來源含有非有限數值")
    return result


def _matches(actual, expected):
    if expected is None:
        return actual is None or actual == "N/A"
    if actual is None or actual == "N/A":
        return False
    return bool(np.isclose(_number(actual), expected, rtol=1e-10, atol=1e-12))


def _check_rows(rows, expected, identity, columns):
    def key(row):
        return tuple(str(row[column]) for column in identity)

    found = {}
    for row in rows:
        current = key(row)
        if current in found:
            raise ReportError("績效表有重複組別／種子")
        found[current] = row
    expected_rows = {key(row): row for row in expected}
    if set(found) != set(expected_rows):
        raise ReportError("績效表的組別／種子與逐日預測不符")
    for current, wanted in expected_rows.items():
        for column in columns:
            if not _matches(found[current][column], wanted[column]):
                raise ReportError(f"績效 {current}/{column} 與逐日 CSV 重算不符")


def _load_bundle(run_dir):
    hashes = {}

    def read(relative):
        data = (run_dir / relative).read_bytes()
        hashes[relative] = sha256(data).hexdigest()
        return data.decode("utf-8")

    def reject_constant(value):
        raise ReportError(f"JSON 含有不合法數值 {value}")

    def document(relative):
        value = json.loads(read(relative), parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ReportError(f"{relative} 必須是 JSON 物件")
        return value

    def table(relative, columns):
        reader = csv.DictReader(StringIO(read(relative)))
        if reader.fieldnames != columns:
            raise ReportError(f"{relative} 欄位或順序不符")
        rows = list(reader)
        if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ReportError(f"{relative} 為空或資料欄數不符")
        return pd.DataFrame(rows, columns=columns)

    document("config.json")
    config = load_config(run_dir / "config.json")
    training_run = document("run.json")
    evaluation = document("evaluation/evaluation.json")
    metadata = document("snapshot.metadata.json")
    environments = {"training": document("environment.json"), "evaluation": document("evaluation/environment.json")}
    groups, seeds = config["features"]["groups"], config["training"]["seeds"]
    if config["evaluation"]["plot_seed"] != 42 or 42 not in seeds:
        raise ReportError("價格圖固定展示 seed 42；不可改用其他或最佳種子")
    expected_pairs = {(name, seed) for name in groups for seed in seeds}
    source_artifacts = evaluation.get("artifact_sha256", {})
    expected_artifacts = {"config.json", "scalers/y.pkl", *(f"scalers/X_{name}.pkl" for name in groups),
                          *(f"{name}/seed_{seed}/best.weights.h5" for name, seed in expected_pairs)}
    if not isinstance(source_artifacts, dict) or set(source_artifacts) != expected_artifacts:
        raise ReportError("評估保存的來源權重／縮放器 hash 清單不完整")
    for digest in [*source_artifacts.values(), metadata["data"]["sha256"], metadata["calendar"]["sha256"]]:
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in digest)):
            raise ReportError("保存的 SHA256 格式無效")
    if (training_run.get("stage") != "training" or training_run.get("status") != "completed"
            or training_run.get("schema_version") != 1 or training_run.get("groups") != list(groups)
            or training_run.get("seeds") != seeds or training_run.get("expected_runs") != len(expected_pairs)
            or evaluation.get("stage") != "evaluation" or evaluation.get("status") != "completed"
            or evaluation.get("schema_version") != 1 or evaluation.get("split") != "test"
            or evaluation.get("groups") != list(groups) or evaluation.get("seeds") != seeds):
        raise ReportError("必須提供設定一致且已完成的訓練與測試集評估")
    if (evaluation.get("config_sha256") != hashes["config.json"]
            or evaluation.get("artifact_sha256", {}).get("config.json") != hashes["config.json"]
            or training_run.get("data_sha256") != metadata["data"]["sha256"]
            or evaluation.get("data_sha256") != metadata["data"]["sha256"]
            or evaluation.get("calendar_sha256") != metadata["calendar"]["sha256"]):
        raise ReportError("設定、來源行情或交易日 hash 與保存紀錄不符")
    parameters = metadata["source"]["parameters"]
    for key, expected in {"tickers": config["data"]["ticker"], "start": config["data"]["start"],
                          "end": config["data"]["end_exclusive"], "interval": config["data"]["interval"],
                          "auto_adjust": config["data"]["auto_adjust"]}.items():
        if parameters[key] != expected:
            raise ReportError("來源下載參數與報告設定不符")
    if metadata["source"]["name"] != "yfinance":
        raise ReportError("來源紀錄必須為 yfinance")

    expected_files = {"predictions_naive.csv", *(f"predictions_{name}_seed_{seed}.csv" for name, seed in expected_pairs)}
    if (not isinstance(evaluation.get("prediction_files"), list)
            or len(evaluation["prediction_files"]) != len(expected_files)
            or set(evaluation["prediction_files"]) != expected_files):
        raise ReportError("逐日預測檔案清單與設定種子不符")

    def predictions(name, seed):
        filename = "predictions_naive.csv" if name == "Naive" else f"predictions_{name}_seed_{seed}.csv"
        frame = table(f"evaluation/{filename}", PREDICTION_COLUMNS)
        if not frame["group"].eq(name).all() or not frame["seed"].eq("N/A" if seed is None else str(seed)).all():
            raise ReportError(f"{filename} 的組別或種子不符")
        for column in ("target_date", "input_end_date"):
            dates = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
            if dates.dt.strftime("%Y-%m-%d").tolist() != frame[column].tolist():
                raise ReportError("逐日日期必須是 ISO YYYY-MM-DD")
            frame[column] = dates
        if (frame["target_date"].duplicated().any() or not frame["target_date"].is_monotonic_increasing
                or not (frame["input_end_date"] < frame["target_date"]).all()
                or (frame["target_date"] < pd.Timestamp(config["split"]["test"]["start"])).any()
                or (frame["target_date"] > pd.Timestamp(config["split"]["test"]["end"])).any()):
            raise ReportError("逐日日期重複、錯序、輸入日錯置或超出測試期間")
        numeric = ["actual_close", "previous_close", "predicted_close", "absolute_error", "squared_error",
                   "actual_direction", "predicted_direction"]
        for column in numeric:
            frame[column] = frame[column].map(_number)
        actual, previous, predicted = (frame[column].to_numpy() for column in ("actual_close", "previous_close", "predicted_close"))
        if (actual <= 0).any() or (previous <= 0).any():
            raise ReportError("逐日真值與前收盤必須是正價格")
        input_days = frame["input_end_date"].to_numpy()
        target_days = frame["target_date"].to_numpy()
        if (input_days[1:] < target_days[:-1]).any():
            raise ReportError("下一筆輸入結束日不可早於前一筆預測目標日")
        adjacent = input_days[1:] == target_days[:-1]
        if not np.array_equal(previous[1:][adjacent], actual[:-1][adjacent]):
            raise ReportError("相鄰交易日的前收盤與已知上一筆真值不符")
        with np.errstate(over="raise", invalid="raise"):
            absolute, squared = np.abs(predicted - actual), np.square(predicted - actual)
        directions = ((actual > previous).astype(int) - (actual < previous).astype(int),
                      (predicted > previous).astype(int) - (predicted < previous).astype(int))
        for column, expected in zip(("absolute_error", "squared_error", "actual_direction", "predicted_direction"),
                                    (absolute, squared, *directions)):
            if not np.allclose(frame[column].to_numpy(), expected, rtol=1e-10, atol=1e-12):
                raise ReportError(f"{filename}/{column} 與原價重算不符")
        if name == "Naive":
            if not np.array_equal(predicted, previous) or not frame["direction_correct"].eq("N/A").all():
                raise ReportError("Naive 必須使用前收盤，方向正確率為 N/A")
        elif not np.array_equal(frame["direction_correct"].map(_number).to_numpy(), directions[0] == directions[1]):
            raise ReportError("逐日方向正確性與原價重算不符")
        return frame, calculate_metrics(actual, predicted, previous, naive=name == "Naive")

    naive_frame, naive_metrics = predictions("Naive", None)
    if (evaluation.get("n_samples") != len(naive_frame)
            or evaluation.get("first_target") != naive_frame["target_date"].iloc[0].strftime("%Y-%m-%d")
            or evaluation.get("last_target") != naive_frame["target_date"].iloc[-1].strftime("%Y-%m-%d")):
        raise ReportError("評估樣本數或日期範圍與逐日資料不符")
    all_predictions, metric_rows = {}, []
    aligned = ["target_date", "input_end_date", "actual_close", "previous_close"]
    for name in groups:
        for seed in seeds:
            frame, metrics = predictions(name, seed)
            if not frame[aligned].equals(naive_frame[aligned]):
                raise ReportError("模型與 Naive 的日期、真值或前收盤未完全對齊")
            all_predictions[name, seed] = frame
            metric_rows.append({"group": name, "seed": seed, "n_samples": len(frame), **metrics,
                                "mae_improvement_pct": error_improvement_pct(metrics["mae"], naive_metrics["mae"]),
                                "rmse_improvement_pct": error_improvement_pct(metrics["rmse"], naive_metrics["rmse"])})
    all_rows = metric_rows + [{"group": "Naive", "seed": None, "n_samples": len(naive_frame), **naive_metrics,
                              "mae_improvement_pct": None, "rmse_improvement_pct": None}]
    summary = summarize_metrics(metric_rows, naive_metrics)
    csv_per_seed = table("evaluation/metrics_per_seed.csv", METRIC_COLUMNS).to_dict("records")
    # CSV uses N/A for the baseline seed; JSON uses null.
    for row in csv_per_seed:
        if row["seed"] == "N/A":
            row["seed"] = None
    _check_rows(csv_per_seed, all_rows, ["group", "seed"], METRIC_COLUMNS[2:])
    _check_rows(evaluation["metrics_per_seed"], all_rows, ["group", "seed"], METRIC_COLUMNS[2:])
    summary_columns = list(summary[0])
    _check_rows(table("evaluation/metrics_summary.csv", summary_columns).to_dict("records"), summary,
                ["group"], summary_columns[1:])
    _check_rows(evaluation["metrics_summary"], summary, ["group"], summary_columns[1:])

    entries = training_run["runs"]
    if len(entries) != len(expected_pairs):
        raise ReportError("訓練清單數量不符")
    histories, training_details = {}, {}
    for entry in entries:
        pair = (entry["group"], entry["seed"])
        if pair not in expected_pairs or pair in histories or entry["status"] != "completed":
            raise ReportError("訓練清單有缺漏、重複或失敗")
        name, seed = pair
        directory = f"{name}/seed_{seed}"
        detail = document(f"{directory}/training.json")
        history = table(f"{directory}/history.csv", ["epoch", "loss", "val_loss"])
        for column in history:
            history[column] = history[column].map(_number)
        if (entry.get("directory") != directory or detail.get("stage") != "training"
                or detail.get("status") != "completed" or detail.get("seed") != seed
                or detail.get("columns") != groups[name]
                or not np.array_equal(history["epoch"].to_numpy(), np.arange(1, len(history) + 1))
                or (history[["loss", "val_loss"]] < 0).any().any()
                or type(detail.get("epochs_trained")) is not int
                or detail["epochs_trained"] != len(history)
                or len(history) > config["training"]["max_epochs"]
                or detail["epochs_trained"] != entry.get("epochs_trained")
                or type(detail.get("best_epoch")) is not int
                or detail["best_epoch"] != int(history["val_loss"].to_numpy().argmin()) + 1
                or detail["best_epoch"] != entry.get("best_epoch")
                or not _matches(detail["best_val_loss"], history["val_loss"].min())
                or not _matches(entry.get("best_val_loss"), history["val_loss"].min())):
            raise ReportError(f"{directory} 的歷程、epochs 或最佳驗證回合不符")
        units, width = config["model"]["lstm_units"], len(groups[name])
        parameter_count = 4 * units * (units + width + 1) + units + 1
        if detail.get("parameter_count") != parameter_count or entry.get("parameter_count") != parameter_count:
            raise ReportError("模型參數量與設定不符")
        histories[pair], training_details[pair] = history, detail
    currency = metadata.get("currency", metadata["source"].get("currency"))
    unit = "TWD" if config["data"]["ticker"] == "2330.TW" else currency if isinstance(currency, str) and currency else "source price unit"
    return {"ticker": config["data"]["ticker"], "price_unit": unit, "seeds": seeds, "plot_seed": 42,
            "predictions": all_predictions, "naive_predictions": naive_frame, "summary": summary,
            "histories": histories, "training_details": training_details, "per_seed": all_rows,
            "config": config, "metadata": metadata, "evaluation": evaluation,
            "environments": environments, "input_hashes": hashes}


def _format(value, *, percent=False):
    return "N/A" if value is None else f"{value * (100 if percent else 1):.6g}"


def _markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def _result_analysis(bundle):
    summaries = {row["group"]: row for row in bundle["summary"]}
    rows = {(row["group"], row["seed"]): row for row in bundle["per_seed"]}
    paragraphs = []
    for title, left, right in (("RQ1：OHLCV 與單變量比較", "B", "A"), ("RQ2：技術指標的影響", "C", "B")):
        left_mae, right_mae = summaries[left]["mae_mean"], summaries[right]["mae_mean"]
        relation = "低於" if left_mae < right_mae else "高於" if left_mae > right_mae else "等於"
        differences = [rows[left, seed]["mae"] - rows[right, seed]["mae"] for seed in bundle["seeds"]]
        improved, tied, worse = sum(value < 0 for value in differences), sum(value == 0 for value in differences), sum(value > 0 for value in differences)
        consistency = "不同種子的比較方向不一致。" if improved and worse else "此計數只描述本次有限種子，不代表統計顯著。"
        paragraphs.append(f"**{title}：**{left} 平均 MAE {_format(left_mae)} {relation} {right} 的 {_format(right_mae)}；"
                          f"逐種子為 {improved} 次改善、{tied} 次平手、{worse} 次變差。{consistency}")
    for name in ("A", "B", "C"):
        row, baseline = summaries[name], summaries["Naive"]
        relation = "優於" if row["mae_mean"] < baseline["mae_mean"] else "劣於" if row["mae_mean"] > baseline["mae_mean"] else "等於"
        differences = [rows[name, seed]["mae"] - baseline["mae_mean"] for seed in bundle["seeds"]]
        mae_improvement = "N/A" if row["mae_improvement_pct"] is None else f"{_format(row['mae_improvement_pct'])}%"
        rmse_improvement = "N/A" if row["rmse_improvement_pct"] is None else f"{_format(row['rmse_improvement_pct'])}%"
        paragraphs.append(f"**RQ3：{name} 與 Naive：**平均 MAE {relation} Naive；MAE 改善率 {mae_improvement}，"
                          f"RMSE 改善率 {rmse_improvement}。逐種子有 {sum(value < 0 for value in differences)} 次低於 Naive、"
                          f"{sum(value == 0 for value in differences)} 次平手、{sum(value > 0 for value in differences)} 次高於 Naive。")
    return "\n\n".join(paragraphs)


def _markdown(bundle, figures):
    config, metadata = bundle["config"], bundle["metadata"]
    training, model = config["training"], config["model"]
    evaluation = bundle["evaluation"]
    formal = (config["data"]["ticker"] == "2330.TW" and config["data"]["start"] == "2015-01-01"
              and config["data"]["end"] == "2025-12-31" and config["sequence"]["lookback"] == 60
              and training["seeds"] == [42, 43, 44] and training["max_epochs"] == 100
              and training["batch_size"] == 32 and training["learning_rate"] == 0.001
              and training["early_stopping"]["patience"] == 10 and model["lstm_units"] == 64
              and config["split"] == {"train": {"start": "2015-01-01", "end": "2022-12-31"},
                                        "validation": {"start": "2023-01-01", "end": "2023-12-31"},
                                        "test": {"start": "2024-01-01", "end": "2025-12-31"}})
    status = ("本次設定符合正式研究預設，資料品質與正式實驗驗收仍須於 Commit 8 確認。" if formal else
              "本次為短測試或自訂實驗，設定與正式研究預設不同，不能視為正式研究結果。")
    per_seed = _markdown_table(["組別", "seed", "MAE", "RMSE", "方向正確率（%）", "MAE 改善（%）", "RMSE 改善（%）"],
        [[row["group"], "N/A" if row["seed"] is None else row["seed"], _format(row["mae"]), _format(row["rmse"]),
          _format(row["directional_accuracy"], percent=True), _format(row["mae_improvement_pct"]), _format(row["rmse_improvement_pct"])]
         for row in bundle["per_seed"]])
    def aggregate(row, key, percent=False):
        value = _format(row[f"{key}_mean"], percent=percent)
        spread = row[f"{key}_std"]
        return value if value == "N/A" else f"{value}（std N/A）" if spread is None else f"{value} ± {_format(spread, percent=percent)}"
    summary = _markdown_table(["組別", "種子數", "MAE 平均±樣本std", "RMSE 平均±樣本std", "方向（%）平均±樣本std", "MAE 改善（%）", "RMSE 改善（%）"],
        [[row["group"], row["n_seeds"], aggregate(row, "mae"), aggregate(row, "rmse"), aggregate(row, "directional_accuracy", True),
          _format(row["mae_improvement_pct"]), _format(row["rmse_improvement_pct"])] for row in bundle["summary"]])
    models = _markdown_table(["組別", "特徵", "欄數", "參數量"],
        [[name, "、".join(columns), len(columns), bundle["training_details"][name, bundle["seeds"][0]]["parameter_count"]]
         for name, columns in config["features"]["groups"].items()])
    epochs = _markdown_table(["組別", "seed", "實際 epochs", "最佳 epoch"],
        [[name, seed, detail["epochs_trained"], detail["best_epoch"]] for (name, seed), detail in bundle["training_details"].items()])
    split = "；".join(f"{name}：{bounds['start']}–{bounds['end']}" for name, bounds in config["split"].items())
    images = "\n\n".join(f"![{label}](figures/{figures[key].name})" for key, label in (
        ("prices", "測試價格與 seed 42 預測"), ("errors", "全種子 MAE／RMSE 比較"),
        ("direction", "全種子方向正確率，Naive N/A"), ("loss", "全部設定種子的訓練與驗證 loss")))
    return f"""# 單變量與多變量 LSTM 股票價格預測比較｜實驗報告

## 一、本次實驗概況

{status} 本報告自動整理保存產物，任何情況均不自動宣告 Commit 8 正式研究完成。

股票：{bundle['ticker']}；設定資料期間：{config['data']['start']}–{config['data']['end']}；實際測試目標：{evaluation['first_target']}–{evaluation['last_target']}，共 {evaluation['n_samples']} 筆。
種子：{', '.join(map(str, bundle['seeds']))}；最大 epochs：{training['max_epochs']}；各次實際 epochs 見下表。

## 二、資料與研究方法

來源：{metadata['source']['name']}，下載時間：{metadata['downloaded_at_utc']}；auto_adjust=True，保留來源成交量，不補週末或休市日。
使用過去 {config['sequence']['lookback']} 個交易日，預測下一交易日調整後 Close。技術指標採向後 MA、Wilder RSI14、SMA 初始化的 MACD 12/26/9；C 的 33 筆暖機與完整視窗決定共同日期。
依目標日期切分：{split}。跨年度輸入保留已知歷史，X 僅以共同訓練輸入位置聯集擬合，y 僅以共同訓練目標擬合；測試只載回縮放器，不重新 fit、不裁切。
Naive 使用前一交易日 Close；方向以同一前收盤判斷上漲、下跌、持平，不加 epsilon；Naive 方向正確率為 N/A。

## 三、模型與訓練設定

三組共用單向 LSTM({model['lstm_units']}) → Dense(1)，Adam learning rate {training['learning_rate']}、MSE、batch size {training['batch_size']}、shuffle=False、CPU。
早停只監看 val_loss，patience={training['early_stopping']['patience']}，還原最佳權重，不以 test 選種子或權重。輸入維度不同，參數量也不同，並非完全等參數比較。

{models}

{epochs}

## 四、逐種子與彙總績效

MAE 為主要比較指標；MAE／RMSE 單位：調整後價格（{bundle['price_unit']}）。方向正確率在 CSV 保存 0–1，本報告以百分比顯示。
彙總為算術平均與樣本標準差（ddof=1）；單種子 std 為 N/A。改善率為 100×(Naive誤差−模型誤差)/Naive誤差；負值表示變差，Naive 誤差為零時為 N/A。Naive 只有單份預測，種子數為 0。

{per_seed}

{summary}

## 五、研究圖表

價格圖固定 seed 42，不挑最佳 seed、不合成 ensemble。誤差與方向圖使用全部設定種子的平均及樣本 std；Naive 方向顯示 N/A。loss 圖顯示所有設定種子，單位為縮放後 Close 的 MSE，與原價 MAE／RMSE 不同。

{images}

## 六、依本次結果回答研究問題

{_result_analysis(bundle)}

以上以平均 MAE 描述本次比較，RMSE 與方向結果另列於表格，沒有進行或宣稱統計顯著性檢定。觀察到的特徵影響不應推論為普遍因果關係。

## 七、研究限制

- 單一股票與固定期間，結果未必適用其他產業、市場或時期。
- 突發事件、基本面及其他未觀測因素未納入，歷史技術特徵不能保證未來預測能力。
- 行情可能修訂，來源缺日、價量品質及交易日曆正確性仍須查證；保存 metadata／hash 不等於資料品質已通過正式研究驗收。
- 僅有 {len(bundle['seeds'])} 個固定種子；有限次重複不代表統計顯著，單種子無法估計樣本標準差。
- 特徵欄數造成模型參數量差異，不能將差異全部歸因於特徵本身。
- 研究只衡量預測誤差，不等同投資報酬、交易策略績效或已完成回測。

## 八、重現與來源紀錄

行情 SHA256：`{evaluation['data_sha256']}`；交易日 SHA256：`{evaluation['calendar_sha256']}`。
設定 SHA256：`{evaluation['config_sha256']}`。訓練與評估環境見 [訓練環境](../environment.json)、[評估環境](../evaluation/environment.json)。
來源詳見 [設定](../config.json)、[metadata](../snapshot.metadata.json)、[逐種子績效](../evaluation/metrics_per_seed.csv)、[彙總績效](../evaluation/metrics_summary.csv)、[評估紀錄](../evaluation/evaluation.json)。
本次報告輸入與輸出 hash、產生環境及狀態保存於 [report.json](report.json)；權重／縮放器沿用評估保存的來源 hash 作追溯，本步驟不需重新載入原始行情、權重或 pickle。

```text
python -m lstm_comparison validate-config --config <設定檔>
python -m lstm_comparison download --config <設定檔>
python -m lstm_comparison run --config <設定檔>
python -m lstm_comparison evaluate --run-dir <實驗目錄>
python -m lstm_comparison report --run-dir <實驗目錄>
```

下載與正式訓練須另行執行並先滿足來源品質要求；本次產生報告沒有進行下載、訓練或重新評估。
"""


def generate_report(run_dir):
    """Generate a new report solely from validated saved evaluation artifacts."""
    run_dir = Path(run_dir).expanduser().resolve()
    output = run_dir / "report"
    if output.exists():
        raise ReportError(f"報告目錄已存在，不覆寫：{output}")
    try:
        bundle = _load_bundle(run_dir)
    except Exception as error:
        raise ReportError(f"無法驗證報告來源：{error}") from error
    try:
        output.mkdir(exist_ok=False)
    except OSError as error:
        raise ReportError(f"無法建立新的報告目錄：{error}") from error
    manifest = {"schema_version": 1, "stage": "report", "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "plot_seed": 42,
                "input_sha256": bundle["input_hashes"],
                "prediction_source_sha256": bundle["evaluation"]["artifact_sha256"]}
    try:
        _write_json(output / "report.json", manifest)
        from .plots import render_figures

        figures = render_figures(bundle, output / "figures")
        if set(figures) != {"prices", "errors", "direction", "loss"}:
            raise ReportError("必須產生價格、誤差、方向與 loss 四張圖")
        for path in figures.values():
            if not Path(path).is_file() or not Path(path).resolve().is_relative_to(output.resolve()):
                raise ReportError("圖檔缺漏或不位於本次報告目錄")
        (output / "report.md").write_text(_markdown(bundle, figures), encoding="utf-8")
        manifest.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "environment": {"python": platform.python_version(), "packages": {
                             name: version(name) for name in ("numpy", "pandas", "matplotlib", "lstm-comparison")}},
                         "output_sha256": {str(path.relative_to(output)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
                                           for path in [output / "report.md", *figures.values()]}})
        _write_json(output / "report.json", manifest)
        return output
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error), "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        manifest.pop("completed_at_utc", None)
        try:
            _write_json(output / "report.json", manifest)
        except OSError:
            pass
        raise ReportError(f"報告產生失敗，保留目錄 {output}：{error}") from error
