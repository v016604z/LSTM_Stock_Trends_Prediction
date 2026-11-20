"""Command-line entry points; help and validation use only the standard library."""

import argparse
import json
import sys

from .config import ConfigError, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m lstm_comparison",
        description="LSTM 股票價格比較研究；提供設定檢查、行情快照與重複訓練。",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("validate-config", "檢查 JSON 設定與解析資料路徑，不下載或訓練"),
        ("download", "下載新行情快照，或離線重用既有快照；不覆寫"),
        ("load-snapshot", "離線讀取快照，檢查來源參數、交易日與 SHA256"),
        ("run", "離線驗證既有快照，執行三組模型重複訓練並保存權重與歷程"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--config", required=True, help="JSON 設定檔路徑")
    report = commands.add_parser("report", help="依保存的評估與訓練歷程產生圖表及中文實驗報告")
    report.add_argument("--run-dir", required=True, help="實驗目錄")
    evaluate = commands.add_parser("evaluate", help="載回已完成訓練，離線評估共同測試日期與 Naive")
    evaluate.add_argument("--run-dir", required=True, help="已完成的訓練實驗目錄")
    args = parser.parse_args(argv)

    if args.command == "report":
        from .report import ReportError, generate_report

        try:
            output = generate_report(args.run_dir)
        except ReportError as error:
            print(f"報告錯誤：{error}", file=sys.stderr)
            return 1
        print(f"研究圖表與實驗報告已產生：{output}")
        print("報告描述保存的實驗結果；不代表 Commit 8 正式研究已完成。")
        return 0

    if args.command == "evaluate":
        from .evaluation import EvaluationError, evaluate_experiment

        try:
            output = evaluate_experiment(args.run_dir)
        except EvaluationError as error:
            print(f"評估錯誤：{error}", file=sys.stderr)
            return 1
        print(f"測試集評估完成：{output}")
        print("已保存逐日預測與績效表；可使用 report --run-dir 產生圖表與中文報告。")
        return 0

    if args.command in ("validate-config", "download", "load-snapshot", "run"):
        try:
            config = load_config(args.config)
        except ConfigError as error:
            print(f"設定錯誤：{error}", file=sys.stderr)
            return 2
        if args.command == "validate-config":
            print("設定檢查通過；尚未下載資料或訓練模型。")
            print(json.dumps(config, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            from .data import SnapshotError
            from .dataset import DatasetError
            from .training import TrainingError, run_experiment

            try:
                run_dir = run_experiment(config)
            except (SnapshotError, DatasetError, TrainingError) as error:
                print(f"訓練錯誤：{error}", file=sys.stderr)
                return 1
            print(f"訓練階段完成：{run_dir}")
            print("已保存權重與歷程；後續階段可使用 evaluate --run-dir 評估，再以 report --run-dir 產生報告。")
            return 0
        from .data import SnapshotError, download_snapshot, load_snapshot

        try:
            operation = download_snapshot if args.command == "download" else load_snapshot
            _, metadata = operation(config)
        except SnapshotError as error:
            print(f"快照錯誤：{error}", file=sys.stderr)
            return 1
        print("快照驗證通過。")
        print(json.dumps({"snapshot_path": config["data"]["snapshot_path"],
                          **metadata["data"], "calendar_version": metadata["calendar"]["version"]},
                         ensure_ascii=False, indent=2))
        return 0
