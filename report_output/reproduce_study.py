"""Reproduce the frozen LSTM study and optional GRU comparison.

Run from the repository root after installing the Python 3.11 environment
described in README.md. Training and evaluation are intentionally not run here.
"""

import argparse
from pathlib import Path

from lstm_comparison.config import load_config
from lstm_comparison.data import download_snapshot
from lstm_comparison.evaluation import evaluate_experiment
from lstm_comparison.gru_experiment import run_gru_experiment
from lstm_comparison.report import generate_report
from lstm_comparison.training import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.json")
    parser.add_argument("--gru", action="store_true", help="also run GRU against this LSTM")
    args = parser.parse_args()

    config = load_config(args.config)
    # Downloads only if no valid frozen snapshot exists; existing snapshots are reused.
    download_snapshot(config)
    lstm_run_dir = Path(run_experiment(config))
    evaluate_experiment(lstm_run_dir)
    generate_report(lstm_run_dir)
    print(f"LSTM results: {lstm_run_dir}")

    if args.gru:
        gru_run_dir = Path(run_gru_experiment(lstm_run_dir))
        print(f"GRU results: {gru_run_dir}")


if __name__ == "__main__":
    main()
