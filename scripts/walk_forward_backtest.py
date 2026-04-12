from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from causal_lstm_stock.config import load_config
from causal_lstm_stock.evaluate.walk_forward import (
    build_backtest_arrays,
    prepare_fused_from_raw,
    run_walk_forward,
)
from causal_lstm_stock.features.modalities import select_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest with periodic retraining.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol to backtest. Default: config ticker.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for UP class.")
    parser.add_argument("--min-train-samples", type=int, default=60, help="Number of earliest samples used for first model fit.")
    parser.add_argument("--retrain-frequency", type=int, default=5, help="Retrain model every N evaluation steps.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epochs override for backtest retraining.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on number of evaluation steps.")

    parser.add_argument(
        "--use-raw-data",
        action="store_true",
        help="Rebuild fused features from raw price/news/causal files before backtest.",
    )
    parser.add_argument("--prices-csv", type=str, default=None, help="Optional raw prices CSV override.")
    parser.add_argument("--news-csv", type=str, default=None, help="Optional raw news CSV override.")
    parser.add_argument("--causal-csv", type=str, default=None, help="Optional raw causal CSV override.")
    parser.add_argument("--macro-csv", type=str, default=None, help="Optional macro panel CSV override for macro_shock_generator.")
    parser.add_argument("--finbert-csv", type=str, default=None, help="Optional FinBERT daily CSV override.")
    parser.add_argument(
        "--save-fused",
        action="store_true",
        help="When using --use-raw-data, persist fused dataset to config fused_output_csv path.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/walk_forward_predictions.csv",
        help="Path to save per-step predictions and strategy returns.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/walk_forward_summary.json",
        help="Path to save summary metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    fused_path = root / cfg.data.paths["fused_output_csv"]
    if args.use_raw_data:
        fused_df = prepare_fused_from_raw(
            root,
            cfg,
            args.prices_csv,
            args.news_csv,
            args.causal_csv,
            args.finbert_csv,
            args.macro_csv,
        )
        if args.save_fused:
            fused_path.parent.mkdir(parents=True, exist_ok=True)
            fused_df.to_csv(fused_path, index=False)
    else:
        if not fused_path.exists():
            raise FileNotFoundError(
                f"Fused dataset not found at {fused_path}. Run scripts/prepare_data.py or pass --use-raw-data."
            )
        fused_df = pd.read_csv(fused_path)

    ticker = args.ticker or cfg.data.ticker
    feature_cols = select_feature_columns(fused_df, modalities=cfg.data.modalities)
    X, y, meta_df = build_backtest_arrays(
        fused_df=fused_df,
        ticker=ticker,
        lookback_window=cfg.data.lookback_window,
        feature_cols=feature_cols,
    )

    pred_df = run_walk_forward(
        X=X,
        y=y,
        meta_df=meta_df,
        cfg=cfg,
        feature_cols=feature_cols,
        architecture=cfg.model.architecture,
        fusion_cfg=cfg.model.fusion,
        min_train_samples=args.min_train_samples,
        retrain_frequency=args.retrain_frequency,
        threshold=args.threshold,
        epochs=args.epochs or cfg.train.epochs,
        max_steps=args.max_steps,
    )

    y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)

    summary = {
        "ticker": ticker,
        "num_predictions": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "final_cum_strategy_return": float(pred_df["cum_strategy_return"].iloc[-1]),
        "final_cum_buy_hold_return": float(pred_df["cum_buy_hold_return"].iloc[-1]),
        "min_train_samples": int(args.min_train_samples),
        "retrain_frequency": int(args.retrain_frequency),
        "threshold": float(args.threshold),
        "epochs": int(args.epochs or cfg.train.epochs),
        "data_source": "raw_recomputed" if args.use_raw_data else "fused_csv",
        "num_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "modalities": cfg.data.modalities,
    }

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)

    summary_json = Path(args.summary_json)
    if not summary_json.is_absolute():
        summary_json = root / summary_json
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Walk-forward backtest summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print(f"Saved predictions CSV to: {output_csv}")
    print(f"Saved summary JSON to: {summary_json}")


if __name__ == "__main__":
    main()
