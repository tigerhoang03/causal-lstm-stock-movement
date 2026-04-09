from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from causal_lstm_stock.config import load_config
from causal_lstm_stock.data.causal_loader import load_causal_signals
from causal_lstm_stock.data.news_loader import load_news
from causal_lstm_stock.data.price_loader import load_prices
from causal_lstm_stock.features.causal_features import build_causal_features
from causal_lstm_stock.features.fusion import fuse_modalities
from causal_lstm_stock.features.modalities import select_feature_columns
from causal_lstm_stock.features.news_features import build_news_features
from causal_lstm_stock.features.price_features import build_price_features
from causal_lstm_stock.models.baseline_lstm import BaselineLSTM
from causal_lstm_stock.models.causal_fusion_lstm import CausalFusionLSTM
from causal_lstm_stock.train import train_model


def _build_model(arch: str, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int):
    if arch == "baseline_lstm":
        return BaselineLSTM(input_dim, hidden_dim, num_layers, dropout, num_classes)
    if arch == "causal_fusion_lstm":
        return CausalFusionLSTM(input_dim, hidden_dim, num_layers, dropout, num_classes)
    raise ValueError(f"Unknown architecture: {arch}")


def _resolve_path(root: Path, default_rel: str, override: str | None) -> Path:
    if override is None:
        return root / default_rel
    p = Path(override)
    return p if p.is_absolute() else root / p


def _prepare_fused_from_raw(
    root: Path,
    cfg: Any,
    prices_csv: str | None,
    news_csv: str | None,
    causal_csv: str | None,
    finbert_csv: str | None,
) -> pd.DataFrame:
    prices_path = _resolve_path(root, cfg.data.paths["prices_csv"], prices_csv)
    news_path = _resolve_path(root, cfg.data.paths["news_csv"], news_csv)
    causal_path = _resolve_path(root, cfg.data.paths["causal_csv"], causal_csv)
    finbert_enabled = bool((cfg.data.finbert or {}).get("enabled", False))
    finbert_default_rel = cfg.data.paths.get("finbert_daily_csv", "data/interim/finbert_daily_features.csv")
    finbert_path = None
    if finbert_csv is not None:
        finbert_path = _resolve_path(root, finbert_default_rel, finbert_csv)
    elif finbert_enabled:
        finbert_path = _resolve_path(root, finbert_default_rel, None)

    prices = load_prices(prices_path)
    news = load_news(news_path, finbert_daily_csv=finbert_path)
    causal = load_causal_signals(causal_path)

    price_feat = build_price_features(prices)
    news_feat = build_news_features(news)
    causal_feat = build_causal_features(causal)

    return fuse_modalities(price_feat, news_feat, causal_feat)


def _build_backtest_arrays(
    fused_df: pd.DataFrame,
    ticker: str,
    lookback_window: int,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = fused_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    g = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(g) <= lookback_window + 1:
        raise ValueError(
            f"Not enough rows for walk-forward backtest on {ticker}. Need > {lookback_window + 1}, found {len(g)}."
        )

    missing_cols = [c for c in feature_cols if c not in g.columns]
    if missing_cols:
        raise ValueError(f"Requested feature columns missing in backtest dataframe: {missing_cols}")

    if not feature_cols:
        raise ValueError("No numeric feature columns found for backtest.")

    next_close = g["close"].shift(-1)
    movement = (next_close > g["close"]).astype(int)
    realized_return = ((next_close - g["close"]) / g["close"]).fillna(0.0)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for i in range(lookback_window, len(g) - 1):
        X_list.append(g.loc[i - lookback_window : i - 1, feature_cols].to_numpy(dtype=np.float32))
        y_list.append(int(movement.iloc[i]))
        meta_rows.append(
            {
                "target_index": i,
                "target_date": g.loc[i, "date"],
                "current_close": float(g.loc[i, "close"]),
                "next_close": float(g.loc[i + 1, "close"]),
                "realized_return": float(realized_return.iloc[i]),
            }
        )

    X = np.stack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)
    return X, y, meta_df


def _run_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: Any,
    min_train_samples: int,
    retrain_frequency: int,
    threshold: float,
    epochs: int,
    max_steps: int | None,
) -> pd.DataFrame:
    if min_train_samples >= len(X):
        raise ValueError(
            f"min_train_samples={min_train_samples} is too large for total samples={len(X)}."
        )

    rows: list[dict[str, Any]] = []
    model = None

    for eval_idx in range(min_train_samples, len(X)):
        if max_steps is not None and len(rows) >= max_steps:
            break

        retrain_due = model is None or ((eval_idx - min_train_samples) % retrain_frequency == 0)
        if retrain_due:
            model = _build_model(
                arch=cfg.model.architecture,
                input_dim=X.shape[-1],
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.num_layers,
                dropout=cfg.model.dropout,
                num_classes=cfg.model.num_classes,
            )
            train_model(
                model=model,
                X=X[:eval_idx],
                y=y[:eval_idx],
                epochs=epochs,
                batch_size=cfg.train.batch_size,
                learning_rate=cfg.train.learning_rate,
                weight_decay=cfg.train.weight_decay,
                seed=cfg.train.seed,
            )

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X[eval_idx : eval_idx + 1], dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        p_down = float(probs[0])
        p_up = float(probs[1]) if len(probs) > 1 else float(1.0 - probs[0])
        y_pred = 1 if p_up >= threshold else 0
        y_true = int(y[eval_idx])

        row = meta_df.iloc[eval_idx].to_dict()
        row.update(
            {
                "eval_sample_index": int(eval_idx),
                "y_true": y_true,
                "y_pred": y_pred,
                "p_up": p_up,
                "p_down": p_down,
                "signal": 1 if y_pred == 1 else -1,
                "strategy_return": (1.0 if y_pred == 1 else -1.0) * float(meta_df.iloc[eval_idx]["realized_return"]),
            }
        )
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise ValueError("No predictions were produced. Reduce min_train_samples or increase data size.")

    pred_df["target_date"] = pd.to_datetime(pred_df["target_date"])
    pred_df = pred_df.sort_values("target_date").reset_index(drop=True)
    pred_df["cum_strategy_return"] = (1.0 + pred_df["strategy_return"]).cumprod() - 1.0
    pred_df["cum_buy_hold_return"] = (1.0 + pred_df["realized_return"]).cumprod() - 1.0
    return pred_df


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
        fused_df = _prepare_fused_from_raw(
            root,
            cfg,
            args.prices_csv,
            args.news_csv,
            args.causal_csv,
            args.finbert_csv,
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
    X, y, meta_df = _build_backtest_arrays(
        fused_df=fused_df,
        ticker=ticker,
        lookback_window=cfg.data.lookback_window,
        feature_cols=feature_cols,
    )

    pred_df = _run_walk_forward(
        X=X,
        y=y,
        meta_df=meta_df,
        cfg=cfg,
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
