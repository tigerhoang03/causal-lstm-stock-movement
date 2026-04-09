from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

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


def _load_state_dict_compat(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported checkpoint format in: {path}")


def _build_latest_window(
    fused_df: pd.DataFrame,
    ticker: str,
    lookback_window: int,
    as_of_date: str | None,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.Timestamp, list[str]]:
    df = fused_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    ticker_df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if as_of_date:
        cutoff = pd.to_datetime(as_of_date)
        ticker_df = ticker_df[ticker_df["date"] <= cutoff].reset_index(drop=True)

    if len(ticker_df) < lookback_window:
        raise ValueError(
            f"Not enough rows for ticker={ticker}. Need at least {lookback_window}, found {len(ticker_df)}."
        )

    missing_cols = [c for c in feature_cols if c not in ticker_df.columns]
    if missing_cols:
        raise ValueError(f"Requested feature columns missing in inference dataframe: {missing_cols}")

    if not feature_cols:
        raise ValueError("No numeric feature columns found for inference.")

    window_df = ticker_df.iloc[-lookback_window:]
    X = window_df[feature_cols].to_numpy(dtype=np.float32)[None, :, :]
    current_date = pd.Timestamp(window_df.iloc[-1]["date"])
    return X, current_date, feature_cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-step model inference for next-day movement.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol to score. Default: config ticker.")
    parser.add_argument("--as-of-date", type=str, default=None, help="Use latest record on or before this date (YYYY-MM-DD).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for UP class.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional checkpoint path override.")

    parser.add_argument(
        "--use-raw-data",
        action="store_true",
        help="Rebuild fused features from raw price/news/causal files before inference.",
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

    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save inference result JSON.")
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
    X, current_date, feature_cols = _build_latest_window(
        fused_df=fused_df,
        ticker=ticker,
        lookback_window=cfg.data.lookback_window,
        as_of_date=args.as_of_date,
        feature_cols=feature_cols,
    )

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else (root / cfg.train.checkpoint_dir / "latest_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run scripts/train_model.py first.")

    model = _build_model(
        arch=cfg.model.architecture,
        input_dim=X.shape[-1],
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    )
    try:
        model.load_state_dict(_load_state_dict_compat(ckpt_path))
    except RuntimeError as err:
        raise RuntimeError(
            "Checkpoint is incompatible with the current feature set. "
            "This usually happens after changing modality toggles or adding/removing FinBERT features. "
            "Retrain with scripts/train_model.py (or run live_predict_job.py --retrain-on-live-data) "
            "using the same feature configuration."
        ) from err
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    p_down = float(probs[0])
    p_up = float(probs[1]) if len(probs) > 1 else float(1.0 - probs[0])
    direction = "UP" if p_up >= args.threshold else "DOWN"

    result = {
        "ticker": ticker,
        "as_of_date": current_date.strftime("%Y-%m-%d"),
        "predicted_next_day_direction": direction,
        "p_up": p_up,
        "p_down": p_down,
        "threshold": args.threshold,
        "lookback_window": cfg.data.lookback_window,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "checkpoint_path": str(ckpt_path),
        "data_source": "raw_recomputed" if args.use_raw_data else "fused_csv",
        "modalities": cfg.data.modalities,
    }

    print("Inference result:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON output to: {output_path}")


if __name__ == "__main__":
    main()
