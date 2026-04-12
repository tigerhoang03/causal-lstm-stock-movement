from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from causal_lstm_stock.data.causal_loader import load_causal_signals
from causal_lstm_stock.data.news_loader import load_news
from causal_lstm_stock.data.price_loader import load_prices
from causal_lstm_stock.features.causal_features import build_causal_features
from causal_lstm_stock.features.fusion import fuse_modalities
from causal_lstm_stock.features.news_features import build_news_features
from causal_lstm_stock.features.price_features import build_price_features
from causal_lstm_stock.models.factory import build_model
from causal_lstm_stock.pipeline import integrate_macro_shock_into_causal
from causal_lstm_stock.train import train_model


def resolve_project_path(root: Path, default_rel: str, override: str | None) -> Path:
    if override is None:
        return root / default_rel
    p = Path(override)
    return p if p.is_absolute() else root / p


def prepare_fused_from_raw(
    root: Path,
    cfg: Any,
    prices_csv: str | None,
    news_csv: str | None,
    causal_csv: str | None,
    finbert_csv: str | None,
    macro_csv: str | None,
) -> pd.DataFrame:
    prices_path = resolve_project_path(root, cfg.data.paths["prices_csv"], prices_csv)
    news_path = resolve_project_path(root, cfg.data.paths["news_csv"], news_csv)
    causal_path = resolve_project_path(root, cfg.data.paths["causal_csv"], causal_csv)
    finbert_enabled = bool((cfg.data.finbert or {}).get("enabled", False))
    finbert_default_rel = cfg.data.paths.get("finbert_daily_csv", "data/interim/finbert_daily_features.csv")
    finbert_path = None
    if finbert_csv is not None:
        finbert_path = resolve_project_path(root, finbert_default_rel, finbert_csv)
    elif finbert_enabled:
        finbert_path = resolve_project_path(root, finbert_default_rel, None)

    macro_path_override: Path | None = None
    if macro_csv is not None:
        p = Path(macro_csv)
        macro_path_override = p if p.is_absolute() else root / p

    prices = load_prices(prices_path)
    news = load_news(news_path, finbert_daily_csv=finbert_path)
    causal = load_causal_signals(causal_path)
    causal = integrate_macro_shock_into_causal(
        root,
        cfg,
        causal,
        prices_df=prices,
        macro_csv_path=macro_path_override,
    )

    price_feat = build_price_features(prices)
    news_feat = build_news_features(news)
    causal_feat = build_causal_features(causal)

    return fuse_modalities(price_feat, news_feat, causal_feat)


def build_backtest_arrays(
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


def run_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: Any,
    feature_cols: list[str],
    architecture: str,
    fusion_cfg: dict[str, Any] | None,
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
            model = build_model(
                architecture=architecture,
                input_dim=X.shape[-1],
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.num_layers,
                dropout=cfg.model.dropout,
                num_classes=cfg.model.num_classes,
                feature_cols=feature_cols,
                fusion_cfg=fusion_cfg,
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
