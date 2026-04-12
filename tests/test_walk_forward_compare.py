from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from causal_lstm_stock.config import load_config
from causal_lstm_stock.evaluate.walk_forward import run_walk_forward


def test_run_walk_forward_baseline_short() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    n = 45
    t = 10
    f = 6
    X = np.random.randn(n, t, f).astype(np.float32)
    y = np.random.randint(0, 2, size=n, dtype=np.int64)
    meta_df = pd.DataFrame(
        {
            "target_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "realized_return": np.random.randn(n) * 0.01,
        }
    )
    feature_cols = [f"f{i}" for i in range(f)]

    pred = run_walk_forward(
        X=X,
        y=y,
        meta_df=meta_df,
        cfg=cfg,
        feature_cols=feature_cols,
        architecture="baseline_lstm",
        fusion_cfg=None,
        min_train_samples=20,
        retrain_frequency=10,
        threshold=0.5,
        epochs=1,
        max_steps=5,
    )
    assert len(pred) <= 5
    assert "cum_strategy_return" in pred.columns


def test_run_walk_forward_causal_short() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    n = 45
    t = 10
    # Order must include macro_shock_signal and sentiment_score for fusion
    feature_cols = [
        "volume",
        "return_1d",
        "macro_shock_signal",
        "sentiment_score",
        "x",
        "y",
    ]
    f = len(feature_cols)
    macro_i = feature_cols.index("macro_shock_signal")
    sent_i = feature_cols.index("sentiment_score")
    X = np.random.randn(n, t, f).astype(np.float32)
    X[:, :, macro_i] = 0.1
    X[:, :, sent_i] = -0.05
    y = np.random.randint(0, 2, size=n, dtype=np.int64)
    meta_df = pd.DataFrame(
        {
            "target_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "realized_return": np.random.randn(n) * 0.01,
        }
    )

    pred = run_walk_forward(
        X=X,
        y=y,
        meta_df=meta_df,
        cfg=cfg,
        feature_cols=feature_cols,
        architecture="causal_fusion_lstm",
        fusion_cfg=cfg.model.fusion,
        min_train_samples=20,
        retrain_frequency=10,
        threshold=0.5,
        epochs=1,
        max_steps=3,
    )
    assert len(pred) <= 3
    assert torch.isfinite(torch.tensor(pred["p_up"].values)).all()
