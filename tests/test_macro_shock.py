from __future__ import annotations

import numpy as np
import pandas as pd

from causal_lstm_stock.causal.macro_shock_generator import apply_macro_shock_generator


def test_dml_macro_shock_overwrites_signal() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    causal = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * 40,
            "intervention_score": np.random.randn(40),
            "confounder_proxy": np.random.randn(40),
            "macro_shock_signal": np.zeros(40),
        }
    )
    macro = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * 40,
            "vix": 13.0 + np.cumsum(np.random.randn(40) * 0.1),
            "fed_funds": 5.0 + np.cumsum(np.random.randn(40) * 0.01),
        }
    )
    gen_cfg = {
        "method": "dml",
        "primary_column": "vix",
        "control_columns": ["fed_funds"],
        "lag_days": 3,
        "ridge_alpha": 1.0,
    }
    out = apply_macro_shock_generator(causal, macro, prices_df=None, gen_cfg=gen_cfg)
    assert "macro_shock_signal" in out.columns
    assert np.all(np.isfinite(out["macro_shock_signal"].to_numpy()))


def test_factory_builds_causal_fusion() -> None:
    import torch

    from causal_lstm_stock.models.factory import build_model

    model = build_model(
        architecture="causal_fusion_lstm",
        input_dim=5,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        num_classes=2,
        feature_cols=["a", "macro_shock_signal", "b", "sentiment_score", "c"],
        fusion_cfg={"macro_column": "macro_shock_signal", "sentiment_column": "sentiment_score"},
    )
    x = torch.randn(2, 7, 5)
    assert model(x).shape == (2, 2)
