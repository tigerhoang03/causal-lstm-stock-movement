from __future__ import annotations

import numpy as np
import pandas as pd

import torch

from causal_lstm_stock.data.dataset_builder import build_sequences
from causal_lstm_stock.models.causal_fusion_lstm import CausalFusionLSTM


def test_sequence_builder_shapes() -> None:
    rows = 64
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "ticker": ["AAPL"] * rows,
            "open": np.random.rand(rows) * 100,
            "high": np.random.rand(rows) * 100,
            "low": np.random.rand(rows) * 100,
            "close": np.random.rand(rows) * 100,
            "volume": np.random.randint(1000, 5000, size=rows),
            "return_1d": np.random.randn(rows),
            "range_pct": np.random.rand(rows),
            "news_count": np.random.randint(0, 10, size=rows),
            "sentiment_score": np.random.randn(rows),
            "intervention_score": np.random.randn(rows),
        }
    )

    ds = build_sequences(df, lookback_window=10)
    assert ds.X.ndim == 3
    assert ds.y.ndim == 1
    assert ds.X.shape[0] == ds.y.shape[0]


def test_causal_fusion_attention_output_shape() -> None:
    b, t, f = 4, 10, 8
    macro_idx, sent_idx = 3, 5
    x = torch.randn(b, t, f)
    model = CausalFusionLSTM(
        input_dim=f,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        num_classes=2,
        macro_feature_index=macro_idx,
        sentiment_feature_index=sent_idx,
    )
    logits = model(x)
    assert logits.shape == (b, 2)
