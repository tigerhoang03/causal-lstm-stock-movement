from __future__ import annotations

import pandas as pd

from causal_lstm_stock.features.modalities import select_feature_columns


def test_select_feature_columns_price_only() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "ticker": ["AAPL"] * 3,
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [10, 11, 12],
            "return_1d": [0.0, 0.1, -0.1],
            "news_count": [1, 2, 3],
            "finbert_pos_mean": [0.2, 0.3, 0.4],
            "intervention_score": [0.0, 1.0, -1.0],
        }
    )

    selected = select_feature_columns(
        df,
        modalities={"price": True, "news": False, "causal": False, "include_other_features": False},
    )

    assert selected == ["volume", "return_1d"]


def test_select_feature_columns_all_modalities() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "ticker": ["AAPL"] * 3,
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [10, 11, 12],
            "news_count": [1, 2, 3],
            "finbert_pos_mean": [0.2, 0.3, 0.4],
            "intervention_score": [0.0, 1.0, -1.0],
        }
    )

    selected = select_feature_columns(df, modalities={"price": True, "news": True, "causal": True})
    assert set(selected) == {"volume", "news_count", "finbert_pos_mean", "intervention_score"}
