from __future__ import annotations

import pandas as pd


KEYS = ["date", "ticker"]


def fuse_modalities(price_features: pd.DataFrame, news_features: pd.DataFrame, causal_features: pd.DataFrame) -> pd.DataFrame:
    fused = price_features.merge(news_features, on=KEYS, how="left", suffixes=("", "_news"))
    fused = fused.merge(causal_features, on=KEYS, how="left", suffixes=("", "_causal"))

    numeric_cols = fused.select_dtypes(include=["number"]).columns
    fused[numeric_cols] = fused[numeric_cols].fillna(0.0)

    fused = fused.sort_values(["ticker", "date"]).reset_index(drop=True)
    return fused
