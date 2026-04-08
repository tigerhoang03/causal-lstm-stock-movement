from __future__ import annotations

import pandas as pd


def build_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["return_1d"] = df.groupby("ticker")["close"].pct_change().fillna(0.0)
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"]).fillna(0.0)
    df["volume_z"] = (
        df.groupby("ticker")["volume"].transform(lambda s: (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-8))
    ).fillna(0.0)
    return df
