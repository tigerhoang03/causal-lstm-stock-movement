from __future__ import annotations

import pandas as pd


def build_causal_features(causal_df: pd.DataFrame) -> pd.DataFrame:
    df = causal_df.copy()
    for col in df.columns:
        if col in {"date", "ticker"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)
    return df
