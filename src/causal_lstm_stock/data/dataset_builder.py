from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray


def _infer_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"date", "ticker", "open", "high", "low", "close"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def build_sequences(fused_df: pd.DataFrame, lookback_window: int) -> SequenceDataset:
    feature_cols = _infer_feature_columns(fused_df)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _, group in fused_df.groupby("ticker"):
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) <= lookback_window:
            continue

        next_close = g["close"].shift(-1)
        movement = (next_close > g["close"]).astype(int)

        for i in range(lookback_window, len(g) - 1):
            window = g.loc[i - lookback_window : i - 1, feature_cols].to_numpy(dtype=np.float32)
            X_list.append(window)
            y_list.append(int(movement.iloc[i]))

    if not X_list:
        return SequenceDataset(X=np.empty((0, lookback_window, 0), dtype=np.float32), y=np.empty((0,), dtype=np.int64))

    X = np.stack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    return SequenceDataset(X=X, y=y)
