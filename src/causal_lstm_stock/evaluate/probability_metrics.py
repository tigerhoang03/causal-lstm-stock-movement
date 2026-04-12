from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def probability_diagnostics(pred_df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    p = pred_df["p_up"].to_numpy(dtype=np.float64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)
    return {
        "mean_p_up": float(np.mean(p)),
        "std_p_up": float(np.std(p)),
        "min_p_up": float(np.min(p)),
        "max_p_up": float(np.max(p)),
        "frac_predict_up": float(np.mean(y_pred == 1)),
        "frac_p_up_above_threshold": float(np.mean(p >= threshold)),
    }


def cross_model_p_up_stats(
    pred_baseline: pd.DataFrame,
    pred_causal: pd.DataFrame,
) -> dict[str, Any]:
    b = pred_baseline.sort_values("eval_sample_index").reset_index(drop=True)
    c = pred_causal.sort_values("eval_sample_index").reset_index(drop=True)
    if len(b) != len(c):
        return {"error": "length_mismatch", "mean_abs_delta_p_up": None, "corr_p_up": None}
    pb = b["p_up"].to_numpy(dtype=np.float64)
    pc = c["p_up"].to_numpy(dtype=np.float64)
    delta = np.abs(pb - pc)
    out: dict[str, Any] = {"mean_abs_delta_p_up": float(np.mean(delta))}
    if np.std(pb) > 1e-12 and np.std(pc) > 1e-12:
        out["corr_p_up"] = float(np.corrcoef(pb, pc)[0, 1])
    else:
        out["corr_p_up"] = None
    return out
