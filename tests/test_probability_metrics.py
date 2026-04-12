from __future__ import annotations

import numpy as np
import pandas as pd

from causal_lstm_stock.evaluate.probability_metrics import cross_model_p_up_stats, probability_diagnostics


def test_probability_diagnostics_finite() -> None:
    pred_df = pd.DataFrame(
        {
            "p_up": [0.45, 0.55, 0.62, 0.48, 0.71],
            "y_pred": [0, 1, 1, 0, 1],
        }
    )
    d = probability_diagnostics(pred_df, threshold=0.5)
    assert np.isfinite(d["mean_p_up"])
    assert np.isfinite(d["std_p_up"])
    assert d["frac_predict_up"] == 0.6
    assert d["frac_p_up_above_threshold"] == 0.6


def test_cross_model_p_up_stats() -> None:
    b = pd.DataFrame({"eval_sample_index": [0, 1, 2], "p_up": [0.5, 0.6, 0.7]})
    c = pd.DataFrame({"eval_sample_index": [0, 1, 2], "p_up": [0.52, 0.55, 0.65]})
    out = cross_model_p_up_stats(b, c)
    assert out["mean_abs_delta_p_up"] is not None
    assert abs(out["mean_abs_delta_p_up"] - np.mean([0.02, 0.05, 0.05])) < 1e-6
    assert out["corr_p_up"] is not None
    assert out["corr_p_up"] > 0.9
