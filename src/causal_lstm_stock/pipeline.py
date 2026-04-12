from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from causal_lstm_stock.causal.macro_shock_generator import apply_macro_shock_generator
from causal_lstm_stock.data.macro_loader import load_macro_panel


def integrate_macro_shock_into_causal(
    root: Path,
    cfg: Any,
    causal_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
    macro_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    When macro_shock_generator.enabled is True, load macro CSV and overwrite
    `macro_shock_signal` on the causal frame. Otherwise return causal_df unchanged.
    """
    data_cfg = cfg.data
    gen_cfg = getattr(data_cfg, "macro_shock_generator", None) or {}
    if not bool(gen_cfg.get("enabled", False)):
        return causal_df

    if macro_csv_path is not None:
        macro_path = Path(macro_csv_path)
    else:
        macro_rel = data_cfg.paths.get("macro_csv")
        if not macro_rel:
            raise ValueError("macro_shock_generator.enabled is True but data.paths.macro_csv is not set.")
        macro_path = Path(macro_rel)
        if not macro_path.is_absolute():
            macro_path = root / macro_path

    if not macro_path.exists():
        raise FileNotFoundError(f"Macro panel not found: {macro_path}")

    method = str(gen_cfg.get("method", "dml")).lower()
    if method == "mlp" and prices_df is None:
        raise ValueError("macro_shock_generator method 'mlp' requires prices_df when integrating into causal.")

    macro_df = load_macro_panel(macro_path)
    return apply_macro_shock_generator(causal_df, macro_df, prices_df, gen_cfg)
