from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _zscore(s: np.ndarray) -> np.ndarray:
    m = np.nanmean(s)
    sd = float(np.nanstd(s))
    if sd < 1e-12:
        return np.zeros_like(s, dtype=np.float64)
    return ((s - m) / sd).astype(np.float64)


def _build_dml_design(
    g: pd.DataFrame,
    primary: str,
    controls: list[str],
    lag_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows aligned with g; valid[i] True when row i can be used to fit (has full lags)."""
    n = len(g)
    n_feat = len(controls) + lag_days * (1 + len(controls))
    X = np.zeros((n, n_feat), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    y = g[primary].to_numpy(dtype=np.float64)

    for i in range(n):
        row: list[float] = [float(g[c].iloc[i]) for c in controls]
        ok = True
        for k in range(1, lag_days + 1):
            if i - k < 0:
                ok = False
                break
            row.append(float(g[primary].iloc[i - k]))
            for c in controls:
                row.append(float(g[c].iloc[i - k]))
        if not ok:
            continue
        X[i, :] = row
        valid[i] = True
    return X, y, valid


def generate_dml_macro_shock(
    merged: pd.DataFrame,
    primary: str,
    controls: list[str],
    lag_days: int,
    ridge_alpha: float,
) -> pd.DataFrame:
    """Adds column `_macro_shock_signal` to a copy of merged (one block per ticker)."""
    out = merged.copy()
    shock = np.zeros(len(out), dtype=np.float64)

    for ticker, grp in out.groupby("ticker"):
        g = grp.sort_values("date").reset_index(drop=True)
        idx = grp.index.to_numpy()
        if primary not in g.columns or any(c not in g.columns for c in controls):
            raise ValueError(f"DML requires columns {primary} and {controls} in merged macro data.")
        X, y, valid = _build_dml_design(g, primary, controls, lag_days)
        if not np.any(valid):
            continue
        n_valid = int(np.sum(valid))
        min_samples = max(3, X.shape[1] + 1)
        if n_valid < min_samples:
            continue
        model = Ridge(alpha=float(ridge_alpha))
        model.fit(X[valid], y[valid])
        pred = model.predict(X)
        resid = np.where(valid, y - pred, 0.0)
        shock[idx] = resid

    out["_macro_shock_signal"] = _zscore(shock)
    return out


def _rolling_mlp_features(
    g: pd.DataFrame,
    macro_cols: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns X_flat (n, window * n_macro), valid mask."""
    n = len(g)
    vals = g[macro_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    vals_s = scaler.fit_transform(vals)
    dim = window * len(macro_cols)
    X = np.zeros((n, dim), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if i - window + 1 < 0:
            continue
        X[i, :] = vals_s[i - window + 1 : i + 1, :].reshape(-1)
        valid[i] = True
    return X, valid


def generate_mlp_macro_shock(
    merged: pd.DataFrame,
    macro_cols: list[str],
    window_days: int,
    hidden_dim: int,
    max_iter: int,
    target: str,
) -> pd.DataFrame:
    """MLP maps rolling macro window to auxiliary target; residuals vs prediction as shock."""
    out = merged.copy()
    shock = np.zeros(len(out), dtype=np.float64)

    for ticker, grp in out.groupby("ticker"):
        g = grp.sort_values("date").reset_index(drop=True)
        idx = grp.index.to_numpy()

        if target == "abs_next_return":
            if "abs_next_ret" not in g.columns:
                raise ValueError("merged dataframe must contain abs_next_ret for MLP target abs_next_return.")
            y_raw = g["abs_next_ret"].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Unknown MLP target: {target}")

        X, valid = _rolling_mlp_features(g, macro_cols, window_days)
        if not np.any(valid):
            continue
        y = y_raw.copy()
        n_valid = int(np.sum(valid))
        min_samples = max(5, X.shape[1] + 1)
        if n_valid < min_samples:
            continue
        model = MLPRegressor(
            hidden_layer_sizes=(int(hidden_dim),),
            max_iter=int(max_iter),
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
        )
        model.fit(X[valid], y[valid])
        pred = model.predict(X)
        resid = np.where(valid, y - pred, 0.0)
        shock[idx] = resid

    out["_macro_shock_signal"] = _zscore(shock)
    return out


def apply_macro_shock_generator(
    causal_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    prices_df: pd.DataFrame | None,
    gen_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Merge macro into causal rows and overwrite `macro_shock_signal` with refined values.
    """
    result = causal_df.copy()
    result["date"] = pd.to_datetime(result["date"])
    macro = macro_df.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    method = str(gen_cfg.get("method", "dml")).lower()
    if method == "passthrough":
        return result

    merged = result.merge(macro, on=["date", "ticker"], how="left", suffixes=("", "_macro_dup"))

    macro_numeric = [c for c in macro.columns if c not in ("date", "ticker")]
    for c in macro_numeric:
        if c in merged.columns:
            merged[c] = merged.groupby("ticker")[c].transform(lambda s: s.ffill().bfill())
            merged[c] = merged[c].fillna(merged[c].mean())

    if method == "dml":
        primary = gen_cfg.get("primary_column", "vix")
        controls = list(gen_cfg.get("control_columns", ["fed_funds"]))
        lag_days = int(gen_cfg.get("lag_days", 3))
        ridge_alpha = float(gen_cfg.get("ridge_alpha", 1.0))
        merged = generate_dml_macro_shock(merged, primary, controls, lag_days, ridge_alpha)
    elif method == "mlp":
        if prices_df is None:
            raise ValueError("macro_shock_generator method 'mlp' requires prices_df (for auxiliary target).")
        px = prices_df.copy()
        px["date"] = pd.to_datetime(px["date"])
        px = px.sort_values(["ticker", "date"])
        px["ret"] = px.groupby("ticker")["close"].pct_change()
        px["abs_next_ret"] = px.groupby("ticker")["ret"].shift(-1).abs()
        aux = px[["date", "ticker", "abs_next_ret"]]
        merged = merged.merge(aux, on=["date", "ticker"], how="left")
        merged["abs_next_ret"] = merged["abs_next_ret"].fillna(0.0)
        mlp_cfg = gen_cfg.get("mlp") or {}
        window_days = int(mlp_cfg.get("window_days", 5))
        hidden_dim = int(mlp_cfg.get("hidden_dim", 32))
        max_iter = int(mlp_cfg.get("max_iter", 500))
        target = str(mlp_cfg.get("target", "abs_next_return"))
        macro_cols = [c for c in macro_numeric if c in merged.columns]
        if not macro_cols:
            raise ValueError("No numeric macro columns available for MLP macro shock.")
        merged = generate_mlp_macro_shock(
            merged,
            macro_cols=macro_cols,
            window_days=window_days,
            hidden_dim=hidden_dim,
            max_iter=max_iter,
            target=target,
        )
    else:
        raise ValueError(f"Unknown macro_shock_generator.method: {method}")

    signal = merged["_macro_shock_signal"].to_numpy(dtype=np.float64)
    result["macro_shock_signal"] = signal
    return result
