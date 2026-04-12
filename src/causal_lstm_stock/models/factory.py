from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from causal_lstm_stock.models.baseline_lstm import BaselineLSTM
from causal_lstm_stock.models.causal_fusion_lstm import CausalFusionLSTM


def _default_fusion_cfg() -> dict[str, str]:
    return {
        "macro_column": "macro_shock_signal",
        "sentiment_column": "sentiment_score",
    }


def _resolve_fusion_indices(feature_cols: list[str], fusion_cfg: dict[str, Any] | None) -> tuple[int, int]:
    cfg = {**_default_fusion_cfg(), **(fusion_cfg or {})}
    macro_col = str(cfg["macro_column"])
    sent_col = str(cfg["sentiment_column"])
    if macro_col not in feature_cols:
        raise ValueError(
            f"Fusion macro_column '{macro_col}' is not in the selected feature columns. "
            "Enable the causal modality or adjust configs/model.yaml fusion settings."
        )
    if sent_col not in feature_cols:
        raise ValueError(
            f"Fusion sentiment_column '{sent_col}' is not in the selected feature columns. "
            "Enable the news modality or adjust configs/model.yaml fusion settings."
        )
    return feature_cols.index(macro_col), feature_cols.index(sent_col)


def build_model(
    architecture: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_classes: int,
    feature_cols: list[str] | None = None,
    fusion_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    if architecture == "baseline_lstm":
        return BaselineLSTM(input_dim, hidden_dim, num_layers, dropout, num_classes)
    if architecture == "causal_fusion_lstm":
        if not feature_cols:
            raise ValueError("causal_fusion_lstm requires feature_cols to resolve fusion column indices.")
        macro_idx, sent_idx = _resolve_fusion_indices(feature_cols, fusion_cfg)
        return CausalFusionLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout,
            num_classes,
            macro_feature_index=macro_idx,
            sentiment_feature_index=sent_idx,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        return blob["state_dict"]
    if isinstance(blob, dict):
        return blob
    raise ValueError(f"Unsupported checkpoint format in: {path}")
