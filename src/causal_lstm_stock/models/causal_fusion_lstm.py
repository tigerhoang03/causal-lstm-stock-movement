from __future__ import annotations

import torch
from torch import nn


class CausalFusionLSTM(nn.Module):
    """
    Starter architecture:
    - Sequence encoder (LSTM)
    - Learned gate that can up/down-weight latent states based on causal channels

    Note: In this starter, causal channels are assumed to already be in the input tensor.
    Future extension: split inputs by modality and implement explicit modality encoders.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.causal_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        gate = self.causal_gate(last_hidden)
        gated_hidden = last_hidden * gate
        logits = self.classifier(gated_hidden)
        return logits
