from __future__ import annotations

import torch
from torch import nn


class CausalFusionLSTM(nn.Module):
    """
    Sequence encoder (LSTM unchanged) + temporal attention over timesteps.
    Attention logits condition on hidden state and explicit macro / sentiment channels
    sliced from the input tensor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 2,
        macro_feature_index: int = 0,
        sentiment_feature_index: int = 0,
    ) -> None:
        super().__init__()
        self.macro_feature_index = int(macro_feature_index)
        self.sentiment_feature_index = int(sentiment_feature_index)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        attn_in = hidden_dim + 2
        self.attention_scorer = nn.Sequential(
            nn.Linear(attn_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        m = x[:, :, self.macro_feature_index : self.macro_feature_index + 1]
        s = x[:, :, self.sentiment_feature_index : self.sentiment_feature_index + 1]
        combined = torch.cat([out, m, s], dim=-1)
        scores = self.attention_scorer(combined).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(alpha.unsqueeze(-1) * out, dim=1)
        logits = self.classifier(context)
        return logits
