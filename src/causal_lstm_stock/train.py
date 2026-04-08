from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class TrainResult:
    train_loss_history: list[float]
    val_loss_history: list[float]


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None)
    train_loader = _to_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    for _ in tqdm(range(epochs), desc="Training"):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))

        train_loss_history.append(float(np.mean(train_losses)) if train_losses else 0.0)
        val_loss_history.append(float(np.mean(val_losses)) if val_losses else 0.0)

    return TrainResult(train_loss_history=train_loss_history, val_loss_history=val_loss_history)
