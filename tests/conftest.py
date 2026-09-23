"""Shared fixtures: a tiny synthetic model/dataset used across unit and integration tests.

Kept deliberately trivial (a 2-layer MLP on synthetic regression data) — these tests
exercise the *framework's* correctness (validation, memory, the trainer loop), not any
particular model's training quality, so the model just needs to produce a real,
differentiable loss quickly on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class TinyRegressionModel(torch.nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mse_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    predictions = model(inputs)
    return torch.nn.functional.mse_loss(predictions, targets)


@pytest.fixture
def tiny_model() -> TinyRegressionModel:
    return TinyRegressionModel()


@pytest.fixture
def tiny_dataloader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(64, 4, generator=generator)
    targets = inputs.sum(dim=1, keepdim=True) + 0.1 * torch.randn(64, 1, generator=generator)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def tmp_sqlite_path(tmp_path: Path) -> str:
    """A throwaway SQLite path under pytest's managed temp directory."""
    return str(tmp_path / "experience.sqlite")
