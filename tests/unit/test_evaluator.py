"""`LossEvalSuite`: mean loss over a held-out loader, a pass/fail gate against an
optional threshold, and — critically — restoring the model's train/eval mode
afterward so a mid-training evaluation never leaves it stuck in `eval()` mode.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from corefinity_adaptive.evaluation.suites import LossEvalSuite
from tests.conftest import TinyRegressionModel, mse_loss


def _eval_loader(
    inputs: torch.Tensor, targets: torch.Tensor, batch_size: int = 4
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


def test_evaluate_computes_mean_loss_over_the_full_loader() -> None:
    model = TinyRegressionModel()
    inputs = torch.zeros(8, 4)
    targets = model(inputs).detach()  # zero loss for every batch, by construction
    suite = LossEvalSuite(eval_loader=_eval_loader(inputs, targets), compute_loss=mse_loss)

    result = suite.evaluate(model, torch.device("cpu"))

    assert result.metrics["eval_loss"] == pytest.approx(0.0, abs=1e-6)


def test_passed_is_true_with_no_threshold_configured() -> None:
    model = TinyRegressionModel()
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 1) * 100  # deliberately high loss
    suite = LossEvalSuite(eval_loader=_eval_loader(inputs, targets), compute_loss=mse_loss)

    result = suite.evaluate(model, torch.device("cpu"))
    assert result.passed


def test_passed_is_false_when_loss_exceeds_threshold() -> None:
    model = TinyRegressionModel()
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 1) * 1000  # guarantees a large loss
    suite = LossEvalSuite(
        eval_loader=_eval_loader(inputs, targets), compute_loss=mse_loss, pass_threshold=1e-6
    )

    result = suite.evaluate(model, torch.device("cpu"))
    assert not result.passed


def test_passed_is_true_when_loss_is_within_threshold() -> None:
    model = TinyRegressionModel()
    inputs = torch.zeros(8, 4)
    targets = model(inputs).detach()  # zero loss
    suite = LossEvalSuite(
        eval_loader=_eval_loader(inputs, targets), compute_loss=mse_loss, pass_threshold=0.01
    )

    result = suite.evaluate(model, torch.device("cpu"))
    assert result.passed


def test_evaluate_restores_training_mode_when_model_was_training() -> None:
    model = TinyRegressionModel()
    model.train()
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 1)
    suite = LossEvalSuite(eval_loader=_eval_loader(inputs, targets), compute_loss=mse_loss)

    suite.evaluate(model, torch.device("cpu"))
    assert model.training is True


def test_evaluate_restores_eval_mode_when_model_was_already_in_eval_mode() -> None:
    model = TinyRegressionModel()
    model.eval()
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 1)
    suite = LossEvalSuite(eval_loader=_eval_loader(inputs, targets), compute_loss=mse_loss)

    suite.evaluate(model, torch.device("cpu"))
    assert model.training is False


def test_empty_eval_loader_does_not_crash() -> None:
    model = TinyRegressionModel()
    suite = LossEvalSuite(eval_loader=[], compute_loss=mse_loss)

    result = suite.evaluate(model, torch.device("cpu"))
    assert math.isnan(result.metrics["eval_loss"])
