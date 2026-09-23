"""`LossEvalSuite`: the reference `Evaluator` — average loss over a held-out data loader.

Generic enough to reuse across model/task types (the caller supplies `compute_loss`,
the same shape `TrainerAdapter` itself takes), while staying small enough to be a
genuinely minimal Phase 1 deliverable rather than a preview of the Phase 3 comparison
harness.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from corefinity_adaptive.evaluation.evaluator import EvalResult


class LossEvalSuite:
    def __init__(
        self,
        *,
        eval_loader: Iterable[Any],
        compute_loss: Callable[[torch.nn.Module, Any, torch.device], torch.Tensor],
        metric_name: str = "eval_loss",
        pass_threshold: float | None = None,
    ) -> None:
        self._eval_loader = eval_loader
        self._compute_loss = compute_loss
        self._metric_name = metric_name
        self._pass_threshold = pass_threshold

    def evaluate(self, model: torch.nn.Module, device: torch.device) -> EvalResult:
        was_training = model.training
        model.eval()
        try:
            total_loss = 0.0
            batch_count = 0
            with torch.no_grad():
                for batch in self._eval_loader:
                    loss = self._compute_loss(model, batch, device)
                    total_loss += float(loss.detach().cpu())
                    batch_count += 1
        finally:
            # Restore whatever mode the caller had the model in — evaluation must never
            # leave a model that was mid-training stuck in eval() for its next step
            # (wrong behavior for any model using dropout/batchnorm).
            model.train(was_training)

        mean_loss = total_loss / batch_count if batch_count > 0 else float("nan")
        passed = self._pass_threshold is None or mean_loss <= self._pass_threshold
        return EvalResult(metrics={self._metric_name: mean_loss}, passed=passed)
