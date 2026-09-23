"""`Evaluator`: the contract for running a fixed evaluation procedure against a model
and getting back comparable metrics.

This is deliberately a single-model, single-call contract — not the multi-controller,
multi-seed comparison harness the research spec's §10 rigor requirements call for (that
needs a real task to be meaningful, not this Mac-sized synthetic model, and stays Phase 3
per the project plan). What lands here is the piece every later phase needs regardless:
a way to actually produce a pass/fail signal for `TrainerAdapter`'s `on_trigger_eval`
hook and the branch-promotion eval gate it feeds.
"""

from __future__ import annotations

from typing import Protocol

import torch
from pydantic import BaseModel, ConfigDict


class EvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    passed: bool
    """Whether this result clears whatever gate the `Evaluator` was configured with.
    `True` when no gate was configured — "no threshold" means nothing to fail."""


class Evaluator(Protocol):
    def evaluate(self, model: torch.nn.Module, device: torch.device) -> EvalResult: ...
