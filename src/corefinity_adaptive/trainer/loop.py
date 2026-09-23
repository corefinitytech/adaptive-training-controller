"""Small, pure helpers used by `TrainerAdapter`'s interval-boundary orchestration.

Kept separate from `adapter.py` so the arithmetic/heuristics involved (gradient norms,
default outcome assessment, config hashing) can be unit-tested without constructing a
full trainer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping

import torch

from corefinity_adaptive.actions.outcome import ActionOutcome, Assessment
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float | None:
    """L2 norm of all gradients currently on `parameters`, or `None` if none have one."""
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is not None:
            found = True
            total += float(parameter.grad.detach().norm(2) ** 2)
    return total**0.5 if found else None


def default_assess(deltas: Mapping[str, float]) -> Assessment:
    """Lower validation loss (or, failing that, training loss) is better.

    A default only — a caller with a different notion of "better" (e.g. accuracy-primary)
    passes their own `assess_fn` to `AdaptiveTrainer`.
    """
    primary = deltas.get("val_loss", deltas.get("train_loss"))
    if primary is None:
        return Assessment.NEUTRAL
    if primary < 0:
        return Assessment.POSITIVE
    if primary > 0:
        return Assessment.NEGATIVE
    return Assessment.NEUTRAL


def compute_outcome(
    pre: MetricSnapshot,
    post: MetricSnapshot,
    *,
    assess_fn: Callable[[Mapping[str, float]], Assessment] = default_assess,
) -> ActionOutcome:
    """The measured effect of whatever happened between two consecutive intervals."""
    deltas: dict[str, float] = {"train_loss": post.train_loss - pre.train_loss}
    if pre.val_loss is not None and post.val_loss is not None:
        deltas["val_loss"] = post.val_loss - pre.val_loss
    if pre.val_accuracy is not None and post.val_accuracy is not None:
        deltas["val_accuracy"] = post.val_accuracy - pre.val_accuracy
    return ActionOutcome(
        observed_at_step=post.step,
        metric_deltas=deltas,
        additional_cost=BudgetUsage(),
        assessment=assess_fn(deltas),
    )


def compute_config_hash(**config: object) -> str:
    """A short, stable fingerprint of the settings that make a run reproducible."""
    canonical = repr(sorted(config.items(), key=lambda kv: kv[0]))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
