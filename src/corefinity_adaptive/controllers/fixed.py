"""`FixedController`: the "do nothing" baseline every comparison is measured against.

Runs through the exact same propose → validate → (no-op) apply → record path as any
intelligent controller, so a Fixed-vs-Adaptive comparison shares the same overhead
measurement — without it, "controller overhead" (research spec §12) couldn't be measured
honestly, only guessed at.
"""

from __future__ import annotations

from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import NoopAction
from corefinity_adaptive.controllers.base import ControllerOutput
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.state.models import BudgetUsage, TrainingState


class FixedController:
    """Always proposes `NoopAction`. No history queried; confidence is trivially 1.0."""

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def requires_history(self) -> bool:
        return False

    def propose(
        self,
        *,
        state: TrainingState,
        history: tuple[ExperienceRecord, ...],
        action_space: ActionSpace,
        budget_remaining: BudgetUsage,
    ) -> ControllerOutput:
        return ControllerOutput(
            action=NoopAction(),
            confidence=1.0,
            rationale="fixed baseline: never intervenes",
        )
