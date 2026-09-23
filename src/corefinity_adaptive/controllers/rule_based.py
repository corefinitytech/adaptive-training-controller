"""`RuleBasedController`: deterministic plateau-triggered reweighting.

Watches its own loss trend across consultations (independent of `TrainingState`'s
bounded per-batch `recent_trend` — see `StatelessController` for the version that reads
from that instead) and, on a plateau, nudges a target data group's weight up. This is
the first non-trivial controller: a real, explainable rule the framework's guardrails
(reweight-delta bound, oscillation detection) actually get exercised against, before any
learned or LLM-backed controller enters the picture.
"""

from __future__ import annotations

from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import ActionKind, NoopAction
from corefinity_adaptive.controllers.policies import is_plateaued, primary_loss, propose_reweight
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.state.history import BoundedHistory
from corefinity_adaptive.state.models import BudgetUsage, TrainingState


class RuleBasedController:
    def __init__(
        self,
        *,
        target_group: str = "difficult",
        plateau_window: int = 3,
        min_relative_improvement: float = 0.0,
        reweight_step: float = 0.1,
        initial_weight: float = 0.5,
        max_weight: float = 1.0,
    ) -> None:
        self._target_group = target_group
        self._min_relative_improvement = min_relative_improvement
        self._reweight_step = reweight_step
        self._initial_weight = initial_weight
        self._max_weight = max_weight
        self._loss_history: BoundedHistory[float] = BoundedHistory(maxlen=plateau_window + 1)

    @property
    def name(self) -> str:
        return "rule_based"

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
        self._loss_history.push(primary_loss(state.current))

        if not action_space.allows(ActionKind.REWEIGHT_DATA):
            return ControllerOutput(
                action=NoopAction(),
                confidence=1.0,
                rationale="REWEIGHT_DATA is not enabled for this run",
            )
        if not is_plateaued(
            self._loss_history.as_tuple(), min_relative_improvement=self._min_relative_improvement
        ):
            return ControllerOutput(
                action=NoopAction(), confidence=1.0, rationale="loss is still improving"
            )

        action = propose_reweight(
            state=state,
            action_space=action_space,
            target_group=self._target_group,
            step=self._reweight_step,
            initial_weight=self._initial_weight,
            max_weight=self._max_weight,
        )
        return ControllerOutput(
            action=action,
            confidence=0.7,
            rationale=(
                f"loss plateaued over the last {len(self._loss_history)} consultations; "
                f"increasing {self._target_group!r} exposure"
            ),
        )
