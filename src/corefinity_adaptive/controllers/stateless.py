"""`StatelessController`: the same plateau-triggered reweighting rule as
`RuleBasedController`, but with zero controller-owned memory.

Every decision is derived solely from the `TrainingState` object handed to `propose()`
this call — specifically its bounded `recent_trend` window — never from anything the
controller itself remembers between calls. This is the deliberate contrast the research
spec's H3 hypothesis needs: `HistoryAwareController` is this same rule plus a real
experience-store lookup, so comparing the two isolates whether *memory* helps, not
whether the underlying rule differs.
"""

from __future__ import annotations

from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import ActionKind, NoopAction
from corefinity_adaptive.controllers.policies import is_plateaued, primary_loss, propose_reweight
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.state.models import BudgetUsage, TrainingState


class StatelessController:
    def __init__(
        self,
        *,
        target_group: str = "difficult",
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

    @property
    def name(self) -> str:
        return "stateless"

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
        loss_series = (*(primary_loss(s) for s in state.recent_trend), primary_loss(state.current))

        if not action_space.allows(ActionKind.REWEIGHT_DATA):
            return ControllerOutput(
                action=NoopAction(),
                confidence=1.0,
                rationale="REWEIGHT_DATA is not enabled for this run",
            )
        if not is_plateaued(loss_series, min_relative_improvement=self._min_relative_improvement):
            return ControllerOutput(
                action=NoopAction(),
                confidence=1.0,
                rationale="loss is still improving within the recent_trend window",
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
            confidence=0.6,
            rationale=(
                f"loss plateaued within the {len(state.recent_trend)}-step trend window; "
                f"increasing {self._target_group!r} exposure"
            ),
        )
