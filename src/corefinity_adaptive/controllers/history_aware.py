"""`HistoryAwareController`: `StatelessController`'s plateau rule, plus a real check
against the experience store before acting on it.

Shares `StatelessController`'s exact plateau signal (derived from `TrainingState.
recent_trend`, not controller-owned memory) so the only variable between the two is
whether experience history changes the decision — directly the comparison the research
spec's H3 hypothesis (§9) wants measured. Before proposing a reweight, this controller
asks "was a similar reweight of this group tried recently, and did it go badly?" — if
so, it holds off instead of repeating a known-bad intervention.
"""

from __future__ import annotations

from corefinity_adaptive.actions.outcome import Assessment
from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import ActionKind, NoopAction, ReweightDataAction
from corefinity_adaptive.controllers.policies import is_plateaued, primary_loss, propose_reweight
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.state.models import BudgetUsage, TrainingState


def _has_bad_precedent(target_group: str, history: tuple[ExperienceRecord, ...]) -> bool:
    """True if a similar past reweight of `target_group` was rejected or scored negative."""
    for record in history:
        action = record.proposed_action
        if not isinstance(action, ReweightDataAction) or target_group not in action.group_weights:
            continue
        if not record.validation_result.approved or record.assessment == Assessment.NEGATIVE:
            return True
    return False


class HistoryAwareController:
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
        return "history_aware"

    @property
    def requires_history(self) -> bool:
        return True

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
        if _has_bad_precedent(self._target_group, history):
            return ControllerOutput(
                action=NoopAction(),
                confidence=0.5,
                rationale=(
                    f"loss plateaued, but a similar {self._target_group!r} reweight was "
                    "rejected or scored negative recently — holding off"
                ),
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
            confidence=0.8,
            rationale=(
                f"loss plateaued and no negative precedent found in the last "
                f"{len(history)} similar records; increasing {self._target_group!r} exposure"
            ),
        )
