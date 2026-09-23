"""`RuleBasedController`, `StatelessController`, `HistoryAwareController`: each proposes
`NoopAction` while loss is still improving or `REWEIGHT_DATA` isn't enabled, and a
`ReweightDataAction` once plateaued — `HistoryAwareController` additionally holds off
when a similar reweight has a bad precedent in its queried history.
"""

from __future__ import annotations

from corefinity_adaptive.actions.outcome import Assessment
from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import ActionKind, NoopAction, ReweightDataAction
from corefinity_adaptive.controllers.history_aware import HistoryAwareController
from corefinity_adaptive.controllers.rule_based import RuleBasedController
from corefinity_adaptive.controllers.stateless import StatelessController
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot, TrainingState
from corefinity_adaptive.validation.results import ValidationResult

_REWEIGHT_SPACE = ActionSpace(
    enabled_kinds=frozenset({ActionKind.REWEIGHT_DATA}), max_reweight_delta=0.5
)
_NOOP_ONLY_SPACE = ActionSpace(enabled_kinds=frozenset({ActionKind.NOOP}))


def _snapshot(step: int, train_loss: float) -> MetricSnapshot:
    return MetricSnapshot(
        step=step,
        epoch=float(step),
        train_loss=train_loss,
        learning_rate=1e-3,
        wall_clock_seconds=float(step),
    )


def _state(
    step: int, train_loss: float, *, recent_trend: tuple[MetricSnapshot, ...] = ()
) -> TrainingState:
    return TrainingState(
        run_id="r",
        branch_id="b",
        step=step,
        wall_clock_seconds=float(step),
        current=_snapshot(step, train_loss),
        recent_trend=recent_trend,
        budget_consumed=BudgetUsage(),
        budget_remaining=BudgetUsage(),
        seed=0,
        config_hash="hash",
    )


def test_rule_based_noop_when_reweight_not_enabled() -> None:
    controller = RuleBasedController()
    result = controller.propose(
        state=_state(1, 1.0),
        history=(),
        action_space=_NOOP_ONLY_SPACE,
        budget_remaining=BudgetUsage(),
    )
    assert isinstance(result.action, NoopAction)


def test_rule_based_noop_while_its_own_history_shows_improvement() -> None:
    controller = RuleBasedController(plateau_window=2)
    for step, loss in enumerate([3.0, 2.0, 1.0], start=1):
        result = controller.propose(
            state=_state(step, loss),
            history=(),
            action_space=_REWEIGHT_SPACE,
            budget_remaining=BudgetUsage(),
        )
    assert isinstance(result.action, NoopAction)


def test_rule_based_reweights_once_its_own_history_plateaus() -> None:
    controller = RuleBasedController(plateau_window=2, reweight_step=0.1)
    losses = [3.0, 2.0, 2.0, 2.0]  # improves once, then flat for plateau_window+1 calls
    result: ControllerOutput | None = None
    for step, loss in enumerate(losses, start=1):
        result = controller.propose(
            state=_state(step, loss),
            history=(),
            action_space=_REWEIGHT_SPACE,
            budget_remaining=BudgetUsage(),
        )
    assert result is not None
    assert isinstance(result.action, ReweightDataAction)


def test_stateless_derives_plateau_from_recent_trend_in_one_call() -> None:
    trend = tuple(_snapshot(s, loss) for s, loss in enumerate([2.0, 2.0, 2.0], start=1))
    controller = StatelessController(reweight_step=0.1)
    result = controller.propose(
        state=_state(4, 2.0, recent_trend=trend),
        history=(),
        action_space=_REWEIGHT_SPACE,
        budget_remaining=BudgetUsage(),
    )
    assert isinstance(result.action, ReweightDataAction)


def test_stateless_noop_when_recent_trend_shows_improvement() -> None:
    trend = tuple(_snapshot(s, loss) for s, loss in enumerate([5.0, 4.0, 3.0], start=1))
    controller = StatelessController()
    result = controller.propose(
        state=_state(4, 1.0, recent_trend=trend),
        history=(),
        action_space=_REWEIGHT_SPACE,
        budget_remaining=BudgetUsage(),
    )
    assert isinstance(result.action, NoopAction)


def _bad_precedent_record(target_group: str) -> ExperienceRecord:
    action = ReweightDataAction(group_weights={target_group: 0.6})
    return ExperienceRecord(
        run_id="r",
        branch_id="b",
        step=1,
        seed=0,
        state=_state(1, 1.0),
        proposed_action=action,
        validation_result=ValidationResult(approved=True, action=action),
        controller_name="history_aware",
        controller_confidence=0.8,
        executed=True,
        assessment=Assessment.NEGATIVE,
    )


def test_history_aware_reweights_when_no_bad_precedent() -> None:
    trend = tuple(_snapshot(s, loss) for s, loss in enumerate([2.0, 2.0, 2.0], start=1))
    controller = HistoryAwareController(target_group="hard", reweight_step=0.1)
    result = controller.propose(
        state=_state(4, 2.0, recent_trend=trend),
        history=(),
        action_space=_REWEIGHT_SPACE,
        budget_remaining=BudgetUsage(),
    )
    assert isinstance(result.action, ReweightDataAction)


def test_history_aware_holds_off_on_bad_precedent() -> None:
    trend = tuple(_snapshot(s, loss) for s, loss in enumerate([3.0, 2.0, 2.0], start=1))
    controller = HistoryAwareController(target_group="hard", reweight_step=0.1)
    result = controller.propose(
        state=_state(4, 2.0, recent_trend=trend),
        history=(_bad_precedent_record("hard"),),
        action_space=_REWEIGHT_SPACE,
        budget_remaining=BudgetUsage(),
    )
    assert isinstance(result.action, NoopAction)


def test_history_aware_ignores_precedent_for_a_different_group() -> None:
    trend = tuple(_snapshot(s, loss) for s, loss in enumerate([2.0, 2.0, 2.0], start=1))
    controller = HistoryAwareController(target_group="hard", reweight_step=0.1)
    result = controller.propose(
        state=_state(4, 2.0, recent_trend=trend),
        history=(_bad_precedent_record("easy"),),
        action_space=_REWEIGHT_SPACE,
        budget_remaining=BudgetUsage(),
    )
    assert isinstance(result.action, ReweightDataAction)
