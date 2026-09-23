"""`is_plateaued` and `propose_reweight`: the pure decision arithmetic shared by
`RuleBasedController`, `StatelessController`, and `HistoryAwareController`.
"""

from __future__ import annotations

from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.controllers.policies import is_plateaued, primary_loss, propose_reweight
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot, TrainingState


def _snapshot(step: int, *, train_loss: float, val_loss: float | None = None) -> MetricSnapshot:
    return MetricSnapshot(
        step=step,
        epoch=float(step),
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=1e-3,
        wall_clock_seconds=float(step),
    )


def test_primary_loss_prefers_val_loss_over_train_loss() -> None:
    assert primary_loss(_snapshot(1, train_loss=9.0, val_loss=1.0)) == 1.0


def test_primary_loss_falls_back_to_train_loss() -> None:
    assert primary_loss(_snapshot(1, train_loss=9.0)) == 9.0


def test_fewer_than_two_points_is_never_a_plateau() -> None:
    assert is_plateaued((), min_relative_improvement=0.0) is False
    assert is_plateaued((1.0,), min_relative_improvement=0.0) is False


def test_no_improvement_is_a_plateau() -> None:
    assert is_plateaued((1.0, 1.0, 1.0), min_relative_improvement=0.0) is True


def test_worsening_is_a_plateau() -> None:
    assert is_plateaued((1.0, 1.2), min_relative_improvement=0.0) is True


def test_any_improvement_clears_a_zero_threshold_plateau() -> None:
    assert is_plateaued((1.0, 0.99), min_relative_improvement=0.0) is False


def test_improvement_below_threshold_still_counts_as_plateau() -> None:
    # 1% improvement, but a 5% improvement is required.
    assert is_plateaued((1.0, 0.99), min_relative_improvement=0.05) is True


def test_improvement_above_threshold_is_not_a_plateau() -> None:
    assert is_plateaued((1.0, 0.9), min_relative_improvement=0.05) is False


def test_non_positive_earliest_value_falls_back_to_non_improvement_check() -> None:
    assert is_plateaued((0.0, 0.0), min_relative_improvement=0.0) is True
    assert is_plateaued((0.0, -1.0), min_relative_improvement=0.0) is False


def _state(data_exposure: dict[str, float] | None = None) -> TrainingState:
    return TrainingState(
        run_id="r",
        branch_id="b",
        step=1,
        wall_clock_seconds=1.0,
        current=_snapshot(1, train_loss=1.0),
        data_exposure=data_exposure or {},
        budget_consumed=BudgetUsage(),
        budget_remaining=BudgetUsage(),
        seed=0,
        config_hash="hash",
    )


def test_propose_reweight_starts_from_initial_weight_when_never_set() -> None:
    action = propose_reweight(
        state=_state(),
        action_space=ActionSpace(max_reweight_delta=0.5),
        target_group="hard",
        step=0.1,
        initial_weight=0.5,
        max_weight=1.0,
    )
    assert action.group_weights == {"hard": 0.6}


def test_propose_reweight_nudges_from_current_state_value() -> None:
    action = propose_reweight(
        state=_state({"hard": 0.3}),
        action_space=ActionSpace(max_reweight_delta=0.5),
        target_group="hard",
        step=0.1,
        initial_weight=0.5,
        max_weight=1.0,
    )
    assert action.group_weights == {"hard": 0.4}


def test_propose_reweight_step_is_bounded_by_action_space_max_delta() -> None:
    action = propose_reweight(
        state=_state({"hard": 0.5}),
        action_space=ActionSpace(max_reweight_delta=0.05),
        target_group="hard",
        step=0.5,  # would overshoot without the ActionSpace-aware clamp
        initial_weight=0.5,
        max_weight=1.0,
    )
    assert action.group_weights == {"hard": 0.55}


def test_propose_reweight_is_bounded_by_max_weight() -> None:
    action = propose_reweight(
        state=_state({"hard": 0.95}),
        action_space=ActionSpace(max_reweight_delta=0.5),
        target_group="hard",
        step=0.5,
        initial_weight=0.5,
        max_weight=1.0,
    )
    assert action.group_weights == {"hard": 1.0}
