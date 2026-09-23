"""`DefaultStateBuilder`: bounded trend window, budget/data-exposure accounting."""

from __future__ import annotations

from corefinity_adaptive.state.budget import BudgetTracker
from corefinity_adaptive.state.builder import DefaultStateBuilder
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot


def _snapshot(step: int, loss: float = 1.0) -> MetricSnapshot:
    return MetricSnapshot(
        step=step,
        epoch=float(step),
        train_loss=loss,
        learning_rate=1e-3,
        wall_clock_seconds=float(step),
    )


def test_recent_trend_stays_bounded_past_window_size() -> None:
    builder = DefaultStateBuilder(
        run_id="r",
        branch_id="b",
        seed=0,
        config_hash="deadbeef",
        budget_tracker=BudgetTracker(total=BudgetUsage(steps=100)),
        trend_window=3,
    )
    state = None
    for step in range(1, 11):
        state = builder.observe(_snapshot(step))
    assert state is not None
    # recent_trend holds snapshots *prior to* the current one, bounded at trend_window
    assert len(state.recent_trend) == 3
    assert [s.step for s in state.recent_trend] == [7, 8, 9]


def test_budget_consumed_and_remaining_track_spend() -> None:
    builder = DefaultStateBuilder(
        run_id="r",
        branch_id="b",
        seed=0,
        config_hash="deadbeef",
        budget_tracker=BudgetTracker(total=BudgetUsage(steps=10, gpu_hours=2.0)),
    )
    builder.record_budget_spend(BudgetUsage(steps=1))
    builder.record_budget_spend(BudgetUsage(steps=1, gpu_hours=0.5))
    state = builder.observe(_snapshot(1))
    assert state.budget_consumed.steps == 2
    assert state.budget_consumed.gpu_hours == 0.5
    assert state.budget_remaining.steps == 8
    assert state.budget_remaining.gpu_hours == 1.5


def test_group_weight_reflects_last_set_value_not_a_running_total() -> None:
    builder = DefaultStateBuilder(
        run_id="r",
        branch_id="b",
        seed=0,
        config_hash="deadbeef",
        budget_tracker=BudgetTracker(total=BudgetUsage()),
    )
    builder.set_group_weight("easy", 0.3)
    builder.set_group_weight("easy", 0.2)  # overwrites, does not accumulate
    builder.set_group_weight("hard", 0.1)
    state = builder.observe(_snapshot(1))
    assert state.data_exposure == {"easy": 0.2, "hard": 0.1}


def test_state_carries_identity_and_reproducibility_fields() -> None:
    builder = DefaultStateBuilder(
        run_id="run-42",
        branch_id="branch-a",
        seed=7,
        config_hash="cafebabe",
        budget_tracker=BudgetTracker(total=BudgetUsage()),
    )
    state = builder.observe(_snapshot(1))
    assert state.run_id == "run-42"
    assert state.branch_id == "branch-a"
    assert state.seed == 7
    assert state.config_hash == "cafebabe"
