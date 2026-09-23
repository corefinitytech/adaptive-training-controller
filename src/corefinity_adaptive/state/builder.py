"""Turns raw per-step measurements into a bounded, controller-readable `TrainingState`."""

from __future__ import annotations

from typing import Protocol

from corefinity_adaptive.state.budget import BudgetTracker
from corefinity_adaptive.state.history import BoundedHistory
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot, TrainingState


class StateBuilder(Protocol):
    """Anything that can turn a new metric snapshot into a `TrainingState`."""

    def observe(self, snapshot: MetricSnapshot) -> TrainingState: ...


class DefaultStateBuilder:
    """The reference `StateBuilder`: maintains a bounded trend window per run/branch.

    One instance is scoped to a single (run, branch) — the `TrainerAdapter` owns its
    lifecycle and constructs a fresh one per branch, sharing the same `BudgetTracker`
    instance with the branch's `ActionValidator` so both report against one ledger.
    """

    def __init__(
        self,
        *,
        run_id: str,
        branch_id: str,
        seed: int,
        config_hash: str,
        budget_tracker: BudgetTracker,
        trend_window: int = 20,
    ) -> None:
        self._run_id = run_id
        self._branch_id = branch_id
        self._seed = seed
        self._config_hash = config_hash
        self._budget_tracker = budget_tracker
        self._data_exposure: dict[str, float] = {}
        self._trend: BoundedHistory[MetricSnapshot] = BoundedHistory(maxlen=trend_window)

    def record_budget_spend(self, delta: BudgetUsage) -> None:
        self._budget_tracker.spend(delta)

    def set_group_weight(self, group: str, weight: float) -> None:
        """Record `group`'s current sampling weight, replacing whatever was set before.

        `weight` is the absolute value a `ReweightDataAction` set (see
        `actions/types.py::ReweightDataAction`), not a delta — `data_exposure` reports
        "what is this group's weight right now," not a running total.
        """
        self._data_exposure[group] = weight

    def _budget_remaining(self) -> BudgetUsage:
        total, used = self._budget_tracker.total, self._budget_tracker.consumed
        return BudgetUsage(
            gpu_hours=_remaining_float(total.gpu_hours, used.gpu_hours),
            tokens=_remaining_int(total.tokens, used.tokens),
            steps=_remaining_int(total.steps, used.steps),
            wall_clock_seconds=_remaining_float(total.wall_clock_seconds, used.wall_clock_seconds),
        )

    def observe(self, snapshot: MetricSnapshot) -> TrainingState:
        state = TrainingState(
            run_id=self._run_id,
            branch_id=self._branch_id,
            step=snapshot.step,
            wall_clock_seconds=snapshot.wall_clock_seconds,
            current=snapshot,
            recent_trend=self._trend.as_tuple(),
            data_exposure=dict(self._data_exposure),
            budget_consumed=self._budget_tracker.consumed,
            budget_remaining=self._budget_remaining(),
            seed=self._seed,
            config_hash=self._config_hash,
        )
        self._trend.push(snapshot)
        return state


def _remaining_float(total: float | None, used: float | None) -> float | None:
    return None if total is None else max(total - (used or 0.0), 0.0)


def _remaining_int(total: int | None, used: int | None) -> int | None:
    return None if total is None else max(total - (used or 0), 0)
