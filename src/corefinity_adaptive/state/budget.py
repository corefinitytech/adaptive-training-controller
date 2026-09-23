"""Tracks cumulative budget spend against a hard cap for one run/branch.

Lives in `state/`, not `validation/`, even though `ActionValidator` is its main
consumer: it's the single ledger both the state layer (reporting `budget_consumed`/
`budget_remaining` on every `TrainingState`) and the validation layer (the hard-budget
guardrail) read from. Putting it in `validation/` instead would make `state/` depend on
`validation/` while `validation/` already depends on `state/` for `BudgetUsage` —  a
two-package cycle. One ledger, owned by the lower layer both need, avoids that.
"""

from __future__ import annotations

from corefinity_adaptive.state.models import BudgetUsage


class BudgetTracker:
    """Mutable running total of consumed budget, checked against a fixed cap.

    Deliberately mutable (unlike the frozen `BudgetUsage` it wraps) — it owns the one
    place spend accumulates for a branch, so every caller (the trainer's per-step
    accounting, `ActionValidator`'s hard-budget check, `StateBuilder`'s reporting) reads
    the same total instead of reconstructing one independently.
    """

    def __init__(self, total: BudgetUsage) -> None:
        self._total = total
        self._consumed = BudgetUsage()

    @property
    def total(self) -> BudgetUsage:
        return self._total

    @property
    def consumed(self) -> BudgetUsage:
        return self._consumed

    def spend(self, delta: BudgetUsage) -> None:
        self._consumed = self._consumed.plus(delta)

    def would_exceed(self, additional: BudgetUsage) -> bool:
        return self._consumed.plus(additional).exceeds(self._total)
