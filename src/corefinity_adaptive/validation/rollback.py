"""Post-hoc regression detection: automatic rollback after a defined regression (§13).

Unlike the pre-execution guardrails in `rules.py`, rollback is evaluated *after* an
interval completes, once the actual `ActionOutcome` is known. A triggered rollback is
itself recorded as a negative `ExperienceRecord` (see `memory/records.py`), which is
exactly what feeds `rules.py::check_oscillation` on later proposals.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from corefinity_adaptive.actions.outcome import ActionOutcome


class RegressionRule(BaseModel):
    """One metric's tolerance for moving in the wrong direction after an intervention."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    lower_is_better: bool
    max_allowed_regression: float
    """Magnitude of movement in the bad direction that's still tolerated. Must be >= 0."""


class RollbackPolicy:
    def __init__(self, rules: tuple[RegressionRule, ...] = ()) -> None:
        self._rules = rules

    def should_rollback(self, outcome: ActionOutcome) -> RegressionRule | None:
        """Return the first violated rule, or `None` if the outcome is within tolerance."""
        for rule in self._rules:
            delta = outcome.metric_deltas.get(rule.metric_name)
            if delta is None:
                continue
            bad_direction_magnitude = delta if rule.lower_is_better else -delta
            if bad_direction_magnitude > rule.max_allowed_regression:
                return rule
        return None
