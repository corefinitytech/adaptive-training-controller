"""`ActionSpace`: the explicit allowed-action list and numeric bounds for one run.

This is the guardrail from the research spec's §13 ("explicit allowed-action list")
expressed as data, consumed by both sides of the safety boundary: a well-behaved
controller uses it to avoid proposing something disallowed, and `ActionValidator` uses it
to never trust the controller not to anyway.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from corefinity_adaptive.actions.types import ActionKind


class ActionSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled_kinds: frozenset[ActionKind] = Field(
        default_factory=lambda: frozenset({ActionKind.NOOP})
    )
    min_lr: PositiveFloat = 1e-6
    max_lr: PositiveFloat = 1.0
    max_reweight_delta: float = Field(default=0.2, gt=0, le=1.0)
    """Largest allowed absolute change to any single data group's weight in one action."""

    def allows(self, kind: ActionKind) -> bool:
        return kind in self.enabled_kinds
