"""What happened after a validated action executed.

`ActionOutcome` is filled in once the *next* interval completes — an action's effect
can't be known at proposal time — and is what turns an `ExperienceRecord` (see
`memory/records.py`) from "here's what was tried" into "here's what happened," which is
the whole point of the experience layer.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from corefinity_adaptive.state.models import BudgetUsage


class Assessment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    PENDING = "pending"
    """Outcome not yet observed — the next interval hasn't completed."""


class ActionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed_at_step: int
    metric_deltas: dict[str, float]
    """Named metric -> change since the action was applied, e.g. {"val_loss": -0.03}."""
    additional_cost: BudgetUsage
    """Compute spent as a consequence of the action, beyond ordinary training cost."""
    assessment: Assessment
