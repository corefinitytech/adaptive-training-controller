"""`Branch`: one experiment branch's metadata, tracked by `ExperimentManager`.

A branch's *budget allocation* is bookkeeping only — it does not gate anything by
itself. Whether an `AllocateComputeAction` is affordable is decided entirely by
`ActionValidator`'s existing hard-budget guardrail against the proposing branch's own
`BudgetTracker`, exactly as for every other action kind. `ExperimentManager` deliberately
does not maintain a second, parallel budget check: two ledgers meant to represent the
same constraint is exactly the bug class fixed in `state/budget.py` — see that module's
docstring. What `ExperimentManager` owns instead is lineage, config, checkpoints, and
lifecycle status: things no other component tracks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corefinity_adaptive.state.models import BudgetUsage


class BranchStatus(StrEnum):
    PENDING = "pending"
    """Registered but not yet run by a `TrainerAdapter`."""
    ACTIVE = "active"
    """Currently being run."""
    TERMINATED = "terminated"
    PROMOTED = "promoted"
    """Terminated specifically because it seeded a new branch during exploit/explore."""


class Branch(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch_id: str
    parent_branch_id: str | None
    config_overrides: dict[str, float | int | str | bool]
    budget: BudgetUsage
    status: BranchStatus
    checkpoint: Any | None = None
    """An opaque snapshot (e.g. a `TrainerAdapter`'s own `{"model": ..., "optimizer":
    ...}` in-memory checkpoint dict) to seed a new `TrainerAdapter` from. Not validated
    or interpreted by `ExperimentManager` itself."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    terminated_reason: str | None = None
