"""Bounded retention: the store-side half of the anti-"history explosion" guardrail (§12).

`recent_trend` on `TrainingState` bounds what a single decision sees (see
`state/history.py`); `RetentionPolicy` bounds what accumulates in the store over an
entire run, so both `query_similar`'s candidate pool and the store's on-disk size stay
flat over long runs instead of growing without limit.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PositiveInt


class RetentionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_records_per_branch: PositiveInt = 5_000
    max_total_records: PositiveInt = 50_000
    protected_branch_ids: frozenset[str] = frozenset()
    """Branches never pruned regardless of age (typically the currently active one)."""
