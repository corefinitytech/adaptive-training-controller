"""`ExperienceStore`: the inspectable, queryable record of every intervention tried.

A `Protocol` so a Postgres- or vector-backed store can replace `SQLiteExperienceStore`
later (Phase 6/7 scale) without any controller, validator, or trainer code changing.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from corefinity_adaptive.actions.outcome import ActionOutcome
from corefinity_adaptive.memory.query import ExperienceFilter
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.memory.retention import RetentionPolicy
from corefinity_adaptive.state.models import TrainingState


class ExperienceStore(Protocol):
    def append(self, record: ExperienceRecord) -> None: ...

    def update_outcome(self, record_id: UUID, outcome: ActionOutcome) -> None: ...

    def query_by_run(self, run_id: str) -> list[ExperienceRecord]: ...

    def query_similar(
        self,
        state: TrainingState,
        *,
        k: int,
        filters: ExperienceFilter | None = None,
    ) -> list[ExperienceRecord]: ...

    def prune(self, policy: RetentionPolicy) -> int:
        """Apply a retention policy, returning the number of records removed."""
        ...
