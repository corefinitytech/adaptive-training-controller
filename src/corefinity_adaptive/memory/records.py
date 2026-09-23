"""`ExperienceRecord`: one intervention, its validation result, and (later) its outcome.

This is the unit the `ExperienceStore` persists and controllers query. A record is
written the moment `ActionValidator.validate()` runs — approved or rejected — and updated
once the outcome is known after the next interval. Recording rejected proposals too is
deliberate: it's what lets a history-aware controller learn "this was tried and refused,"
not just "this worked."
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from corefinity_adaptive.actions.outcome import ActionOutcome, Assessment
from corefinity_adaptive.actions.types import Action
from corefinity_adaptive.state.models import TrainingState
from corefinity_adaptive.validation.results import ValidationResult


class ExperienceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    branch_id: str
    step: int
    seed: int
    state: TrainingState
    proposed_action: Action
    validation_result: ValidationResult
    controller_name: str
    controller_confidence: float | None
    executed: bool
    """False if the validator rejected the proposal — the record is still kept."""
    outcome: ActionOutcome | None = None
    assessment: Assessment = Assessment.PENDING
    comparison_group_id: str | None = None
    """Ties this record to a controlled multi-controller/multi-seed comparison run, so
    the evaluator can distinguish a controlled comparison from a simple before/after."""

    def with_outcome(self, outcome: ActionOutcome) -> ExperienceRecord:
        """Return a copy of this record with its outcome (and derived assessment) filled in.

        Records are immutable, so updating a stored record means replacing it — see
        `ExperienceStore.update_outcome`.
        """
        return self.model_copy(update={"outcome": outcome, "assessment": outcome.assessment})
