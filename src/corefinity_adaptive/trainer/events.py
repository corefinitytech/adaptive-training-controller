"""Training loop events and the read-only callback protocol.

Modeled on MosaicML Composer's Event+State split: callbacks here are strictly read-only
(logging, introspection) — the only code path that may mutate trainer state is
`TrainerAdapter` applying a validator-approved `Action`. This keeps "what changed and
why" always traceable to exactly one place. See `trainer/adapter.py`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from corefinity_adaptive.state.models import TrainingState


class TrainingEvent(StrEnum):
    INTERVAL_START = "interval_start"
    BATCH_END = "batch_end"
    INTERVAL_END = "interval_end"
    """The controller is consulted here."""
    EVAL_END = "eval_end"
    CHECKPOINT_SAVED = "checkpoint_saved"
    BRANCH_CREATED = "branch_created"
    RUN_END = "run_end"


class TrainerCallback(Protocol):
    """A read-only observer of the training loop. May log; may never mutate state."""

    def on_event(self, event: TrainingEvent, state: TrainingState) -> None: ...
