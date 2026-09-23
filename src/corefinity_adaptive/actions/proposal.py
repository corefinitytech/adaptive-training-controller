"""`ControllerOutput`: a controller's proposal, decoupled from any specific controller.

Lives alongside the `Action` taxonomy rather than in `controllers/`, deliberately:
`ActionValidator` (in `validation/`) needs this type but must never depend on
`controllers/` (a controller depends on the validator's guardrails existing, not the
other way around), and `ExperienceRecord` (in `memory/`) needs `Action` but not
`ControllerOutput` at all. Keeping this type here — next to `Action`, which both
`controllers/` and `validation/` already depend on — is what keeps that dependency graph
acyclic: `controllers` -> `actions`, `validation` -> `actions`, and neither of those two
depends on the other.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corefinity_adaptive.actions.types import Action


class ControllerOutput(BaseModel):
    """A controller's proposal: a typed action, never a free-form string or dict.

    `confidence` is informational only — see `validation/validator.py`, which never
    branches a safety check on it. `raw_controller_metadata` carries whatever a specific
    controller wants to preserve for later analysis (e.g. Jev's full probability
    distribution over options) without polluting the common `Action`/`confidence` shape.
    """

    model_config = ConfigDict(frozen=True)

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None
    raw_controller_metadata: dict[str, Any] = Field(default_factory=dict)
