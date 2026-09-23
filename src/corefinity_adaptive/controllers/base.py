"""The controller contract every strategy — baseline or intelligent — implements.

`ControllerStrategy` is a `Protocol`, not an ABC, so vendor-neutrality (research spec
§6/§17) is structural: any object with this shape works as a controller, including one
defined in a downstream project with no import dependency on this package.
`FixedController`, `RuleBasedController`, `HistoryAwareController`, and `JevController`
(later phases) all satisfy it identically, and `TrainerAdapter` never knows which one it's
holding.

Note this is deliberately *not* named `Controller` — that name is reserved for the public
factory namespace (`corefinity_adaptive.Controller.fixed()`, `.jev(...)`, matching the
research spec's illustrative API) in `controllers/factory.py`, which constructs and
returns objects satisfying this protocol.
"""

from __future__ import annotations

from typing import Protocol

from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.state.models import BudgetUsage, TrainingState

# `ControllerOutput` is defined in `actions/proposal.py` (see that module's docstring for
# why) and re-exported here since it's what a `ControllerStrategy.propose()` returns —
# most callers reasonably look for it alongside the controller protocol itself.
__all__ = ["ControllerOutput", "ControllerStrategy"]


class ControllerStrategy(Protocol):
    """Anything that can look at training state and propose an action."""

    @property
    def name(self) -> str:
        """A short, stable identifier used in audit log entries and experience records."""
        ...

    @property
    def requires_history(self) -> bool:
        """Whether the caller should query the experience store before calling `propose`.

        Lets a stateless controller (or `FixedController`) skip a memory query that it
        would ignore anyway.
        """
        ...

    def propose(
        self,
        *,
        state: TrainingState,
        history: tuple[ExperienceRecord, ...],
        action_space: ActionSpace,
        budget_remaining: BudgetUsage,
    ) -> ControllerOutput: ...
