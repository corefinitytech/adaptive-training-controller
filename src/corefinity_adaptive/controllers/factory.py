"""`Controller`: the public factory namespace matching the research spec's illustrative
API (`Controller.fixed()`, `Controller.jev(api_key=...)`, ...).

Only `Controller.fixed()` exists in this phase. `rule_based`, `stateless`,
`history_aware`, and `jev` are added as their implementations land in later build phases
(see the project plan) — each one simply another classmethod here returning an object
that satisfies `ControllerStrategy`.
"""

from __future__ import annotations

from corefinity_adaptive.controllers.base import ControllerStrategy
from corefinity_adaptive.controllers.fixed import FixedController


class Controller:
    """Namespace of factory methods for the controllers this library ships."""

    @staticmethod
    def fixed() -> ControllerStrategy:
        """The no-op baseline every comparison is measured against."""
        return FixedController()
