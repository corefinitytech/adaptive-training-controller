"""`Controller`: the public factory namespace matching the research spec's illustrative
API (`Controller.fixed()`, `Controller.jev(api_key=...)`, ...).

`jev` is added once its implementation lands in Phase 2 (see the project plan) — another
classmethod here returning an object that satisfies `ControllerStrategy`, same as every
controller below.
"""

from __future__ import annotations

from corefinity_adaptive.controllers.base import ControllerStrategy
from corefinity_adaptive.controllers.fixed import FixedController
from corefinity_adaptive.controllers.history_aware import HistoryAwareController
from corefinity_adaptive.controllers.rule_based import RuleBasedController
from corefinity_adaptive.controllers.stateless import StatelessController


class Controller:
    """Namespace of factory methods for the controllers this library ships."""

    @staticmethod
    def fixed() -> ControllerStrategy:
        """The no-op baseline every comparison is measured against."""
        return FixedController()

    @staticmethod
    def rule_based(
        *,
        target_group: str = "difficult",
        plateau_window: int = 3,
        min_relative_improvement: float = 0.0,
        reweight_step: float = 0.1,
        initial_weight: float = 0.5,
        max_weight: float = 1.0,
    ) -> ControllerStrategy:
        """Deterministic plateau-triggered reweighting, watching its own consultation
        history (see `RuleBasedController`)."""
        return RuleBasedController(
            target_group=target_group,
            plateau_window=plateau_window,
            min_relative_improvement=min_relative_improvement,
            reweight_step=reweight_step,
            initial_weight=initial_weight,
            max_weight=max_weight,
        )

    @staticmethod
    def stateless(
        *,
        target_group: str = "difficult",
        min_relative_improvement: float = 0.0,
        reweight_step: float = 0.1,
        initial_weight: float = 0.5,
        max_weight: float = 1.0,
    ) -> ControllerStrategy:
        """The same plateau rule as `rule_based`, but derived only from `TrainingState.
        recent_trend` — no controller-owned memory (see `StatelessController`)."""
        return StatelessController(
            target_group=target_group,
            min_relative_improvement=min_relative_improvement,
            reweight_step=reweight_step,
            initial_weight=initial_weight,
            max_weight=max_weight,
        )

    @staticmethod
    def history_aware(
        *,
        target_group: str = "difficult",
        min_relative_improvement: float = 0.0,
        reweight_step: float = 0.1,
        initial_weight: float = 0.5,
        max_weight: float = 1.0,
    ) -> ControllerStrategy:
        """`stateless`'s plateau rule, plus a check against the experience store for a
        bad precedent before acting on it (see `HistoryAwareController`)."""
        return HistoryAwareController(
            target_group=target_group,
            min_relative_improvement=min_relative_improvement,
            reweight_step=reweight_step,
            initial_weight=initial_weight,
            max_weight=max_weight,
        )
