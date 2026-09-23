"""Controllers: the pluggable strategies that propose interventions.

`Controller` is the public factory namespace (`Controller.fixed()`, `.rule_based(...)`,
`.stateless(...)`, `.history_aware(...)`, later `.jev(...)`); `ControllerStrategy` is the
structural protocol every concrete controller implements.
"""

from corefinity_adaptive.controllers.base import ControllerOutput, ControllerStrategy
from corefinity_adaptive.controllers.factory import Controller
from corefinity_adaptive.controllers.fixed import FixedController
from corefinity_adaptive.controllers.history_aware import HistoryAwareController
from corefinity_adaptive.controllers.rule_based import RuleBasedController
from corefinity_adaptive.controllers.stateless import StatelessController

__all__ = [
    "Controller",
    "ControllerOutput",
    "ControllerStrategy",
    "FixedController",
    "HistoryAwareController",
    "RuleBasedController",
    "StatelessController",
]
