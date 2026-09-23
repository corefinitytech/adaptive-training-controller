"""Controllers: the pluggable strategies that propose interventions.

`Controller` is the public factory namespace (`Controller.fixed()`, later `.jev(...)`
etc.); `ControllerStrategy` is the structural protocol every concrete controller
implements.
"""

from corefinity_adaptive.controllers.base import ControllerOutput, ControllerStrategy
from corefinity_adaptive.controllers.factory import Controller
from corefinity_adaptive.controllers.fixed import FixedController

__all__ = ["Controller", "ControllerOutput", "ControllerStrategy", "FixedController"]
