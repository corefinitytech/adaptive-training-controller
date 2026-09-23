"""corefinity-adaptive: a vendor-neutral framework for closed-loop adaptive training.

The public surface is deliberately small — most of the package's types are reachable
from here or from their owning subpackage (`corefinity_adaptive.actions`,
`.controllers`, `.memory`, `.validation`, `.trainer`) for callers who need the lower-level
pieces (e.g. to assemble a `TrainerAdapter` directly, or write a new controller).
"""

from corefinity_adaptive._version import __version__
from corefinity_adaptive.actions import Action, ActionKind, ActionSpace
from corefinity_adaptive.controllers import Controller, ControllerOutput, ControllerStrategy
from corefinity_adaptive.state import BudgetUsage, TrainingState
from corefinity_adaptive.trainer import AdaptiveTrainer, RunReport, RunResult

__all__ = [
    "Action",
    "ActionKind",
    "ActionSpace",
    "AdaptiveTrainer",
    "BudgetUsage",
    "Controller",
    "ControllerOutput",
    "ControllerStrategy",
    "RunReport",
    "RunResult",
    "TrainingState",
    "__version__",
]
