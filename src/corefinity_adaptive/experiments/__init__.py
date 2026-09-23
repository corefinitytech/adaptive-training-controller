"""Experiment branching: lineage/config/checkpoint bookkeeping and a PBT-style
exploit/explore policy. Branches run sequentially in this build phase — see
`ExperimentManager`'s docstring.
"""

from corefinity_adaptive.experiments.branch import Branch, BranchStatus
from corefinity_adaptive.experiments.manager import ExperimentManager, UnknownBranchError
from corefinity_adaptive.experiments.policy import perturb_learning_rate, rank_for_exploit

__all__ = [
    "Branch",
    "BranchStatus",
    "ExperimentManager",
    "UnknownBranchError",
    "perturb_learning_rate",
    "rank_for_exploit",
]
