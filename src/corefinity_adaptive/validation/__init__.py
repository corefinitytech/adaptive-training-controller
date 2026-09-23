"""The action validator ("shield"), its guardrail rules, and rollback.

`BudgetTracker` itself lives in `state/budget.py` (see that module's docstring for why)
and is re-exported here since the hard-budget guardrail is its main consumer.
"""

from corefinity_adaptive.state.budget import BudgetTracker
from corefinity_adaptive.validation.results import RuleViolation, ValidationResult
from corefinity_adaptive.validation.rollback import RegressionRule, RollbackPolicy
from corefinity_adaptive.validation.rules import DEFAULT_RULES, RuleContext
from corefinity_adaptive.validation.validator import ActionValidator, ValidatorStateSnapshot

__all__ = [
    "DEFAULT_RULES",
    "ActionValidator",
    "BudgetTracker",
    "RegressionRule",
    "RollbackPolicy",
    "RuleContext",
    "RuleViolation",
    "ValidationResult",
    "ValidatorStateSnapshot",
]
