"""Pure result types shared by the validator and the audit log.

Split out from `validator.py` so `observability/audit_log.py` can depend on the *shape*
of a validation result without importing the validator itself (which depends on the
audit log to write entries) — keeps the dependency graph acyclic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from corefinity_adaptive.actions.types import Action


class RuleViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_name: str
    message: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    action: Action
    """The action to execute if `approved`; the original proposal echoed back otherwise."""
    violations: tuple[RuleViolation, ...] = ()
    requires_checkpoint_first: bool = False
    requires_eval_gate: bool = False

    @property
    def triggered_rules(self) -> tuple[str, ...]:
        return tuple(v.rule_name for v in self.violations)

    @property
    def rejection_reason(self) -> str | None:
        if self.approved or not self.violations:
            return None
        return "; ".join(f"{v.rule_name}: {v.message}" for v in self.violations)
