"""`ActionValidator`: the sole gate between any controller and the trainer.

Modeled as a safety *shield* (per safe-RL shielding practice), not a schema checker: it
owns the state needed to detect oscillation and enforce the evaluation gate, it is the
only place budget is actually charged, and every decision it makes — approved or
rejected — is written to the audit log before it returns. No rule here, and no caller of
this class, may skip a check because a controller reported high confidence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from corefinity_adaptive.actions.outcome import ActionOutcome
from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import Action, AdjustLearningRateAction, ReweightDataAction
from corefinity_adaptive.observability.audit_log import AuditLog, AuditLogEntry
from corefinity_adaptive.state.budget import BudgetTracker
from corefinity_adaptive.state.history import BoundedHistory
from corefinity_adaptive.state.models import TrainingState
from corefinity_adaptive.validation.results import RuleViolation, ValidationResult
from corefinity_adaptive.validation.rollback import RegressionRule, RollbackPolicy
from corefinity_adaptive.validation.rules import (
    DEFAULT_RULES,
    RuleCheck,
    RuleContext,
    estimate_cost,
    requires_checkpoint,
    requires_eval_gate,
)


class ValidatorStateSnapshot(BaseModel):
    """The validator's oscillation-detection baseline at one point in time.

    Captured alongside a trainer checkpoint and restored together with it on rollback —
    without this, the validator would keep treating a rolled-back action's value as
    "current" even after the model/optimizer physically reverted, and reject the
    controller's correct attempt to undo it as a false "oscillation".
    """

    model_config = ConfigDict(frozen=True)

    lr_history: tuple[float, ...]
    group_weight_history: dict[str, tuple[float, ...]]
    latest_eval_passed: bool | None


class ActionValidator:
    def __init__(
        self,
        *,
        action_space: ActionSpace,
        budget_tracker: BudgetTracker,
        audit_log: AuditLog,
        initial_lr: float,
        rollback_policy: RollbackPolicy | None = None,
        lr_history_window: int = 10,
        group_weight_history_window: int = 10,
        rules: tuple[RuleCheck, ...] = DEFAULT_RULES,
    ) -> None:
        self._action_space = action_space
        self._budget_tracker = budget_tracker
        self._audit_log = audit_log
        self._rollback_policy = rollback_policy or RollbackPolicy()
        self._rules = rules
        self._lr_history: BoundedHistory[float] = BoundedHistory(maxlen=lr_history_window)
        self._lr_history.push(initial_lr)
        self._group_weight_window = group_weight_history_window
        self._group_weight_history: dict[str, BoundedHistory[float]] = {}
        self._latest_eval_passed: bool | None = None

    def _build_context(self) -> RuleContext:
        return RuleContext(
            action_space=self._action_space,
            budget_tracker=self._budget_tracker,
            lr_history=self._lr_history.as_tuple(),
            group_weight_history={
                group: hist.as_tuple() for group, hist in self._group_weight_history.items()
            },
            latest_eval_passed=self._latest_eval_passed,
        )

    def validate(
        self,
        proposed: ControllerOutput,
        state: TrainingState,
        *,
        controller_name: str,
    ) -> ValidationResult:
        """Run every rule against the proposal and record the outcome, win or lose."""
        context = self._build_context()
        violations: list[RuleViolation] = []
        for rule in self._rules:
            violation = rule(proposed.action, state, context)
            if violation is not None:
                violations.append(violation)

        approved = not violations
        result = ValidationResult(
            approved=approved,
            action=proposed.action,
            violations=tuple(violations),
            requires_checkpoint_first=approved and requires_checkpoint(proposed.action, context),
            requires_eval_gate=requires_eval_gate(proposed.action),
        )

        self._audit_log.record(
            AuditLogEntry(
                run_id=state.run_id,
                branch_id=state.branch_id,
                step=state.step,
                controller_name=controller_name,
                controller_confidence=proposed.confidence,
                proposed_action=proposed.action,
                result=result,
            )
        )
        return result

    def record_applied(self, action: Action) -> None:
        """Update budget spend and per-dimension history for an action just applied.

        Must only be called for an action that was both approved by `validate()` and
        actually applied by the trainer — a validated-but-never-applied proposal must not
        consume budget or shift the oscillation-detection baseline.
        """
        self._budget_tracker.spend(estimate_cost(action))
        if isinstance(action, AdjustLearningRateAction):
            self._lr_history.push(action.new_lr)
        elif isinstance(action, ReweightDataAction):
            for group, weight in action.group_weights.items():
                history = self._group_weight_history.setdefault(
                    group, BoundedHistory(maxlen=self._group_weight_window)
                )
                history.push(weight)

    def record_eval_result(self, *, passed: bool) -> None:
        self._latest_eval_passed = passed

    def evaluate_rollback(self, outcome: ActionOutcome) -> RegressionRule | None:
        """Return the violated regression rule, if the outcome warrants a rollback."""
        return self._rollback_policy.should_rollback(outcome)

    def capture_state(self) -> ValidatorStateSnapshot:
        """Snapshot the oscillation-detection baseline, to restore on rollback."""
        return ValidatorStateSnapshot(
            lr_history=self._lr_history.as_tuple(),
            group_weight_history={
                group: hist.as_tuple() for group, hist in self._group_weight_history.items()
            },
            latest_eval_passed=self._latest_eval_passed,
        )

    def restore_state(self, snapshot: ValidatorStateSnapshot) -> None:
        """Undo `record_applied`'s bookkeeping back to a prior `capture_state()` point."""
        self._lr_history.replace(snapshot.lr_history)
        for group, history in snapshot.group_weight_history.items():
            bounded = self._group_weight_history.setdefault(
                group, BoundedHistory(maxlen=self._group_weight_window)
            )
            bounded.replace(history)
        self._latest_eval_passed = snapshot.latest_eval_passed
