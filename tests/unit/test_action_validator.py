"""`ActionValidator`: every guardrail rejects the proposal it targets, an approved
action passes through unmodified, and every `validate()` call — either outcome — writes
exactly one audit log entry.
"""

from __future__ import annotations

from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import (
    ActionKind,
    AdjustLearningRateAction,
    AllocateComputeAction,
    BranchExperimentAction,
    EarlyStopAction,
    NoopAction,
    ReweightDataAction,
)
from corefinity_adaptive.controllers.base import ControllerOutput
from corefinity_adaptive.observability.audit_log import InMemoryAuditLog
from corefinity_adaptive.state.budget import BudgetTracker
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot, TrainingState
from corefinity_adaptive.validation.validator import ActionValidator


def _state(step: int = 1) -> TrainingState:
    return TrainingState(
        run_id="r",
        branch_id="b",
        step=step,
        wall_clock_seconds=float(step),
        current=MetricSnapshot(
            step=step,
            epoch=float(step),
            train_loss=1.0,
            learning_rate=1e-3,
            wall_clock_seconds=float(step),
        ),
        budget_consumed=BudgetUsage(),
        budget_remaining=BudgetUsage(),
        seed=0,
        config_hash="hash",
    )


def _validator(
    *,
    enabled_kinds: frozenset[ActionKind],
    min_lr: float = 1e-6,
    max_lr: float = 1.0,
    max_reweight_delta: float = 0.2,
    budget_total: BudgetUsage | None = None,
    initial_lr: float = 1e-3,
) -> tuple[ActionValidator, InMemoryAuditLog]:
    audit_log = InMemoryAuditLog()
    budget_total = budget_total or BudgetUsage(steps=100)
    validator = ActionValidator(
        action_space=ActionSpace(
            enabled_kinds=enabled_kinds,
            min_lr=min_lr,
            max_lr=max_lr,
            max_reweight_delta=max_reweight_delta,
        ),
        budget_tracker=BudgetTracker(total=budget_total),
        audit_log=audit_log,
        initial_lr=initial_lr,
    )
    return validator, audit_log


def test_disallowed_action_kind_is_rejected() -> None:
    validator, _ = _validator(enabled_kinds=frozenset({ActionKind.NOOP}))
    proposal = ControllerOutput(action=AdjustLearningRateAction(new_lr=1e-2), confidence=0.9)
    result = validator.validate(proposal, _state(), controller_name="test")
    assert not result.approved
    assert "check_allowed_action_kind" in result.triggered_rules


def test_lr_out_of_bounds_is_rejected() -> None:
    validator, _ = _validator(
        enabled_kinds=frozenset({ActionKind.ADJUST_LEARNING_RATE}), max_lr=1e-2
    )
    proposal = ControllerOutput(action=AdjustLearningRateAction(new_lr=1.0), confidence=0.9)
    result = validator.validate(proposal, _state(), controller_name="test")
    assert not result.approved
    assert "check_lr_bounds" in result.triggered_rules


def test_lr_within_bounds_is_approved() -> None:
    validator, _ = _validator(enabled_kinds=frozenset({ActionKind.ADJUST_LEARNING_RATE}))
    proposal = ControllerOutput(action=AdjustLearningRateAction(new_lr=5e-3), confidence=0.9)
    result = validator.validate(proposal, _state(), controller_name="test")
    assert result.approved
    assert result.action == proposal.action
    assert result.triggered_rules == ()


def test_excessive_reweight_delta_is_rejected() -> None:
    validator, _ = _validator(
        enabled_kinds=frozenset({ActionKind.REWEIGHT_DATA}), max_reweight_delta=0.1
    )
    # first assignment for "hard" has no history, so it's allowed through once...
    first = ControllerOutput(action=ReweightDataAction(group_weights={"hard": 0.5}), confidence=0.9)
    result = validator.validate(first, _state(1), controller_name="test")
    assert result.approved
    validator.record_applied(result.action)

    # ...but a large jump from that baseline is rejected.
    second = ControllerOutput(
        action=ReweightDataAction(group_weights={"hard": 0.9}), confidence=0.9
    )
    result = validator.validate(second, _state(2), controller_name="test")
    assert not result.approved
    assert "check_max_reweight_delta" in result.triggered_rules


def test_hard_budget_exceeded_is_rejected() -> None:
    validator, _ = _validator(
        enabled_kinds=frozenset({ActionKind.ALLOCATE_COMPUTE}),
        budget_total=BudgetUsage(gpu_hours=1.0),
    )
    proposal = ControllerOutput(
        action=AllocateComputeAction(branch_id="b2", additional_gpu_hours=2.0), confidence=0.9
    )
    result = validator.validate(proposal, _state(), controller_name="test")
    assert not result.approved
    assert "check_hard_budget" in result.triggered_rules


def test_oscillation_reverses_immediately_is_rejected() -> None:
    validator, _ = _validator(enabled_kinds=frozenset({ActionKind.ADJUST_LEARNING_RATE}))
    up = ControllerOutput(action=AdjustLearningRateAction(new_lr=2e-3), confidence=0.9)
    result = validator.validate(up, _state(1), controller_name="test")
    assert result.approved
    validator.record_applied(result.action)

    down = ControllerOutput(action=AdjustLearningRateAction(new_lr=1e-3), confidence=0.9)
    result = validator.validate(down, _state(2), controller_name="test")
    assert not result.approved
    assert "check_oscillation" in result.triggered_rules


def test_branch_action_without_eval_gate_is_rejected() -> None:
    validator, _ = _validator(enabled_kinds=frozenset({ActionKind.BRANCH_EXPERIMENT}))
    proposal = ControllerOutput(
        action=BranchExperimentAction(parent_branch_id="b", config_overrides={}),
        confidence=0.9,
    )
    result = validator.validate(proposal, _state(), controller_name="test")
    assert not result.approved
    assert "check_branch_promotion_eval_gate" in result.triggered_rules


def test_branch_action_after_passed_eval_is_approved() -> None:
    validator, _ = _validator(enabled_kinds=frozenset({ActionKind.BRANCH_EXPERIMENT}))
    validator.record_eval_result(passed=True)
    proposal = ControllerOutput(
        action=BranchExperimentAction(parent_branch_id="b", config_overrides={}),
        confidence=0.9,
    )
    result = validator.validate(proposal, _state(), controller_name="test")
    assert result.approved
    assert result.requires_eval_gate


def test_major_lr_change_requires_checkpoint_first() -> None:
    validator, _ = _validator(
        enabled_kinds=frozenset({ActionKind.ADJUST_LEARNING_RATE}), initial_lr=1e-3
    )
    proposal = ControllerOutput(action=AdjustLearningRateAction(new_lr=1e-2), confidence=0.9)
    result = validator.validate(proposal, _state(), controller_name="test")
    assert result.approved
    assert result.requires_checkpoint_first


def test_early_stop_is_exempt_from_the_hard_budget_check() -> None:
    """Regression test: once a run is over budget, EarlyStopAction must still be
    approvable — it's the one action meant to honor the budget by ending the run;
    rejecting it would make the hard-budget guardrail block its own exit ramp."""
    tracker = BudgetTracker(total=BudgetUsage(steps=10))
    tracker.spend(BudgetUsage(steps=100))  # simulate already well over budget
    validator = ActionValidator(
        action_space=ActionSpace(enabled_kinds=frozenset({ActionKind.EARLY_STOP})),
        budget_tracker=tracker,
        audit_log=InMemoryAuditLog(),
        initial_lr=1e-3,
    )
    proposal = ControllerOutput(action=EarlyStopAction(reason="over budget"), confidence=1.0)
    result = validator.validate(proposal, _state(), controller_name="test")
    assert result.approved
    assert "check_hard_budget" not in result.triggered_rules


def test_restore_state_lets_validator_forget_a_reverted_action() -> None:
    """Regression test: after a rollback restores the model/optimizer to a prior
    checkpoint, the validator must forget the reverted action too — otherwise it keeps
    treating the undone value as "current" and rejects the controller's correct attempt
    to go back to the known-good value as a false "oscillation"."""
    validator, _ = _validator(enabled_kinds=frozenset({ActionKind.ADJUST_LEARNING_RATE}))
    baseline = validator.capture_state()

    up = ControllerOutput(action=AdjustLearningRateAction(new_lr=2e-3), confidence=0.9)
    result = validator.validate(up, _state(1), controller_name="test")
    assert result.approved
    validator.record_applied(result.action)

    down = ControllerOutput(action=AdjustLearningRateAction(new_lr=1e-3), confidence=0.9)
    still_thinks_it_oscillated = validator.validate(down, _state(2), controller_name="test")
    assert not still_thinks_it_oscillated.approved
    assert "check_oscillation" in still_thinks_it_oscillated.triggered_rules

    validator.restore_state(baseline)  # what `TrainerAdapter._rollback` does
    after_restore = validator.validate(down, _state(3), controller_name="test")
    assert after_restore.approved


def test_every_validate_call_writes_exactly_one_audit_entry() -> None:
    validator, audit_log = _validator(enabled_kinds=frozenset({ActionKind.NOOP}))
    approved_proposal = ControllerOutput(action=NoopAction(), confidence=1.0)
    rejected_proposal = ControllerOutput(
        action=AdjustLearningRateAction(new_lr=1e-2), confidence=0.9
    )

    validator.validate(approved_proposal, _state(1), controller_name="test")
    validator.validate(rejected_proposal, _state(2), controller_name="test")

    entries = audit_log.query_by_run("r")
    assert len(entries) == 2
    assert entries[0].result.approved
    assert not entries[1].result.approved
