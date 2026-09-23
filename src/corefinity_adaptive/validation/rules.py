"""Individual guardrail checks, composed by `ActionValidator`.

Each rejection rule is a pure function `(action, state, context) -> RuleViolation | None`
so it can be unit-tested in isolation from the rest of the validator (see
`tests/unit/test_action_validator.py`). Two small non-rejecting helpers
(`requires_checkpoint`, `requires_eval_gate`) compute the flags `ActionValidator` attaches
to an otherwise-approved result.

Each rule traces back to either an explicit guardrail in the research spec (§13) or a
named failure mode (§12) — see the docstring on each function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import (
    Action,
    ActionKind,
    AdjustLearningRateAction,
    AllocateComputeAction,
    EarlyStopAction,
    ReweightDataAction,
)
from corefinity_adaptive.state.budget import BudgetTracker
from corefinity_adaptive.state.models import BudgetUsage, TrainingState
from corefinity_adaptive.validation.results import RuleViolation

_EVAL_GATED_KINDS = frozenset({ActionKind.BRANCH_EXPERIMENT, ActionKind.ALLOCATE_COMPUTE})
_CHECKPOINT_REQUIRED_KINDS = frozenset(
    {ActionKind.BRANCH_EXPERIMENT, ActionKind.TERMINATE_BRANCH, ActionKind.ALLOCATE_COMPUTE}
)
_LR_MAJOR_CHANGE_FRACTION = 0.5
"""A learning-rate change of >= 50% relative to the current value is "major"."""


@dataclass(frozen=True)
class RuleContext:
    """Everything a rule needs beyond the proposed action and current state."""

    action_space: ActionSpace
    budget_tracker: BudgetTracker
    lr_history: tuple[float, ...] = ()
    """Chronological actual learning rates, most recent last (includes the current one)."""
    group_weight_history: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    """Per data-group, chronological actual weights, most recent last."""
    latest_eval_passed: bool | None = None
    """`None` if no evaluation has run yet this branch."""


def estimate_cost(action: Action) -> BudgetUsage:
    """Budget an action directly consumes beyond ordinary per-step training cost.

    Only actions that explicitly request additional compute (`AllocateComputeAction`)
    have a nonzero estimate here; branch creation's ongoing cost is tracked by the
    `ExperimentManager` once it exists, not estimated at proposal time.
    """
    if isinstance(action, AllocateComputeAction):
        return BudgetUsage(gpu_hours=action.additional_gpu_hours)
    return BudgetUsage()


def requires_checkpoint(action: Action, context: RuleContext) -> bool:
    """Whether the trainer must checkpoint before applying this action, once approved."""
    if action.kind in _CHECKPOINT_REQUIRED_KINDS:
        return True
    if isinstance(action, AdjustLearningRateAction) and context.lr_history:
        current = context.lr_history[-1]
        if current > 0 and abs(action.new_lr - current) / current >= _LR_MAJOR_CHANGE_FRACTION:
            return True
    return False


def requires_eval_gate(action: Action) -> bool:
    return action.kind in _EVAL_GATED_KINDS


def check_allowed_action_kind(
    action: Action, state: TrainingState, context: RuleContext
) -> RuleViolation | None:
    """§13 "explicit allowed-action list": reject any kind the run didn't enable."""
    if context.action_space.allows(action.kind):
        return None
    return RuleViolation(
        rule_name="check_allowed_action_kind",
        message=f"{action.kind} is not in this run's enabled action kinds",
    )


def check_lr_bounds(
    action: Action, state: TrainingState, context: RuleContext
) -> RuleViolation | None:
    """§13 "minimum/maximum learning-rate bounds"."""
    if not isinstance(action, AdjustLearningRateAction):
        return None
    space = context.action_space
    if space.min_lr <= action.new_lr <= space.max_lr:
        return None
    return RuleViolation(
        rule_name="check_lr_bounds",
        message=f"new_lr={action.new_lr} outside [{space.min_lr}, {space.max_lr}]",
    )


def check_max_reweight_delta(
    action: Action, state: TrainingState, context: RuleContext
) -> RuleViolation | None:
    """§13 "maximum dataset reweighting".

    A group with no prior recorded weight has nothing to compare against, so its first
    assignment is not treated as a "change" — only a group being *moved* is bounded.
    """
    if not isinstance(action, ReweightDataAction):
        return None
    max_delta = context.action_space.max_reweight_delta
    for group, new_weight in action.group_weights.items():
        history = context.group_weight_history.get(group, ())
        if not history:
            continue
        delta = abs(new_weight - history[-1])
        if delta > max_delta:
            return RuleViolation(
                rule_name="check_max_reweight_delta",
                message=(
                    f"group {group!r} weight would change by {delta:.4f}, "
                    f"exceeding max_reweight_delta={max_delta}"
                ),
            )
    return None


def check_hard_budget(
    action: Action, state: TrainingState, context: RuleContext
) -> RuleViolation | None:
    """§13 "hard compute budget".

    `EarlyStopAction` is exempt: it costs nothing to execute, and rejecting it once a
    run is already over budget would block the one action meant to honor that budget —
    the guardrail would otherwise keep the run going past its own hard cap forever.
    """
    if isinstance(action, EarlyStopAction):
        return None
    cost = estimate_cost(action)
    if not context.budget_tracker.would_exceed(cost):
        return None
    return RuleViolation(
        rule_name="check_hard_budget",
        message=f"estimated cost {cost} would exceed the remaining budget",
    )


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _direction_reversal(
    history: tuple[float, ...], proposed: float, label: str
) -> RuleViolation | None:
    if len(history) < 2:
        return None
    prior_diff = _sign(history[-1] - history[-2])
    proposed_diff = _sign(proposed - history[-1])
    if prior_diff != 0 and proposed_diff != 0 and prior_diff != proposed_diff:
        return RuleViolation(
            rule_name="check_oscillation",
            message=f"{label} would reverse direction immediately after the previous change",
        )
    return None


def check_oscillation(
    action: Action, state: TrainingState, context: RuleContext
) -> RuleViolation | None:
    """Mitigates the "oscillation" failure mode (§12): repeatedly flipping a parameter.

    Rejects a proposal that would immediately reverse the direction of the previous
    change to the same control dimension (learning rate, or a specific data group's
    weight).
    """
    if isinstance(action, AdjustLearningRateAction):
        return _direction_reversal(context.lr_history, action.new_lr, "learning rate")
    if isinstance(action, ReweightDataAction):
        for group, new_weight in action.group_weights.items():
            history = context.group_weight_history.get(group, ())
            violation = _direction_reversal(history, new_weight, f"data group {group!r} weight")
            if violation is not None:
                return violation
    return None


def check_branch_promotion_eval_gate(
    action: Action, state: TrainingState, context: RuleContext
) -> RuleViolation | None:
    """§13 "evaluation gate before branch promotion"."""
    if not requires_eval_gate(action):
        return None
    if context.latest_eval_passed is True:
        return None
    return RuleViolation(
        rule_name="check_branch_promotion_eval_gate",
        message=f"{action.kind} requires a passed evaluation gate on this branch first",
    )


RuleCheck = Callable[[Action, TrainingState, RuleContext], RuleViolation | None]

DEFAULT_RULES: tuple[RuleCheck, ...] = (
    check_allowed_action_kind,
    check_lr_bounds,
    check_max_reweight_delta,
    check_hard_budget,
    check_oscillation,
    check_branch_promotion_eval_gate,
)
