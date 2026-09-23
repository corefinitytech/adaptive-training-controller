"""The typed action taxonomy.

Every intervention a controller can propose is one of the variants below, combined into
`Action` — a Pydantic discriminated union. This is a structural guarantee, not a
convention: there is no string or free-form-dict path from a controller into the trainer.
A controller that wants to do something outside this taxonomy cannot express it, and the
`ActionValidator` never has to guess what an untyped payload "means" before deciding
whether it's safe.

Adding a new kind of intervention means adding a variant here *and* a guardrail rule in
`validation/rules.py` before any controller may propose it (see CONTRIBUTING.md).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionKind(StrEnum):
    NOOP = "noop"
    REWEIGHT_DATA = "reweight_data"
    ADJUST_LEARNING_RATE = "adjust_learning_rate"
    TRIGGER_EVAL = "trigger_eval"
    EARLY_STOP = "early_stop"
    BRANCH_EXPERIMENT = "branch_experiment"
    TERMINATE_BRANCH = "terminate_branch"
    ALLOCATE_COMPUTE = "allocate_compute"


class _ActionBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class NoopAction(_ActionBase):
    """Do nothing this interval. Returned by `FixedController` and as a safe fallback."""

    kind: Literal[ActionKind.NOOP] = ActionKind.NOOP


class ReweightDataAction(_ActionBase):
    """Change the relative sampling weight of one or more named data groups.

    `group_weights` gives the *new* weight for each named group; groups not mentioned are
    left unchanged. Validated against `ActionSpace.max_reweight_delta` — see
    `validation/rules.py::check_max_reweight_delta`.
    """

    kind: Literal[ActionKind.REWEIGHT_DATA] = ActionKind.REWEIGHT_DATA
    group_weights: dict[str, float]


class AdjustLearningRateAction(_ActionBase):
    """Set a new learning rate, bounded by `ActionSpace.min_lr`/`max_lr`."""

    kind: Literal[ActionKind.ADJUST_LEARNING_RATE] = ActionKind.ADJUST_LEARNING_RATE
    new_lr: float = Field(gt=0)


class TriggerEvalAction(_ActionBase):
    """Run the configured evaluation suite out of the regular schedule."""

    kind: Literal[ActionKind.TRIGGER_EVAL] = ActionKind.TRIGGER_EVAL
    suite_name: str | None = None


class EarlyStopAction(_ActionBase):
    """Stop the current branch now."""

    kind: Literal[ActionKind.EARLY_STOP] = ActionKind.EARLY_STOP
    reason: str


class BranchExperimentAction(_ActionBase):
    """Fork the current branch into a new one with config overrides.

    Requires a passed evaluation gate — see
    `validation/rules.py::check_branch_promotion_eval_gate`.
    """

    kind: Literal[ActionKind.BRANCH_EXPERIMENT] = ActionKind.BRANCH_EXPERIMENT
    parent_branch_id: str
    config_overrides: dict[str, float | int | str | bool]


class TerminateBranchAction(_ActionBase):
    """Stop a branch and return its remaining budget to the pool."""

    kind: Literal[ActionKind.TERMINATE_BRANCH] = ActionKind.TERMINATE_BRANCH
    branch_id: str
    reason: str


class AllocateComputeAction(_ActionBase):
    """Shift budget allocation from one branch to another."""

    kind: Literal[ActionKind.ALLOCATE_COMPUTE] = ActionKind.ALLOCATE_COMPUTE
    branch_id: str
    additional_gpu_hours: float = Field(ge=0)


Action = Annotated[
    NoopAction
    | ReweightDataAction
    | AdjustLearningRateAction
    | TriggerEvalAction
    | EarlyStopAction
    | BranchExperimentAction
    | TerminateBranchAction
    | AllocateComputeAction,
    Field(discriminator="kind"),
]

ACTION_VARIANTS: tuple[type[_ActionBase], ...] = (
    NoopAction,
    ReweightDataAction,
    AdjustLearningRateAction,
    TriggerEvalAction,
    EarlyStopAction,
    BranchExperimentAction,
    TerminateBranchAction,
    AllocateComputeAction,
)
