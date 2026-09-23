"""`TrainerAdapter.__init__` rejects a misconfigured `ActionSpace`/hook combination up
front, rather than letting the validator approve (and checkpoint!) a proposal that
`_apply_action` can only crash on later. Regression tests for that construction-time
guard — see `TrainerAdapter._validate_action_space`.
"""

from __future__ import annotations

import pytest
import torch

from corefinity_adaptive import ActionSpace, BudgetUsage
from corefinity_adaptive.actions.types import ActionKind
from corefinity_adaptive.controllers import Controller
from corefinity_adaptive.trainer.adapter import TrainerAdapter, UnsupportedActionError
from tests.conftest import TinyRegressionModel, mse_loss


def _build(
    *,
    enabled_kinds: frozenset[ActionKind],
    on_reweight: object = None,
    on_trigger_eval: object = None,
) -> None:
    model = TinyRegressionModel()
    TrainerAdapter(
        model=model,
        train_loader=[],
        optimizer=torch.optim.Adam(model.parameters()),
        compute_loss=mse_loss,
        controller=Controller.fixed(),
        budget=BudgetUsage(steps=10),
        action_space=ActionSpace(enabled_kinds=enabled_kinds),
        on_reweight=on_reweight,  # type: ignore[arg-type]
        on_trigger_eval=on_trigger_eval,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "kind",
    [ActionKind.BRANCH_EXPERIMENT, ActionKind.TERMINATE_BRANCH, ActionKind.ALLOCATE_COMPUTE],
)
def test_enabling_an_unimplemented_action_kind_is_rejected_at_construction(
    kind: ActionKind,
) -> None:
    with pytest.raises(UnsupportedActionError, match=kind.value):
        _build(enabled_kinds=frozenset({kind}))


def test_reweight_data_without_a_hook_is_rejected_at_construction() -> None:
    with pytest.raises(UnsupportedActionError, match="on_reweight"):
        _build(enabled_kinds=frozenset({ActionKind.REWEIGHT_DATA}))


def test_reweight_data_with_a_hook_is_accepted() -> None:
    _build(enabled_kinds=frozenset({ActionKind.REWEIGHT_DATA}), on_reweight=lambda weights: None)


def test_trigger_eval_without_a_hook_is_rejected_at_construction() -> None:
    with pytest.raises(UnsupportedActionError, match="on_trigger_eval"):
        _build(enabled_kinds=frozenset({ActionKind.TRIGGER_EVAL}))


def test_trigger_eval_with_a_hook_is_accepted() -> None:
    _build(enabled_kinds=frozenset({ActionKind.TRIGGER_EVAL}), on_trigger_eval=lambda: True)


def test_noop_only_action_space_needs_no_hooks() -> None:
    _build(enabled_kinds=frozenset({ActionKind.NOOP}))
