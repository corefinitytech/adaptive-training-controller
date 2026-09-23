"""The `Action` discriminated union: every variant round-trips through JSON, and an
untyped/malformed payload is rejected rather than silently accepted."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from corefinity_adaptive.actions.types import (
    ACTION_VARIANTS,
    Action,
    ActionKind,
    AdjustLearningRateAction,
    NoopAction,
    ReweightDataAction,
)

_action_adapter: TypeAdapter[Action] = TypeAdapter(Action)


def test_every_variant_round_trips_through_json() -> None:
    examples: list[Action] = [
        NoopAction(),
        ReweightDataAction(group_weights={"easy": 0.3, "hard": 0.7}),
        AdjustLearningRateAction(new_lr=1e-3),
    ]
    for action in examples:
        payload = action.model_dump_json()
        restored = _action_adapter.validate_json(payload)
        assert restored == action


def test_discriminator_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        _action_adapter.validate_python({"kind": "not_a_real_action"})


def test_discriminator_rejects_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        _action_adapter.validate_python({"kind": ActionKind.ADJUST_LEARNING_RATE.value})


def test_all_action_kinds_have_a_variant() -> None:
    covered = {variant.model_fields["kind"].default for variant in ACTION_VARIANTS}
    assert covered == set(ActionKind)
