"""`rank_for_exploit` and `perturb_learning_rate`: the pure PBT decision logic, kept
separate from `ExperimentManager`'s stateful branch bookkeeping.
"""

from __future__ import annotations

import random

from corefinity_adaptive.experiments.policy import perturb_learning_rate, rank_for_exploit


def test_rank_for_exploit_orders_best_first() -> None:
    ranked, _ = rank_for_exploit({"a": 0.5, "b": 0.9, "c": 0.1}, bottom_fraction=0.34)
    assert ranked == ["b", "a", "c"]


def test_rank_for_exploit_bottom_never_includes_the_top_performer() -> None:
    _, bottom = rank_for_exploit({"a": 0.5, "b": 0.9}, bottom_fraction=1.0)
    assert "b" not in bottom
    assert bottom == ["a"]


def test_rank_for_exploit_single_branch_has_nothing_to_exploit() -> None:
    ranked, bottom = rank_for_exploit({"a": 0.5}, bottom_fraction=0.5)
    assert ranked == ["a"]
    assert bottom == []


def test_rank_for_exploit_no_branches() -> None:
    ranked, bottom = rank_for_exploit({}, bottom_fraction=0.5)
    assert ranked == []
    assert bottom == []


def test_rank_for_exploit_rounds_bottom_fraction_to_at_least_one() -> None:
    _, bottom = rank_for_exploit({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, bottom_fraction=0.1)
    assert len(bottom) == 1


def test_perturb_learning_rate_scales_within_factor_range() -> None:
    rng = random.Random(0)
    result = perturb_learning_rate({"learning_rate": 1e-3}, factor_range=(0.5, 1.5), rng=rng)
    assert 0.5e-3 <= result["learning_rate"] <= 1.5e-3


def test_perturb_learning_rate_does_not_mutate_input() -> None:
    original = {"learning_rate": 1e-3}
    perturb_learning_rate(original, rng=random.Random(0))
    assert original == {"learning_rate": 1e-3}


def test_perturb_learning_rate_leaves_other_keys_untouched() -> None:
    result = perturb_learning_rate(
        {"learning_rate": 1e-3, "target_group": "hard"}, rng=random.Random(0)
    )
    assert result["target_group"] == "hard"


def test_perturb_learning_rate_no_op_when_key_missing() -> None:
    result = perturb_learning_rate({"other": 1}, rng=random.Random(0))
    assert result == {"other": 1}


def test_perturb_learning_rate_no_op_for_non_numeric_value() -> None:
    result = perturb_learning_rate({"learning_rate": "auto"}, rng=random.Random(0))
    assert result == {"learning_rate": "auto"}
