"""`ExperimentManager`: branch lifecycle bookkeeping and the PBT exploit/explore step."""

from __future__ import annotations

import pytest

from corefinity_adaptive.experiments.branch import BranchStatus
from corefinity_adaptive.experiments.manager import ExperimentManager, UnknownBranchError


def test_root_branch_exists_and_is_active_on_construction() -> None:
    manager = ExperimentManager(root_branch_id="main")
    root = manager.get("main")
    assert root.status is BranchStatus.ACTIVE
    assert root.parent_branch_id is None


def test_create_branch_registers_a_pending_branch() -> None:
    manager = ExperimentManager()
    branch = manager.create_branch(parent_branch_id="main", config_overrides={"lr": 1e-3})
    assert branch.status is BranchStatus.PENDING
    assert branch.parent_branch_id == "main"
    assert branch.config_overrides == {"lr": 1e-3}
    assert branch in manager.pending_branches()


def test_create_branch_with_unknown_parent_raises() -> None:
    manager = ExperimentManager()
    with pytest.raises(UnknownBranchError):
        manager.create_branch(parent_branch_id="does-not-exist", config_overrides={})


def test_terminate_branch_marks_terminated_with_reason() -> None:
    manager = ExperimentManager()
    branch = manager.create_branch(parent_branch_id="main", config_overrides={})
    updated = manager.terminate_branch(branch.branch_id, reason="underperforming")
    assert updated.status is BranchStatus.TERMINATED
    assert updated.terminated_reason == "underperforming"
    assert updated not in manager.pending_branches()


def test_terminate_unknown_branch_raises() -> None:
    manager = ExperimentManager()
    with pytest.raises(UnknownBranchError):
        manager.terminate_branch("does-not-exist", reason="x")


def test_allocate_compute_accumulates_on_the_branch() -> None:
    manager = ExperimentManager()
    branch = manager.create_branch(parent_branch_id="main", config_overrides={})
    manager.allocate_compute(branch.branch_id, additional_gpu_hours=2.0)
    updated = manager.allocate_compute(branch.branch_id, additional_gpu_hours=1.5)
    assert updated.budget.gpu_hours == 3.5


def test_activate_branch_transitions_pending_to_active() -> None:
    manager = ExperimentManager()
    branch = manager.create_branch(parent_branch_id="main", config_overrides={})
    updated = manager.activate_branch(branch.branch_id)
    assert updated.status is BranchStatus.ACTIVE
    assert updated in manager.active_branches()
    assert updated not in manager.pending_branches()


def test_exploit_and_explore_needs_at_least_two_active_branches() -> None:
    manager = ExperimentManager(root_branch_id="a")  # "a" is ACTIVE from construction
    new_branches = manager.exploit_and_explore({"a": 0.2}, bottom_fraction=0.5)
    assert new_branches == ()


def test_exploit_and_explore_ignores_scores_for_inactive_or_unknown_ids() -> None:
    manager = ExperimentManager(root_branch_id="a")
    pending = manager.create_branch(parent_branch_id="a", config_overrides={})
    # pending isn't ACTIVE yet, and "nonexistent" was never registered — neither should
    # count toward having "two active branches to compare".
    scores = {"a": 0.2, pending.branch_id: 0.9, "nonexistent": 0.5}
    new_branches = manager.exploit_and_explore(scores, bottom_fraction=0.5)
    assert new_branches == ()


def test_exploit_and_explore_terminates_bottom_and_promotes_top() -> None:
    manager = ExperimentManager(root_branch_id="a")
    branch_b = manager.create_branch(parent_branch_id="a", config_overrides={"learning_rate": 1e-3})
    manager.activate_branch(branch_b.branch_id)  # now "a" and branch_b are both ACTIVE

    new_branches = manager.exploit_and_explore(
        {"a": 0.9, branch_b.branch_id: 0.1}, bottom_fraction=0.5
    )

    assert len(new_branches) == 1
    promoted = new_branches[0]
    assert promoted.status is BranchStatus.PENDING
    assert promoted.parent_branch_id == "a"  # cloned from the top performer, not itself

    terminated = manager.get(branch_b.branch_id)
    assert terminated.status is BranchStatus.TERMINATED
    assert terminated.terminated_reason is not None


def test_exploit_and_explore_promoted_branch_inherits_top_performers_checkpoint() -> None:
    manager = ExperimentManager()  # root "main" stays ACTIVE but unscored below
    top = manager.create_branch(parent_branch_id="main", config_overrides={}, checkpoint={"w": 1})
    bottom = manager.create_branch(parent_branch_id="main", config_overrides={})
    manager.activate_branch(top.branch_id)
    manager.activate_branch(bottom.branch_id)

    (promoted,) = manager.exploit_and_explore(
        {top.branch_id: 0.9, bottom.branch_id: 0.1}, bottom_fraction=1.0
    )
    assert promoted.checkpoint == {"w": 1}
    assert promoted.parent_branch_id == top.branch_id
