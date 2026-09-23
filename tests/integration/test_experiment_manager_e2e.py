"""`BranchExperimentAction`, `AllocateComputeAction`, and `TerminateBranchAction` running
through a real `TrainerAdapter.fit()` call, delegating to a real `ExperimentManager` —
the piece that used to unconditionally raise `UnsupportedActionError` before this round.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from corefinity_adaptive import ActionSpace, BudgetUsage
from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.types import (
    ActionKind,
    AllocateComputeAction,
    BranchExperimentAction,
    TerminateBranchAction,
    TriggerEvalAction,
)
from corefinity_adaptive.experiments.branch import BranchStatus
from corefinity_adaptive.experiments.manager import ExperimentManager
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.observability.audit_log import InMemoryAuditLog
from corefinity_adaptive.state.models import TrainingState
from corefinity_adaptive.trainer import TrainerAdapter
from tests.conftest import TinyRegressionModel, mse_loss

_ROOT_BRANCH_ID = "main"


class _ScriptedExperimentController:
    """Triggers an eval (to clear the eval gate), branches, self-allocates more
    compute, then terminates its own branch — in that order, one action per interval.
    """

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "scripted_experiment"

    @property
    def requires_history(self) -> bool:
        return False

    def propose(
        self,
        *,
        state: TrainingState,
        history: tuple[ExperienceRecord, ...],
        action_space: ActionSpace,
        budget_remaining: BudgetUsage,
    ) -> ControllerOutput:
        self.calls += 1
        if self.calls == 1:
            return ControllerOutput(action=TriggerEvalAction(), confidence=1.0)
        if self.calls == 2:
            action = BranchExperimentAction(
                parent_branch_id=_ROOT_BRANCH_ID, config_overrides={"learning_rate": 5e-3}
            )
            return ControllerOutput(action=action, confidence=0.9)
        if self.calls == 3:
            action = AllocateComputeAction(branch_id=_ROOT_BRANCH_ID, additional_gpu_hours=0.5)
            return ControllerOutput(action=action, confidence=0.9)
        action = TerminateBranchAction(branch_id=_ROOT_BRANCH_ID, reason="scripted end")
        return ControllerOutput(action=action, confidence=1.0)


@pytest.mark.integration
def test_experiment_actions_run_end_to_end(
    tiny_model: TinyRegressionModel,
    tiny_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    tmp_sqlite_path: str,
) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    audit_log = InMemoryAuditLog()
    experiment_manager = ExperimentManager(root_branch_id=_ROOT_BRANCH_ID)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)

    adapter = TrainerAdapter(
        model=tiny_model,
        train_loader=tiny_dataloader,
        optimizer=optimizer,
        compute_loss=mse_loss,
        controller=_ScriptedExperimentController(),
        budget=BudgetUsage(steps=100, gpu_hours=10.0),
        action_space=ActionSpace(
            enabled_kinds=frozenset(
                {
                    ActionKind.TRIGGER_EVAL,
                    ActionKind.BRANCH_EXPERIMENT,
                    ActionKind.ALLOCATE_COMPUTE,
                    ActionKind.TERMINATE_BRANCH,
                }
            )
        ),
        experience_store=store,
        audit_log=audit_log,
        run_id="experiment-test",
        branch_id=_ROOT_BRANCH_ID,
        interval_steps=4,
        max_steps=100,  # the scripted controller terminates itself at interval 4
        seed=0,
        on_trigger_eval=lambda: True,
        experiment_manager=experiment_manager,
    )

    result = adapter.fit()

    assert result.stopped_early
    assert result.stop_reason == "scripted end"

    records = store.query_by_run("experiment-test")
    assert [type(r.proposed_action).__name__ for r in records] == [
        "TriggerEvalAction",
        "BranchExperimentAction",
        "AllocateComputeAction",
        "TerminateBranchAction",
    ]
    assert all(r.validation_result.approved for r in records)

    # ExperimentManager actually recorded the branch/allocate/terminate side effects.
    new_branches = [b for b in experiment_manager.all_branches() if b.branch_id != _ROOT_BRANCH_ID]
    assert len(new_branches) == 1
    assert new_branches[0].status is BranchStatus.PENDING
    assert new_branches[0].parent_branch_id == _ROOT_BRANCH_ID
    assert new_branches[0].config_overrides == {"learning_rate": 5e-3}
    # A checkpoint was taken before this (non-Noop, approved) action, per the earlier fix.
    assert new_branches[0].checkpoint is not None

    root = experiment_manager.get(_ROOT_BRANCH_ID)
    assert root.budget.gpu_hours == pytest.approx(0.5)
    assert root.status is BranchStatus.TERMINATED
    assert root.terminated_reason == "scripted end"
