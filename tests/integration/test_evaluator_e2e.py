"""`LossEvalSuite` wired into `TrainerAdapter.on_trigger_eval`: a controller proposing
`TriggerEvalAction` should actually run the eval suite and feed its pass/fail into the
validator's eval gate — not just the hardcoded-lambda version already exercised by
`test_experiment_manager_e2e.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from corefinity_adaptive import ActionSpace, BudgetUsage
from corefinity_adaptive.actions.proposal import ControllerOutput
from corefinity_adaptive.actions.types import ActionKind, TriggerEvalAction
from corefinity_adaptive.device import resolve_device
from corefinity_adaptive.evaluation.suites import LossEvalSuite
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.observability.audit_log import InMemoryAuditLog
from corefinity_adaptive.state.models import TrainingState
from corefinity_adaptive.trainer import TrainerAdapter
from tests.conftest import TinyRegressionModel, mse_loss


class _AlwaysTriggerEvalController:
    """Proposes TRIGGER_EVAL every interval — enough to exercise the on_trigger_eval
    wiring without needing any other decision logic."""

    @property
    def name(self) -> str:
        return "eval_trigger"

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
        return ControllerOutput(action=TriggerEvalAction(), confidence=1.0)


@pytest.mark.integration
def test_loss_eval_suite_drives_the_trainer_eval_gate(
    tiny_model: TinyRegressionModel,
    tiny_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    tmp_sqlite_path: str,
) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    audit_log = InMemoryAuditLog()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)
    device = resolve_device()

    # A held-out set the model has zero chance of fitting well in a handful of steps,
    # and a threshold no real loss could clear: the eval genuinely runs and genuinely
    # fails, rather than trivially passing.
    eval_inputs = torch.randn(8, 4)
    eval_targets = torch.randn(8, 1) * 50
    eval_loader = DataLoader(TensorDataset(eval_inputs, eval_targets), batch_size=4)
    suite = LossEvalSuite(eval_loader=eval_loader, compute_loss=mse_loss, pass_threshold=1e-9)

    adapter = TrainerAdapter(
        model=tiny_model,
        train_loader=tiny_dataloader,
        optimizer=optimizer,
        compute_loss=mse_loss,
        controller=_AlwaysTriggerEvalController(),
        budget=BudgetUsage(steps=100),
        action_space=ActionSpace(enabled_kinds=frozenset({ActionKind.TRIGGER_EVAL})),
        experience_store=store,
        audit_log=audit_log,
        run_id="eval-suite-test",
        interval_steps=4,
        max_steps=4,
        seed=0,
        on_trigger_eval=lambda: suite.evaluate(tiny_model, device).passed,
    )

    result = adapter.fit()

    assert result.total_steps == 4
    records = store.query_by_run("eval-suite-test")
    assert len(records) == 1
    assert isinstance(records[0].proposed_action, TriggerEvalAction)
    assert records[0].validation_result.approved

    # The model must be back in train() mode after fit() despite the eval suite
    # switching it to eval() mid-run — the next training step (or a later fit() call)
    # needs train() mode, and this is a real correctness requirement, not a convenience.
    assert tiny_model.training is True
