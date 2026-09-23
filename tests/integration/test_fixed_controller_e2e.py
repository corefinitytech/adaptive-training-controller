"""The Phase 1 milestone test: a full run through `AdaptiveTrainer` with the `Fixed`
baseline, on a tiny synthetic model, asserting the whole safety/observability chain
actually produces the artifacts the research spec requires — experience records and
audit log entries for every interval, with no interventions (the fixed baseline never
proposes one).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from corefinity_adaptive import ActionSpace, BudgetUsage, Controller
from corefinity_adaptive.actions.types import ActionKind
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.observability.audit_log import SQLiteAuditLog
from corefinity_adaptive.trainer import TrainerAdapter
from tests.conftest import TinyRegressionModel, mse_loss


@pytest.mark.integration
def test_fixed_controller_runs_end_to_end(
    tiny_model: TinyRegressionModel,
    tiny_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    tmp_sqlite_path: str,
) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    audit_log = SQLiteAuditLog(tmp_sqlite_path.replace("experience.sqlite", "audit_log.sqlite"))
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)

    adapter = TrainerAdapter(
        model=tiny_model,
        train_loader=tiny_dataloader,
        optimizer=optimizer,
        compute_loss=mse_loss,
        controller=Controller.fixed(),
        budget=BudgetUsage(steps=100),
        action_space=ActionSpace(enabled_kinds=frozenset({ActionKind.NOOP})),
        experience_store=store,
        audit_log=audit_log,
        run_id="test-run",
        interval_steps=4,
        max_steps=24,
        seed=0,
    )

    result = adapter.fit()

    assert result.run_id == "test-run"
    assert result.total_steps == 24
    assert not result.stopped_early
    assert result.stop_reason is None
    assert result.budget_consumed.steps == 24

    records = store.query_by_run("test-run")
    # one interval consultation every 4 steps, over 24 steps => 6 intervals
    assert len(records) == 6
    assert all(record.controller_name == "fixed" for record in records)
    assert all(record.validation_result.approved for record in records)
    assert all(record.executed for record in records)

    audit_entries = audit_log.query_by_run("test-run")
    assert len(audit_entries) == len(records)
    assert all(entry.result.approved for entry in audit_entries)

    report = adapter.report()
    assert report.interventions_proposed == 6
    assert report.interventions_approved == 6
    assert report.interventions_rejected == 0
