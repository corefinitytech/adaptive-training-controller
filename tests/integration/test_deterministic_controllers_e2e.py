"""`RuleBasedController`, `StatelessController`, and `HistoryAwareController` each run
end-to-end through the exact same `TrainerAdapter` harness as `FixedController` (see
`test_fixed_controller_e2e.py`) — the Phase 1 "done" criterion per the project plan.

`min_relative_improvement` is set high enough that a plateau is detected on virtually
every consultation regardless of real training dynamics (loss can't realistically
improve by 100%+ in a few steps), so the reweight path is deterministically exercised
without depending on — or being flaky because of — actual stochastic training behavior.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from corefinity_adaptive import ActionSpace, BudgetUsage, Controller, ControllerStrategy
from corefinity_adaptive.actions.types import ActionKind, ReweightDataAction
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.observability.audit_log import SQLiteAuditLog
from corefinity_adaptive.trainer import TrainerAdapter
from tests.conftest import TinyRegressionModel, mse_loss

_CONTROLLER_FACTORIES: dict[str, ControllerStrategy] = {
    "rule_based": Controller.rule_based(
        target_group="hard", plateau_window=1, min_relative_improvement=1.0
    ),
    "stateless": Controller.stateless(target_group="hard", min_relative_improvement=1.0),
    "history_aware": Controller.history_aware(target_group="hard", min_relative_improvement=1.0),
}


@pytest.mark.integration
@pytest.mark.parametrize("controller_name", list(_CONTROLLER_FACTORIES))
def test_deterministic_controller_runs_end_to_end(
    controller_name: str,
    tiny_model: TinyRegressionModel,
    tiny_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    tmp_sqlite_path: str,
) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    audit_log = SQLiteAuditLog(tmp_sqlite_path.replace("experience.sqlite", "audit_log.sqlite"))
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)
    reweight_calls: list[dict[str, float]] = []

    adapter = TrainerAdapter(
        model=tiny_model,
        train_loader=tiny_dataloader,
        optimizer=optimizer,
        compute_loss=mse_loss,
        controller=_CONTROLLER_FACTORIES[controller_name],
        budget=BudgetUsage(steps=100),
        action_space=ActionSpace(
            enabled_kinds=frozenset({ActionKind.NOOP, ActionKind.REWEIGHT_DATA}),
            max_reweight_delta=0.5,
        ),
        experience_store=store,
        audit_log=audit_log,
        run_id=f"{controller_name}-run",
        interval_steps=4,
        max_steps=16,
        seed=0,
        on_reweight=lambda weights: reweight_calls.append(dict(weights)),
    )

    result = adapter.fit()

    assert result.run_id == f"{controller_name}-run"
    assert not result.stopped_early

    records = store.query_by_run(f"{controller_name}-run")
    assert len(records) == 4  # one consultation per interval, 16 steps / 4 = 4
    assert all(r.controller_name == controller_name for r in records)
    assert all(r.validation_result.approved for r in records)

    reweight_records = [r for r in records if isinstance(r.proposed_action, ReweightDataAction)]
    assert reweight_records, "expected at least one reweight given the lenient plateau threshold"
    assert reweight_calls, "on_reweight should have been invoked for the approved reweight(s)"
