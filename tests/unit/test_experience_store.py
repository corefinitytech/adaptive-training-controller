"""`SQLiteExperienceStore`: round-trip persistence, similarity query, and retention."""

from __future__ import annotations

from corefinity_adaptive.actions.outcome import ActionOutcome, Assessment
from corefinity_adaptive.actions.types import NoopAction
from corefinity_adaptive.memory.query import ExperienceFilter
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.memory.retention import RetentionPolicy
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot, TrainingState
from corefinity_adaptive.validation.results import ValidationResult


def _state(
    step: int, *, run_id: str = "r", branch_id: str = "b", loss: float = 1.0
) -> TrainingState:
    return TrainingState(
        run_id=run_id,
        branch_id=branch_id,
        step=step,
        wall_clock_seconds=float(step),
        current=MetricSnapshot(
            step=step,
            epoch=float(step),
            train_loss=loss,
            learning_rate=1e-3,
            wall_clock_seconds=float(step),
        ),
        budget_consumed=BudgetUsage(),
        budget_remaining=BudgetUsage(),
        seed=0,
        config_hash="hash",
    )


def _record(
    step: int, *, run_id: str = "r", branch_id: str = "b", loss: float = 1.0
) -> ExperienceRecord:
    action = NoopAction()
    return ExperienceRecord(
        run_id=run_id,
        branch_id=branch_id,
        step=step,
        seed=0,
        state=_state(step, run_id=run_id, branch_id=branch_id, loss=loss),
        proposed_action=action,
        validation_result=ValidationResult(approved=True, action=action),
        controller_name="fixed",
        controller_confidence=1.0,
        executed=True,
    )


def test_append_and_query_by_run_round_trips(tmp_sqlite_path: str) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    store.append(_record(1))
    store.append(_record(2))
    store.append(_record(1, run_id="other"))

    records = store.query_by_run("r")
    assert [r.step for r in records] == [1, 2]


def test_update_outcome_replaces_assessment(tmp_sqlite_path: str) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    record = _record(1)
    store.append(record)

    outcome = ActionOutcome(
        observed_at_step=2,
        metric_deltas={"train_loss": -0.1},
        additional_cost=BudgetUsage(),
        assessment=Assessment.POSITIVE,
    )
    store.update_outcome(record.id, outcome)

    (updated,) = store.query_by_run("r")
    assert updated.assessment == Assessment.POSITIVE
    assert updated.outcome is not None
    assert updated.outcome.metric_deltas == {"train_loss": -0.1}


def test_query_similar_ranks_by_metric_distance(tmp_sqlite_path: str) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    store.append(_record(1, loss=1.0))
    store.append(_record(2, loss=5.0))
    store.append(_record(3, loss=1.3))

    # loss=1.05 is unambiguously closer to record 1 (distance 0.05) than record 3
    # (distance 0.25) or record 2 (distance 3.95) — no tie to break.
    results = store.query_similar(_state(4, loss=1.05), k=2)
    assert [r.step for r in results] == [1, 3]


def test_query_similar_respects_branch_filter(tmp_sqlite_path: str) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    store.append(_record(1, branch_id="a", loss=1.0))
    store.append(_record(2, branch_id="b", loss=1.0))

    results = store.query_similar(_state(3, loss=1.0), k=5, filters=ExperienceFilter(branch_id="a"))
    assert [r.branch_id for r in results] == ["a"]


def test_prune_caps_records_per_branch(tmp_sqlite_path: str) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    for step in range(1, 6):
        store.append(_record(step))

    deleted = store.prune(RetentionPolicy(max_records_per_branch=3, max_total_records=100))

    assert deleted == 2
    remaining = store.query_by_run("r")
    assert len(remaining) == 3
    assert [r.step for r in remaining] == [3, 4, 5]


def test_prune_never_removes_protected_branch(tmp_sqlite_path: str) -> None:
    store = SQLiteExperienceStore(tmp_sqlite_path)
    for step in range(1, 6):
        store.append(_record(step, branch_id="active"))

    deleted = store.prune(
        RetentionPolicy(
            max_records_per_branch=1,
            max_total_records=100,
            protected_branch_ids=frozenset({"active"}),
        )
    )

    assert deleted == 0
    assert len(store.query_by_run("r")) == 5
