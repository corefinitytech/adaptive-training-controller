"""The default `ExperienceStore`: local-first, file-backed SQLite.

Chosen over a server-backed store for Phase 1 because it's queryable with off-the-shelf
tools (`sqlite3` CLI, `pandas.read_sql`, DBeaver) with zero setup — directly serving the
research spec's requirement that the experience layer stay "inspectable" — and because it
fits a single researcher running experiments on their own machine (this project's
Mac-first target) without forcing a database server dependency on every user.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

from corefinity_adaptive.actions.outcome import ActionOutcome
from corefinity_adaptive.memory.query import ExperienceFilter, state_distance
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.memory.retention import RetentionPolicy
from corefinity_adaptive.state.models import TrainingState

_SIMILARITY_CANDIDATE_LIMIT = 2_000
"""Upper bound on rows pulled into Python before ranking by distance, so `query_similar`
stays cheap even as a run's history grows — the store-side half of the bound is
`RetentionPolicy`, this is the query-side half."""


class SQLiteExperienceStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experience (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                controller_name TEXT NOT NULL,
                assessment TEXT NOT NULL,
                executed INTEGER NOT NULL,
                comparison_group_id TEXT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        for column in ("run_id", "branch_id", "comparison_group_id", "created_at"):
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_experience_{column} ON experience({column})"
            )
        self._conn.commit()

    def append(self, record: ExperienceRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO experience
                (id, run_id, branch_id, step, controller_name, assessment,
                 executed, comparison_group_id, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.id),
                record.run_id,
                record.branch_id,
                record.step,
                record.controller_name,
                record.assessment.value,
                int(record.executed),
                record.comparison_group_id,
                record.created_at.isoformat(),
                record.model_dump_json(),
            ),
        )
        self._conn.commit()

    def _load(self, record_id: UUID) -> ExperienceRecord:
        row = self._conn.execute(
            "SELECT payload FROM experience WHERE id = ?", (str(record_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"no experience record with id {record_id}")
        return ExperienceRecord.model_validate_json(row["payload"])

    def update_outcome(self, record_id: UUID, outcome: ActionOutcome) -> None:
        updated = self._load(record_id).with_outcome(outcome)
        self._conn.execute(
            "UPDATE experience SET assessment = ?, payload = ? WHERE id = ?",
            (updated.assessment.value, updated.model_dump_json(), str(record_id)),
        )
        self._conn.commit()

    def query_by_run(self, run_id: str) -> list[ExperienceRecord]:
        rows = self._conn.execute(
            "SELECT payload FROM experience WHERE run_id = ? ORDER BY step ASC", (run_id,)
        ).fetchall()
        return [ExperienceRecord.model_validate_json(row["payload"]) for row in rows]

    def query_similar(
        self,
        state: TrainingState,
        *,
        k: int,
        filters: ExperienceFilter | None = None,
    ) -> list[ExperienceRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if filters is not None:
            if filters.run_id is not None:
                clauses.append("run_id = ?")
                params.append(filters.run_id)
            if filters.branch_id is not None:
                clauses.append("branch_id = ?")
                params.append(filters.branch_id)
            if filters.controller_name is not None:
                clauses.append("controller_name = ?")
                params.append(filters.controller_name)
            if filters.assessment is not None:
                clauses.append("assessment = ?")
                params.append(filters.assessment.value)
            if filters.comparison_group_id is not None:
                clauses.append("comparison_group_id = ?")
                params.append(filters.comparison_group_id)
            if filters.executed_only:
                clauses.append("executed = 1")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT payload FROM experience {where} ORDER BY created_at DESC LIMIT ?"
        rows = self._conn.execute(query, (*params, _SIMILARITY_CANDIDATE_LIMIT)).fetchall()
        candidates = [ExperienceRecord.model_validate_json(row["payload"]) for row in rows]
        candidates.sort(key=lambda record: state_distance(state, record.state))
        return candidates[:k]

    def prune(self, policy: RetentionPolicy) -> int:
        deleted = 0
        deleted += self._prune_per_branch(policy)
        deleted += self._prune_total(policy)
        self._conn.commit()
        return deleted

    def _prune_per_branch(self, policy: RetentionPolicy) -> int:
        deleted = 0
        branches = self._conn.execute("SELECT DISTINCT branch_id FROM experience").fetchall()
        for row in branches:
            branch_id = row["branch_id"]
            if branch_id in policy.protected_branch_ids:
                continue
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM experience WHERE branch_id = ?", (branch_id,)
            ).fetchone()
            overflow = count - policy.max_records_per_branch
            if overflow <= 0:
                continue
            ids = self._conn.execute(
                "SELECT id FROM experience WHERE branch_id = ? ORDER BY created_at ASC LIMIT ?",
                (branch_id, overflow),
            ).fetchall()
            deleted += self._delete_ids([r["id"] for r in ids])
        return deleted

    def _prune_total(self, policy: RetentionPolicy) -> int:
        (total,) = self._conn.execute("SELECT COUNT(*) FROM experience").fetchone()
        overflow = total - policy.max_total_records
        if overflow <= 0:
            return 0
        protected = policy.protected_branch_ids
        placeholders = ",".join("?" for _ in protected) if protected else None
        exclude = f"WHERE branch_id NOT IN ({placeholders})" if placeholders else ""
        ids = self._conn.execute(
            f"SELECT id FROM experience {exclude} ORDER BY created_at ASC LIMIT ?",
            (*protected, overflow),
        ).fetchall()
        return self._delete_ids([r["id"] for r in ids])

    def _delete_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(f"DELETE FROM experience WHERE id IN ({placeholders})", ids)
        return len(ids)

    def close(self) -> None:
        self._conn.close()
