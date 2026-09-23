"""The append-only decision audit trail.

Every call to `ActionValidator.validate()` — approved or rejected — writes exactly one
`AuditLogEntry` here. This is what makes the research spec's "complete audit log"
requirement (§13) a structural property of the validator rather than something a caller
has to remember to do separately.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from corefinity_adaptive.actions.types import Action
from corefinity_adaptive.validation.results import ValidationResult


class AuditLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    branch_id: str
    step: int
    controller_name: str
    controller_confidence: float | None
    proposed_action: Action
    result: ValidationResult


class AuditLog(Protocol):
    """Write-once, read-many decision log."""

    def record(self, entry: AuditLogEntry) -> None: ...

    def query_by_run(self, run_id: str) -> list[AuditLogEntry]: ...


class InMemoryAuditLog:
    """Process-local audit log. Fine for tests and short scripts; not for real runs."""

    def __init__(self) -> None:
        self._entries: list[AuditLogEntry] = []

    def record(self, entry: AuditLogEntry) -> None:
        self._entries.append(entry)

    def query_by_run(self, run_id: str) -> list[AuditLogEntry]:
        return [e for e in self._entries if e.run_id == run_id]


class SQLiteAuditLog:
    """Durable, file-backed audit log — the default for real runs.

    Each entry is stored as one JSON-serialized row, indexed by `run_id`/`step`. Uses the
    same "queryable with off-the-shelf tools" rationale as `SQLiteExperienceStore`.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                approved INTEGER NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_log(run_id)")
        self._conn.commit()

    def record(self, entry: AuditLogEntry) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (id, run_id, branch_id, step, timestamp, approved, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(entry.id),
                entry.run_id,
                entry.branch_id,
                entry.step,
                entry.timestamp.isoformat(),
                int(entry.result.approved),
                entry.model_dump_json(),
            ),
        )
        self._conn.commit()

    def query_by_run(self, run_id: str) -> list[AuditLogEntry]:
        rows = self._conn.execute(
            "SELECT payload FROM audit_log WHERE run_id = ? ORDER BY step ASC", (run_id,)
        ).fetchall()
        return [AuditLogEntry.model_validate_json(row[0]) for row in rows]

    def close(self) -> None:
        self._conn.close()
