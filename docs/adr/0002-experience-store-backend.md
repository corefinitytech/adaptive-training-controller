# 0002 — SQLite as the default experience store backend

## Status

Accepted.

## Context

The research spec requires the experience layer to be "inspectable" (§4) and warns
explicitly against "history explosion" (§12) — unbounded growth making decisions
expensive or noisy. The first-experiment target (§11, and this project's own scope
decision) is a single researcher running experiments on their own machine — this
project's own first target is a Mac with no GPU cluster, no database server.

## Decision

`SQLiteExperienceStore` (`memory/sqlite_store.py`) is the default and only
`ExperienceStore` implementation in this phase: one file-backed SQLite database, JSON
columns for the full record payload, indexed columns (`run_id`, `branch_id`,
`comparison_group_id`, `created_at`) for the query patterns the store actually serves.
`ExperienceStore` itself is a `Protocol` (`memory/store.py`), not a concrete base class.

`query_similar` uses a simple weighted-feature Euclidean distance over the current
`MetricSnapshot` (`memory/query.py`), computed in Python over a bounded SQL-side
candidate window (`_SIMILARITY_CANDIDATE_LIMIT`) — not a vector index.

## Consequences

- Zero-ops for a single researcher: `sqlite3`, `pandas.read_sql`, or any off-the-shelf
  SQLite browser can inspect a run's full history with no server to stand up.
- The store's growth is bounded from two directions: `RetentionPolicy`
  (`memory/retention.py`) caps what accumulates on disk, and the similarity query's
  candidate window caps what a single decision costs to make, independent of total
  history size.
- A Postgres- or vector-backed store can replace this later (Phase 6/7, at real scale)
  by implementing the same `Protocol` — no controller, validator, or trainer code
  changes, only the object passed to `TrainerAdapter(experience_store=...)`.
- We deliberately did *not* reach for a vector database now. Revisit only once
  `benchmarks/` shows the Python-side distance computation is an actual bottleneck at
  realistic history sizes — premature optimization here would cost a real dependency for
  a problem we don't have yet.
