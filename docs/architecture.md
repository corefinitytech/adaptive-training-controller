# Architecture

This is the map of how `corefinity_adaptive` fits together, phase by phase. For the
reasoning behind individual decisions, see the numbered ADRs in `docs/adr/`. For the
original product brief, see the research spec PDF at the repository root.

## The loop

```
Dataset / Model / Config
        │
        ▼
  Training Executor  ──▶  Metrics + Gradient Stats
        ▲                          │
        │                          ▼
        │                  State Builder / Memory
        │                          │
        │                          ▼
        │                  Decision Controller
        │                          │
        │                          ▼
        │                  Action Validator ──▶ (rejected → nothing happens, audit logged)
        │                          │ (approved)
        └──────────── Next Training Interval
```

At each interval boundary, `TrainerAdapter` (`trainer/adapter.py`):

1. Builds a `TrainingState` from the interval's metrics (`state/builder.py`).
2. If the controller needs history, queries the `ExperienceStore` for similar past states
   (`memory/sqlite_store.py`).
3. Asks the controller to `propose()` an `Action` (`controllers/`).
4. Passes the proposal to `ActionValidator.validate()` (`validation/validator.py`) — the
   **sole** gate between any controller and the training loop.
5. If approved, applies the action itself (the only code path that mutates trainer
   state), checkpointing first if the validator flagged it as major.
6. Appends an `ExperienceRecord` either way — approved or rejected — and logs an audit
   entry.
7. Once the *next* interval completes, computes the `ActionOutcome` for the previous
   action, checks it against the `RollbackPolicy`, and rolls back if it regressed.

## Safety boundary

No controller — including Jev — ever touches the model, optimizer, or dataset directly.
Every controller (`ControllerStrategy` in `controllers/base.py`) only ever returns a typed
`ControllerOutput` (an `Action` from the closed discriminated union in
`actions/types.py`, plus a confidence score that is informational only). The validator
never special-cases a controller by name or skips a check because confidence is high.

## Why these specific choices

See the ADRs:

- [0001 — Action taxonomy](adr/0001-action-taxonomy.md)
- [0002 — Experience store backend](adr/0002-experience-store-backend.md)
- [0003 — Trainer hook model](adr/0003-trainer-hook-model.md)
- [0004 — Jev integration boundary](adr/0004-jev-integration-boundary.md)

## Build phases

See `CHANGELOG.md` for what's shipped so far, and the project plan for the full phase
breakdown (Phase 0 research design → Phase 1 prototype → Phase 2 Jev integration →
Phase 3 evaluation → Phase 4 public library hardening → Phase 5 dashboard → Phase 6/7
research expansion and scaling). This repository is currently in **Phase 1**.
