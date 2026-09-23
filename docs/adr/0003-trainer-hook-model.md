# 0003 — Composer-style Event+State, not HuggingFace-style TrainerControl

## Status

Accepted.

## Context

Two well-established precedents exist for hooking into a training loop:

- **HuggingFace `TrainerCallback`**: callbacks are read-only except for a shared
  `TrainerControl` object with a handful of boolean flags
  (`should_save`/`should_evaluate`/`should_training_stop`/...) that callbacks may flip.
- **MosaicML Composer**: a two-way callback system split into read-only `Callback`s
  (logging, introspection) and state-mutating `Algorithm`s, each keyed to specific
  training `Event`s, operating on a shared mutable `State`.

CoreFinity's controller doesn't set a few flags — it returns an arbitrary typed `Action`
from a taxonomy that's expected to grow (data reweighting, LR changes, branching,
compute allocation, ...). A HF-style `TrainerControl` object would need either a field
per `ActionKind` (duplicating the `Action` union in a second place) or an untyped
catch-all bag — which reintroduces exactly the free-form-output problem ADR 0001 rules
out.

## Decision

`TrainerAdapter` follows the Composer split: `TrainerCallback`s
(`trainer/events.py::TrainerCallback`) are strictly read-only, observing `TrainingEvent`s
(`INTERVAL_START`, `BATCH_END`, `INTERVAL_END`, `EVAL_END`, `CHECKPOINT_SAVED`,
`BRANCH_CREATED`, `RUN_END`). The *only* code path that mutates trainer state
(optimizer LR, sampler weights, ...) is `TrainerAdapter._apply_action`, called exclusively
on an `ActionValidator`-approved `Action`.

## Consequences

- There is exactly one place in the codebase that mutates trainer state from a
  controller's decision — makes the audit log's "what changed and why" traceable by
  construction, not by convention.
- New `Action` variants require adding a branch to `_apply_action`; nothing about the
  event/callback system needs to change to support them.
- Callbacks added later (metrics export, MLflow sink, a future dashboard's live feed)
  are guaranteed side-effect-free with respect to training correctness, by the type
  system (`TrainerCallback.on_event` returns `None` and receives an immutable
  `TrainingState`).
