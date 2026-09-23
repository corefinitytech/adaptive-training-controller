# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `0.1.0`.

## [Unreleased]

### Added

- Project scaffold: packaging, lint/type-check/test tooling, CI.
- Core types: `TrainingState`, the `Action` discriminated union, `ControllerOutput`.
- `ActionValidator` guardrail engine with an append-only audit log.
- `SQLiteExperienceStore` for local, inspectable experience records.
- `TrainerAdapter` / `AdaptiveTrainer` event-driven training loop with accelerator
  resolution (`cpu`/`mps`/`cuda`).
- `FixedController` no-op baseline.
- `RuleBasedController`, `StatelessController`, `HistoryAwareController`: three
  deterministic controllers sharing one plateau-triggered reweighting rule
  (`controllers/policies.py`), differing only in where the plateau signal and
  precedent-checking come from — controller-owned memory, `TrainingState.recent_trend`,
  and the experience store, respectively. Reachable via `Controller.rule_based()`,
  `.stateless()`, `.history_aware()`.
- `ExperimentManager`: branch lineage/config/checkpoint/budget-allocation bookkeeping
  for `BranchExperimentAction`/`TerminateBranchAction`/`AllocateComputeAction`, plus a
  PBT-style exploit/explore step (`experiments/policy.py`: rank active branches by
  score, terminate the bottom fraction, clone the top performer's checkpoint/config
  into a new branch with a perturbed hyperparameter). Branches run *sequentially* in
  this build phase — see the module's docstring for the intended orchestration pattern.
  `TrainerAdapter` now executes these three action kinds (previously always rejected at
  construction) when an `experiment_manager` is supplied.

### Fixed

Found by a multi-pass code review before the first push; all confirmed against the
actual code, not just reported:

- Checkpointing now happens before *any* approved mutating action, not only ones the
  validator flagged as "major" — previously `ReweightDataAction` and minor LR changes
  were never checkpointed, so rollback silently no-op'd for the actions most likely to
  actually execute and regress.
- Rollback now restores `ActionValidator`'s LR/group-weight history alongside the
  model/optimizer (`ActionValidator.capture_state`/`restore_state`) — previously the
  validator kept treating a rolled-back value as "current" and rejected the
  controller's correct follow-up attempt to undo it as a false oscillation.
- `TrainerAdapter.__init__` now rejects an `ActionSpace` that enables
  `BRANCH_EXPERIMENT`/`TERMINATE_BRANCH`/`ALLOCATE_COMPUTE` (no execution path exists
  yet) or `REWEIGHT_DATA`/`TRIGGER_EVAL` without their required hook, instead of
  letting the validator approve and checkpoint a proposal that then crashes `fit()`
  with a permanently-misleading `executed=True` audit record.
- `ReweightDataAction`'s absolute new weight is now stored as-is in
  `TrainingState.data_exposure` (`DefaultStateBuilder.set_group_weight`), instead of
  being fed into a delta-accumulating method that made the reported weight grow
  unboundedly on repeated reweights.
- `check_hard_budget` now exempts `EarlyStopAction` — previously, once a run went over
  budget, the guardrail rejected *every* proposal including the one meant to end the
  run, defeating its own purpose.
- Documented (in `TrainerAdapter`'s docstring and a `fit()` code comment) that `seed`
  governs randomness during `fit()` only — it cannot retroactively seed a model's
  already-drawn initial weights or an already-constructed `DataLoader`'s shuffle order,
  since both happen before `TrainerAdapter` exists.
