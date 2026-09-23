"""`TrainerAdapter` (internal engine) and `AdaptiveTrainer` (public façade).

`TrainerAdapter` wraps a standard PyTorch training loop and, at each interval boundary:
builds state, consults the controller, validates the proposal, and — only if approved —
is the one place that applies the resulting mutation. `AdaptiveTrainer` is a thin façade
matching the research spec's illustrative public API
(`AdaptiveTrainer(model=..., dataset=..., controller=..., budget=...)`, `.fit()`,
`.report()`) so that surface stays stable while `TrainerAdapter`'s internals evolve.
"""

from __future__ import annotations

import copy
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict

from corefinity_adaptive.actions.outcome import ActionOutcome, Assessment
from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import (
    Action,
    ActionKind,
    AdjustLearningRateAction,
    AllocateComputeAction,
    BranchExperimentAction,
    EarlyStopAction,
    NoopAction,
    ReweightDataAction,
    TerminateBranchAction,
    TriggerEvalAction,
)
from corefinity_adaptive.controllers.base import ControllerStrategy
from corefinity_adaptive.device import resolve_device
from corefinity_adaptive.experiments.manager import ExperimentManager
from corefinity_adaptive.memory.query import ExperienceFilter
from corefinity_adaptive.memory.records import ExperienceRecord
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.memory.store import ExperienceStore
from corefinity_adaptive.observability.audit_log import AuditLog, SQLiteAuditLog
from corefinity_adaptive.observability.logging_config import get_logger
from corefinity_adaptive.state.budget import BudgetTracker
from corefinity_adaptive.state.builder import DefaultStateBuilder
from corefinity_adaptive.state.models import BudgetUsage, MetricSnapshot, TrainingState
from corefinity_adaptive.trainer.events import TrainerCallback, TrainingEvent
from corefinity_adaptive.trainer.loop import (
    compute_config_hash,
    compute_grad_norm,
    compute_outcome,
    default_assess,
)
from corefinity_adaptive.validation.results import ValidationResult
from corefinity_adaptive.validation.rollback import RollbackPolicy
from corefinity_adaptive.validation.validator import ActionValidator, ValidatorStateSnapshot

_logger = get_logger("trainer")

_EXPERIMENT_MANAGER_REQUIRED_KINDS = frozenset(
    {ActionKind.BRANCH_EXPERIMENT, ActionKind.TERMINATE_BRANCH, ActionKind.ALLOCATE_COMPUTE}
)
"""Kinds `_apply_action` can only execute by delegating to an `ExperimentManager`.
Enabling one of these without supplying `experiment_manager` is rejected at construction
time (see `TrainerAdapter.__init__`) rather than left to crash `fit()` mid-run after a
proposal for one of them has already been approved and checkpointed."""


class UnsupportedActionError(RuntimeError):
    """Raised when an `ActionSpace`/hook configuration enables a kind that
    `TrainerAdapter` cannot execute.

    `__init__` rejects such a configuration up front (see `_validate_action_space`), so
    an approved proposal should never actually reach `_apply_action`'s own raises for
    these cases — those remain as defense-in-depth, not the primary guard, since a
    construction-time check can't prove exhaustiveness the way the type system would.
    """


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    branch_id: str
    total_steps: int
    final_train_loss: float
    stopped_early: bool
    stop_reason: str | None
    budget_consumed: BudgetUsage


class RunReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    branch_id: str
    controller_name: str
    total_steps: int
    interventions_proposed: int
    interventions_approved: int
    interventions_rejected: int


class TrainerAdapter:
    """Wraps a PyTorch training loop with closed-loop adaptive control.

    Owns the model/optimizer/data loop and every safety-critical dependency (validator,
    experience store, audit log). A fresh instance is scoped to one (run, branch).

    `seed` governs reproducibility of what happens *inside* `fit()` — it does not (and
    structurally cannot) retroactively seed `model`'s initial weights or `train_loader`'s
    shuffle order, since both are already constructed by the caller before this object
    exists. See the note in `fit()`.

    `experiment_manager`, if supplied, must have been constructed with
    `root_branch_id=branch_id` — `_apply_action` passes `TerminateBranchAction`'s and
    `AllocateComputeAction`'s `branch_id` straight through to it, so a mismatch surfaces
    as `UnknownBranchError` the first time this branch tries to act on itself. Branches
    run sequentially in this build phase: creating one here only registers its metadata
    as `PENDING` (see `ExperimentManager`'s docstring) — it does not start training it.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader: Iterable[Any],
        optimizer: torch.optim.Optimizer,
        compute_loss: Callable[[torch.nn.Module, Any, torch.device], torch.Tensor],
        controller: ControllerStrategy,
        budget: BudgetUsage,
        action_space: ActionSpace | None = None,
        experience_store: ExperienceStore | None = None,
        audit_log: AuditLog | None = None,
        run_id: str | None = None,
        branch_id: str = "main",
        seed: int = 0,
        interval_steps: int = 50,
        max_steps: int | None = None,
        device: torch.device | str | None = None,
        history_k: int = 5,
        callbacks: Sequence[TrainerCallback] = (),
        on_reweight: Callable[[Mapping[str, float]], None] | None = None,
        on_trigger_eval: Callable[[], bool] | None = None,
        assess_fn: Callable[[Mapping[str, float]], Assessment] | None = None,
        comparison_group_id: str | None = None,
        rollback_policy: RollbackPolicy | None = None,
        experiment_manager: ExperimentManager | None = None,
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._optimizer = optimizer
        self._compute_loss = compute_loss
        self._controller = controller
        self._action_space = action_space or ActionSpace()
        self._validate_action_space(
            self._action_space,
            on_reweight=on_reweight,
            on_trigger_eval=on_trigger_eval,
            experiment_manager=experiment_manager,
        )
        self._experiment_manager = experiment_manager
        self._run_id = run_id or str(uuid4())
        self._branch_id = branch_id
        self._seed = seed
        self._interval_steps = interval_steps
        self._max_steps = max_steps
        self._device = resolve_device(str(device) if device is not None else None)
        self._history_k = history_k
        self._callbacks = tuple(callbacks)
        self._on_reweight = on_reweight
        self._on_trigger_eval = on_trigger_eval
        self._assess_fn = assess_fn
        self._comparison_group_id = comparison_group_id

        self._experience_store = experience_store or SQLiteExperienceStore(
            Path(tempfile.mkdtemp(prefix="corefinity_")) / "experience.sqlite"
        )
        self._audit_log = audit_log or SQLiteAuditLog(
            Path(tempfile.mkdtemp(prefix="corefinity_audit_")) / "audit_log.sqlite"
        )

        config_hash = compute_config_hash(
            interval_steps=interval_steps,
            max_steps=max_steps,
            seed=seed,
            controller=controller.name,
            action_space=sorted(k.value for k in self._action_space.enabled_kinds),
            min_lr=self._action_space.min_lr,
            max_lr=self._action_space.max_lr,
            max_reweight_delta=self._action_space.max_reweight_delta,
        )
        self._budget_tracker = BudgetTracker(total=budget)
        self._state_builder = DefaultStateBuilder(
            run_id=self._run_id,
            branch_id=self._branch_id,
            seed=seed,
            config_hash=config_hash,
            budget_tracker=self._budget_tracker,
        )
        self._validator = ActionValidator(
            action_space=self._action_space,
            budget_tracker=self._budget_tracker,
            audit_log=self._audit_log,
            initial_lr=_current_lr(optimizer),
            rollback_policy=rollback_policy,
        )
        self._checkpoint: dict[str, Any] | None = None
        self._validator_checkpoint: ValidatorStateSnapshot | None = None

    @staticmethod
    def _validate_action_space(
        action_space: ActionSpace,
        *,
        on_reweight: Callable[[Mapping[str, float]], None] | None,
        on_trigger_eval: Callable[[], bool] | None,
        experiment_manager: ExperimentManager | None,
    ) -> None:
        """Reject at construction time any enabled kind `_apply_action` can't execute.

        Without this, the validator can fully approve (and checkpoint!) a proposal that
        `_apply_action` then has no choice but to crash on — after the checkpoint's cost
        was already paid and the "approved" `ExperienceRecord` already persisted. See
        `UnsupportedActionError`.
        """
        needs_experiment_manager = action_space.enabled_kinds & _EXPERIMENT_MANAGER_REQUIRED_KINDS
        if needs_experiment_manager and experiment_manager is None:
            raise UnsupportedActionError(
                f"action kind(s) {sorted(k.value for k in needs_experiment_manager)} are "
                "enabled but no experiment_manager was provided to TrainerAdapter"
            )
        if ActionKind.REWEIGHT_DATA in action_space.enabled_kinds and on_reweight is None:
            raise UnsupportedActionError(
                "ActionKind.REWEIGHT_DATA is enabled but no on_reweight hook was "
                "provided to TrainerAdapter"
            )
        if ActionKind.TRIGGER_EVAL in action_space.enabled_kinds and on_trigger_eval is None:
            raise UnsupportedActionError(
                "ActionKind.TRIGGER_EVAL is enabled but no on_trigger_eval hook was "
                "provided to TrainerAdapter"
            )

    # -- public API -----------------------------------------------------------------

    def fit(self) -> RunResult:
        # NOTE on reproducibility: this seeds RNG state used *during* fit() (e.g. any
        # stochastic ops in compute_loss). It does NOT retroactively seed the model's
        # initial weights or an already-constructed DataLoader's shuffle order — both
        # are already resolved by the time `model`/`train_loader` reach this
        # constructor. For full reproducibility, seed the RNG yourself (and construct a
        # DataLoader with an explicit `generator=`) before building the model/loader.
        torch.manual_seed(self._seed)
        self._model.to(self._device)
        self._model.train()

        step = 0
        wall_clock_total = 0.0
        stopped_early = False
        stop_reason: str | None = None
        last_train_loss = float("nan")
        pending_record_id = None
        pending_pre_snapshot: MetricSnapshot | None = None

        for batch in self._iterate_batches():
            step += 1
            start = time.monotonic()
            loss_value = self._train_step(batch)
            elapsed = time.monotonic() - start
            wall_clock_total += elapsed
            last_train_loss = loss_value

            snapshot = MetricSnapshot(
                step=step,
                epoch=step / max(1, self._steps_per_epoch()),
                train_loss=loss_value,
                learning_rate=_current_lr(self._optimizer),
                grad_norm=compute_grad_norm(self._model.parameters()),
                wall_clock_seconds=wall_clock_total,
            )
            self._state_builder.record_budget_spend(
                BudgetUsage(steps=1, wall_clock_seconds=elapsed)
            )
            state = self._state_builder.observe(snapshot)
            self._fire(TrainingEvent.BATCH_END, state)

            if step % self._interval_steps != 0:
                continue

            self._fire(TrainingEvent.INTERVAL_START, state)

            if pending_record_id is not None and pending_pre_snapshot is not None:
                outcome = compute_outcome(
                    pending_pre_snapshot,
                    snapshot,
                    assess_fn=self._assess_fn or default_assess,
                )
                self._experience_store.update_outcome(pending_record_id, outcome)
                regression = self._validator.evaluate_rollback(outcome)
                if regression is not None:
                    _logger.warning(
                        "rollback triggered on run=%s branch=%s step=%d: %s",
                        self._run_id,
                        self._branch_id,
                        step,
                        regression.metric_name,
                    )
                    self._rollback(state)
                pending_record_id = None
                pending_pre_snapshot = None

            history: tuple[ExperienceRecord, ...] = ()
            if self._controller.requires_history:
                history = tuple(
                    self._experience_store.query_similar(
                        state,
                        k=self._history_k,
                        filters=ExperienceFilter(run_id=self._run_id, branch_id=self._branch_id),
                    )
                )

            proposal = self._controller.propose(
                state=state,
                history=history,
                action_space=self._action_space,
                budget_remaining=state.budget_remaining,
            )
            result = self._validator.validate(
                proposal, state, controller_name=self._controller.name
            )
            record = ExperienceRecord(
                run_id=self._run_id,
                branch_id=self._branch_id,
                step=step,
                seed=self._seed,
                state=state,
                proposed_action=proposal.action,
                validation_result=result,
                controller_name=self._controller.name,
                controller_confidence=proposal.confidence,
                executed=result.approved,
                comparison_group_id=self._comparison_group_id,
            )
            self._experience_store.append(record)

            if result.approved:
                # Checkpoint before *any* mutating action, not only ones the validator
                # flagged as "major" (result.requires_checkpoint_first): a rollback can
                # be triggered by a regression after *any* approved action once a
                # RollbackPolicy is configured, so every one of them needs something to
                # roll back to. NoopAction changes nothing, so it's the only exemption.
                if not isinstance(result.action, NoopAction):
                    self._save_checkpoint(state)
                self._apply_action(result.action)
                self._validator.record_applied(result.action)
                pending_record_id = record.id
                pending_pre_snapshot = snapshot
                if isinstance(result.action, EarlyStopAction):
                    stopped_early = True
                    stop_reason = result.action.reason
                elif (
                    isinstance(result.action, TerminateBranchAction)
                    and result.action.branch_id == self._branch_id
                ):
                    # This branch terminated itself — nothing left to train towards.
                    stopped_early = True
                    stop_reason = result.action.reason

            self._fire(TrainingEvent.INTERVAL_END, state)
            if stopped_early:
                break
            if self._max_steps is not None and step >= self._max_steps:
                break

        self._fire(
            TrainingEvent.RUN_END,
            self._state_builder.observe(
                MetricSnapshot(
                    step=step,
                    epoch=step / max(1, self._steps_per_epoch()),
                    train_loss=last_train_loss,
                    learning_rate=_current_lr(self._optimizer),
                    wall_clock_seconds=wall_clock_total,
                )
            ),
        )

        return RunResult(
            run_id=self._run_id,
            branch_id=self._branch_id,
            total_steps=step,
            final_train_loss=last_train_loss,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            budget_consumed=self._budget_tracker.consumed,
        )

    def report(self) -> RunReport:
        records = self._experience_store.query_by_run(self._run_id)
        approved = sum(1 for r in records if r.validation_result.approved)
        return RunReport(
            run_id=self._run_id,
            branch_id=self._branch_id,
            controller_name=self._controller.name,
            total_steps=max((r.step for r in records), default=0),
            interventions_proposed=len(records),
            interventions_approved=approved,
            interventions_rejected=len(records) - approved,
        )

    # -- internals --------------------------------------------------------------

    def _fire(self, event: TrainingEvent, state: TrainingState) -> None:
        for callback in self._callbacks:
            callback.on_event(event, state)

    def _steps_per_epoch(self) -> int:
        try:
            return max(1, len(self._train_loader))  # type: ignore[arg-type]
        except TypeError:
            return 1

    def _iterate_batches(self) -> Iterator[Any]:
        if self._max_steps is None:
            yield from self._train_loader
            return
        emitted = 0
        while emitted < self._max_steps:
            for batch in self._train_loader:
                emitted += 1
                yield batch
                if emitted >= self._max_steps:
                    return

    def _train_step(self, batch: Any) -> float:
        self._optimizer.zero_grad(set_to_none=True)
        loss = self._compute_loss(self._model, batch, self._device)
        loss.backward()  # type: ignore[no-untyped-call]  # torch's stub for this is incomplete
        self._optimizer.step()
        return float(loss.detach().cpu())

    def _save_checkpoint(self, state: TrainingState) -> None:
        self._checkpoint = {
            "model": copy.deepcopy(self._model.state_dict()),
            "optimizer": copy.deepcopy(self._optimizer.state_dict()),
        }
        # Captured together so a rollback can restore both atomically: without this,
        # the validator would keep treating the rolled-back action's value as "current"
        # after the model/optimizer already reverted, and reject the controller's
        # correct follow-up proposal to undo it as a false oscillation.
        self._validator_checkpoint = self._validator.capture_state()
        self._fire(TrainingEvent.CHECKPOINT_SAVED, state)

    def _rollback(self, state: TrainingState) -> None:
        if self._checkpoint is None or self._validator_checkpoint is None:
            _logger.warning(
                "rollback requested on run=%s branch=%s but no checkpoint exists; "
                "continuing without rollback",
                self._run_id,
                self._branch_id,
            )
            return
        self._model.load_state_dict(self._checkpoint["model"])
        self._optimizer.load_state_dict(self._checkpoint["optimizer"])
        self._validator.restore_state(self._validator_checkpoint)
        reversal = ExperienceRecord(
            run_id=self._run_id,
            branch_id=self._branch_id,
            step=state.step,
            seed=self._seed,
            state=state,
            proposed_action=NoopAction(),
            validation_result=ValidationResult(approved=True, action=NoopAction()),
            controller_name="rollback",
            controller_confidence=None,
            executed=True,
            outcome=ActionOutcome(
                observed_at_step=state.step,
                metric_deltas={},
                additional_cost=BudgetUsage(),
                assessment=Assessment.NEGATIVE,
            ),
            assessment=Assessment.NEGATIVE,
            comparison_group_id=self._comparison_group_id,
        )
        self._experience_store.append(reversal)

    def _apply_action(self, action: Action) -> None:
        if isinstance(action, NoopAction):
            return
        if isinstance(action, AdjustLearningRateAction):
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = action.new_lr
            return
        if isinstance(action, ReweightDataAction):
            if self._on_reweight is None:
                raise UnsupportedActionError(
                    "REWEIGHT_DATA was proposed but no on_reweight hook was configured"
                )
            self._on_reweight(action.group_weights)
            for group, weight in action.group_weights.items():
                self._state_builder.set_group_weight(group, weight)
            return
        if isinstance(action, TriggerEvalAction):
            if self._on_trigger_eval is not None:
                passed = self._on_trigger_eval()
                self._validator.record_eval_result(passed=passed)
            return
        if isinstance(action, EarlyStopAction):
            return
        if isinstance(action, BranchExperimentAction):
            if self._experiment_manager is None:
                raise UnsupportedActionError(
                    "BRANCH_EXPERIMENT was proposed but no experiment_manager was configured"
                )
            self._experiment_manager.create_branch(
                parent_branch_id=action.parent_branch_id,
                config_overrides=action.config_overrides,
                checkpoint=self._checkpoint,
            )
            return
        if isinstance(action, TerminateBranchAction):
            if self._experiment_manager is None:
                raise UnsupportedActionError(
                    "TERMINATE_BRANCH was proposed but no experiment_manager was configured"
                )
            self._experiment_manager.terminate_branch(action.branch_id, reason=action.reason)
            return
        if isinstance(action, AllocateComputeAction):
            if self._experiment_manager is None:
                raise UnsupportedActionError(
                    "ALLOCATE_COMPUTE was proposed but no experiment_manager was configured"
                )
            self._experiment_manager.allocate_compute(
                action.branch_id, additional_gpu_hours=action.additional_gpu_hours
            )
            return
        raise UnsupportedActionError(f"unhandled action kind: {action.kind}")


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


class AdaptiveTrainer:
    """The public entrypoint, matching the research spec's illustrative API.

    A thin façade over `TrainerAdapter`: constructs the default validator, experience
    store, and state builder from a compact set of arguments so the common case (start
    from a model, a data loader, a loss function, and a controller) stays simple, while
    `TrainerAdapter` remains available directly for callers who need to supply every
    dependency themselves (e.g. the evaluation harness, in a later phase).
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        dataset: Iterable[Any],
        compute_loss: Callable[[torch.nn.Module, Any, torch.device], torch.Tensor],
        controller: ControllerStrategy,
        budget: BudgetUsage | Mapping[str, float],
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate: float = 1e-3,
        **adapter_kwargs: Any,
    ) -> None:
        resolved_budget = (
            budget if isinstance(budget, BudgetUsage) else BudgetUsage.model_validate(dict(budget))
        )
        resolved_optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate)
        self._adapter = TrainerAdapter(
            model=model,
            train_loader=dataset,
            optimizer=resolved_optimizer,
            compute_loss=compute_loss,
            controller=controller,
            budget=resolved_budget,
            **adapter_kwargs,
        )

    def fit(self) -> RunResult:
        return self._adapter.fit()

    def report(self) -> RunReport:
        return self._adapter.report()
