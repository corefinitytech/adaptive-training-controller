"""`ExperimentManager`: branch lineage, config, checkpoints, and lifecycle — the piece
`TrainerAdapter` hands `BranchExperimentAction`/`TerminateBranchAction`/
`AllocateComputeAction` off to once one is approved.

Branches run *sequentially* in this build phase (see the project plan) — creating a
branch here registers its metadata as `PENDING`; it does not spawn concurrent training.
A caller drives the actual sequence: pull `pending_branches()`, call `activate_branch()`,
construct a `TrainerAdapter` for it (seeded from `Branch.checkpoint`/`config_overrides`),
run it, then feed the result back in (e.g. via `exploit_and_explore`) before picking the
next one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from corefinity_adaptive.experiments.branch import Branch, BranchStatus
from corefinity_adaptive.experiments.policy import perturb_learning_rate, rank_for_exploit
from corefinity_adaptive.state.models import BudgetUsage


class UnknownBranchError(KeyError):
    """Raised when a branch id doesn't exist — a genuine bug (a stale/typo'd id), not a
    guardrail rejection (those are `ActionValidator`'s job)."""


class ExperimentManager:
    def __init__(self, *, root_branch_id: str = "main") -> None:
        self._branches: dict[str, Branch] = {
            root_branch_id: Branch(
                branch_id=root_branch_id,
                parent_branch_id=None,
                config_overrides={},
                budget=BudgetUsage(),
                status=BranchStatus.ACTIVE,
            )
        }

    def _require(self, branch_id: str) -> Branch:
        try:
            return self._branches[branch_id]
        except KeyError as exc:
            raise UnknownBranchError(branch_id) from exc

    def get(self, branch_id: str) -> Branch:
        return self._require(branch_id)

    def all_branches(self) -> tuple[Branch, ...]:
        return tuple(self._branches.values())

    def pending_branches(self) -> tuple[Branch, ...]:
        return tuple(b for b in self._branches.values() if b.status is BranchStatus.PENDING)

    def active_branches(self) -> tuple[Branch, ...]:
        return tuple(b for b in self._branches.values() if b.status is BranchStatus.ACTIVE)

    def create_branch(
        self,
        *,
        parent_branch_id: str,
        config_overrides: dict[str, float | int | str | bool],
        checkpoint: Any | None = None,
    ) -> Branch:
        self._require(parent_branch_id)  # fail fast on a stale/typo'd parent id
        branch = Branch(
            branch_id=str(uuid4()),
            parent_branch_id=parent_branch_id,
            config_overrides=config_overrides,
            budget=BudgetUsage(),
            status=BranchStatus.PENDING,
            checkpoint=checkpoint,
        )
        self._branches[branch.branch_id] = branch
        return branch

    def activate_branch(self, branch_id: str) -> Branch:
        """Transition a `PENDING` branch to `ACTIVE`.

        Call this once a caller actually starts running the branch (e.g. constructs a
        `TrainerAdapter` seeded from `Branch.checkpoint`/`config_overrides` and calls
        `.fit()`) — `ExperimentManager` never does this on its own, since branches run
        sequentially and only the caller knows when that's actually happening.
        """
        branch = self._require(branch_id)
        updated = branch.model_copy(update={"status": BranchStatus.ACTIVE})
        self._branches[branch_id] = updated
        return updated

    def terminate_branch(self, branch_id: str, *, reason: str) -> Branch:
        branch = self._require(branch_id)
        updated = branch.model_copy(
            update={"status": BranchStatus.TERMINATED, "terminated_reason": reason}
        )
        self._branches[branch_id] = updated
        return updated

    def allocate_compute(self, branch_id: str, *, additional_gpu_hours: float) -> Branch:
        """Record that `additional_gpu_hours` was allocated to `branch_id`.

        Purely bookkeeping — see this module's docstring. Whether the proposing branch
        can *afford* to give away that much has already been decided by
        `ActionValidator.validate()` before `TrainerAdapter` ever calls this.
        """
        branch = self._require(branch_id)
        updated = branch.model_copy(
            update={"budget": branch.budget.plus(BudgetUsage(gpu_hours=additional_gpu_hours))}
        )
        self._branches[branch_id] = updated
        return updated

    def exploit_and_explore(
        self,
        scores: dict[str, float],
        *,
        bottom_fraction: float = 0.2,
        perturb: Callable[
            [dict[str, float | int | str | bool]], dict[str, float | int | str | bool]
        ] = perturb_learning_rate,
    ) -> tuple[Branch, ...]:
        """One PBT step: terminate the bottom `bottom_fraction` of `scores` (higher is
        better), and for each, create a new `PENDING` branch cloned from the top
        performer's checkpoint/config with `perturb` applied.

        Only ids present in both `scores` and this manager's `ACTIVE` branches are
        considered — an id that isn't currently active (already terminated, or unknown)
        is silently skipped rather than treated as an error, since a caller comparing
        scores across a whole run's history may naturally include ids that have since
        moved on.
        """
        active_ids = {b.branch_id for b in self.active_branches()}
        scoped_scores = {bid: score for bid, score in scores.items() if bid in active_ids}
        ranked, bottom = rank_for_exploit(scoped_scores, bottom_fraction=bottom_fraction)
        if not bottom:
            return ()

        top_branch = self._branches[ranked[0]]
        new_branches = []
        for branch_id in bottom:
            self.terminate_branch(
                branch_id, reason=f"PBT exploit: bottom {bottom_fraction:.0%} by score"
            )
            new_branches.append(
                self.create_branch(
                    parent_branch_id=top_branch.branch_id,
                    config_overrides=perturb(dict(top_branch.config_overrides)),
                    checkpoint=top_branch.checkpoint,
                )
            )
        return tuple(new_branches)
