"""Typed representations of training state.

`TrainingState` is the single structured payload every controller receives, and the only
form the trainer's raw metrics take once they cross into controller/validator code. It is
a snapshot (immutable) rather than a live view, so a controller's decision can always be
traced back to the exact state it was made against.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt


class BudgetUsage(BaseModel):
    """A point in one or more compute-budget dimensions.

    Used both for "consumed so far" and "remaining" readings, and for the hard budget
    declared on a run. All fields are optional because not every run tracks every
    dimension (e.g. a CPU-only run has no `gpu_hours`).
    """

    model_config = ConfigDict(frozen=True)

    gpu_hours: NonNegativeFloat | None = None
    tokens: NonNegativeInt | None = None
    steps: NonNegativeInt | None = None
    wall_clock_seconds: NonNegativeFloat | None = None

    def exceeds(self, limit: BudgetUsage) -> bool:
        """True if any dimension present on both sides exceeds the corresponding limit.

        A dimension that `limit` doesn't track is treated as unconstrained, not as zero.
        """
        for field in self.__class__.model_fields:
            used = getattr(self, field)
            cap = getattr(limit, field)
            if used is not None and cap is not None and used > cap:
                return True
        return False

    def plus(self, delta: BudgetUsage) -> BudgetUsage:
        """Return a new `BudgetUsage` with each dimension summed, treating unset as zero."""
        return BudgetUsage(
            gpu_hours=_add_float(self.gpu_hours, delta.gpu_hours),
            tokens=_add_int(self.tokens, delta.tokens),
            steps=_add_int(self.steps, delta.steps),
            wall_clock_seconds=_add_float(self.wall_clock_seconds, delta.wall_clock_seconds),
        )


def _add_float(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)


def _add_int(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


class MetricSnapshot(BaseModel):
    """The raw, deterministic measurements taken at one point in training."""

    model_config = ConfigDict(frozen=True)

    step: NonNegativeInt
    epoch: NonNegativeFloat
    train_loss: float
    val_loss: float | None = None
    val_accuracy: float | None = None
    learning_rate: NonNegativeFloat
    grad_norm: NonNegativeFloat | None = None
    grad_variance: NonNegativeFloat | None = None
    throughput_samples_per_sec: NonNegativeFloat | None = None
    wall_clock_seconds: NonNegativeFloat


class TrainingState(BaseModel):
    """The structured state a controller reasons over.

    `recent_trend` is bounded at construction time by `StateBuilder` (see
    `state/builder.py`), never left to grow unbounded — this is the state-side half of
    the guardrail against the "history explosion" failure mode; the memory-side half is
    `memory/retention.py`.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    branch_id: str
    step: NonNegativeInt
    wall_clock_seconds: NonNegativeFloat
    current: MetricSnapshot
    recent_trend: tuple[MetricSnapshot, ...] = Field(default_factory=tuple)
    data_exposure: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per data-group current sampling weight, from the last ReweightDataAction "
            "applied to each group (absent if a group has never been reweighted)."
        ),
    )
    budget_consumed: BudgetUsage
    budget_remaining: BudgetUsage
    seed: int
    config_hash: str
