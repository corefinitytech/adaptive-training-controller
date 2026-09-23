"""Shared decision primitives for the plateau-triggered reweighting rule used by
`RuleBasedController`, `StatelessController`, and `HistoryAwareController`.

Pulled out into pure functions (rather than duplicated per controller) so the actual
arithmetic — what counts as a plateau, how the next weight is computed and bounded — has
exactly one implementation to test and reason about, even though the three controllers
differ in *where* they source their loss series from and whether they consult the
experience store.
"""

from __future__ import annotations

from collections.abc import Sequence

from corefinity_adaptive.actions.space import ActionSpace
from corefinity_adaptive.actions.types import ReweightDataAction
from corefinity_adaptive.state.models import MetricSnapshot, TrainingState


def primary_loss(snapshot: MetricSnapshot) -> float:
    """Validation loss if available, else training loss — the one number these
    controllers treat as "the metric to watch"."""
    return snapshot.val_loss if snapshot.val_loss is not None else snapshot.train_loss


def is_plateaued(loss_series: Sequence[float], *, min_relative_improvement: float) -> bool:
    """True if `loss_series` hasn't improved by at least `min_relative_improvement`
    (as a fraction of its first value) between its first and last points.

    Fewer than two points is never a plateau — there's nothing yet to compare.
    """
    if len(loss_series) < 2:
        return False
    earliest, latest = loss_series[0], loss_series[-1]
    if earliest <= 0:
        # A non-positive loss has no meaningful "relative" improvement; fall back to a
        # plain non-improvement check.
        return latest >= earliest
    relative_improvement = (earliest - latest) / earliest
    return relative_improvement <= min_relative_improvement


def propose_reweight(
    *,
    state: TrainingState,
    action_space: ActionSpace,
    target_group: str,
    step: float,
    initial_weight: float,
    max_weight: float,
) -> ReweightDataAction:
    """The next weight for `target_group`: nudged up by `step` from its current value
    (read from `state.data_exposure`, defaulting to `initial_weight` if never set),
    bounded by both `max_weight` and the run's own `max_reweight_delta` guardrail — so a
    well-behaved controller doesn't get needlessly rejected for a delta it could have
    seen coming from the `ActionSpace` it was already given.
    """
    current = state.data_exposure.get(target_group, initial_weight)
    bounded_step = min(step, action_space.max_reweight_delta)
    new_weight = min(current + bounded_step, max_weight)
    return ReweightDataAction(group_weights={target_group: new_weight})
