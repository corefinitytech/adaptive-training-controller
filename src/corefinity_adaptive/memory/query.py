"""Filtering and similarity search over `ExperienceRecord`s.

`state_distance` is a simple, explainable weighted-feature distance over the current
`MetricSnapshot` — not a learned embedding or vector index. Per the plan this project
follows: start with the simplest thing that's correct, and only reach for a vector store
once `benchmarks/` shows this is actually a bottleneck at realistic history sizes.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from corefinity_adaptive.actions.outcome import Assessment
from corefinity_adaptive.state.models import TrainingState

_METRIC_WEIGHTS: dict[str, float] = {
    "train_loss": 1.0,
    "val_loss": 1.0,
    "val_accuracy": 1.0,
    "learning_rate": 0.25,
}


class ExperienceFilter(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str | None = None
    branch_id: str | None = None
    controller_name: str | None = None
    assessment: Assessment | None = None
    comparison_group_id: str | None = None
    executed_only: bool = False


def state_distance(a: TrainingState, b: TrainingState) -> float:
    """A small, explainable Euclidean-ish distance over current metric values.

    Missing values on either side drop that term rather than penalizing it — an
    incomplete metric isn't evidence of dissimilarity.
    """
    total = 0.0
    for field, weight in _METRIC_WEIGHTS.items():
        value_a: float | None = getattr(a.current, field)
        value_b: float | None = getattr(b.current, field)
        if value_a is None or value_b is None:
            continue
        total += weight * (value_a - value_b) ** 2
    return math.sqrt(total)
