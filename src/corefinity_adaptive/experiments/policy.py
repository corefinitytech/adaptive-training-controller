"""The PBT-style exploit/explore policy: pure ranking and perturbation, kept separate
from `ExperimentManager`'s stateful branch bookkeeping so the actual decision logic is
independently testable (same split as `controllers/policies.py`).
"""

from __future__ import annotations

import random


def rank_for_exploit(
    scores: dict[str, float], *, bottom_fraction: float
) -> tuple[list[str], list[str]]:
    """Rank branches by score (higher is better — negate a loss before calling this).

    Returns `(ranked_best_first, bottom_ids_to_terminate)`. With fewer than two scored
    branches there's nothing to exploit (a single branch has no peer to compare
    against), so the bottom list is always empty in that case.
    """
    ranked = sorted(scores, key=lambda branch_id: scores[branch_id], reverse=True)
    if len(ranked) < 2:
        return ranked, []
    n_bottom = max(1, round(len(ranked) * bottom_fraction))
    bottom = ranked[-n_bottom:]
    # Never terminate the top performer, even if bottom_fraction is large enough that
    # the ranges would otherwise overlap on a small population.
    bottom = [branch_id for branch_id in bottom if branch_id != ranked[0]]
    return ranked, bottom


def perturb_learning_rate(
    config: dict[str, float | int | str | bool],
    *,
    key: str = "learning_rate",
    factor_range: tuple[float, float] = (0.8, 1.2),
    rng: random.Random | None = None,
) -> dict[str, float | int | str | bool]:
    """The classic PBT "explore" step for a single numeric hyperparameter: multiply it
    by a random factor in `factor_range`. Returns a new dict; `config` is left untouched.

    If `key` isn't present in `config`, the config is returned unchanged — perturbing a
    hyperparameter that was never set has nothing to perturb.
    """
    if key not in config:
        return dict(config)
    current = config[key]
    if not isinstance(current, int | float):
        return dict(config)
    generator = rng or random.Random()
    factor = generator.uniform(*factor_range)
    return {**config, key: current * factor}
