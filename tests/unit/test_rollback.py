"""`RollbackPolicy`: regression detection respects each metric's "better" direction and
tolerance.
"""

from __future__ import annotations

from corefinity_adaptive.actions.outcome import ActionOutcome, Assessment
from corefinity_adaptive.state.models import BudgetUsage
from corefinity_adaptive.validation.rollback import RegressionRule, RollbackPolicy


def _outcome(**deltas: float) -> ActionOutcome:
    return ActionOutcome(
        observed_at_step=1,
        metric_deltas=deltas,
        additional_cost=BudgetUsage(),
        assessment=Assessment.PENDING,
    )


def test_lower_is_better_metric_triggers_on_increase_beyond_tolerance() -> None:
    policy = RollbackPolicy(
        (RegressionRule(metric_name="val_loss", lower_is_better=True, max_allowed_regression=0.1),)
    )
    assert policy.should_rollback(_outcome(val_loss=0.5)) is not None
    assert policy.should_rollback(_outcome(val_loss=0.05)) is None
    assert policy.should_rollback(_outcome(val_loss=-0.5)) is None  # improvement


def test_higher_is_better_metric_triggers_on_decrease_beyond_tolerance() -> None:
    policy = RollbackPolicy(
        (
            RegressionRule(
                metric_name="val_accuracy", lower_is_better=False, max_allowed_regression=0.02
            ),
        )
    )
    assert policy.should_rollback(_outcome(val_accuracy=-0.1)) is not None
    assert policy.should_rollback(_outcome(val_accuracy=-0.01)) is None
    assert policy.should_rollback(_outcome(val_accuracy=0.1)) is None  # improvement


def test_missing_metric_is_ignored_not_treated_as_violation() -> None:
    policy = RollbackPolicy(
        (RegressionRule(metric_name="val_loss", lower_is_better=True, max_allowed_regression=0.1),)
    )
    assert policy.should_rollback(_outcome(train_loss=999.0)) is None


def test_no_rules_never_rolls_back() -> None:
    policy = RollbackPolicy()
    assert policy.should_rollback(_outcome(val_loss=999.0)) is None


def test_returns_the_first_violated_rule() -> None:
    policy = RollbackPolicy(
        (
            RegressionRule(
                metric_name="val_loss", lower_is_better=True, max_allowed_regression=0.1
            ),
            RegressionRule(
                metric_name="val_accuracy", lower_is_better=False, max_allowed_regression=0.02
            ),
        )
    )
    violated = policy.should_rollback(_outcome(val_loss=0.5, val_accuracy=-0.5))
    assert violated is not None
    assert violated.metric_name == "val_loss"
