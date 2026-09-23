# CoreFinity Adaptive Training Controller

`corefinity-adaptive` is a vendor-neutral Python framework for **closed-loop adaptive
training**: instead of following a fixed recipe (dataset → epoch 1 → epoch 2 → … → done),
a pluggable *controller* observes training state at regular intervals and proposes bounded
interventions — reweighting data, adjusting the learning rate, allocating or terminating
compute, branching experiments, stopping early. A deterministic *action validator* gates
every proposal before it can execute, and every intervention is recorded as an inspectable
*experience record* so later decisions can draw on what happened before.

This project treats closed-loop adaptive training as a **hypothesis to test, not a proven
result**. See [`docs/architecture.md`](docs/architecture.md) and the
[architecture decision records](docs/adr/) for the reasoning behind each design choice, and
the [research spec](CoreFinity_Adaptive_Training_Controller_Research_and_Product_Spec.pdf)
for the original product brief this library implements.

## Status

Early development (Phase 1 — Prototype). The core types, action validator, experience
store, and trainer adapter exist behind a `FixedController` baseline. Intelligent
controllers (rule-based, history-aware, [Jev](https://typesafe.ai)) and the evaluation
harness land in subsequent phases — see the roadmap in `docs/architecture.md`.

## Design principles

- **Deterministic baselines before intelligent control.** Every controller — including
  Jev — implements the same `Controller` protocol and is validated by the same guardrails.
  A `FixedController` (no-op) always exists so any adaptive controller can be measured
  against doing nothing, on the same code path.
- **The validator is the safety authority, not the controller.** No controller's output
  reaches the training loop unvalidated, and no validation rule is skipped because a
  controller reports high confidence.
- **Every decision is inspectable.** Every validated or rejected proposal is written to an
  audit log and an experience record, queryable from a local SQLite store — no black-box
  history.

## Installation

Requires Python 3.11+.

```bash
uv sync                    # core install
uv sync --extra jev        # + Jev (TypeSafe AI) controller support
uv sync --extra mlflow     # + optional MLflow experience sink
uv sync --extra dev        # + lint/type-check/test tooling
```

## Quickstart

```python
from corefinity_adaptive import AdaptiveTrainer, Controller

trainer = AdaptiveTrainer(
    model=model,
    dataset=dataset,
    controller=Controller.fixed(),  # no-op baseline; swap for an adaptive controller later
    budget={"gpu_hours": 1},
)
trainer.fit()
trainer.report()
```

## Development

```bash
uv sync --extra dev
uv run pre-commit install
uv run ruff check .
uv run mypy src
uv run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
