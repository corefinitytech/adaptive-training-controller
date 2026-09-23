# Contributing

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## Before opening a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

All four must pass locally; CI enforces the same checks plus a CPU smoke run of
`examples/first_experiment` and a scheduled `pip-audit` dependency scan.

## Design rules (non-negotiable)

These follow directly from the project's own research spec (§17) and its safety
architecture — PRs that violate them will be asked to change regardless of test coverage:

1. **No controller output reaches the training loop unvalidated.** Every new `Action`
   variant needs a corresponding guardrail in `validation/rules.py` before it can be
   proposed by any controller.
2. **No validation rule may special-case a specific controller** (by name, by confidence
   threshold, or otherwise). The validator is the sole safety authority; a controller's
   self-reported confidence is informational only.
3. **Deterministic baselines before intelligent control.** New controllers must be
   exercised through the same end-to-end test harness as `FixedController` before being
   considered complete.
4. **No free-form/untyped controller output.** New action types are added to the
   `Action` discriminated union, not passed through as strings or unstructured dicts.
5. **Every validated or rejected proposal is audit-logged.** If you add a new code path
   that mutates trainer state, it must go through `ActionValidator.validate()` first.

## Commit style

Concise, present-tense summary line; explain *why*, not *what* (the diff already shows
what changed).
