# 0004 — Jev is a `ControllerStrategy`, isolated under `controllers/jev/`

## Status

Accepted (design only — implementation lands in Phase 2).

## Context

Jev (TypeSafe AI's "System One" model, launched 2026-09-15) answers typed `choice`,
`score`, and `noul` questions with calibrated confidence scores rather than generating
text. Its request/response shape is a natural fit for this project's `Action` taxonomy —
but the research spec (§6, §17) requires controllers stay vendor-neutral: Jev must be
swappable for the fixed/rule-based/stateless/history-aware controllers (or a future one)
without touching the trainer, validator, or memory layer.

## Decision

`JevController` lives entirely under `controllers/jev/`. Nothing outside that package
imports `typesafe_sdk` or knows Jev's request/response shape. `question_mapper.py`
translates the run's current `ActionSpace` into one Jev request per interval (a `choice`
question over enabled `ActionKind`s, plus conditional `score`/`noul` questions for that
action's parameters) and translates typed answers back into an `Action` — falling back to
`NoopAction` with `confidence=0.0` on any malformed/out-of-range answer, never raising
into the trainer loop. `client.py` isolates the actual `typesafe-sdk` (or raw HTTP)
call behind a `JevClient` Protocol, so `question_mapper.py` is unit-testable against a
`FakeJevClient` with no network access.

Jev's `confidence` score is carried into `ControllerOutput.confidence` and is
**informational only** — see ADR 0001 and `validation/validator.py`: no rule in
`validation/rules.py` branches on confidence, and none is permitted to special-case
`controller_name == "jev"`.

## Consequences

- Swapping `Controller.jev(...)` for any other controller in `TrainerAdapter`'s
  constructor requires zero other code changes. If it ever does, that's a signal the
  abstraction leaked and should be fixed before shipping.
- Jev's exact `typesafe-sdk` method names are **not** guessed in this ADR or anywhere
  else in the codebase. The first task of Phase 2 is verifying them against
  `docs.typesafe.ai` / the installed package. ⚠️ Get API keys only from
  `console.typesafe.ai` — third-party pages (`jevapi.org`, `tokenra.io`) surfaced
  inconsistent, unverifiable claims about Jev's endpoint and key-issuance domain during
  this project's research and should not be trusted as a source of truth.
- Jev is integrated *after* the deterministic baselines (Fixed, Rule-based, Stateless,
  History-aware) are proven end-to-end, per the research spec's own binding instruction
  (§17) — not because Jev is untrusted, but because a controlled comparison against those
  baselines is the only way to know whether Jev is actually contributing value (§10).
