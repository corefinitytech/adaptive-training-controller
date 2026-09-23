# 0001 — Closed discriminated union for the action taxonomy

## Status

Accepted.

## Context

The research spec (§6) is explicit that "free-form output must never be executed
directly." A controller — especially an LLM- or model-backed one like Jev — must not be
able to hand the trainer a string or an unstructured dict that gets interpreted at
execution time; that reopens exactly the prompt-injection-shaped attack surface typed
tool calling exists to close.

## Decision

Every intervention a controller can propose is one variant of `Action`
(`actions/types.py`), a Pydantic discriminated union on a `kind: ActionKind` field:
`NoopAction`, `ReweightDataAction`, `AdjustLearningRateAction`, `TriggerEvalAction`,
`EarlyStopAction`, `BranchExperimentAction`, `TerminateBranchAction`,
`AllocateComputeAction`. There is no "other" or "raw payload" variant.

`ActionSpace` declares which kinds are enabled and their numeric bounds for a given run,
consumed by both the controller (so a well-behaved one avoids proposing something
disallowed) and `ActionValidator` (so it never trusts the controller not to anyway).

## Consequences

- Adding a new kind of intervention is a two-step change: add a variant here, then add a
  guardrail rule in `validation/rules.py` before any controller may propose it. This is
  enforced by convention (see `CONTRIBUTING.md`), not by tooling, in this phase.
- A controller (including Jev, once integrated) cannot express an action outside this
  taxonomy — full stop. `JevController`'s translation layer maps Jev's typed
  choice/score/noul answers *into* one of these variants; it can't do anything else.
