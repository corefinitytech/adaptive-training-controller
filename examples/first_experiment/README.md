# First experiment

Not started yet. Per the research spec (§11) and the project plan, this is a small,
reproducible task — a tiny char-level transformer on a small text corpus, sized to run
on CPU/MPS in minutes — comparing the `Fixed` baseline against `RuleBasedController`
with fixed seeds. It lands once those controllers exist (see `docs/architecture.md` for
the current build phase); Jev is added as a further arm only after that, per the
project's binding instruction to prove deterministic baselines first.

For a working (if intentionally trivial) end-to-end example today, see
`examples/minimal_pytorch_loop/`.
