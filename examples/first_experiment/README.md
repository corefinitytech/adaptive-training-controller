# First experiment

The research spec's §11 first experiment: a tiny, from-scratch character-level
transformer (`model.py`) trained on a real text corpus (`data/corpus.txt`, a 200KB slice
of the standard "tinyshakespeare" dataset), comparing the `Fixed` baseline against the
three deterministic controllers built in Phase 1 — `RuleBasedController`,
`StatelessController`, `HistoryAwareController` — with a fixed seed per controller.

```bash
uv run python examples/first_experiment/run.py            # ~70s on this project's Mac
uv run python examples/first_experiment/run.py --smoke-test  # ~8s, what CI runs
```

## What this is (and isn't)

This is a **single-seed, informal comparison** meant to prove the framework's wiring end
to end on a real task — not the Phase 3 multi-seed comparison harness with the
statistical rigor the research spec's §10 demands (repeated runs, variance bands, a
quality-vs-compute frontier). Treat any numbers this prints as anecdotal, not a finding,
until that harness exists.

## A real result from running this

At 500 steps/controller, `rule_based` never proposed a reweight (its own
consultation-time loss history never showed a plateau in this budget) and matched
`fixed` exactly. `stateless` and `history_aware` did propose — and have approved —
`ReweightDataAction`s partway through, and their final training loss measurably
diverged from `fixed`/`rule_based`'s. This is a genuine effect, not a coincidence: the
`GroupWeightedSampler` backing `on_reweight` only re-reads its weights once per
DataLoader epoch, so it's deliberately sized to about one `TrainerAdapter` interval's
worth of batches — otherwise a reweight proposed mid-run has nothing left of the current
"epoch" to actually affect. Per the project's own instruction (§17) never to fabricate
results, this README reports what one specific run actually showed, not a claim that any
controller here "won."

## Data groups

Each training example (a `block_size`-character window) is classified `"easy"` or
`"hard"` by average word length in that window — a simple, deterministic, illustrative
proxy, not a validated difficulty measure. `RuleBasedController`/`StatelessController`/
`HistoryAwareController` are all configured to reweight the `"hard"` group.

For the minimal (non-real-task) end-to-end example this repo also ships, see
`examples/minimal_pytorch_loop/`.
