"""A single-model evaluation contract (`Evaluator`, `EvalResult`) and a reference
implementation (`LossEvalSuite`) — the piece that gives `TrainerAdapter`'s
`on_trigger_eval` hook and branch-promotion eval gate something real to call.

The multi-controller, multi-seed comparison harness and quality-vs-compute frontier the
research spec's §10 rigor requirements call for are not here yet — that needs a real
task to be meaningful (not this Mac-sized synthetic model) and lands in Phase 3, once
there are at least two controllers and a real first experiment to compare, per the
project's build-sequence plan (`docs/adr/`).
"""

from corefinity_adaptive.evaluation.evaluator import EvalResult, Evaluator
from corefinity_adaptive.evaluation.suites import LossEvalSuite

__all__ = ["EvalResult", "Evaluator", "LossEvalSuite"]
