"""The research spec's §11 first experiment: a tiny char-level transformer trained on a
real text corpus (a 200KB slice of the standard "tinyshakespeare" dataset), comparing
the `Fixed` baseline against the three deterministic controllers built in Phase 1, with
a fixed seed per controller for reproducibility.

This is a single-seed, informal comparison meant to prove the framework's wiring end to
end on a real task — it is explicitly NOT the Phase 3 multi-seed comparison harness with
the statistical rigor the research spec's §10 demands (repeated runs, variance bands,
etc.). Treat any numbers this prints as anecdotal, not a finding, until that harness
exists. Run with `--smoke-test` for a fast few-step sanity check (used by CI); omit it
for a real (still Mac-sized) run.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
from dataset import CharDataset, GroupWeightedSampler
from model import TinyCharTransformer
from torch.utils.data import DataLoader

from corefinity_adaptive import ActionSpace, BudgetUsage, Controller, ControllerStrategy
from corefinity_adaptive.actions.types import ActionKind
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.observability.audit_log import SQLiteAuditLog
from corefinity_adaptive.trainer import RunReport, RunResult, TrainerAdapter

_CORPUS_PATH = Path(__file__).parent / "data" / "corpus.txt"
_BLOCK_SIZE = 64
_BATCH_SIZE = 32
_TARGET_GROUP = "hard"

_CONTROLLER_FACTORIES: dict[str, Callable[[], ControllerStrategy]] = {
    "fixed": Controller.fixed,
    "rule_based": lambda: Controller.rule_based(
        target_group=_TARGET_GROUP, plateau_window=3, reweight_step=0.1
    ),
    "stateless": lambda: Controller.stateless(target_group=_TARGET_GROUP, reweight_step=0.1),
    "history_aware": lambda: Controller.history_aware(
        target_group=_TARGET_GROUP, reweight_step=0.1
    ),
}


def compute_loss(
    model: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> torch.Tensor:
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )


def run_controller(
    name: str,
    factory: Callable[[], ControllerStrategy],
    *,
    dataset: CharDataset,
    steps: int,
    seed: int,
    run_dir: Path,
) -> tuple[RunResult, RunReport]:
    torch.manual_seed(seed)
    model = TinyCharTransformer(vocab_size=dataset.vocab_size, block_size=_BLOCK_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    interval_steps = 25 if steps >= 100 else 5

    # GroupWeightedSampler reads `on_reweight`'s weights fresh once per DataLoader
    # epoch (see its docstring) — sized to roughly one TrainerAdapter interval's worth
    # of batches, not the whole dataset, so a reweight proposed at one interval boundary
    # actually reaches the sampler before the run ends, rather than the draw order
    # having already been fixed for a "epoch" far longer than this run's whole budget.
    sampler = GroupWeightedSampler(
        dataset.group_labels,
        num_samples=interval_steps * _BATCH_SIZE,
        generator=torch.Generator().manual_seed(seed),
    )
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=_BATCH_SIZE, sampler=sampler
    )

    trainer = TrainerAdapter(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        compute_loss=compute_loss,
        controller=factory(),
        budget=BudgetUsage(steps=steps),
        action_space=ActionSpace(
            enabled_kinds=frozenset({ActionKind.NOOP, ActionKind.REWEIGHT_DATA}),
            max_reweight_delta=0.5,
        ),
        experience_store=SQLiteExperienceStore(run_dir / f"{name}_experience.sqlite"),
        audit_log=SQLiteAuditLog(run_dir / f"{name}_audit_log.sqlite"),
        run_id=name,
        interval_steps=interval_steps,
        max_steps=steps,
        seed=seed,
        on_reweight=sampler.set_weights,
    )
    result = trainer.fit()
    report = trainer.report()
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true", help="run a trivially small number of steps, for CI"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    steps = 20 if args.smoke_test else 500
    text = _CORPUS_PATH.read_text()
    dataset = CharDataset(text, block_size=_BLOCK_SIZE)
    hard_fraction = dataset.group_labels.count("hard") / len(dataset.group_labels)
    print(
        f"corpus: {len(text)} chars, vocab_size={dataset.vocab_size}, "
        f"{len(dataset)} training examples ({hard_fraction:.0%} classified {_TARGET_GROUP!r})"
    )

    run_dir = Path(tempfile.mkdtemp(prefix="corefinity_first_experiment_"))
    print(f"{'controller':<15} {'final_train_loss':>18} {'proposed':>10} {'approved':>10}")
    for name, factory in _CONTROLLER_FACTORIES.items():
        result, report = run_controller(
            name, factory, dataset=dataset, steps=steps, seed=args.seed, run_dir=run_dir
        )
        print(
            f"{name:<15} {result.final_train_loss:>18.4f} "
            f"{report.interventions_proposed:>10} {report.interventions_approved:>10}"
        )

    print(f"\nper-controller experience stores/audit logs: {run_dir}")


if __name__ == "__main__":
    main()
