"""The smallest possible `AdaptiveTrainer` run: a tiny MLP, the `Fixed` baseline
controller, on CPU.

Exists to prove the whole wiring (trainer, validator, experience store, audit log) works
end to end outside of the test suite — and to serve as CI's smoke test. Pass
`--smoke-test` to run just a handful of steps.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from corefinity_adaptive import ActionSpace, BudgetUsage, Controller
from corefinity_adaptive.actions.types import ActionKind
from corefinity_adaptive.memory.sqlite_store import SQLiteExperienceStore
from corefinity_adaptive.observability.audit_log import SQLiteAuditLog
from corefinity_adaptive.trainer import TrainerAdapter


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_loss(
    model: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> torch.Tensor:
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    return torch.nn.functional.mse_loss(model(inputs), targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a trivially small number of steps, for CI",
    )
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(64, 4, generator=generator)
    targets = inputs.sum(dim=1, keepdim=True)
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        TensorDataset(inputs, targets), batch_size=8, shuffle=True
    )

    model = TinyModel()
    run_dir = Path(tempfile.mkdtemp(prefix="corefinity_minimal_"))

    trainer = TrainerAdapter(
        model=model,
        train_loader=loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        compute_loss=compute_loss,
        controller=Controller.fixed(),
        budget=BudgetUsage(steps=200),
        action_space=ActionSpace(enabled_kinds=frozenset({ActionKind.NOOP})),
        interval_steps=4,
        max_steps=8 if args.smoke_test else 200,
        experience_store=SQLiteExperienceStore(run_dir / "experience.sqlite"),
        audit_log=SQLiteAuditLog(run_dir / "audit_log.sqlite"),
    )

    result = trainer.fit()
    report = trainer.report()
    print(
        f"run_id={result.run_id} steps={result.total_steps} "
        f"final_train_loss={result.final_train_loss:.4f}"
    )
    print(
        f"interventions_proposed={report.interventions_proposed} "
        f"approved={report.interventions_approved} rejected={report.interventions_rejected}"
    )
    print(f"experience store: {run_dir / 'experience.sqlite'}")


if __name__ == "__main__":
    main()
