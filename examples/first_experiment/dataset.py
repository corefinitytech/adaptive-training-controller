"""Character-level dataset with two illustrative data groups, and a sampler that
`ReweightDataAction`'s `on_reweight` hook can actually change the effect of.

The "easy"/"hard" split (by average word length in each training block) is a simple,
deterministic, illustrative proxy for difficulty — not a scientifically validated
measure. The research spec treats data grouping as something to define per-task, and
what matters for this first experiment is that reweighting a group has a real, visible
effect on sampling, not that the specific difficulty heuristic is principled.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, Sampler

_HARD_AVG_WORD_LENGTH_THRESHOLD = 4.5


class CharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, text: str, *, block_size: int) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.num_examples = max(0, len(self.data) - block_size)
        self.group_labels: list[str] = [
            self._classify(text[start : start + block_size]) for start in range(self.num_examples)
        ]

    @staticmethod
    def _classify(chunk: str) -> str:
        words = chunk.split()
        if not words:
            return "easy"
        avg_word_length = sum(len(w) for w in words) / len(words)
        return "hard" if avg_word_length > _HARD_AVG_WORD_LENGTH_THRESHOLD else "easy"

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]

    def decode(self, indices: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in indices)


class GroupWeightedSampler(Sampler[int]):
    """Samples dataset indices with probability proportional to their group's weight.

    Weights are read fresh on every `__iter__` call, so `set_weights` (the
    `on_reweight` hook `TrainerAdapter` calls) changes sampling starting from the next
    epoch — no need to rebuild the `DataLoader` mid-run.
    """

    def __init__(
        self,
        group_labels: list[str],
        *,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> None:
        self._group_labels = group_labels
        self._num_samples = num_samples
        self._generator = generator
        self._weights: dict[str, float] = {}

    def set_weights(self, weights: dict[str, float]) -> None:
        self._weights.update(weights)

    def __iter__(self) -> Iterator[int]:
        default_weight = 1.0
        per_index_weights = torch.tensor(
            [self._weights.get(label, default_weight) for label in self._group_labels],
            dtype=torch.double,
        )
        indices = torch.multinomial(
            per_index_weights, self._num_samples, replacement=True, generator=self._generator
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self._num_samples
