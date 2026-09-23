"""A bounded rolling window used to build `TrainingState.recent_trend`.

Kept as a tiny, dependency-free type so the "never let history grow unbounded" rule is
enforced by the container itself, not by callers remembering to slice.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedHistory(Generic[T]):
    """A fixed-capacity FIFO buffer: pushing past capacity evicts the oldest item."""

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen = maxlen
        self._items: deque[T] = deque(maxlen=maxlen)

    def push(self, item: T) -> None:
        self._items.append(item)

    def as_tuple(self) -> tuple[T, ...]:
        return tuple(self._items)

    def replace(self, items: tuple[T, ...]) -> None:
        """Discard current contents and reload from `items` (oldest first).

        Used to restore a prior snapshot (e.g. `ActionValidator` resyncing its LR/weight
        history after a rollback) — `items` is truncated to the most recent `maxlen`
        entries, matching what `push`-ing them one at a time would have left behind.
        """
        self._items.clear()
        self._items.extend(items[-self._maxlen :])

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)
