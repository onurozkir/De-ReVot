"""Bounded async queues with drop, coalesce, and overload policies."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")


class BoundedQueue(Generic[T]):
    """Bounded async queue enforcing strict capacity and backlog prevention."""

    def __init__(
        self,
        maxsize: int = 10,
        drop_oldest_on_full: bool = False,
        replace_key: Optional[Callable[[T], str]] = None,
    ):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self.drop_oldest_on_full = drop_oldest_on_full
        self.replace_key = replace_key
        
        self._items: List[tuple[int, T]] = []
        self._cond = asyncio.Condition()
        self._dropped_count = 0
        self._replaced_count = 0

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def replaced_count(self) -> int:
        return self._replaced_count

    def qsize(self) -> int:
        return len(self._items)

    def is_full(self) -> bool:
        return len(self._items) >= self.maxsize

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def oldest_age_ms(self) -> float:
        if not self._items:
            return 0.0
        return (time.monotonic_ns() - self._items[0][0]) / 1e6

    def snapshot(self, name: str, drop_policy: str) -> dict:
        return {
            "name": name,
            "current": self.qsize(),
            "max": self.maxsize,
            "oldest_age_ms": self.oldest_age_ms(),
            "dropped": self.dropped_count,
            "replaced": self.replaced_count,
            "drop_policy": drop_policy,
        }

    async def put(self, item: T) -> bool:
        """Put item into queue according to bounding policy.
        
        Returns True if item was accepted/replaced, False if rejected on full.
        """
        async with self._cond:
            if self.replace_key is not None:
                key = self.replace_key(item)
                for i, (_, existing) in enumerate(self._items):
                    if self.replace_key(existing) == key:
                        self._items[i] = (time.monotonic_ns(), item)
                        self._replaced_count += 1
                        self._cond.notify_all()
                        return True

            if len(self._items) >= self.maxsize:
                if self.drop_oldest_on_full:
                    if self._items:
                        self._items.pop(0)
                        self._dropped_count += 1
                else:
                    return False  # Overload, rejected

            self._items.append((time.monotonic_ns(), item))
            self._cond.notify()
            return True

    async def get(self) -> T:
        """Get the next item from the queue."""
        async with self._cond:
            while not self._items:
                await self._cond.wait()
            return self._items.pop(0)[1]

    def get_nowait(self) -> Optional[T]:
        """Non-blocking get. Returns None if empty."""
        if not self._items:
            return None
        return self._items.pop(0)[1]

    async def clear(self) -> None:
        async with self._cond:
            self._items.clear()
