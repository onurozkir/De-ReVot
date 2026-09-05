"""Unit tests for BoundedQueue policies."""

import asyncio
from voice_translator.core.bounded_queue import BoundedQueue


def test_bounded_queue_basic():
    async def _run():
        q: BoundedQueue[str] = BoundedQueue(maxsize=2)
        assert q.is_empty()

        assert await q.put("a")
        assert not q.is_empty()
        assert q.qsize() == 1

        assert await q.put("b")
        assert q.is_full()

        # Overload: rejects when drop_oldest is False
        assert not await q.put("c")
        assert q.qsize() == 2

        item = await q.get()
        assert item == "a"
        assert q.qsize() == 1

    asyncio.run(_run())


def test_bounded_queue_reports_oldest_age_and_policy_snapshot(monkeypatch):
    ticks = iter((1_000_000_000, 1_005_000_000))
    monkeypatch.setattr("voice_translator.core.bounded_queue.time.monotonic_ns", lambda: next(ticks))

    async def _run():
        q: BoundedQueue[str] = BoundedQueue(maxsize=2)
        await q.put("a")
        snapshot = q.snapshot("committed", "reject_new")
        assert snapshot["current"] == 1
        assert snapshot["max"] == 2
        assert snapshot["oldest_age_ms"] == 5.0
        assert snapshot["drop_policy"] == "reject_new"

    asyncio.run(_run())


def test_bounded_queue_drop_oldest():
    async def _run():
        q: BoundedQueue[str] = BoundedQueue(maxsize=2, drop_oldest_on_full=True)
        assert await q.put("a")
        assert await q.put("b")
        assert await q.put("c")  # Drops 'a'

        assert q.dropped_count == 1
        assert q.qsize() == 2

        item = await q.get()
        assert item == "b"

    asyncio.run(_run())


def test_bounded_queue_replace_key():
    async def _run():
        # Key on username before colon
        q: BoundedQueue[str] = BoundedQueue(maxsize=3, replace_key=lambda s: s.split(":")[0])
        assert await q.put("user1: partial 1")
        assert await q.put("user2: partial 1")
        assert await q.put("user1: partial 2")  # Replaces user1 in-place

        assert q.replaced_count == 1
        assert q.qsize() == 2

        first = await q.get()
        assert first == "user1: partial 2"
        second = await q.get()
        assert second == "user2: partial 1"

    asyncio.run(_run())
