"""Non-blocking asynchronous SQLite persistence worker."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from typing import List, Optional

from voice_translator.config.models import PersistenceConfig
from voice_translator.core.bounded_queue import BoundedQueue
from voice_translator.core.types import LatencyEvent, UtteranceEvent
from voice_translator.persistence.schema import initialize_database

logger = logging.getLogger(__name__)


class PersistenceWorker:
    """Async background database writer with bounded batching."""

    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.queue: BoundedQueue[dict] = BoundedQueue(maxsize=100, drop_oldest_on_full=True)
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._conn: Optional[sqlite3.Connection] = None

    async def start(self):
        if not self.config.enabled:
            logger.info("Meeting persistence is disabled in configuration.")
            return

        self._conn = initialize_database(self.config.database_path)
        self.is_running = True
        self._task = asyncio.create_task(self._worker_loop())
        logger.info(f"PersistenceWorker started. Database: {self.config.database_path}")

    def record_utterance(self, event: UtteranceEvent):
        if not self.config.enabled or not self.is_running:
            return
        asyncio.run_coroutine_threadsafe(
            self.queue.put({
                "type": "utterance",
                "id": event.utterance_id,
                "meeting_id": event.meeting_id,
                "direction": event.direction.value,
                "sequence": event.sequence_id,
                "source_language": event.source_language,
                "text": event.text,
                "state": event.state.value,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }),
            asyncio.get_event_loop(),
        )

    def record_latency(self, event: LatencyEvent):
        if not self.config.enabled or not self.is_running:
            return
        asyncio.run_coroutine_threadsafe(
            self.queue.put({
                "type": "latency",
                "meeting_id": event.meeting_id,
                "utterance_id": event.utterance_id,
                "direction": event.direction.value if event.direction else None,
                "event_type": event.event_type,
                "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "monotonic_ns": event.monotonic_ns,
                "duration_ms": event.duration_ms,
                "queue_age_ms": event.queue_age_ms,
                "metadata_json": json.dumps(event.metadata),
            }),
            asyncio.get_event_loop(),
        )

    async def _worker_loop(self):
        batch: List[dict] = []
        last_flush = time.monotonic()

        while self.is_running:
            try:
                item = self.queue.get_nowait()
                if item:
                    batch.append(item)

                now = time.monotonic()
                if batch and (len(batch) >= self.config.batch_size or (now - last_flush) >= self.config.flush_interval_sec):
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = now

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Persistence worker error: {e}")

        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[dict]):
        if not self._conn or not batch:
            return
        try:
            with self._conn:
                for item in batch:
                    if item["type"] == "utterance":
                        self._conn.execute(
                            """
                            INSERT INTO utterances (id, meeting_id, direction, sequence, source_language, text, state, started_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(meeting_id, direction, sequence) DO UPDATE SET
                                text = excluded.text,
                                state = excluded.state,
                                updated_at = excluded.updated_at
                            """,
                            (
                                item["id"],
                                item["meeting_id"],
                                item["direction"],
                                item["sequence"],
                                item["source_language"],
                                item["text"],
                                item["state"],
                                item["timestamp"],
                                item["timestamp"],
                            ),
                        )
                    elif item["type"] == "latency":
                        self._conn.execute(
                            """
                            INSERT INTO latency_events (meeting_id, utterance_id, direction, event_type, occurred_at, monotonic_ns, duration_ms, queue_age_ms, metadata_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                item["meeting_id"],
                                item["utterance_id"],
                                item["direction"],
                                item["event_type"],
                                item["occurred_at"],
                                item["monotonic_ns"],
                                item["duration_ms"],
                                item["queue_age_ms"],
                                item["metadata_json"],
                            ),
                        )
        except Exception as e:
            logger.error(f"Failed to execute database batch insert: {e}")

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            self._task = None
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        logger.info("PersistenceWorker stopped.")

