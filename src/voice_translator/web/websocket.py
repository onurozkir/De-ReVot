"""WebSocket manager for streaming events to web UI."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Manages active WebSocket connections to the local UI."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        self._loop = asyncio.get_running_loop()
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.debug("WebSocket client connected.")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.debug("WebSocket client disconnected.")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        dead = []
        payload = json.dumps(message)
        for ws in self.active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections.discard(ws)

    def publish(self, message: dict) -> None:
        """Thread-safe bridge for events produced by audio/model worker threads."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(lambda: asyncio.create_task(self.broadcast(message)))
