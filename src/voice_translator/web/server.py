"""FastAPI Server for Voice Translator Web UI."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from voice_translator.streaming.orchestrator import MeetingOrchestrator
from voice_translator.web.routes import create_routes
from voice_translator.web.websocket import WebSocketConnectionManager


def create_app(orchestrator: MeetingOrchestrator) -> FastAPI:
    app = FastAPI(title="De-ReVot", version="0.1.0")

    ws_manager = WebSocketConnectionManager()

    # Subscribe orchestrator events to WebSocket broadcast
    def _broadcast_to_ws(event: dict):
        ws_manager.publish(event)

    orchestrator.subscribe_events(_broadcast_to_ws)

    # Register REST API
    api_router = create_routes(orchestrator)
    app.include_router(api_router)

    # Static assets directory
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        # Send initial status
        await websocket.send_json({
            "type": "status_change",
            "status": orchestrator.status.value,
            "meeting_id": orchestrator.current_meeting_id,
            "error": orchestrator.last_start_error,
            "controls": orchestrator.controls_snapshot(),
        })
        try:
            while True:
                data = await websocket.receive_text()
                # Handle client ping if needed
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    @app.get("/")
    async def serve_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"status": "ok", "message": "De-ReVot API is running."}

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
