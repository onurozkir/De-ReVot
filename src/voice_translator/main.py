"""Main CLI entrypoint for Realtime Voice Translator."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import webbrowser
from pathlib import Path

# Ensure src directory is in sys.path for direct script invocation
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn
from voice_translator.audio.devices import AudioDeviceManager
from voice_translator.config.loader import load_config
from voice_translator.streaming.orchestrator import MeetingOrchestrator
from voice_translator.web.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice_translator")


def list_devices():
    mgr = AudioDeviceManager()
    devices = mgr.list_devices(wasapi_only=False)
    print("\n--- Audio Devices (WASAPI / Windows) ---")
    for d in devices:
        kind = []
        if d.is_input:
            kind.append("INPUT")
        if d.is_output:
            kind.append("OUTPUT")
        if d.is_loopback:
            kind.append("LOOPBACK")
        print(
            f"[{d.index:2d}] {d.name} | stable_id={d.stable_id} | API: {d.host_api_name} "
            f"(hostApi={d.host_api}) | Rate: {d.default_sample_rate}Hz | "
            f"in={d.max_input_channels} out={d.max_output_channels} | {'/'.join(kind)}"
        )
    
    print("\n--- Detected Defaults ---")
    def_mic = mgr.find_default_mic()
    def_loop = mgr.find_default_loopback()
    def_cable = mgr.find_vbcable_render()
    def_cable_capture = mgr.find_vbcable_capture()
    def_speaker = mgr.find_render_for_loopback(def_loop)
    print(f"Default Mic:      {def_mic.name if def_mic else 'None'} [Index {def_mic.index if def_mic else 'N/A'}]")
    print(f"Default Loopback: {def_loop.name if def_loop else 'None'} [Index {def_loop.index if def_loop else 'N/A'}]")
    print(f"VB-CABLE Render:  {def_cable.name if def_cable else 'None'} [Index {def_cable.index if def_cable else 'N/A'}]")
    print(f"Physical Speaker: {def_speaker.name if def_speaker else 'Unresolved'} [Index {def_speaker.index if def_speaker else 'N/A'}]")
    print(f"VB-CABLE Capture: {def_cable_capture.name if def_cable_capture else 'None'} [Index {def_cable_capture.index if def_cable_capture else 'N/A'}]")
    mgr.close()


def run_server(mock: bool = False, port: int = 8000, host: str = "127.0.0.1"):
    config = load_config()
    orchestrator = MeetingOrchestrator(config=config, use_mocks=mock)

    async def _startup():
        try:
            await orchestrator.initialize_and_warmup()
        except Exception as e:
            logger.error(f"Orchestrator warmup error: {e}")
            logger.info("Server will start in DEGRADED/STOPPED state.")

    app = create_app(orchestrator)

    @app.on_event("startup")
    async def on_startup():
        asyncio.create_task(_startup())
        if config.server.open_browser and not mock:
            webbrowser.open(f"http://{host}:{port}")

    @app.on_event("shutdown")
    async def on_shutdown():
        await orchestrator.shutdown()

    print(f"\n========================================================")
    print(f" Starting Realtime Voice Translator on http://{host}:{port}")
    print(f" Mock Mode: {mock}")
    print(f"========================================================\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Realtime Voice Translator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Start the Web UI translator server")
    run_parser.add_argument("--mock", action="store_true", help="Run with mock ASR/MT/TTS (no GPU/models needed)")
    run_parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    run_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")

    # devices command
    subparsers.add_parser("devices", help="List audio input, loopback, and VB-CABLE devices")

    args = parser.parse_args()

    if args.command == "devices":
        list_devices()
    elif args.command == "run" or args.command is None:
        mock = getattr(args, "mock", False)
        port = getattr(args, "port", 8000)
        host = getattr(args, "host", "127.0.0.1")
        run_server(mock=mock, port=port, host=host)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
