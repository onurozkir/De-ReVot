"""Capture a selected WASAPI speaker-loopback endpoint and save diagnostic WAV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np

from voice_translator.audio.diagnostic import (
    capture_endpoint,
    dominant_frequency,
    frame_level_summary,
    write_pcm16_wav,
)
from voice_translator.audio.devices import AudioDeviceManager
from voice_translator.audio.render import AudioRenderEngine
from voice_translator.config.loader import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="stable_id, exact name, or current PyAudio index")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-signal", action="store_true")
    parser.add_argument("--prepare-seconds", type=float, default=2.0)
    parser.add_argument("--min-rms", type=float, default=0.001)
    parser.add_argument("--min-active-ms", type=int, default=200)
    parser.add_argument("--test-tone", action="store_true", help="play a known tone through the matching speaker")
    parser.add_argument("--render-device", help="speaker stable_id/name/index used with --test-tone")
    parser.add_argument("--frequency", type=float, default=440.0)
    args = parser.parse_args()
    output = args.output or Path("recordings/diagnostics") / f"loopback-{datetime.now():%Y%m%d-%H%M%S}.wav"

    manager = AudioDeviceManager()
    try:
        configured = args.device or load_config().audio.loopback_device_id
        device = manager.resolve_required(configured, "loopback") if configured else manager.find_default_loopback()
        if device is None:
            print("ERROR: no WASAPI loopback endpoint found.")
            return 2
        print("SELECTED_ENDPOINT=" + json.dumps(device.to_dict(), ensure_ascii=False))
        renderer = None
        on_started = None
        if args.test_tone:
            render_device = (
                manager.resolve_required(args.render_device, "render")
                if args.render_device else manager.find_render_for_loopback(device)
            )
            if render_device is None:
                print("ERROR: matching physical speaker could not be resolved; pass --render-device.")
                return 4
            print("RENDER_ENDPOINT=" + json.dumps(render_device.to_dict(), ensure_ascii=False))
            renderer = AudioRenderEngine(render_device, sample_rate=48000)
            renderer.start()
            tone_duration = max(0.5, args.duration - 0.5)
            timeline = np.arange(int(48000 * tone_duration), dtype=np.float32) / 48000.0
            tone = (0.05 * np.sin(2.0 * np.pi * args.frequency * timeline)).astype(np.float32)
            on_started = lambda: renderer.push_pcm(tone, source_rate=48000)
        else:
            print("Play Teams/test audio through the matching physical speaker during capture.")
        print(f"Capture starts after {args.prepare_seconds:.1f}s.")
        time.sleep(max(0.0, args.prepare_seconds))
        try:
            pcm, diagnostics = capture_endpoint(device, args.duration, on_started=on_started)
        finally:
            if renderer is not None:
                renderer.stop()
        levels = frame_level_summary(pcm, active_rms_threshold=args.min_rms)
        diagnostics["recording"]["frame_levels"] = levels
        diagnostics["recording"]["dominant_hz"] = dominant_frequency(pcm)
        write_pcm16_wav(output, pcm)
        print("RESULT=" + json.dumps(diagnostics, ensure_ascii=False))
        print(f"WAV={output.resolve()}")
        recording = diagnostics["recording"]
        if args.require_signal and levels["active_duration_ms"] < args.min_active_ms:
            print(
                "FAIL: loopback stream had insufficient active PCM. "
                f"callbacks={diagnostics['callback_count']}, samples={recording['samples']}, "
                f"active={levels['active_duration_ms']}ms. Confirm Teams speaker endpoint or use --test-tone."
            )
            return 3
        if args.test_tone and abs(recording["dominant_hz"] - args.frequency) > 10.0:
            print("FAIL: captured signal does not contain the expected test-tone frequency.")
            return 5
        print("PASS: loopback capture completed; inspect WAV and metrics.")
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
