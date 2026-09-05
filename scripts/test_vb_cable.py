"""Render known PCM to a selected VB-CABLE input without ASR, MT, or TTS."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from voice_translator.audio.devices import AudioDeviceManager
from voice_translator.audio.capture import AudioCaptureEngine
from voice_translator.audio.render import AudioRenderEngine
from voice_translator.audio.signal import signal_levels
from voice_translator.config.loader import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="stable_id, exact name, or current PyAudio index")
    parser.add_argument("--capture-device", help="optional CABLE Output stable_id/name/index")
    parser.add_argument("--skip-capture-verification", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--frequency", type=float, default=440.0)
    parser.add_argument("--level", type=float, default=0.15)
    args = parser.parse_args()

    manager = AudioDeviceManager()
    try:
        configured = args.device or load_config().audio.render_device_id
        device = manager.resolve_required(configured, "render") if configured else manager.find_vbcable_render()
        if device is None:
            print("ERROR: VB-CABLE render endpoint not found. Use the UI/API device inventory.")
            return 2
        print("SELECTED_ENDPOINT=" + json.dumps(device.to_dict(), ensure_ascii=False))
        renderer = AudioRenderEngine(device, sample_rate=48000, ring_buffer_sec=max(4.0, args.duration + 1.0))
        capture = None
        captured_chunks: list[np.ndarray] = []
        if not args.skip_capture_verification:
            capture_device = (
                manager.resolve_required(args.capture_device, "vb_capture")
                if args.capture_device else manager.find_vbcable_capture()
            )
            if capture_device is None:
                print("ERROR: CABLE Output capture endpoint not found; use --skip-capture-verification only for render-only testing.")
                return 3
            print("CAPTURE_ENDPOINT=" + json.dumps(capture_device.to_dict(), ensure_ascii=False))
            capture = AudioCaptureEngine(capture_device, sample_rate=48000, ring_buffer_sec=max(4.0, args.duration + 1.0))
        try:
            if capture is not None:
                capture.start()
            renderer.start()
            sample_count = int(48000 * args.duration)
            timeline = np.arange(sample_count, dtype=np.float32) / 48000.0
            tone = (args.level * np.sin(2.0 * np.pi * args.frequency * timeline)).astype(np.float32)
            renderer.push_pcm(tone, source_rate=48000)
            deadline = time.monotonic() + args.duration + 0.25
            while time.monotonic() < deadline:
                if capture is not None:
                    chunk = capture.read_samples(capture.frame_size)
                    if len(chunk):
                        captured_chunks.append(chunk)
                time.sleep(0.005)
        finally:
            renderer.stop()
            if capture is not None:
                tail = capture.read_samples()
                if len(tail):
                    captured_chunks.append(tail)
                capture.stop()
        print("RESULT=" + json.dumps(renderer.get_diagnostics(), ensure_ascii=False))
        if capture is not None:
            captured = np.concatenate(captured_chunks) if captured_chunks else np.empty(0, dtype=np.float32)
            rms, peak, dbfs = signal_levels(captured)
            dominant_hz = 0.0
            if len(captured) >= 4800:
                spectrum = np.abs(np.fft.rfft(captured * np.hanning(len(captured))))
                dominant_hz = float(np.fft.rfftfreq(len(captured), 1.0 / 48000)[int(np.argmax(spectrum))])
            capture_result = capture.get_diagnostics()
            capture_result["recording"] = {
                "samples": len(captured), "rms": rms, "peak": peak,
                "dbfs": dbfs, "dominant_hz": dominant_hz,
            }
            print("CAPTURE_RESULT=" + json.dumps(capture_result, ensure_ascii=False))
            if rms < max(0.001, args.level * 0.1):
                print("FAIL: known PCM did not appear at CABLE Output capture.")
                return 4
            if abs(dominant_hz - args.frequency) > 10.0:
                print("FAIL: CABLE Output signal does not contain the expected tone frequency.")
                return 5
        print("PASS: known PCM was written to the selected render endpoint.")
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
