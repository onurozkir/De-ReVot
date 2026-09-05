"""Capture raw physical-microphone PCM before VAD/Whisper and save diagnostic WAV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from voice_translator.audio.diagnostic import capture_endpoint, frame_level_summary, write_pcm16_wav
from voice_translator.audio.devices import AudioDeviceManager
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
    args = parser.parse_args()
    output = args.output or Path("recordings/diagnostics") / f"microphone-{datetime.now():%Y%m%d-%H%M%S}.wav"

    manager = AudioDeviceManager()
    try:
        configured = args.device or load_config().audio.mic_device_id
        device = manager.resolve_required(configured, "mic") if configured else manager.find_default_mic()
        if device is None:
            print("ERROR: no physical microphone endpoint found.")
            return 2
        print("SELECTED_ENDPOINT=" + json.dumps(device.to_dict(), ensure_ascii=False))
        print(f"Speak continuously after {args.prepare_seconds:.1f}s preparation delay.")
        time.sleep(max(0.0, args.prepare_seconds))
        pcm, diagnostics = capture_endpoint(device, args.duration)
        levels = frame_level_summary(pcm, active_rms_threshold=args.min_rms)
        diagnostics["recording"]["frame_levels"] = levels
        write_pcm16_wav(output, pcm)
        print("RESULT=" + json.dumps(diagnostics, ensure_ascii=False))
        print(f"WAV={output.resolve()}")
        if args.require_signal and levels["active_duration_ms"] < args.min_active_ms:
            print(
                "FAIL: stream opened, but usable mic signal was too short/quiet. "
                f"Need >= {args.min_active_ms}ms above RMS {args.min_rms}; "
                f"measured {levels['active_duration_ms']}ms, max frame RMS {levels['max_frame_rms']:.6f}."
            )
            print("Check Windows Input device, mute switch, Microphone volume/gain, then speak during the capture window.")
            return 3
        print("PASS: raw microphone capture completed; inspect WAV and metrics.")
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
