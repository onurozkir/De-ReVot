"""Explicit offline ASR comparison. No microphone capture or weight downloads.

Examples:
  uv run python scripts/benchmark_asr_acoustics.py --model large_v3 --audio voices/onur-default/reference.wav
  uv run python scripts/benchmark_asr_acoustics.py --model turbo --corpus tests/fixtures/asr-corpus.example.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soundfile as sf
import soxr

from voice_translator.asr.whisper_backend import WhisperASRAdapter
from voice_translator.config.loader import load_config
from voice_translator.config.presets import ASR_PRESETS
from voice_translator.core.types import Direction
from voice_translator.streaming.hallucination_guard import SpeechEvidence
from voice_translator.streaming.pipeline_runtime import build_guard, build_vad


def distribution(values):
    return {"status": "MEASURED", "count": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))} if values else {"status": "UNKNOWN"}


def gpu_memory():
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {"status": "MEASURED", "scope": "whole GPU, includes other processes",
                    "name": pynvml.nvmlDeviceGetName(handle), "used_mb": memory.used / 2**20,
                    "total_mb": memory.total / 2**20}
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        return {"status": "UNKNOWN", "reason": str(exc)}


def synthetic_cases():
    silence = np.zeros(16000, dtype=np.float32)
    clicks = silence.copy()
    for index in (3200, 8000, 12800):
        clicks[index:index + 80] = 0.15 * np.hanning(80)
    noise = np.random.default_rng(26).normal(0, 0.004, 16000).astype(np.float32)
    return [("synthetic_silence", silence, "tr", "", "non_speech"),
            ("synthetic_clicks", clicks, "tr", "", "non_speech"),
            ("synthetic_noise", noise, "tr", "", "non_speech")]


def read_audio(path: Path, seconds: float = 30, offset: float = 0):
    with sf.SoundFile(path) as stream:
        rate = stream.samplerate
        stream.seek(min(len(stream), round(offset * rate)))
        audio = stream.read(frames=round(seconds * rate), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        audio = soxr.resample(audio, rate, 16000).astype(np.float32)
    if not len(audio) or len(audio) > 16000 * 30:
        raise ValueError(f"Benchmark clips must contain 0–30 seconds of audio: {path}")
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["configured", *ASR_PRESETS], default="configured")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--audio-seconds", type=float, default=4, help="Explicit bounded reference excerpt, max 30 seconds")
    parser.add_argument("--audio-offset", type=float, default=0)
    parser.add_argument("--language", default="tr")
    parser.add_argument("--corpus", type=Path, help="JSON array: path, language, text (optional), kind=speech/non_speech")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 100:
        parser.error("repetitions must be between 1 and 100")
    if not 0 < args.audio_seconds <= 30 or args.audio_offset < 0:
        parser.error("audio-seconds must be in (0, 30]; audio-offset must be nonnegative")

    config = load_config(project_root=str(ROOT))
    if args.model != "configured":
        for key in ("backend", "model_path", "compute_type"):
            setattr(config.asr, key, ASR_PRESETS[args.model][key])
    model_path = Path(config.asr.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    adapter = WhisperASRAdapter(partial_interval_ms=config.asr.partial_interval_ms,
                                min_audio_rms=config.asr.min_audio_rms, beam_size=config.asr.beam_size)
    cases = synthetic_cases()
    if args.audio:
        cases.append((f"{args.audio.name}@{args.audio_offset}s+{args.audio_seconds}s", read_audio(args.audio, args.audio_seconds, args.audio_offset), args.language, None, "speech"))
    if args.corpus:
        for item in json.loads(args.corpus.read_text(encoding="utf-8")):
            path = args.corpus.parent / item["path"]
            cases.append((item["path"], read_audio(path), item.get("language", "tr"), item.get("text"), item.get("kind", "speech")))

    report = {"scope": "offline ASR and acoustic guard; not a live call or ASR+MT+TTS soak",
              "platform": platform.platform(), "model_path": str(model_path),
              "revision": "UNKNOWN", "config": config.asr.model_dump(),
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "faster-whisper", "ctranslate2", "transformers")},
              "gpu_before": gpu_memory(), "cases": [], "full_duplex_soak": "UNKNOWN",
              "end_to_end_latency": "UNKNOWN", "live_audio_queue_age": "UNKNOWN",
              "partial_decode_target_ms": {"status": "TARGET", "p50": 250}}
    manifest = model_path / "download-manifest.json"
    if manifest.is_file():
        report["model_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        report["revision"] = report["model_manifest"]["revision"]
    started = time.perf_counter()
    try:
        adapter.initialize(str(model_path), config.asr.device, config.asr.compute_type)
        adapter.warmup()
        report["warmup_ms"] = (time.perf_counter() - started) * 1000
        report["gpu_warm"] = gpu_memory()
        guard = build_guard(config.streaming)
        partial_ms, partial_rtf = [], []
        for name, audio, language, reference, kind in cases:
            vad = build_vad(config.streaming)
            voiced_samples = 0
            for offset in range(0, len(audio), 320):
                frame = audio[offset:offset + 320]
                result = vad.process(frame)
                if result.probability >= vad.end_threshold:
                    voiced_samples += len(frame)
            evidence = SpeechEvidence(len(audio) / 16, voiced_samples / 16, voiced_samples / len(audio))
            timings, outputs = [], []
            for _ in range(args.repetitions):
                start = time.perf_counter()
                text, info = adapter._decode_audio(audio, language, direction=Direction.OUTGOING, is_final=True,
                                                    audio_end_ns=time.monotonic_ns())
                timings.append((time.perf_counter() - start) * 1000)
                decision = guard.evaluate(text, evidence, info)
                outputs.append({"raw_text": text, "accepted": decision.accepted, "reason": decision.reason})
            case = {"name": name, "kind": kind, "audio_seconds": len(audio) / 16000,
                    "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(), "evidence": asdict(evidence),
                    "final_decode_ms": distribution(timings), "rtf": distribution([t / (len(audio) / 16) for t in timings]),
                    "outputs": outputs, "wer": "UNKNOWN", "cer": "UNKNOWN"}
            if reference:
                from jiwer import wer, cer
                case["wer"] = wer(reference, outputs[-1]["raw_text"])
                case["cer"] = cer(reference, outputs[-1]["raw_text"])
            report["cases"].append(case)
            if kind == "speech":
                for size in range(4800, min(len(audio), 64000) + 1, adapter.partial_interval_samples):
                    for _ in range(args.repetitions):
                        start = time.perf_counter()
                        adapter._decode_audio(audio[:size], language, direction=Direction.OUTGOING,
                                              audio_end_ns=time.monotonic_ns())
                        duration = (time.perf_counter() - start) * 1000
                        partial_ms.append(duration)
                        partial_rtf.append(duration / (size / 16))

        report["partial_decode_ms"] = distribution(partial_ms)
        report["partial_rtf"] = distribution(partial_rtf)
        speech = next((case for case in cases if case[-1] == "speech"), None)
        if speech:
            audio = speech[1][:32000]
            barrier = threading.Barrier(2)
            contention = []

            def decode_pair(direction):
                barrier.wait()
                start = time.perf_counter()
                _, info = adapter._decode_audio(audio, speech[2], direction=direction, audio_end_ns=time.monotonic_ns())
                return {"direction": direction.value, "decode_ms": (time.perf_counter() - start) * 1000,
                        "scheduler_wait_ms": info.get("asr_inference_wait_ms", 0),
                        "deadline_miss_ms": info.get("asr_inference_deadline_miss_ms", 0)}

            with ThreadPoolExecutor(max_workers=2) as pool:
                for _ in range(args.repetitions):
                    futures = [pool.submit(decode_pair, direction) for direction in (Direction.OUTGOING, Direction.INCOMING)]
                    contention.extend(f.result() for f in futures)
            report["two_session_asr"] = {"scope": "shared weights; same clip, not full pipeline", "samples": contention,
                "decode_ms": distribution([x["decode_ms"] for x in contention]),
                "scheduler_wait_ms": distribution([x["scheduler_wait_ms"] for x in contention])}
        report["gpu_after"] = gpu_memory()
    finally:
        adapter.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "partial_decode_ms": report["partial_decode_ms"],
                      "gpu_warm": report["gpu_warm"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
