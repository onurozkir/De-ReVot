"""Explicit offline XTTS comparison; no microphone, playback or model downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soundfile as sf
from voice_translator.tts.xtts_backend import XTTSv2Adapter, torch
from voice_translator.tts.conditioning import VoiceProfileManager


TEXTS = [
    "Hello, I am testing the voice translator.",
    "We will compare three different solutions and review twenty four files before the meeting.",
    "I am testing push to talk in meetings and games. My sentences should stay together. Please translate the complete recording after I release the key.",
]


def distribution(values):
    return {"status": "MEASURED", "count": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))} if values else {"status": "UNKNOWN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/tts/xtts-v2")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20 or any(not 2 <= n <= 40 for n in args.chunk_sizes):
        parser.error("repetitions must be 1–20 and chunk sizes 2–40")
    profiles = VoiceProfileManager(str(ROOT / "voices"))
    profile = profiles.profiles.get(args.profile)
    if profile is None:
        parser.error(f"Local voice profile missing: {args.profile}")
    if not torch.cuda.is_available():
        parser.error("This target-hardware benchmark requires CUDA")
    report = {
        "scope": "offline TTS only; no ASR/MT or physical playback",
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
        "packages": {p: importlib.metadata.version(p) for p in ["coqui-tts", "torch", "transformers"]},
        "model": str(args.model.resolve()), "profile": args.profile,
        "reference_hash": VoiceProfileManager.compute_audio_hash(profile.all_reference_paths),
        "corpus": TEXTS, "seed": 42,
        "ttfa_target_ms": {"status": "TARGET", "value": 300},
        "live_queue_age_ms": "UNKNOWN", "full_duplex_soak": "UNKNOWN",
        "speaker_similarity": "UNKNOWN", "human_listening_acceptance": "UNKNOWN", "runs": [],
        "chatterbox": {"status": "UNKNOWN", "eligible": False,
            "reason": "Local adapter is a silence stub; model weights absent. Upstream 0.1.7 pins torch/torchaudio 2.6.0 and numpy<2 on Python 3.12, conflicting with this CUDA 12.8/Torch>=2.7/numpy>=2.1 runtime. No performance ranking is claimed.",
            "source": "https://github.com/resemble-ai/chatterbox/blob/master/pyproject.toml"},
    }
    manifest = args.model / "download-manifest.json"
    if manifest.is_file():
        report["model_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    adapter = XTTSv2Adapter(stream_chunk_size=args.chunk_sizes[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        start = time.perf_counter()
        adapter.initialize(str(args.model.resolve()))
        adapter.prepare_voice_profile(profile)
        adapter.warmup()
        report["load_conditioning_warmup_ms"] = (time.perf_counter() - start) * 1000
        key = adapter._prepared_profiles[(profile.id, tuple(profile.all_reference_paths))]
        cond, speaker = adapter._latents_cache[key]
        for mode, size in [("pseudo_streaming", 0), *[("true_streaming", n) for n in args.chunk_sizes]]:
            if size:
                adapter.stream_chunk_size = size
                adapter.warmup()
            else:
                adapter.model.inference(text="Test warmup.", language="en", gpt_cond_latent=cond,
                                        speaker_embedding=speaker, temperature=adapter.temperature,
                                        speed=adapter.speed, top_p=adapter.top_p,
                                        repetition_penalty=adapter.repetition_penalty, enable_text_splitting=False)
            runs = []
            for text_index, text in enumerate(TEXTS):
                for repetition in range(args.repetitions):
                    torch.manual_seed(42)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    pcm, arrivals = [], []
                    if mode == "pseudo_streaming":
                        out = adapter.model.inference(text=text, language="en", gpt_cond_latent=cond,
                            speaker_embedding=speaker, temperature=adapter.temperature, speed=adapter.speed,
                            top_p=adapter.top_p, repetition_penalty=adapter.repetition_penalty, enable_text_splitting=False)
                        wave = np.asarray(out["wav"], np.float32)
                        peak = float(np.max(np.abs(wave)))
                        pcm = [wave * (.89125 / peak) if peak > 1e-6 else wave]
                        arrivals.append((time.perf_counter() - start) * 1000)
                    else:
                        for chunk in adapter.synthesize_committed(text, profile, "en"):
                            arrivals.append((time.perf_counter() - start) * 1000)
                            pcm.append(chunk)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if not pcm:
                        raise RuntimeError(f"No PCM for {mode}")
                    wave = np.concatenate(pcm)
                    if not np.isfinite(wave).all():
                        raise RuntimeError("Non-finite benchmark PCM")
                    duration = len(wave) / adapter.sample_rate
                    run = {"text_index": text_index, "repetition": repetition, "ttfa_ms": arrivals[0],
                        "total_ms": elapsed * 1000, "audio_seconds": duration, "rtf": elapsed / duration,
                        "chunk_samples": [len(p) for p in pcm], "arrivals_ms": arrivals,
                        "pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                        "peak": float(np.max(np.abs(wave))), "rms": float(np.sqrt(np.mean(wave ** 2))),
                        "peak_torch_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
                    runs.append(run)
                    if args.save_audio and repetition == 0:
                        path = args.output.parent / f"{mode}-{size}-{text_index}.wav"
                        sf.write(path, wave, adapter.sample_rate)
            result = {"mode": mode, "stream_chunk_size": size, "samples": runs,
                "ttfa_ms": distribution([r["ttfa_ms"] for r in runs]),
                "rtf": distribution([r["rtf"] for r in runs]),
                "peak_torch_allocated_mb": max(r["peak_torch_allocated_mb"] for r in runs)}
            report["runs"].append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        adapter.shutdown()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
