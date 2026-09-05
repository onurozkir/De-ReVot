"""B2 shared-ASR contention benchmark for sequential and simultaneous TX/RX."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_translator.asr.whisper_backend import WhisperASRAdapter
from voice_translator.core.types import Direction


def _percentiles(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def _decode_once(
    adapter: WhisperASRAdapter,
    audio: np.ndarray,
    direction: Direction,
    language: str,
    barrier: threading.Barrier | None = None,
) -> dict[str, float]:
    if barrier is not None:
        barrier.wait()
    audio_end_ns = time.monotonic_ns()
    started_ns = time.monotonic_ns()
    _, model_info = adapter._decode_audio(
        audio,
        language,
        direction=direction,
        is_final=False,
        audio_end_ns=audio_end_ns,
    )
    finished_ns = time.monotonic_ns()
    return {
        "wall_ms": (finished_ns - started_ns) / 1e6,
        "wait_ms": float(model_info["asr_inference_wait_ms"]),
        "deadline_miss_ms": float(model_info["asr_inference_deadline_miss_ms"]),
        "queue_depth": float(model_info["asr_inference_queue_depth"]),
    }


def _print_summary(mode: str, direction: str, samples: list[dict[str, float]]) -> None:
    wall_p50, wall_p95 = _percentiles([sample["wall_ms"] for sample in samples])
    wait_p50, wait_p95 = _percentiles([sample["wait_ms"] for sample in samples])
    miss_p50, miss_p95 = _percentiles([sample["deadline_miss_ms"] for sample in samples])
    max_depth = max(sample["queue_depth"] for sample in samples)
    print(
        f"{mode} {direction}: n={len(samples)}, "
        f"decode wall P50/P95={wall_p50:.2f}/{wall_p95:.2f} ms, "
        f"scheduler wait P50/P95={wait_p50:.2f}/{wait_p95:.2f} ms, "
        f"deadline miss P50/P95={miss_p50:.2f}/{miss_p95:.2f} ms, "
        f"max queue depth={max_depth:.0f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/asr/whisper-large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(
            f"Model path '{model_path}' does not exist. "
            "Download model weights manually before running this benchmark."
        )
        return 1
    if args.iterations < 1 or args.duration_sec <= 0:
        parser.error("--iterations and --duration-sec must be greater than zero")

    sample_count = int(16000 * args.duration_sec)
    timeline = np.arange(sample_count, dtype=np.float32) / 16000.0
    rng = np.random.default_rng(42)
    audio = (
        0.08 * np.sin(2 * np.pi * 220.0 * timeline)
        + 0.01 * rng.standard_normal(sample_count)
    ).astype(np.float32)

    adapter = WhisperASRAdapter()
    print("=" * 72)
    print("B2 ASR SHARED-WEIGHT CONTENTION BENCHMARK")
    print("=" * 72)
    adapter.initialize(
        model_path=str(model_path),
        device=args.device,
        compute_type=args.compute_type,
    )
    adapter.warmup()

    sequential = {"outgoing": [], "incoming": []}
    simultaneous = {"outgoing": [], "incoming": []}
    pair_elapsed_ms: list[float] = []
    try:
        for _ in range(args.iterations):
            sequential["outgoing"].append(
                _decode_once(adapter, audio, Direction.OUTGOING, "tr")
            )
            sequential["incoming"].append(
                _decode_once(adapter, audio, Direction.INCOMING, "en")
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            for _ in range(args.iterations):
                barrier = threading.Barrier(2)
                pair_started_ns = time.monotonic_ns()
                outgoing = executor.submit(
                    _decode_once,
                    adapter,
                    audio,
                    Direction.OUTGOING,
                    "tr",
                    barrier,
                )
                incoming = executor.submit(
                    _decode_once,
                    adapter,
                    audio,
                    Direction.INCOMING,
                    "en",
                    barrier,
                )
                simultaneous["outgoing"].append(outgoing.result())
                simultaneous["incoming"].append(incoming.result())
                pair_elapsed_ms.append((time.monotonic_ns() - pair_started_ns) / 1e6)
    finally:
        adapter.shutdown()

    print(
        f"Status: MEASURED contention microbenchmark; model={model_path}, "
        f"backend={adapter.backend_type}, device={adapter.device}, "
        f"iterations={args.iterations}, audio={args.duration_sec:.2f} s/direction"
    )
    for mode, measurements in (("Sequential", sequential), ("Simultaneous", simultaneous)):
        _print_summary(mode, "outgoing", measurements["outgoing"])
        _print_summary(mode, "incoming", measurements["incoming"])
    pair_p50, pair_p95 = _percentiles(pair_elapsed_ms)
    print(f"Simultaneous pair elapsed P50/P95={pair_p50:.2f}/{pair_p95:.2f} ms")
    print(
        "Scope: scheduler contention only. WER/CER, VRAM, real audio edges, "
        "30-minute full-duplex soak, and Phase H acceptance remain UNKNOWN."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
