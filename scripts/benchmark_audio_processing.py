"""Offline CPU NS/AEC evidence. Never records devices or downloads models."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_translator.audio.processing import EchoReferenceBuffer, MicrophoneProcessor
from voice_translator.config.models import AudioConfig


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def measure(mic, reference=None, ns=True, rate=48000):
    history = EchoReferenceBuffer(rate, 10)
    config = AudioConfig(sample_rate=rate, frame_duration_ms=10, noise_suppression=ns,
                         echo_cancellation=reference is not None, echo_delay_ms=50)
    processor = MicrophoneProcessor(config, history)
    output = np.zeros_like(mic)
    timings = []
    size = rate // 100
    for offset in range(0, len(mic) - size + 1, size):
        at_ns = round((offset + size) * 1e9 / rate)
        if reference is not None:
            history.push(reference[offset:offset + size], at_ns)
        started = time.perf_counter_ns()
        frames = processor.process(mic[offset:offset + size], at_ns)
        timings.append((time.perf_counter_ns() - started) / 1e6)
        output[offset:offset + size] = frames[0][0]
    # Exclude initial adaptation from signal attenuation, retain all timing.
    start = min(rate * 2, len(mic) // 2)
    return output, dict(status="MEASURED", frames=len(timings),
                        p50_ms=float(np.percentile(timings, 50)), p95_ms=float(np.percentile(timings, 95)),
                        rtf=float(sum(timings) / (len(mic) / rate * 1000)),
                        attenuation_db=float(20 * np.log10((rms(mic[start:]) + 1e-12) / (rms(output[start:]) + 1e-12))),
                        reference_missing_frames=processor.reference_misses,
                        vram_bytes=0, queue_age_ms="UNKNOWN: offline synchronous processing",
                        full_duplex_soak="UNKNOWN: no ASR/MT/TTS or physical audio")


def main():
    parser = argparse.ArgumentParser(description="De-ReVot offline microphone DSP benchmark")
    parser.add_argument("--speech", help="Optional existing local speech WAV; never recorded automatically")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rate, seconds = 48000, 12
    rng = np.random.default_rng(16)
    noise = rng.normal(0, 0.02, rate * seconds).astype(np.float32)
    _, noise_report = measure(noise)
    # Broadband time-varying echo control with known 50ms delay and a short room tail.
    far = rng.normal(0, 0.08, rate * seconds).astype(np.float32)
    delay = rate // 20
    echo = np.zeros_like(far)
    echo[delay:] = 0.6 * far[:-delay]
    echo[delay + 96:] += 0.2 * far[:-delay - 96]
    _, echo_report = measure(echo, far, ns=False)
    report = dict(platform=platform.platform(), package="aec-audio-processing==1.0.1",
                  sample_rate=rate, seconds=seconds, synthetic_noise=noise_report, synthetic_echo=echo_report,
                  outdoor_vehicle_hallucinations="UNKNOWN: requires labeled vehicle/traffic recordings and ASR",
                  speech_quality="UNKNOWN: listener evaluation required")
    if args.speech:
        path = Path(args.speech).resolve()
        speech, source_rate = sf.read(path, dtype="float32", always_2d=True)
        speech = soxr.resample(speech.mean(axis=1), source_rate, rate)
        speech = np.tile(speech, int(np.ceil(len(noise) / max(1, len(speech)))))[:len(noise)].astype(np.float32)
        speech *= 0.05 / max(rms(speech), 1e-8)
        mixed = speech + echo + noise
        enhanced, stats = measure(mixed, far)
        report["speech_corpus_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        stats["input_snr_db"] = float(20 * np.log10(rms(speech) / rms(mixed - speech)))
        # WebRTC filtering introduces phase/group delay, so do not label raw
        # sample-wise output error as perceptual SNR improvement.
        report["synthetic_double_talk_with_local_speech"] = stats
        destination = Path(args.output).with_suffix(".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, enhanced, rate)
        report["listening_output"] = str(destination)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
