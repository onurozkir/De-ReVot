"""Offline Silero VAD with hysteresis, confirmation frames, and speech evidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from voice_translator.audio.signal import signal_levels

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from silero_vad import load_silero_vad
except ImportError:
    load_silero_vad = None  # type: ignore


@dataclass(slots=True)
class VADResult:
    active: bool
    probability: float
    phase: str
    transition: Optional[str]
    rms: float
    peak: float
    dbfs: float
    utterance_ms: float
    voiced_ms: float
    voiced_ratio: float
    model_backend: str
    silence_ms: float = 0.0


class SileroVAD:
    """Stateful 16 kHz VAD. Model chunks never mix with per-frame energy decisions."""

    CHUNK_SAMPLES = 512

    def __init__(
        self,
        threshold: float = 0.5,
        end_threshold: float = 0.35,
        min_speech_duration_ms: int = 160,
        min_silence_duration_ms: int = 320,
        start_confirm_frames: int = 3,
        end_confirm_frames: int = 6,
        hangover_ms: int = 80,
        energy_start_dbfs: float = -40.0,
        energy_end_dbfs: float = -46.0,
        sample_rate: int = 16000,
        model: Any = None,
        load_model: bool = True,
    ):
        if end_threshold >= threshold:
            raise ValueError("VAD end_threshold must be lower than threshold for hysteresis.")
        self.threshold = float(threshold)
        self.end_threshold = float(end_threshold)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.start_confirm_frames = max(1, int(start_confirm_frames))
        self.end_confirm_frames = max(1, int(end_confirm_frames))
        self.hangover_ms = max(0, int(hangover_ms))
        self.energy_start_dbfs = float(energy_start_dbfs)
        self.energy_end_dbfs = float(energy_end_dbfs)
        self.sample_rate = int(sample_rate)

        self._model = model
        self._buffer = np.empty(0, dtype=np.float32)
        self._last_probability = 0.0
        self.is_speech_active = False
        self.phase = "idle"
        self._start_frames = 0
        self._end_frames = 0
        self._candidate_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        self._voiced_ms = 0.0
        self._last_result = VADResult(False, 0.0, "idle", None, 0.0, 0.0, -120.0, 0.0, 0.0, 0.0, "energy")
        if self._model is None and load_model:
            self._init_model()

    @property
    def backend(self) -> str:
        return "silero" if self._model is not None and torch is not None else "energy"

    def _init_model(self) -> None:
        if torch is None or load_silero_vad is None:
            logger.warning("Silero VAD package unavailable; using deterministic energy fallback.")
            return
        try:
            self._model = load_silero_vad(onnx=False)
            logger.info("Silero VAD initialized from the installed offline package.")
        except Exception as exc:
            logger.warning("Silero VAD local load failed; using energy fallback: %s", exc)

    def process(self, audio_frame_16k: np.ndarray, timestamp_ms: float | None = None) -> VADResult:
        del timestamp_ms
        audio = np.asarray(audio_frame_16k, dtype=np.float32).reshape(-1)
        rms, peak, dbfs = signal_levels(audio)
        if len(audio) == 0:
            return self._last_result

        transition: Optional[str] = None
        if self.backend == "silero":
            self._buffer = np.concatenate((self._buffer, audio))
            while len(self._buffer) >= self.CHUNK_SAMPLES:
                chunk = self._buffer[: self.CHUNK_SAMPLES]
                self._buffer = self._buffer[self.CHUNK_SAMPLES :]
                probability = self._model_probability(chunk)
                maybe_transition = self._advance(probability, self.CHUNK_SAMPLES / self.sample_rate * 1000.0)
                transition = transition or maybe_transition
                self._last_probability = probability
        else:
            probability = self._energy_probability(dbfs)
            transition = self._advance(probability, len(audio) / self.sample_rate * 1000.0)
            self._last_probability = probability

        ratio = self._voiced_ms / self._utterance_ms if self._utterance_ms > 0 else 0.0
        result = VADResult(
            active=self.is_speech_active,
            probability=self._last_probability,
            phase=self.phase,
            transition=transition,
            rms=rms,
            peak=peak,
            dbfs=dbfs,
            utterance_ms=self._utterance_ms,
            voiced_ms=self._voiced_ms,
            voiced_ratio=ratio,
            model_backend=self.backend,
            silence_ms=self._silence_ms,
        )
        self._last_result = result
        return result

    def is_speech(self, audio_frame_16k: np.ndarray, timestamp_ms: float) -> bool:
        return self.process(audio_frame_16k, timestamp_ms).active

    def _model_probability(self, chunk: np.ndarray) -> float:
        try:
            tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            with torch.inference_mode():
                return float(self._model(tensor, self.sample_rate).item())
        except Exception as exc:
            logger.warning("Silero VAD inference failed; switching to energy fallback: %s", exc)
            self._model = None
            _, _, dbfs = signal_levels(chunk)
            return self._energy_probability(dbfs)

    def _energy_probability(self, dbfs: float) -> float:
        low = self.energy_end_dbfs
        high = self.energy_start_dbfs
        if dbfs <= low:
            return 0.0
        if dbfs >= high:
            return 1.0
        return float((dbfs - low) / max(high - low, 1e-6))

    def _advance(self, probability: float, duration_ms: float) -> Optional[str]:
        if not self.is_speech_active:
            if probability >= self.threshold:
                self._start_frames += 1
                self._candidate_ms += duration_ms
                self._utterance_ms += duration_ms
                self._voiced_ms += duration_ms
                self.phase = "start_confirm"
                if self._start_frames >= self.start_confirm_frames and self._candidate_ms >= self.min_speech_duration_ms:
                    self.is_speech_active = True
                    self.phase = "speech"
                    self._end_frames = 0
                    self._silence_ms = 0.0
                    return "started"
            else:
                self._reset_utterance_counters()
                self.phase = "idle"
            return None

        self._utterance_ms += duration_ms
        if probability >= self.end_threshold:
            self._voiced_ms += duration_ms
            self._end_frames = 0
            self._silence_ms = 0.0
            self.phase = "speech"
            return None

        self._end_frames += 1
        self._silence_ms += duration_ms
        self.phase = "end_confirm"
        required_silence = self.min_silence_duration_ms + self.hangover_ms
        if self._end_frames >= self.end_confirm_frames and self._silence_ms >= required_silence:
            self.is_speech_active = False
            self.phase = "ended"
            return "ended"
        return None

    def _reset_utterance_counters(self) -> None:
        self._start_frames = 0
        self._end_frames = 0
        self._candidate_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        self._voiced_ms = 0.0

    def reset(self) -> None:
        self.is_speech_active = False
        self.phase = "idle"
        self._buffer = np.empty(0, dtype=np.float32)
        self._last_probability = 0.0
        self._reset_utterance_counters()
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

