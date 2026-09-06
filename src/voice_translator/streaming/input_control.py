"""Timestamped, bounded microphone admission. Called outside audio callbacks."""

from collections import deque
from dataclasses import dataclass
from threading import Lock
import time

import numpy as np


@dataclass
class GateAction:
    kind: str
    at_ns: int
    audio: np.ndarray | None = None
    input_mode: str = "vad"


class InputGate:
    """Split captured PCM at key edges; retain exactly the configured pre-roll.

    Key release ends input, but already committed cloned playback must finish.
    Control overload fails closed and is reported, never loses a release silently.
    """

    def __init__(self, sample_rate: int, mode: str = "vad", preroll_ms: int = 150):
        self.sample_rate = sample_rate
        self.mode = mode
        self.pressed = False
        self.paused = False
        self.muted = False
        self._preroll_samples = sample_rate * preroll_ms // 1000
        self._preroll = np.empty(0, dtype=np.float32)
        self._changes = deque()
        self._lock = Lock()
        self._overflow = False

    def change(self, kind: str, value, at_ns: int | None = None) -> None:
        if kind not in ("ptt", "mode", "pause", "mute"):
            raise ValueError(f"Unknown input control: {kind}")
        if kind == "mode" and value not in ("vad", "ptt"):
            raise ValueError(f"Invalid input mode: {value}")
        with self._lock:
            if len(self._changes) >= 64:
                self._changes.clear()
                self._overflow = True
            self._changes.append((at_ns if at_ns is not None else time.monotonic_ns(), kind, value))

    def route(self, audio: np.ndarray, end_ns: int) -> list[GateAction]:
        start_ns = end_ns - round(len(audio) * 1e9 / self.sample_rate)
        with self._lock:
            changes = []
            while self._changes and self._changes[0][0] <= end_ns:
                changes.append(self._changes.popleft())
            overflow, self._overflow = self._overflow, False
        actions = []
        if overflow:
            self.muted = True
            self.pressed = False
            self._preroll = np.empty(0, dtype=np.float32)
            return [GateAction("control_overload", end_ns)]

        cursor = 0
        for at_ns, kind, value in changes:
            index = max(cursor, min(len(audio), round((at_ns - start_ns) * self.sample_rate / 1e9)))
            self._audio(audio[cursor:index], start_ns + round(index * 1e9 / self.sample_rate), actions)
            cursor = index
            if kind == "ptt":
                if bool(value) == self.pressed:
                    continue
                if self.mode == "ptt" and not self.paused and not self.muted:
                    if value:
                        actions.append(GateAction("ptt_press", at_ns))
                        if len(self._preroll):
                            actions.append(GateAction("audio", max(start_ns, at_ns), self._preroll, "ptt"))
                    elif not value:
                        actions.append(GateAction("ptt_release", at_ns))
                self.pressed = bool(value)
            else:
                attr = {"mode": "mode", "pause": "paused", "mute": "muted"}[kind]
                if getattr(self, attr) == value:
                    continue
                setattr(self, attr, value)
                self.pressed = False
                actions.append(GateAction("reset", at_ns))
            self._preroll = np.empty(0, dtype=np.float32)
        self._audio(audio[cursor:], end_ns, actions)
        return actions

    def _audio(self, audio: np.ndarray, end_ns: int, actions: list[GateAction]) -> None:
        if not len(audio):
            return
        if self.paused or self.muted:
            return
        if self.mode == "vad" or self.pressed:
            actions.append(GateAction("audio", end_ns, audio, self.mode))
        elif self._preroll_samples:
            self._preroll = np.concatenate((self._preroll, audio))[-self._preroll_samples:]

    def discard_until(self, end_ns: int) -> list[GateAction]:
        """Apply controls across lost PCM without admitting audio or pre-roll."""
        actions = self.route(np.empty(0, np.float32), end_ns)
        self._preroll = np.empty(0, np.float32)
        return actions
