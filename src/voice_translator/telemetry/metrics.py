"""Latency statistics, rolling percentiles (P50/P95/P99), and backlog monitoring."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional
import numpy as np

from voice_translator.core.types import LatencyEvent


class TelemetryTracker:
    """Calculates rolling P50/P95 latency percentiles and queues metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._samples: Dict[str, deque] = {
            "incoming_partial": deque(maxlen=window_size),
            "incoming_committed": deque(maxlen=window_size),
            "outgoing_pcm": deque(maxlen=window_size),
            "mt_duration": deque(maxlen=window_size),
            "tts_first_pcm": deque(maxlen=window_size),
            "asr_wait_outgoing": deque(maxlen=window_size),
            "asr_wait_incoming": deque(maxlen=window_size),
        }
        self.total_events = 0
        self.underruns = 0
        self.overruns = 0

    def record_event(self, event: LatencyEvent):
        self.total_events += 1
        dur = event.duration_ms
        if dur is None or dur < 0:
            return

        if event.event_type == "asr_inference_wait":
            metric = "asr_wait_outgoing" if event.direction.value == "outgoing" else "asr_wait_incoming"
            self._samples[metric].append(dur)
        elif "incoming_partial" in event.event_type:
            self._samples["incoming_partial"].append(dur)
        elif "incoming_committed" in event.event_type:
            self._samples["incoming_committed"].append(dur)
        elif "tts_first_pcm" in event.event_type:
            self._samples["tts_first_pcm"].append(dur)
            self._samples["outgoing_pcm"].append(dur)
        elif "mt_duration" in event.event_type:
            self._samples["mt_duration"].append(dur)

    def get_percentiles(self, metric_name: str) -> Dict[str, float]:
        samples = self._samples.get(metric_name)
        if not samples or len(samples) == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}

        arr = np.array(samples)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "count": len(samples),
        }

    def get_snapshot(self) -> Dict[str, dict]:
        return {
            "incoming_partial": self.get_percentiles("incoming_partial"),
            "incoming_committed": self.get_percentiles("incoming_committed"),
            "outgoing_pcm": self.get_percentiles("outgoing_pcm"),
            "mt_duration": self.get_percentiles("mt_duration"),
            "tts_first_pcm": self.get_percentiles("tts_first_pcm"),
            "asr_wait_outgoing": self.get_percentiles("asr_wait_outgoing"),
            "asr_wait_incoming": self.get_percentiles("asr_wait_incoming"),
            "counters": {
                "total_events": self.total_events,
                "underruns": self.underruns,
                "overruns": self.overruns,
            },
        }
