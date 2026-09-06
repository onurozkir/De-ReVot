"""Incoming Pipeline: selected speaker loopback -> guarded EN ASR -> live Turkish text."""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from dataclasses import asdict, replace
from typing import Callable, Optional

import numpy as np

from voice_translator.asr.base import ASRAdapter, ASRSession
from voice_translator.audio.capture import AudioCaptureEngine
from voice_translator.audio.devices import DeviceInfo
from voice_translator.audio.resampler import AudioResampler
from voice_translator.config.languages import asr_prompt, defaults as language_defaults
from voice_translator.config.models import AppConfig
from voice_translator.core.bounded_queue import BoundedQueue
from voice_translator.core.types import Direction, LatencyEvent, UtteranceEvent, UtteranceState
from voice_translator.streaming.commit_policy import CommitController
from voice_translator.streaming.pipeline_runtime import (
    build_guard,
    build_vad,
    carried_partial_context,
    merge_carried_partial,
    reset_asr_utterance,
    speech_evidence,
)
from voice_translator.streaming.vad import VADResult
from voice_translator.translation.base import MTAdapter

logger = logging.getLogger("voice_translator.incoming")


class IncomingPipeline:
    """Realtime incoming subtitle path with bounded, coalesced partial work."""

    def __init__(
        self,
        meeting_id: str,
        loopback_device: DeviceInfo,
        asr_adapter: ASRAdapter,
        mt_adapter: MTAdapter,
        config: AppConfig,
        on_event_callback: Optional[Callable[[dict], None]] = None,
        on_latency_callback: Optional[Callable[[LatencyEvent], None]] = None,
    ):
        self.meeting_id = meeting_id
        self.loopback_device = loopback_device
        self.asr_adapter = asr_adapter
        self.mt_adapter = mt_adapter
        self.config = config
        default_source, default_target = language_defaults(config)
        self.meeting_language: str = default_target  # ASR language of remote speakers
        self.display_language: str = default_source  # Subtitle language for the user
        self.on_event_callback = on_event_callback
        self.on_latency_callback = on_latency_callback
        self.vad = build_vad(config.streaming)
        self.guard = build_guard(config.streaming)
        self.commit_controller = CommitController(
            min_words=config.streaming.commit_min_words,
            max_wait_ms=config.streaming.commit_max_wait_ms,
            stable_prefix_min_count=config.streaming.stable_prefix_min_count,
            enable_adaptive_sov=False,
        )
        self.resampler_in = AudioResampler(in_rate=config.audio.sample_rate, out_rate=16000, streaming=True)
        self.asr_session: Optional[ASRSession] = None
        self.capture_engine: Optional[AudioCaptureEngine] = None
        self.partial_queue: BoundedQueue[UtteranceEvent] = BoundedQueue(
            maxsize=config.streaming.max_partial_queue_size,
            drop_oldest_on_full=True,
            replace_key=lambda event: event.stream_id,
        )
        self.committed_queue: BoundedQueue[UtteranceEvent] = BoundedQueue(
            maxsize=config.streaming.max_committed_queue_size,
            drop_oldest_on_full=False,
        )
        self.is_running = False
        self._tasks: list[asyncio.Task] = []
        self._sequence_counter = 0
        self._in_speech = False
        self._context_history: collections.deque[str] = collections.deque(maxlen=2)
        self._preroll = collections.deque(maxlen=max(1, 200 // config.audio.frame_duration_ms))
        self._last_vad_result = VADResult(False, 0.0, "idle", None, 0.0, 0.0, -120.0, 0.0, 0.0, 0.0, "unknown")
        self._current_max_queue_age_ms = 0.0
        self._max_queue_age_seen_ms = 0.0
        self._dropped_audio_samples = 0
        self._last_rejection: Optional[dict] = None
        self._overloaded = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.paused = False
        self._delivery_generation = 0
        self._reset_requested = False
        self.echo_reference = None

    async def start(self) -> None:
        if self.is_running:
            return
        self._loop = asyncio.get_running_loop()
        self.is_running = True
        self.capture_engine = AudioCaptureEngine(
            device_info=self.loopback_device,
            sample_rate=self.config.audio.sample_rate,
            frame_duration_ms=self.config.audio.frame_duration_ms,
            ring_buffer_sec=self.config.audio.ring_buffer_duration_sec,
        )
        self.asr_session = self.asr_adapter.create_session(
            stream_id=f"rx_{self.meeting_id}",
            direction=Direction.INCOMING,
            language=self.meeting_language,
            initial_prompt=asr_prompt(
                self.config, self.meeting_language,
                "Hello. English business, technical and meeting conversation.",
            ),
        )
        self.asr_session.metadata["meeting_id"] = self.meeting_id
        self.capture_engine.echo_reference = self.echo_reference
        try:
            self.capture_engine.start()
        except Exception:
            self.is_running = False
            self.capture_engine.stop()
            raise
        self._tasks = [
            asyncio.create_task(self._audio_worker(), name="incoming-audio"),
            asyncio.create_task(self._partial_mt_worker(), name="incoming-partial-mt"),
            asyncio.create_task(self._committed_mt_worker(), name="incoming-committed-mt"),
        ]
        logger.info("Incoming pipeline: %s", self.loopback_device.name)

    async def _audio_worker(self) -> None:
        assert self.capture_engine is not None
        frame_size = self.capture_engine.frame_size
        while self.is_running:
            available = self.capture_engine.ring_buffer.available_read
            if available < frame_size:
                await asyncio.sleep(0.005)
                continue
            queue_age_ms = available / self.capture_engine.sample_rate * 1000.0
            self._max_queue_age_seen_ms = max(self._max_queue_age_seen_ms, queue_age_ms)
            if queue_age_ms > self.config.streaming.max_audio_queue_age_ms:
                discard = max(0, available - frame_size)
                if discard:
                    self.capture_engine.read_samples(discard)
                    self._dropped_audio_samples += discard
                    self._reject_current("stale_audio", self.asr_session.last_partial_text if self.asr_session else "")
                    self.vad.reset()
                    self._in_speech = False
                    self._preroll.clear()
                queue_age_ms = frame_size / self.capture_engine.sample_rate * 1000.0
            frame = self.capture_engine.read_samples(frame_size)
            if len(frame) < frame_size:
                continue
            captured_at_ns = time.monotonic_ns() - int(queue_age_ms * 1e6)
            await asyncio.to_thread(self._process_audio_frame, frame, captured_at_ns, queue_age_ms)

    def _process_audio_frame(self, frame_48k: np.ndarray, captured_at_ns: int, queue_age_ms: float) -> None:
        if not self.is_running or self.asr_session is None:
            return
        if self._reset_requested:
            self._reset_requested = False
            reset_asr_utterance(self.asr_session)
            self.commit_controller.reset()
            self.vad.reset()
            self._preroll.clear()
            self._in_speech = False
            self._current_max_queue_age_ms = 0.0
        if self.paused:
            return
        generation = self._delivery_generation
        frame_16k = self.resampler_in.process(frame_48k)
        vad_result = self.vad.process(frame_16k)
        self._last_vad_result = vad_result
        if vad_result.phase != "idle":
            self._current_max_queue_age_ms = max(self._current_max_queue_age_ms, queue_age_ms)

        if vad_result.transition == "started":
            self._in_speech = True
            while self._preroll:
                self.asr_adapter.process_audio(self.asr_session, self._preroll.popleft(), captured_at_ns)

        if vad_result.active:
            event = self.asr_adapter.process_audio(self.asr_session, frame_16k, captured_at_ns)
            if self.paused or generation != self._delivery_generation:
                return
            if event is not None and event.text.strip():
                event.model_info["delivery_generation"] = generation
                event.text = merge_carried_partial(self.asr_session, event.text)
                evidence = speech_evidence(vad_result, self._current_max_queue_age_ms)
                guard = self.guard.evaluate(event.text, evidence, event.model_info)
                if guard.accepted:
                    decision = self.commit_controller.evaluate(
                        event.text,
                        now_ms=time.monotonic() * 1000.0,
                        silence_ms=getattr(vad_result, "silence_ms", 0.0),
                        language=self.meeting_language,
                    )
                    if decision.should_commit and self._handle_commit(
                        decision.committed_text,
                        event.audio_start_ns,
                        captured_at_ns,
                        event.model_info,
                        remaining_partial_text=decision.remaining_partial_text,
                    ):
                        if decision.remaining_partial_text:
                            event = replace(
                                event,
                                utterance_id=f"{self.asr_session.stream_id}_{self._sequence_counter}",
                                sequence_id=self._sequence_counter,
                                revision=1,
                                text=decision.remaining_partial_text,
                            )
                        else:
                            reset_asr_utterance(self.asr_session)
                            return
                    self._submit_queue(self.partial_queue, event, "incoming_partial")
                elif guard.reason in {
                    "implausible_speech_rate", "repetitive_text", "whisper_no_speech",
                    "whisper_low_logprob", "whisper_repetition", "stale_audio",
                }:
                    self._note_rejection(guard.reason, event.text)
            return

        if vad_result.transition == "ended" and self._in_speech:
            self._in_speech = False
            carried_text, carried_start_ns, carried_model_info = carried_partial_context(self.asr_session)
            final_event = self.asr_adapter.flush_session(self.asr_session)
            if self.paused or generation != self._delivery_generation:
                return
            if final_event is not None and final_event.text.strip():
                final_event.text = merge_carried_partial(self.asr_session, final_event.text)
                self._handle_commit(
                    final_event.text,
                    carried_start_ns or final_event.audio_start_ns,
                    captured_at_ns,
                    final_event.model_info,
                )
            elif carried_text:
                self._handle_commit(
                    carried_text,
                    carried_start_ns or captured_at_ns,
                    captured_at_ns,
                    carried_model_info,
                )
            else:
                self._reject_current("no_asr_text", "")
            self.commit_controller.reset()
            reset_asr_utterance(self.asr_session)
            self.vad.reset()
            self._current_max_queue_age_ms = 0.0
            return

        self._preroll.append(frame_16k)

    def _handle_commit(
        self,
        text: str,
        audio_start_ns: int,
        audio_end_ns: int,
        model_info: dict,
        *,
        remaining_partial_text: str = "",
    ) -> bool:
        if self.paused:
            return False
        evidence = speech_evidence(self._last_vad_result, self._current_max_queue_age_ms)
        decision = self.guard.evaluate(text, evidence, model_info)
        if not decision.accepted:
            self._reject_current(decision.reason, text)
            return False
        assert self.asr_session is not None
        accepted_info = dict(model_info)
        accepted_info.update({"speech_evidence_accepted": True, "speech_evidence": asdict(evidence)})
        accepted_info["delivery_generation"] = self._delivery_generation
        event = UtteranceEvent(
            meeting_id=self.meeting_id, stream_id=self.asr_session.stream_id,
            direction=Direction.INCOMING,
            utterance_id=f"{self.asr_session.stream_id}_{self._sequence_counter}",
            sequence_id=self._sequence_counter, revision=1, state=UtteranceState.COMMITTED,
            source_language=self.meeting_language, text=text.strip(), audio_start_ns=audio_start_ns,
            audio_end_ns=audio_end_ns, is_final=True, model_info=accepted_info,
        )
        self._sequence_counter += 1
        self.asr_session.sequence_id = self._sequence_counter
        reset_asr_utterance(
            self.asr_session,
            carried_partial_text=remaining_partial_text,
            carried_audio_start_ns=audio_start_ns,
            carried_model_info=model_info,
        )
        self.commit_controller.reset()
        self._submit_queue(self.committed_queue, event, "incoming_committed")
        return True

    def _reject_current(self, reason: str, text: str) -> None:
        self._note_rejection(reason, text)
        reset_asr_utterance(self.asr_session)
        self.commit_controller.reset()

    def _note_rejection(self, reason: str, text: str) -> None:
        self._last_rejection = {"reason": reason, "text": text, "timestamp_ns": time.monotonic_ns()}
        self._emit_event({"type": "asr_rejected", "direction": "incoming", **self._last_rejection})

    def _submit_queue(self, queue: BoundedQueue, item, name: str) -> None:
        if self._loop is None or not self._loop.is_running():
            self._overloaded = True
            return
        future = asyncio.run_coroutine_threadsafe(queue.put(item), self._loop)

        def _done(done_future) -> None:
            try:
                accepted = done_future.result()
            except Exception as exc:
                accepted = False
                logger.error("%s queue submission failed: %s", name, exc)
            if not accepted:
                self._overloaded = True
                self._emit_event({"type": "queue_overload", "queue": name, "direction": "incoming"})

        future.add_done_callback(_done)

    async def _partial_mt_worker(self) -> None:
        while self.is_running:
            event = await self.partial_queue.get()
            if self._subtitle_cancelled(event) or event.sequence_id < self._sequence_counter:
                continue
            translated = await asyncio.to_thread(
                self.mt_adapter.translate, event.text, self.meeting_language, self.display_language, True)
            if self._subtitle_cancelled(event) or event.sequence_id < self._sequence_counter:
                continue
            now_ns = time.monotonic_ns()
            self._record_latency(event, "incoming_partial", now_ns, (now_ns - event.audio_end_ns) / 1e6)
            self._emit_event({
                "type": "incoming_partial", "direction": "incoming",
                "source_text": event.text, "translated_text": translated,
                "sequence_id": event.sequence_id, "revision": event.revision, "timestamp_ns": now_ns,
                "source_language": self.meeting_language, "target_language": self.display_language,
            })

    async def _committed_mt_worker(self) -> None:
        while self.is_running:
            event = await self.committed_queue.get()
            if self._subtitle_cancelled(event):
                continue
            prev_context = (
                self._context_history[-1]
                if (getattr(self.config.translation, "enable_context_priming", True) and self._context_history)
                else None
            )
            glossary = getattr(self.config.translation, "glossary", None)
            translated = await asyncio.to_thread(
                self.mt_adapter.translate_event,
                event,
                self.display_language,
                context=prev_context,
                glossary=glossary,
            )
            if self._subtitle_cancelled(event):
                continue
            self._context_history.append(event.text)
            now_ns = time.monotonic_ns()
            self._record_latency(event, "incoming_committed", now_ns, (now_ns - event.audio_end_ns) / 1e6)
            self._emit_event({
                "type": "incoming_committed", "direction": "incoming",
                "source_text": translated.source_text, "translated_text": translated.translated_text,
                "sequence_id": translated.sequence_id, "timestamp_ns": now_ns,
                "source_language": self.meeting_language, "target_language": self.display_language,
            })

    def _record_latency(self, event: UtteranceEvent, event_type: str, now_ns: int, duration_ms: float) -> None:
        if self.on_latency_callback:
            self.on_latency_callback(LatencyEvent(
                meeting_id=self.meeting_id, utterance_id=event.utterance_id,
                direction=Direction.INCOMING, event_type=event_type,
                monotonic_ns=now_ns, duration_ms=duration_ms,
            ))

    def get_diagnostics(self) -> dict:
        vad = asdict(self._last_vad_result)
        return {
            "direction": "incoming", "asr_state": vad["phase"], "vad": vad,
            "last_rejection": self._last_rejection, "overloaded": self._overloaded,
            "audio_dropped_samples": self._dropped_audio_samples,
            "max_audio_queue_age_ms": self._max_queue_age_seen_ms,
            "capture": self.capture_engine.get_diagnostics() if self.capture_engine else None,
            "queues": [
                self.partial_queue.snapshot("incoming_partial", "coalesce_latest"),
                self.committed_queue.snapshot("incoming_committed", "reject_new"),
            ],
        }

    def set_paused(self, paused: bool) -> None:
        if paused and not self.paused:
            self._delivery_generation += 1
            self._reset_requested = True
        self.paused = paused

    def set_languages(self, meeting_language: str, display_language: str) -> None:
        """Apply incoming ASR and subtitle languages; resets the current utterance boundary."""
        meeting = meeting_language.lower().strip()
        display = display_language.lower().strip()
        if meeting == self.meeting_language and display == self.display_language:
            return
        self.meeting_language = meeting
        self.display_language = display
        if self.asr_session is not None:
            self.asr_session.language = meeting
            self.asr_session.initial_prompt = asr_prompt(
                self.config, meeting, "Hello. English business, technical and meeting conversation.")
            reset_asr_utterance(self.asr_session)
            self.commit_controller.reset()
            self.vad.reset()
            self._preroll.clear()
            self._in_speech = False
            self._current_max_queue_age_ms = 0.0
        logger.info("Incoming languages updated: ASR '%s', subtitles '%s'", self.meeting_language, self.display_language)

    def _subtitle_cancelled(self, event) -> bool:
        cancelled = self.paused or event.model_info.get("delivery_generation", 0) != self._delivery_generation
        if cancelled:
            self._emit_event({"type": "routing_cancelled", "direction": "incoming",
                              "sequence_id": event.sequence_id, "reason": "user_pause"})
        return cancelled

    def _emit_event(self, data: dict) -> None:
        if self.on_event_callback:
            try:
                self.on_event_callback(data)
            except Exception:
                logger.debug("Incoming event subscriber failed", exc_info=True)

    async def stop(self) -> None:
        self.is_running = False
        if self.capture_engine:
            self.capture_engine.stop()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self.asr_session:
            self.asr_adapter.close_session(self.asr_session)
        self.capture_engine = None
        self.asr_session = None
        logger.info("Incoming pipeline stopped.")
