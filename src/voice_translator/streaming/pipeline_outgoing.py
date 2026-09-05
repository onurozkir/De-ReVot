"""Outgoing Pipeline: Turkish Mic -> guarded ASR -> MT -> cloned TTS -> VB-CABLE."""

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
from voice_translator.audio.render import AudioRenderEngine
from voice_translator.audio.resampler import AudioResampler
from voice_translator.config.languages import asr_prompt, defaults as language_defaults
from voice_translator.config.models import AppConfig
from voice_translator.config.presets import resolve_input_mode
from voice_translator.streaming.input_control import InputGate
from voice_translator.core.bounded_queue import BoundedQueue
from voice_translator.core.types import Direction, LatencyEvent, TranslationEvent, UtteranceEvent, UtteranceState
from voice_translator.streaming.commit_policy import CommitController
from voice_translator.streaming.pipeline_runtime import (
    build_guard,
    build_vad,
    carried_partial_context,
    merge_carried_partial,
    next_pcm_chunk,
    reset_asr_utterance,
    speech_evidence,
)
from voice_translator.streaming.vad import VADResult
from voice_translator.translation.base import MTAdapter
from voice_translator.tts.base import TTSAdapter, VoiceProfile

logger = logging.getLogger("voice_translator.outgoing")


class OutgoingPipeline:
    """Realtime outgoing pipeline with a non-inference PortAudio callback boundary."""

    def __init__(
        self,
        meeting_id: str,
        mic_device: DeviceInfo,
        render_device: DeviceInfo,
        asr_adapter: ASRAdapter,
        mt_adapter: MTAdapter,
        tts_adapter: TTSAdapter,
        voice_profile: VoiceProfile,
        config: AppConfig,
        on_event_callback: Optional[Callable[[dict], None]] = None,
        on_latency_callback: Optional[Callable[[LatencyEvent], None]] = None,
    ):
        self.meeting_id = meeting_id
        self.mic_device = mic_device
        self.render_device = render_device
        self.asr_adapter = asr_adapter
        self.mt_adapter = mt_adapter
        self.tts_adapter = tts_adapter
        self.voice_profile = voice_profile
        self.config = config
        default_source, default_target = language_defaults(config)
        self.source_language: str = default_source
        self.target_language: str = default_target
        self.on_event_callback = on_event_callback
        self.on_latency_callback = on_latency_callback

        self.vad = build_vad(config.streaming)
        self.guard = build_guard(config.streaming)
        self.commit_controller = CommitController(
            min_words=config.streaming.commit_min_words,
            max_wait_ms=config.streaming.commit_max_wait_ms,
            stable_prefix_min_count=config.streaming.stable_prefix_min_count,
            enable_adaptive_sov=getattr(config.streaming, "enable_adaptive_sov", True),
            sov_min_silence_ms=getattr(config.streaming, "sov_min_silence_ms", 200),
        )
        self.resampler_in = AudioResampler(in_rate=config.audio.sample_rate, out_rate=16000, streaming=True)
        self.asr_session: Optional[ASRSession] = None
        self.capture_engine: Optional[AudioCaptureEngine] = None
        self.render_engine: Optional[AudioRenderEngine] = None
        self.committed_queue: BoundedQueue[UtteranceEvent] = BoundedQueue(
            maxsize=config.streaming.max_committed_queue_size,
            drop_oldest_on_full=False,
        )
        self.tts_queue: BoundedQueue[TranslationEvent] = BoundedQueue(
            maxsize=config.streaming.max_tts_queue_size,
            drop_oldest_on_full=False,
        )

        self.is_running = False
        self._tasks: list[asyncio.Task] = []
        self._sequence_counter = 0
        self._in_speech = False
        self._context_history: collections.deque[str] = collections.deque(maxlen=2)
        self._preroll = collections.deque(maxlen=max(1, 300 // config.audio.frame_duration_ms))
        self._last_vad_result = VADResult(False, 0.0, "idle", None, 0.0, 0.0, -120.0, 0.0, 0.0, 0.0, "unknown")
        self._current_max_queue_age_ms = 0.0
        self._max_queue_age_seen_ms = 0.0
        self._dropped_audio_samples = 0
        self._last_rejection: Optional[dict] = None
        self._overloaded = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.input_gate = InputGate(config.audio.sample_rate,
                                    resolve_input_mode(config.controls.app_preset, config.controls.input_mode),
                                    config.controls.ptt_preroll_ms)
        self._routing_muted = False
        self._delivery_generation = 0
        self._audio_inflight: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self.is_running:
            return
        self._loop = asyncio.get_running_loop()
        self.is_running = True
        self.capture_engine = AudioCaptureEngine(
            device_info=self.mic_device,
            sample_rate=self.config.audio.sample_rate,
            frame_duration_ms=self.config.audio.frame_duration_ms,
            ring_buffer_sec=self.config.audio.ring_buffer_duration_sec,
        )
        self.render_engine = AudioRenderEngine(
            device_info=self.render_device,
            sample_rate=self.config.audio.sample_rate,
            frame_duration_ms=self.config.audio.frame_duration_ms,
        )
        self.asr_session = self.asr_adapter.create_session(
            stream_id=f"tx_{self.meeting_id}",
            direction=Direction.OUTGOING,
            language=self.source_language,
            initial_prompt=asr_prompt(self.config, self.source_language, getattr(self.config.asr, "initial_prompt", "")),
        )
        self.asr_session.metadata["meeting_id"] = self.meeting_id
        try:
            self.capture_engine.start()
            self.render_engine.start()
        except Exception:
            self.is_running = False
            if self.capture_engine:
                self.capture_engine.stop()
            if self.render_engine:
                self.render_engine.stop()
            raise
        self._tasks = [
            asyncio.create_task(self._audio_worker(), name="outgoing-audio"),
            asyncio.create_task(self._mt_worker(), name="outgoing-mt"),
            asyncio.create_task(self._tts_worker(), name="outgoing-tts"),
        ]
        logger.info("Outgoing pipeline: %s -> %s", self.mic_device.name, self.render_device.name)

    async def _audio_worker(self) -> None:
        """Drain bounded PCM outside PortAudio, then run VAD/Whisper in a worker thread."""
        assert self.capture_engine is not None
        frame_size = self.capture_engine.frame_size
        while self.is_running:
            available = self.capture_engine.ring_buffer.available_read
            if available == 0:
                await self._run_audio_work(np.empty(0, dtype=np.float32), time.monotonic_ns(), 0.0)
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
            frame = self.capture_engine.read_samples(min(frame_size, available))
            captured_at_ns = time.monotonic_ns() - int(queue_age_ms * 1e6) + round(len(frame) * 1e9 / self.capture_engine.sample_rate)
            await self._run_audio_work(frame, captured_at_ns, queue_age_ms)

    async def _run_audio_work(self, frame, captured_at_ns, queue_age_ms) -> None:
        self._audio_inflight = asyncio.create_task(asyncio.to_thread(
            self._process_gated_audio, frame, captured_at_ns, queue_age_ms))
        await asyncio.shield(self._audio_inflight)

    def _process_gated_audio(self, frame, captured_at_ns, queue_age_ms) -> None:
        for action in self.input_gate.route(frame, captured_at_ns):
            if not self.is_running:
                break
            if action.kind == "audio":
                self._process_audio_frame(action.audio, action.at_ns, queue_age_ms)
            elif action.kind == "ptt_release":
                self._flush_endpoint(action.at_ns, "ptt_release")
            else:
                self._reject_current(action.kind, "")
                self.vad.reset()
                self._in_speech = False
                self._preroll.clear()
                self._current_max_queue_age_ms = 0.0
                if action.kind == "control_overload":
                    self.set_routing_muted(True)
                    self._overloaded = True

    def request_input(self, kind: str, value, at_ns: int | None = None) -> None:
        self.input_gate.change(kind, value, at_ns)

    def set_routing_muted(self, muted: bool) -> None:
        if muted and not self._routing_muted:
            self._delivery_generation += 1
        self._routing_muted = muted
        if self.render_engine is not None:
            self.render_engine.set_muted(muted)

    def _process_audio_frame(self, frame_48k: np.ndarray, captured_at_ns: int, queue_age_ms: float) -> None:
        if not self.is_running or self.asr_session is None:
            return
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
            if event is not None and event.text.strip():
                event.text = merge_carried_partial(self.asr_session, event.text)
                evidence = speech_evidence(vad_result, self._current_max_queue_age_ms)
                decision = self.guard.evaluate(event.text, evidence, event.model_info)
                if decision.accepted:
                    commit = self.commit_controller.evaluate(
                        event.text,
                        now_ms=time.monotonic() * 1000.0,
                        silence_ms=getattr(vad_result, "silence_ms", 0.0),
                        language=self.source_language,
                    )
                    if commit.should_commit and self._handle_commit(
                        commit.committed_text,
                        event.audio_start_ns,
                        captured_at_ns,
                        event.model_info,
                        remaining_partial_text=commit.remaining_partial_text,
                    ):
                        if commit.remaining_partial_text:
                            event = replace(
                                event,
                                utterance_id=f"{self.asr_session.stream_id}_{self._sequence_counter}",
                                sequence_id=self._sequence_counter,
                                revision=1,
                                text=commit.remaining_partial_text,
                            )
                        else:
                            reset_asr_utterance(self.asr_session)
                            return
                    self._emit_event({
                        "type": "asr_partial", "direction": "outgoing", "text": event.text,
                        "sequence_id": event.sequence_id, "revision": event.revision,
                        "timestamp_ns": captured_at_ns,
                        "source_language": self.source_language, "target_language": self.target_language,
                    })
                elif decision.reason in {
                    "implausible_speech_rate", "repetitive_text", "whisper_no_speech",
                    "whisper_low_logprob", "whisper_repetition", "stale_audio",
                }:
                    self._note_rejection(decision.reason, event.text)
            return

        if vad_result.transition == "ended" and self._in_speech:
            self._flush_endpoint(captured_at_ns, "vad_endpoint")
            return

        self._preroll.append(frame_16k)

    def _flush_endpoint(self, captured_at_ns: int, reason: str) -> None:
        if self.asr_session is None:
            return
        self._in_speech = False
        carried_text, carried_start_ns, carried_model_info = carried_partial_context(self.asr_session)
        final_event = self.asr_adapter.flush_session(self.asr_session)
        if final_event is not None and final_event.text.strip():
            final_event.text = merge_carried_partial(self.asr_session, final_event.text)
            self._handle_commit(final_event.text, carried_start_ns or final_event.audio_start_ns,
                                final_event.audio_end_ns, {**final_event.model_info, "commit_reason": reason})
        elif carried_text:
            self._handle_commit(carried_text, carried_start_ns or captured_at_ns, captured_at_ns,
                                {**carried_model_info, "commit_reason": reason})
        elif self.asr_session.total_audio_samples:
            self._reject_current("no_asr_text", "")
        self.commit_controller.reset()
        reset_asr_utterance(self.asr_session)
        self.vad.reset()
        self._preroll.clear()
        self._current_max_queue_age_ms = 0.0
        self._emit_event({"type": "input_endpoint", "direction": "outgoing", "reason": reason,
                          "audio_end_ns": captured_at_ns, "timestamp_ns": time.monotonic_ns()})

    def _handle_commit(
        self,
        text: str,
        audio_start_ns: int,
        audio_end_ns: int,
        model_info: dict,
        *,
        remaining_partial_text: str = "",
    ) -> bool:
        if self._routing_muted:
            self._reject_current("routing_muted", text)
            return False
        evidence = speech_evidence(self._last_vad_result, self._current_max_queue_age_ms)
        decision = self.guard.evaluate(text, evidence, model_info)
        if not decision.accepted:
            self._reject_current(decision.reason, text)
            return False
        accepted_info = dict(model_info)
        accepted_info.update({"speech_evidence_accepted": True, "speech_evidence": asdict(evidence)})
        accepted_info["delivery_generation"] = self._delivery_generation
        assert self.asr_session is not None
        event = UtteranceEvent(
            meeting_id=self.meeting_id, stream_id=self.asr_session.stream_id,
            direction=Direction.OUTGOING,
            utterance_id=f"{self.asr_session.stream_id}_{self._sequence_counter}",
            sequence_id=self._sequence_counter, revision=1, state=UtteranceState.COMMITTED,
            source_language=self.source_language, text=text.strip(), audio_start_ns=audio_start_ns,
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
        self._submit_queue(self.committed_queue, event, "outgoing_committed")
        self._emit_event({
            "type": "asr_committed", "direction": "outgoing", "text": event.text,
            "sequence_id": event.sequence_id, "timestamp_ns": time.monotonic_ns(),
        })
        return True

    def _reject_current(self, reason: str, text: str) -> None:
        self._note_rejection(reason, text)
        reset_asr_utterance(self.asr_session)
        self.commit_controller.reset()

    def _note_rejection(self, reason: str, text: str) -> None:
        self._last_rejection = {"reason": reason, "text": text, "timestamp_ns": time.monotonic_ns()}
        self._emit_event({"type": "asr_rejected", "direction": "outgoing", **self._last_rejection})

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
                self._emit_event({"type": "queue_overload", "queue": name, "direction": "outgoing"})

        future.add_done_callback(_done)

    async def _mt_worker(self) -> None:
        while self.is_running:
            event = await self.committed_queue.get()
            if self._delivery_cancelled(event):
                continue
            t0 = time.monotonic_ns()
            prev_context = (
                self._context_history[-1]
                if (getattr(self.config.translation, "enable_context_priming", True) and self._context_history)
                else None
            )
            glossary = getattr(self.config.translation, "glossary", None)
            translated = await asyncio.to_thread(
                self.mt_adapter.translate_event,
                event,
                self.target_language,
                context=prev_context,
                glossary=glossary,
            )
            t1 = time.monotonic_ns()
            if self._delivery_cancelled(event):
                continue
            self._context_history.append(event.text)
            self._record_latency(event, "mt_duration", t1, (t1 - t0) / 1e6)
            self._emit_event({
                "type": "mt_committed", "direction": "outgoing",
                "source_text": translated.source_text, "translated_text": translated.translated_text,
                "sequence_id": translated.sequence_id, "timestamp_ns": t1,
                "source_language": self.source_language, "target_language": self.target_language,
            })
            if not await self.tts_queue.put(translated):
                self._overloaded = True
                self._emit_event({"type": "queue_overload", "queue": "outgoing_tts", "direction": "outgoing"})

    async def _tts_worker(self) -> None:
        while self.is_running:
            event = await self.tts_queue.get()
            if (
                self._delivery_cancelled(event)
                or event.state != UtteranceState.COMMITTED
                or not event.model_info.get("speech_evidence_accepted")
                or not event.translated_text.strip()
            ):
                self._emit_event({"type": "tts_rejected", "direction": "outgoing", "sequence_id": event.sequence_id})
                continue
            t0 = time.monotonic_ns()
            iterator = self.tts_adapter.synthesize_committed(event.translated_text, self.voice_profile, self.target_language)
            first_pcm = True
            pcm_routed = False
            while self.is_running:
                has_chunk, pcm = await asyncio.to_thread(next_pcm_chunk, iterator)
                if not has_chunk:
                    break
                if self._delivery_cancelled(event):
                    break
                now_ns = time.monotonic_ns()
                if first_pcm:
                    first_pcm = False
                    self._record_latency(event, "tts_first_pcm", now_ns, (now_ns - t0) / 1e6)
                if self.render_engine is not None:
                    source_rate = int(getattr(self.tts_adapter, "sample_rate", self.config.tts.sample_rate))
                    self.render_engine.push_pcm(pcm, source_rate=source_rate)
                    if not pcm_routed:
                        self._emit_event({
                            "type": "tts_started", "direction": "outgoing",
                            "source_text": event.source_text,
                            "translated_text": event.translated_text,
                            "sequence_id": event.sequence_id,
                            "timestamp_ns": now_ns,
                            "source_language": self.source_language, "target_language": self.target_language,
                        })
                        # Event means first PCM reached bounded render path, once.
                        pcm_routed = True
            if not pcm_routed:
                self._emit_event({
                    "type": "tts_rejected", "direction": "outgoing",
                    "sequence_id": event.sequence_id,
                    "reason": "no_pcm" if first_pcm else "render_unavailable",
                })
            if self.render_engine is not None and not self._delivery_cancelled(event):
                source_rate = int(getattr(self.tts_adapter, "sample_rate", self.config.tts.sample_rate))
                self.render_engine.flush_source(source_rate)

    def _delivery_cancelled(self, event) -> bool:
        cancelled = self._routing_muted or event.model_info.get("delivery_generation", 0) != self._delivery_generation
        if cancelled:
            self._emit_event({"type": "routing_cancelled", "direction": "outgoing",
                              "sequence_id": event.sequence_id, "reason": "user_mute_or_pause"})
        return cancelled

    def _record_latency(self, event, event_type: str, now_ns: int, duration_ms: float) -> None:
        if self.on_latency_callback:
            self.on_latency_callback(LatencyEvent(
                meeting_id=self.meeting_id, utterance_id=event.utterance_id,
                direction=Direction.OUTGOING, event_type=event_type,
                monotonic_ns=now_ns, duration_ms=duration_ms,
            ))

    def get_diagnostics(self) -> dict:
        vad = asdict(self._last_vad_result)
        return {
            "direction": "outgoing", "asr_state": vad["phase"], "vad": vad,
            "input_mode": self.input_gate.mode, "ptt_pressed": self.input_gate.pressed,
            "routing_muted": self._routing_muted,
            "last_rejection": self._last_rejection, "overloaded": self._overloaded,
            "audio_dropped_samples": self._dropped_audio_samples,
            "max_audio_queue_age_ms": self._max_queue_age_seen_ms,
            "capture": self.capture_engine.get_diagnostics() if self.capture_engine else None,
            "render": self.render_engine.get_diagnostics() if self.render_engine else None,
            "queues": [
                self.committed_queue.snapshot("outgoing_committed", "reject_new"),
                self.tts_queue.snapshot("outgoing_tts", "reject_new"),
            ],
        }

    def _emit_event(self, data: dict) -> None:
        if self.on_event_callback:
            try:
                self.on_event_callback(data)
            except Exception:
                logger.debug("Outgoing event subscriber failed", exc_info=True)

    async def stop(self) -> None:
        self.is_running = False
        if self.capture_engine:
            self.capture_engine.stop()
        if self.render_engine:
            self.render_engine.stop()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._audio_inflight:
            await asyncio.shield(self._audio_inflight)
        self._tasks.clear()
        if self.asr_session:
            self.asr_adapter.close_session(self.asr_session)
        self.capture_engine = None
        self.render_engine = None
        self.asr_session = None
        logger.info("Outgoing pipeline stopped.")

    def set_voice_profile(self, profile: VoiceProfile) -> None:
        self.voice_profile = profile
        logger.info("Outgoing pipeline voice profile updated to '%s' (%s)", profile.display_name, profile.id)

    def set_source_language(self, source_language: str) -> None:
        """Apply the speaking language to ASR; resets the current utterance boundary."""
        code = source_language.lower().strip()
        if code == self.source_language:
            return
        self.source_language = code
        if self.asr_session is not None:
            self.asr_session.language = code
            self.asr_session.initial_prompt = asr_prompt(
                self.config, code, getattr(self.config.asr, "initial_prompt", ""))
            reset_asr_utterance(self.asr_session)
            self.commit_controller.reset()
            self.vad.reset()
            self._preroll.clear()
            self._in_speech = False
            self._current_max_queue_age_ms = 0.0
        logger.info("Outgoing pipeline source language updated to '%s'", self.source_language)

    def set_target_language(self, target_language: str) -> None:
        self.target_language = target_language.lower().strip()
        logger.info("Outgoing pipeline target language updated to '%s'", self.target_language)
