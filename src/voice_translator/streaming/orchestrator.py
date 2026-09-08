"""Full-duplex Meeting Orchestrator and lifecycle coordinator."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Callable, Dict, List, Optional

from voice_translator.asr.base import ASRAdapter, ASRSession
from voice_translator.asr.mock_backend import MockASRAdapter
from voice_translator.asr.whisper_backend import WhisperASRAdapter
from voice_translator.audio.devices import AudioDeviceManager, DeviceInfo
from voice_translator.audio.processing import EchoReferenceBuffer, MicrophoneProcessor
from voice_translator.config.languages import (
    defaults as language_defaults,
    language_choices,
    opus_pair_paths,
    validate_language,
    validate_pair,
    xtts_supported,
)
from voice_translator.config.models import AppConfig
from voice_translator.config.presets import APP_PRESETS, ASR_PRESETS, asr_choices, resolve_input_mode
from voice_translator.desktop.hotkeys import GlobalHotkeys, validate_bindings
from voice_translator.desktop.overlay import DesktopOverlay
from voice_translator.core.types import LatencyEvent, MeetingStatus
from voice_translator.persistence.database import PersistenceWorker
from voice_translator.streaming.pipeline_incoming import IncomingPipeline
from voice_translator.streaming.pipeline_outgoing import OutgoingPipeline
from voice_translator.telemetry.metrics import TelemetryTracker
from voice_translator.translation.base import MTAdapter
from voice_translator.translation.ctranslate_backend import CTranslate2MTAdapter
from voice_translator.translation.mock_backend import MockMTAdapter
from voice_translator.tts.base import TTSAdapter, VoiceProfile
from voice_translator.tts.conditioning import VoiceProfileManager
from voice_translator.tts.mock_backend import MockTTSAdapter
from voice_translator.tts.xtts_backend import XTTSv2Adapter

logger = logging.getLogger(__name__)


class MeetingOrchestrator:
    """Coordinates hardware devices, adapters, outgoing and incoming pipelines."""

    def __init__(self, config: AppConfig, use_mocks: bool = False):
        self.config = config
        self.use_mocks = use_mocks
        self.status = MeetingStatus.STOPPED
        self.current_meeting_id: Optional[str] = None

        self.device_manager = AudioDeviceManager()
        self.profile_manager = VoiceProfileManager(config.voice.profiles_root)
        self.telemetry = TelemetryTracker(window_size=config.telemetry.window_size)
        self.persistence = PersistenceWorker(config.persistence)

        self.asr_adapter: Optional[ASRAdapter] = None
        self.mt_adapter: Optional[MTAdapter] = None
        self.tts_adapter: Optional[TTSAdapter] = None

        self.outgoing_pipeline: Optional[OutgoingPipeline] = None
        self.incoming_pipeline: Optional[IncomingPipeline] = None

        self.event_subscribers: List[Callable[[dict], None]] = []
        self.resolved_devices: Dict[str, Optional[dict]] = {
            "physical_mic": None,
            "physical_speaker": None,
            "speaker_loopback": None,
            "vb_cable_render": None,
            "vb_cable_capture": None,
        }
        self.last_start_error: Optional[str] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lifecycle_lock = asyncio.Lock()
        self.overlay = DesktopOverlay(config.overlay)
        self.hotkeys: Optional[GlobalHotkeys] = None
        self._hotkey_task: Optional[asyncio.Task] = None
        self.input_mode = resolve_input_mode(config.controls.app_preset, config.controls.input_mode)
        default_source, default_target = language_defaults(config)
        self.source_language = default_source
        self.target_language = default_target
        self.ptt_pressed = False
        self.paused = False
        self.muted = False
        self.desktop_error: Optional[str] = None

    def subscribe_events(self, callback: Callable[[dict], None]):
        self.event_subscribers.append(callback)

    def _broadcast_event(self, data: dict):
        self.overlay.state.update(data)
        for sub in self.event_subscribers:
            try:
                sub(data)
            except Exception:
                pass

    def _on_latency_event(self, event: LatencyEvent):
        self.telemetry.record_event(event)
        self.persistence.record_latency(event)
        self._broadcast_event({
            "type": "latency_update",
            "metrics": self.telemetry.get_snapshot(),
        })

    def _on_asr_inference_wait(self, session: ASRSession, sample: dict) -> None:
        meeting_id = str(session.metadata.get("meeting_id") or self.current_meeting_id or "local_session")
        event = LatencyEvent(
            meeting_id=meeting_id,
            utterance_id=f"{session.stream_id}_{session.sequence_id}",
            direction=session.direction,
            event_type="asr_inference_wait",
            monotonic_ns=time.monotonic_ns(),
            duration_ms=float(sample["wait_ms"]),
            metadata={
                "stream_id": session.stream_id,
                "queue_depth": int(sample["queue_depth"]),
                "deadline_miss_ms": float(sample["deadline_miss_ms"]),
                "is_final": bool(sample["is_final"]),
            },
        )
        if self._event_loop is not None and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._on_latency_event, event)
        else:
            self._on_latency_event(event)

    async def initialize_and_warmup(self):
        """Initializes and warms all models before marking READY."""
        self._event_loop = asyncio.get_running_loop()
        self.last_start_error = None
        self.status = MeetingStatus.STARTING
        self._broadcast_status()

        try:
            logger.info("Initializing adapters...")
            if self.use_mocks:
                self.asr_adapter = MockASRAdapter()
                self.mt_adapter = MockMTAdapter()
                self.tts_adapter = MockTTSAdapter()
            else:
                self.asr_adapter = WhisperASRAdapter(
                    partial_interval_ms=self.config.asr.partial_interval_ms,
                    min_audio_rms=self.config.asr.min_audio_rms,
                    beam_size=self.config.asr.beam_size,
                    on_inference_wait=self._on_asr_inference_wait,
                    backend_preference={"whisper_turbo": "transformers", "faster_whisper": "faster_whisper", "auto": "auto"}[self.config.asr.backend],
                )
                self.mt_adapter = CTranslate2MTAdapter(
                    beam_size=self.config.translation.beam_size,
                )
                self.tts_adapter = XTTSv2Adapter(
                    temperature=self.config.tts.temperature,
                    speed=self.config.tts.speed,
                    top_p=self.config.tts.top_p,
                    repetition_penalty=self.config.tts.repetition_penalty,
                    peak_normalization=self.config.tts.peak_normalization,
                    stream_chunk_size=self.config.tts.stream_chunk_size,
                )

            # Initialize models offline
            self.asr_adapter.initialize(
                model_path=self.config.asr.model_path,
                device=self.config.asr.device,
                compute_type=self.config.asr.compute_type,
            )
            nllb_path = getattr(self.config.translation, "nllb_model_path", None)
            self.mt_adapter.initialize(
                pair_paths=opus_pair_paths(self.config),
                nllb_model_path=nllb_path if getattr(self.config.translation, "model_type", "auto") in ("auto", "nllb") else None,
                device=self.config.translation.device,
                compute_type=self.config.translation.compute_type,
            )
            self.tts_adapter.initialize(
                model_path=self.config.tts.model_path,
                device=self.config.tts.device,
                sample_rate=self.config.tts.sample_rate,
            )

            self.status = MeetingStatus.WARMING
            self._broadcast_status()
            logger.info("Warming up models...")

            # Model warming (hard invariant before READY)
            self.asr_adapter.warmup()
            self.mt_adapter.warmup()
            self.tts_adapter.warmup()

            # Warm voice profile
            default_profile = self.profile_manager.get_default_profile()
            if not self.use_mocks and default_profile is None:
                raise RuntimeError("No default voice profile is available for outgoing TTS.")
            if default_profile:
                self.tts_adapter.prepare_voice_profile(default_profile)

            await self.persistence.start()

            self.status = MeetingStatus.READY
            self._broadcast_status()
            logger.info("MeetingOrchestrator is READY.")
        except Exception as e:
            self.last_start_error = str(e)
            self.status = MeetingStatus.ERROR
            self._broadcast_status()
            logger.error(f"Failed to initialize models/adapters: {e}")
            raise

    async def start_meeting(self, **kwargs):
        async with self._lifecycle_lock:
            await self._start_meeting(**kwargs)

    async def _start_meeting(
        self,
        mic_id: Optional[str] = None,
        loopback_id: Optional[str] = None,
        render_id: Optional[str] = None,
        voice_profile_id: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        save_meeting: bool = False,
        context_prompt: Optional[str] = None,
        app_preset: Optional[str] = None,
        input_mode: Optional[str] = None,
        ptt_key: Optional[str] = None,
        overlay_enabled: Optional[bool] = None,
        asr_model: str = "configured",
        noise_suppression: Optional[bool] = None,
        echo_cancellation: Optional[bool] = None,
    ):
        if context_prompt and context_prompt.strip():
            self.config.asr.initial_prompt = context_prompt.strip()

        source = validate_language(self.config, source_language or self.source_language)
        target = validate_language(self.config, target_language or self.target_language)
        if not self.use_mocks:
            if not xtts_supported(self.config, target):
                raise RuntimeError(
                    f"XTTS-v2 cannot synthesize '{target}'. Pick a target language from the supported matrix."
                )
            validate_pair(self.config, source, target, require_models=True)
            validate_pair(self.config, target, source, require_models=True)
        self.source_language = source
        self.target_language = target

        if self.status != MeetingStatus.READY:
            if self.status == MeetingStatus.ERROR:
                raise RuntimeError(
                    self.last_start_error
                    or "Models failed to initialize or are missing. If models are not yet downloaded, launch with '--mock' (python src/voice_translator/main.py run --mock) or download models manually."
                )
            if self.status in (MeetingStatus.STARTING, MeetingStatus.WARMING):
                raise RuntimeError("Models are still warming up. Please wait for the READY status.")
            if self.status == MeetingStatus.RUNNING:
                raise RuntimeError("An active meeting session is already running.")
            raise RuntimeError(f"Cannot start meeting while status is '{self.status.value}'.")

        if not self.asr_adapter or not self.mt_adapter or not self.tts_adapter:
            raise RuntimeError("Adapters are not initialized. Please restart the service.")

        controls_data = self.config.controls.model_dump()
        controls_data.update({k: v for k, v in {"app_preset": app_preset, "input_mode": input_mode, "ptt_key": ptt_key}.items() if v is not None})
        controls = type(self.config.controls).model_validate(controls_data)
        bindings = {"ptt": controls.ptt_key, "mode": controls.toggle_mode_key,
                    "pause": controls.pause_key, "mute": controls.mute_key}
        validate_bindings(bindings)
        if asr_model != "configured":
            await self._select_asr(asr_model)
        self.config.controls = controls
        for key, value in (("noise_suppression", noise_suppression), ("echo_cancellation", echo_cancellation)):
            if value is not None:
                setattr(self.config.audio, key, value)
        self.input_mode = resolve_input_mode(controls.app_preset, controls.input_mode)
        if overlay_enabled is not None:
            self.config.overlay.enabled = overlay_enabled
        self.paused = self.muted = self.ptt_pressed = False
        self.desktop_error = None

        mic_selector = mic_id or self.config.audio.mic_device_id
        loop_selector = loopback_id or self.config.audio.loopback_device_id
        render_selector = render_id or self.config.audio.render_device_id

        # Explicit selectors must fail visibly; presets never redirect audio silently.
        mic_dev = self.device_manager.resolve_required(mic_selector, "mic") if mic_selector else self.device_manager.find_default_mic()
        loop_dev = self.device_manager.resolve_required(loop_selector, "loopback") if loop_selector else self.device_manager.find_default_loopback()
        ren_dev = self.device_manager.resolve_required(render_selector, "render") if render_selector else self.device_manager.find_vbcable_render()
        speaker_dev = self.device_manager.find_render_for_loopback(loop_dev)
        cable_capture = self.device_manager.find_vbcable_capture()
        self.resolved_devices = {
            "physical_mic": mic_dev.to_dict() if mic_dev else None,
            "physical_speaker": speaker_dev.to_dict() if speaker_dev else None,
            "speaker_loopback": loop_dev.to_dict() if loop_dev else None,
            "vb_cable_render": ren_dev.to_dict() if ren_dev else None,
            "vb_cable_capture": cable_capture.to_dict() if cable_capture else None,
        }
        if not self.use_mocks and (not mic_dev or not loop_dev or not ren_dev):
            missing = [
                role for role, device in (("physical mic", mic_dev), ("speaker loopback", loop_dev), ("VB-CABLE render", ren_dev))
                if device is None
            ]
            raise RuntimeError("Required audio endpoints are missing: " + ", ".join(missing))

        self.current_meeting_id = f"m_{uuid.uuid4().hex[:8]}"
        self.last_start_error = None

        profile = self.profile_manager.get_profile(voice_profile_id or self.config.tts.voice_profile_id) or self.profile_manager.get_default_profile()
        if not self.use_mocks and profile is None:
            self.current_meeting_id = None
            raise RuntimeError("No valid voice profile is available for outgoing TTS.")
        if not self.use_mocks:
            try:
                self.tts_adapter.prepare_voice_profile(profile)
            except Exception as exc:
                self.last_start_error = str(exc)
                self.current_meeting_id = None
                raise

        # A separate bounded loopback tap feeds AEC even while incoming ASR is
        # busy or subtitles are paused. DSP remains in the outgoing worker.
        echo_reference = None
        microphone_processor = None
        if not self.use_mocks and (self.config.audio.noise_suppression or self.config.audio.echo_cancellation):
            try:
                if self.config.audio.echo_cancellation:
                    echo_reference = EchoReferenceBuffer(self.config.audio.sample_rate, self.config.audio.frame_duration_ms)
                microphone_processor = MicrophoneProcessor(self.config.audio, echo_reference)
            except Exception as exc:
                self.last_start_error = str(exc)
                self.current_meeting_id = None
                raise

        # Outgoing Pipeline
        if mic_dev and ren_dev:
            self.outgoing_pipeline = OutgoingPipeline(
                meeting_id=self.current_meeting_id,
                mic_device=mic_dev,
                render_device=ren_dev,
                asr_adapter=self.asr_adapter,
                mt_adapter=self.mt_adapter,
                tts_adapter=self.tts_adapter,
                voice_profile=profile,
                config=self.config,
                on_event_callback=self._broadcast_event,
                on_latency_callback=self._on_latency_event,
            )
            self.outgoing_pipeline.set_source_language(source)
            self.outgoing_pipeline.microphone_processor = microphone_processor
            self.outgoing_pipeline.set_target_language(target)
            try:
                await self.outgoing_pipeline.start()
            except Exception as exc:
                self.last_start_error = str(exc)
                self.outgoing_pipeline = None
                self.current_meeting_id = None
                self.status = MeetingStatus.READY
                self._broadcast_status()
                raise

        # Incoming Pipeline
        if loop_dev:
            self.incoming_pipeline = IncomingPipeline(
                meeting_id=self.current_meeting_id,
                loopback_device=loop_dev,
                asr_adapter=self.asr_adapter,
                mt_adapter=self.mt_adapter,
                config=self.config,
                on_event_callback=self._broadcast_event,
                on_latency_callback=self._on_latency_event,
            )
            self.incoming_pipeline.set_languages(target, source)
            self.incoming_pipeline.echo_reference = echo_reference
            try:
                await self.incoming_pipeline.start()
            except Exception as exc:
                self.last_start_error = str(exc)
                if self.outgoing_pipeline:
                    await self.outgoing_pipeline.stop()
                    self.outgoing_pipeline = None
                self.incoming_pipeline = None
                self.current_meeting_id = None
                self.status = MeetingStatus.READY
                self._broadcast_status()
                raise

        try:
            if not self.use_mocks:
                if controls.global_hotkeys:
                    self.hotkeys = GlobalHotkeys(bindings)
                    await asyncio.to_thread(self.hotkeys.start)
                if self.config.overlay.enabled:
                    await asyncio.to_thread(self.overlay.start)
            self._hotkey_task = asyncio.create_task(self._hotkey_worker(), name="session-hotkeys")
        except Exception as exc:
            self.desktop_error = str(exc)
            await self._stop_meeting()
            raise RuntimeError(f"Desktop controls failed: {exc}") from exc

        self.status = MeetingStatus.RUNNING
        self._broadcast_status()
        logger.info(f"Meeting session '{self.current_meeting_id}' started.")

    def get_audio_diagnostics(self) -> dict:
        return {
            "status": self.status.value,
            "meeting_id": self.current_meeting_id,
            "configured": {
                "mic_device_id": self.config.audio.mic_device_id,
                "loopback_device_id": self.config.audio.loopback_device_id,
                "render_device_id": self.config.audio.render_device_id,
            },
            "resolved": self.resolved_devices,
            "last_start_error": self.last_start_error,
            "outgoing": self.outgoing_pipeline.get_diagnostics() if self.outgoing_pipeline else None,
            "incoming": self.incoming_pipeline.get_diagnostics() if self.incoming_pipeline else None,
        }

    async def stop_meeting(self):
        async with self._lifecycle_lock:
            await self._stop_meeting()

    async def _stop_meeting(self):
        if self._hotkey_task:
            self._hotkey_task.cancel()
            await asyncio.gather(self._hotkey_task, return_exceptions=True)
            self._hotkey_task = None
        if self.hotkeys:
            await asyncio.to_thread(self.hotkeys.stop)
            self.hotkeys = None
        await asyncio.to_thread(self.overlay.stop)
        self.ptt_pressed = self.paused = self.muted = False
        if self.outgoing_pipeline:
            await self.outgoing_pipeline.stop()
            self.outgoing_pipeline = None

        if self.incoming_pipeline:
            await self.incoming_pipeline.stop()
            self.incoming_pipeline = None

        self.status = MeetingStatus.READY
        self.current_meeting_id = None
        self._broadcast_status()
        logger.info("Meeting stopped.")

    def controls_snapshot(self) -> dict:
        return {"app_preset": self.config.controls.app_preset, "input_mode": self.input_mode,
                "ptt_key": self.config.controls.ptt_key, "ptt_pressed": self.ptt_pressed,
                "paused": self.paused, "muted": self.muted,
                "overlay_enabled": self.config.overlay.enabled, "overlay_running": self.overlay.running,
                "hotkeys_running": bool(self.hotkeys and self.hotkeys.running),
                "desktop_error": self.desktop_error or self.overlay.error,
                "toggle_mode_key": self.config.controls.toggle_mode_key,
                "pause_key": self.config.controls.pause_key, "mute_key": self.config.controls.mute_key}

    def session_options(self) -> dict:
        return {"controls": self.controls_snapshot(), "asr_models": asr_choices(self.config.asr.model_path),
                "presets": [{"id": key, "label": data[0], "input_mode": data[1], "guidance": data[2]}
                            for key, data in APP_PRESETS.items()],
                "audio_processing": {"noise_suppression": self.config.audio.noise_suppression,
                                     "echo_cancellation": self.config.audio.echo_cancellation},
                "languages": language_choices(self.config),
                "language_defaults": {"source": self.source_language, "target": self.target_language}}

    async def update_controls(self, *, input_mode=None, ptt_pressed=None, paused=None,
                              muted=None, overlay_enabled=None, at_ns=None):
        if self.status != MeetingStatus.RUNNING:
            raise RuntimeError("Start Meeting before changing live controls.")
        if input_mode is not None:
            mode = resolve_input_mode(self.config.controls.app_preset, input_mode)
            self.input_mode = mode
            self.ptt_pressed = False
            if self.outgoing_pipeline:
                self.outgoing_pipeline.request_input("mode", mode, at_ns)
        if ptt_pressed is not None:
            if self.input_mode != "ptt":
                raise ValueError("Push-to-talk is available in PTT input mode.")
            self.ptt_pressed = bool(ptt_pressed) and not (self.paused or self.muted)
            if self.outgoing_pipeline:
                self.outgoing_pipeline.request_input("ptt", self.ptt_pressed, at_ns)
        for name, value in (("pause", paused), ("mute", muted)):
            if value is not None:
                setattr(self, "paused" if name == "pause" else "muted", bool(value))
                self.ptt_pressed = False
                if self.outgoing_pipeline:
                    self.outgoing_pipeline.request_input(name, bool(value), at_ns)
        if self.outgoing_pipeline:
            self.outgoing_pipeline.set_routing_muted(self.paused or self.muted)
        if self.incoming_pipeline:
            self.incoming_pipeline.set_paused(self.paused)
        if self.paused:
            self.overlay.state.clear()
        if overlay_enabled is not None:
            if overlay_enabled and not self.use_mocks:
                await asyncio.to_thread(self.overlay.start)
            else:
                await asyncio.to_thread(self.overlay.stop)
            self.config.overlay.enabled = bool(overlay_enabled)
        self._broadcast_event({"type": "controls_change", "controls": self.controls_snapshot()})
        return self.controls_snapshot()

    async def _hotkey_worker(self):
        while self.status == MeetingStatus.RUNNING:
            if self.hotkeys:
                if not self.hotkeys.running:
                    self.desktop_error = self.hotkeys.error or "Global hotkey thread stopped; routing muted."
                    await self.update_controls(muted=True)
                    return
                for action, value, at_ns in self.hotkeys.drain():
                    if action == "ptt" and self.input_mode == "ptt":
                        await self.update_controls(ptt_pressed=value, at_ns=at_ns)
                    elif action == "mode":
                        await self.update_controls(input_mode="vad" if self.input_mode == "ptt" else "ptt", at_ns=at_ns)
                    elif action == "pause":
                        await self.update_controls(paused=not self.paused, at_ns=at_ns)
                    elif action == "mute":
                        await self.update_controls(muted=not self.muted, at_ns=at_ns)
                    elif action == "fault":
                        self.desktop_error = "Global hotkey queue overflow; routing muted."
                        await self.update_controls(muted=True, at_ns=at_ns)
            await asyncio.sleep(0.01)

    async def _select_asr(self, key: str):
        if key not in ASR_PRESETS:
            raise ValueError(f"Unknown ASR model: {key}")
        selected = ASR_PRESETS[key]
        if selected["model_path"] == self.config.asr.model_path:
            return
        if not self.use_mocks and not next(c for c in asr_choices(self.config.asr.model_path) if c["id"] == key)["available"]:
            raise RuntimeError(f"Local model missing at '{selected['model_path']}'. Run: uv run python scripts/download_models.py {selected['download']}")
        previous = self.config.asr.model_copy(deep=True)
        candidate = previous.model_copy(update={k: selected[k] for k in ("backend", "model_path", "compute_type")})
        if candidate.device == "cpu":
            candidate.compute_type = "int8" if key == "large_v3" else "float32"
        self.status = MeetingStatus.WARMING
        self._broadcast_status()
        try:
            if self.asr_adapter:
                await asyncio.to_thread(self.asr_adapter.shutdown)
            self.config.asr = candidate
            await asyncio.to_thread(self._load_selected_asr)
        except Exception as exc:
            self.config.asr = previous
            self.last_start_error = f"ASR switch failed: {exc}"
            try:
                if self.asr_adapter:
                    await asyncio.to_thread(self.asr_adapter.shutdown)
                await asyncio.to_thread(self._load_selected_asr)
                self.status = MeetingStatus.READY
            except Exception as restore_exc:
                self.last_start_error += f"; previous model restore failed: {restore_exc}"
                self.status = MeetingStatus.ERROR
            self._broadcast_status()
            raise
        self.status = MeetingStatus.READY
        self._broadcast_status()

    def _load_selected_asr(self):
        self.asr_adapter = MockASRAdapter() if self.use_mocks else WhisperASRAdapter(
            partial_interval_ms=self.config.asr.partial_interval_ms, min_audio_rms=self.config.asr.min_audio_rms,
            beam_size=self.config.asr.beam_size, on_inference_wait=self._on_asr_inference_wait,
            backend_preference={"whisper_turbo": "transformers", "faster_whisper": "faster_whisper", "auto": "auto"}[self.config.asr.backend])
        self.asr_adapter.initialize(model_path=self.config.asr.model_path, device=self.config.asr.device,
                                    compute_type=self.config.asr.compute_type)
        self.asr_adapter.warmup()

    def switch_voice_profile(self, profile_id: str) -> bool:
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            raise RuntimeError(f"Voice profile '{profile_id}' not found.")
        if not self.use_mocks:
            if self.tts_adapter is None:
                raise RuntimeError("TTS adapter is not initialized.")
            self.tts_adapter.prepare_voice_profile(profile)
        if self.outgoing_pipeline:
            self.outgoing_pipeline.set_voice_profile(profile)
        logger.info("Switched active voice profile to '%s' (%s)", profile.display_name, profile_id)
        self._broadcast_event({
            "type": "profile_switched",
            "profile_id": profile_id,
            "display_name": profile.display_name,
        })
        return True

    def switch_languages(self, source_language: str, target_language: str) -> bool:
        """Live switch of speaking and meeting languages during an active session."""
        source = validate_language(self.config, source_language)
        target = validate_language(self.config, target_language)
        if not self.use_mocks:
            if not xtts_supported(self.config, target):
                raise RuntimeError(
                    f"XTTS-v2 cannot synthesize '{target}'. Pick a target language from the supported matrix."
                )
            validate_pair(self.config, source, target, require_models=True)
            validate_pair(self.config, target, source, require_models=True)
        self.source_language = source
        self.target_language = target
        self.config.languages.default_source = source
        self.config.languages.default_target = target
        if self.outgoing_pipeline:
            self.outgoing_pipeline.set_source_language(source)
            self.outgoing_pipeline.set_target_language(target)
        if self.incoming_pipeline:
            self.incoming_pipeline.set_languages(target, source)
        logger.info("Switched languages to source '%s', target '%s'", source, target)
        self._broadcast_event({
            "type": "languages_switched",
            "source_language": source,
            "target_language": target,
        })
        return True

    def switch_target_language(self, target_language: str) -> bool:
        """Legacy single-knob switch: keeps the current source language."""
        return self.switch_languages(self.source_language, target_language)

    def _broadcast_status(self):
        self._broadcast_event({
            "type": "status_change",
            "status": self.status.value,
            "meeting_id": self.current_meeting_id,
            "error": self.last_start_error,
            "controls": self.controls_snapshot(),
        })

    async def shutdown(self):
        await self.stop_meeting()
        await self.persistence.stop()
        if self.asr_adapter:
            self.asr_adapter.shutdown()
        if self.mt_adapter:
            self.mt_adapter.shutdown()
        if self.tts_adapter:
            self.tts_adapter.shutdown()
        self.device_manager.close()
        self._event_loop = None
        self.status = MeetingStatus.STOPPED
        logger.info("MeetingOrchestrator shut down completely.")
