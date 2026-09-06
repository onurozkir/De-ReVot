import asyncio
import pytest
from voice_translator.config.loader import load_config
from voice_translator.core.types import MeetingStatus
from voice_translator.streaming.orchestrator import MeetingOrchestrator


@pytest.mark.parametrize("preset,mode", [("teams", "vad"), ("gaming", "ptt")])
def test_full_pipeline_lifecycle_mock(monkeypatch, preset, mode):
    async def _run():
        config = load_config()
        config.audio.mic_device_id = ""
        config.audio.loopback_device_id = ""
        config.audio.render_device_id = ""
        orchestrator = MeetingOrchestrator(config=config, use_mocks=True)
        monkeypatch.setattr(orchestrator.device_manager, "find_default_mic", lambda: None)
        monkeypatch.setattr(orchestrator.device_manager, "find_default_loopback", lambda: None)
        monkeypatch.setattr(orchestrator.device_manager, "find_vbcable_render", lambda: None)
        monkeypatch.setattr(orchestrator.device_manager, "find_vbcable_capture", lambda: None)

        # 1. Warmup and Ready
        await orchestrator.initialize_and_warmup()
        assert orchestrator.status == MeetingStatus.READY

        # 2. Start Meeting
        await orchestrator.start_meeting(app_preset=preset, input_mode="auto")
        assert orchestrator.status == MeetingStatus.RUNNING
        assert orchestrator.current_meeting_id is not None
        assert orchestrator.input_mode == mode
        assert orchestrator.config.overlay.enabled

        await orchestrator.update_controls(input_mode="ptt")
        await orchestrator.update_controls(ptt_pressed=True)
        await orchestrator.update_controls(ptt_pressed=False)
        assert orchestrator.status == MeetingStatus.RUNNING
        assert not orchestrator.paused

        # Wait 0.5s for async workers
        await asyncio.sleep(0.5)

        # 3. Stop Meeting
        await orchestrator.stop_meeting()
        assert orchestrator.status == MeetingStatus.READY

        # 4. Shutdown
        await orchestrator.shutdown()
        assert orchestrator.status == MeetingStatus.STOPPED

    asyncio.run(_run())


def test_acoustic_frontend_lifecycle_uses_independent_shared_reference(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from voice_translator.tts.base import VoiceProfile
    from voice_translator.tts.mock_backend import MockTTSAdapter
    from voice_translator.asr.mock_backend import MockASRAdapter
    from voice_translator.translation.mock_backend import MockMTAdapter
    import voice_translator.streaming.orchestrator as module

    class Pipeline:
        def __init__(self, **kwargs):
            self.started = False

        def set_source_language(self, *args): pass
        def set_target_language(self, *args): pass
        def set_languages(self, *args): pass
        async def start(self): self.started = True
        async def stop(self): self.started = False

    async def run():
        config = load_config()
        config.audio.mic_device_id = ""
        config.audio.loopback_device_id = ""
        config.audio.render_device_id = ""
        config.voice.profiles_root = str(tmp_path)
        config.controls.global_hotkeys = False
        config.overlay.enabled = False
        instance = MeetingOrchestrator(config, use_mocks=False)
        instance.status = MeetingStatus.READY
        instance.asr_adapter, instance.mt_adapter, instance.tts_adapter = MockASRAdapter(), MockMTAdapter(), MockTTSAdapter()
        instance.profile_manager.profiles["six"] = VoiceProfile("six", "Six", "mock", reference_audio_paths=[f"voice_{i}.wav" for i in range(6)])
        device = SimpleNamespace(to_dict=lambda: {}, name="Fake")
        for name in ("find_default_mic", "find_default_loopback", "find_vbcable_render", "find_vbcable_capture"):
            monkeypatch.setattr(instance.device_manager, name, lambda: device)
        monkeypatch.setattr(instance.device_manager, "find_render_for_loopback", lambda _: device)
        monkeypatch.setattr(module, "validate_pair", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "OutgoingPipeline", Pipeline)
        monkeypatch.setattr(module, "IncomingPipeline", Pipeline)
        monkeypatch.setattr(module, "MicrophoneProcessor", lambda config, reference: SimpleNamespace(reference=reference))
        await instance.start_meeting(voice_profile_id="six")
        outgoing, incoming = instance.outgoing_pipeline, instance.incoming_pipeline
        assert outgoing.started and incoming.started
        assert outgoing.microphone_processor.reference is incoming.echo_reference
        assert incoming.echo_reference._blocks.maxlen == 50
        await instance.stop_meeting()
        assert not outgoing.started and not incoming.started
        assert instance.status == MeetingStatus.READY
        instance.device_manager.close()

    asyncio.run(run())
