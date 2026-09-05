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
