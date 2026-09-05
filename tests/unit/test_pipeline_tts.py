import asyncio

import pytest

from voice_translator.asr.mock_backend import MockASRAdapter
from voice_translator.audio.devices import DeviceInfo
from voice_translator.config.loader import load_config
from voice_translator.core.types import Direction, TranslationEvent, UtteranceState
from voice_translator.streaming.pipeline_outgoing import OutgoingPipeline
from voice_translator.translation.mock_backend import MockMTAdapter
from voice_translator.tts.base import VoiceProfile
from voice_translator.tts.mock_backend import MockTTSAdapter


class FakeRender:
    def __init__(self):
        self.chunks = []
        self.flushes = []

    def push_pcm(self, pcm, source_rate):
        self.chunks.append((pcm, source_rate))

    def flush_source(self, source_rate):
        self.flushes.append(source_rate)


@pytest.mark.parametrize("translation", ["Hello", "Thank you for watching", "Please subscribe"])
def test_outgoing_history_event_is_emitted_once_after_first_pcm_is_routed(translation):
    async def _run():
        config = load_config()
        mic = DeviceInfo(8, "Microphone", 1, "Windows DirectSound", 2, 0, 48000, False)
        render = DeviceInfo(14, "CABLE Input", 2, "Windows WASAPI", 0, 2, 48000, False)
        profile = VoiceProfile("test", "Test", "mock", "reference.wav")
        events = []
        pipeline = OutgoingPipeline(
            "test", mic, render, MockASRAdapter(), MockMTAdapter(), MockTTSAdapter(),
            profile, config, on_event_callback=events.append,
        )
        pipeline.render_engine = FakeRender()
        pipeline.is_running = True
        event = TranslationEvent(
            meeting_id="test", utterance_id="tx_0", sequence_id=0, revision=1,
            direction=Direction.OUTGOING, source_language="tr", target_language="en",
            source_text="Merhaba", translated_text=translation, state=UtteranceState.COMMITTED,
            model_info={"speech_evidence_accepted": True},
        )
        await pipeline.tts_queue.put(event)
        worker = asyncio.create_task(pipeline._tts_worker())

        for _ in range(100):
            if any(item.get("type") == "tts_started" for item in events):
                break
            await asyncio.sleep(0.005)
        pipeline.is_running = False
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

        routed = [item for item in events if item.get("type") == "tts_started"]
        assert len(routed) == 1
        assert routed[0]["source_text"] == "Merhaba"
        assert routed[0]["translated_text"] == translation
        assert pipeline.render_engine.chunks

    asyncio.run(_run())
