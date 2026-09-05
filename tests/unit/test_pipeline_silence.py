import time

import numpy as np

from voice_translator.asr.mock_backend import MockASRAdapter
from voice_translator.audio.devices import DeviceInfo
from voice_translator.config.loader import load_config
from voice_translator.core.types import Direction
from voice_translator.streaming.pipeline_incoming import IncomingPipeline
from voice_translator.streaming.pipeline_outgoing import OutgoingPipeline
from voice_translator.streaming.vad import SileroVAD
from voice_translator.streaming.vad import VADResult
from voice_translator.translation.mock_backend import MockMTAdapter
from voice_translator.tts.base import VoiceProfile
from voice_translator.tts.mock_backend import MockTTSAdapter


class CountingASR(MockASRAdapter):
    def __init__(self):
        super().__init__()
        self.process_calls = 0

    def process_audio(self, session, audio_chunk_16k, captured_at_ns):
        self.process_calls += 1
        return super().process_audio(session, audio_chunk_16k, captured_at_ns)


def test_incoming_silence_never_reaches_asr_or_subtitle_queue():
    config = load_config()
    adapter = CountingASR()
    device = DeviceInfo(3, "Speaker (Loopback)", 2, "Windows WASAPI", 2, 0, 48000, True)
    pipeline = IncomingPipeline("test", device, adapter, MockMTAdapter(), config)
    pipeline.vad = SileroVAD(load_model=False)
    pipeline.asr_session = adapter.create_session("rx_test", Direction.INCOMING, "en")
    pipeline.is_running = True
    for _ in range(150):
        pipeline._process_audio_frame(np.zeros(960, dtype=np.float32), time.monotonic_ns(), 20.0)
    assert adapter.process_calls == 0
    assert pipeline.partial_queue.qsize() == 0
    assert pipeline.committed_queue.qsize() == 0
    assert pipeline.get_diagnostics()["asr_state"] == "idle"


def test_silence_hallucination_cannot_enter_commit_or_tts_queue():
    config = load_config()
    adapter = CountingASR()
    mic = DeviceInfo(18, "Microphone", 2, "Windows WASAPI", 2, 0, 48000, False)
    render = DeviceInfo(14, "CABLE Input", 2, "Windows WASAPI", 0, 2, 48000, False)
    profile = VoiceProfile("test", "Test", "mock", "reference.wav")
    pipeline = OutgoingPipeline(
        "test", mic, render, adapter, MockMTAdapter(), MockTTSAdapter(), profile, config
    )
    pipeline.asr_session = adapter.create_session("tx_test", Direction.OUTGOING, "tr")
    pipeline._last_vad_result = VADResult(
        active=False, probability=0.0, phase="ended", transition="ended",
        rms=0.0, peak=0.0, dbfs=-120.0,
        utterance_ms=700.0, voiced_ms=0.0, voiced_ratio=0.0, model_backend="energy",
    )
    accepted = pipeline._handle_commit("İzlediğiniz için teşekkürler", 1, 2, {})
    assert not accepted
    assert pipeline.committed_queue.qsize() == 0
    assert pipeline.tts_queue.qsize() == 0
    assert pipeline.get_diagnostics()["last_rejection"]["reason"] == "insufficient_voiced_audio"


def test_replaceable_partial_rejection_does_not_discard_current_audio():
    config = load_config()
    adapter = CountingASR()
    mic = DeviceInfo(18, "Microphone", 2, "Windows WASAPI", 2, 0, 48000, False)
    render = DeviceInfo(14, "CABLE Input", 2, "Windows WASAPI", 0, 2, 48000, False)
    profile = VoiceProfile("test", "Test", "mock", "reference.wav")
    pipeline = OutgoingPipeline(
        "test", mic, render, adapter, MockMTAdapter(), MockTTSAdapter(), profile, config
    )
    pipeline.asr_session = adapter.create_session("tx_test", Direction.OUTGOING, "tr")
    pipeline.asr_session.audio_buffer = [np.ones(320, dtype=np.float32)]
    pipeline.asr_session.total_audio_samples = 320

    pipeline._note_rejection("known_hallucination_pattern", "İzlediğiniz için teşekkür ederim.")

    assert pipeline.asr_session.total_audio_samples == 320
    assert len(pipeline.asr_session.audio_buffer) == 1
