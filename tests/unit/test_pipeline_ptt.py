import numpy as np

from voice_translator.asr.whisper_backend import WhisperASRAdapter
from voice_translator.audio.devices import DeviceInfo
from voice_translator.config.models import AppConfig
from voice_translator.core.types import Direction
from voice_translator.streaming.pipeline_outgoing import OutgoingPipeline
from voice_translator.streaming.vad import SileroVAD
from voice_translator.translation.mock_backend import MockMTAdapter
from voice_translator.tts.mock_backend import MockTTSAdapter
from voice_translator.tts.base import VoiceProfile


def make_pipeline():
    config = AppConfig()
    config.audio.sample_rate = 16000
    config.controls.input_mode = "ptt"
    config.streaming.commit_min_words = 50
    config.streaming.commit_max_wait_ms = 10000
    adapter = WhisperASRAdapter(min_audio_rms=0)
    adapter.model = object()
    calls = []

    def decode(audio, language, **kwargs):
        calls.append((len(audio), kwargs.get("is_final")))
        return "Teşekkür ederim", {"avg_logprob": -0.2}

    adapter._decode_audio = decode
    device = DeviceInfo(1, "Test", 1, "Windows WASAPI", 1, 1, 16000, False)
    events, commits = [], []
    pipeline = OutgoingPipeline("test", device, device, adapter, MockMTAdapter(), MockTTSAdapter(),
                                VoiceProfile("test", "Test", "mock", "reference.wav"), config, events.append)
    pipeline.asr_session = adapter.create_session("tx", Direction.OUTGOING, "tr")
    pipeline.is_running = True
    pipeline.resampler_in.process = lambda audio: audio
    pipeline.vad = SileroVAD(load_model=False)
    pipeline._submit_queue = lambda queue, item, name: commits.append(item)
    return pipeline, calls, commits, events


def test_ptt_release_decodes_tail_and_commits_without_waiting_for_silence():
    pipeline, calls, commits, events = make_pipeline()
    pipeline.request_input("ptt", True, 200_000_000)
    pipeline.request_input("ptt", False, 800_000_000)
    frame = np.ones(320, dtype=np.float32) * 0.1
    for i in range(40):
        pipeline._process_gated_audio(frame, (i + 1) * 20_000_000, 20)
    assert len(commits) == 1
    assert commits[0].text == "Teşekkür ederim"
    assert commits[0].model_info["commit_reason"] == "ptt_release"
    assert commits[0].audio_end_ns == 800_000_000
    assert calls[-1][1] is True
    count = len(calls)
    for i in range(40, 100):
        pipeline._process_gated_audio(frame, (i + 1) * 20_000_000, 20)
    assert len(calls) == count
    assert [e["reason"] for e in events if e["type"] == "input_endpoint"] == ["ptt_release"]


def test_held_ptt_does_not_turn_silence_into_speech_evidence():
    pipeline, calls, commits, _ = make_pipeline()
    pipeline.request_input("ptt", True, 0)
    pipeline.request_input("ptt", False, 1_000_000_000)
    for i in range(50):
        pipeline._process_gated_audio(np.zeros(320, dtype=np.float32), (i + 1) * 20_000_000, 20)
    assert calls == []
    assert commits == []


def test_mute_invalidates_already_committed_work_even_after_unmute():
    pipeline, _, commits, _ = make_pipeline()
    pipeline.request_input("ptt", True, 0)
    pipeline.request_input("ptt", False, 800_000_000)
    for i in range(40):
        pipeline._process_gated_audio(np.ones(320, dtype=np.float32) * 0.1, (i + 1) * 20_000_000, 20)
    assert commits
    pipeline.set_routing_muted(True)
    pipeline.set_routing_muted(False)
    assert pipeline._delivery_cancelled(commits[0])


def test_microphone_processing_runs_before_ptt_gate_and_vad():
    pipeline, calls, commits, _ = make_pipeline()
    processed = []

    class Processor:
        def process(self, frame, at_ns):
            processed.append(at_ns)
            return [(np.zeros_like(frame), at_ns)]

    pipeline.microphone_processor = Processor()
    pipeline.request_input("ptt", True, 200_000_000)
    pipeline.request_input("ptt", False, 800_000_000)
    for i in range(50):
        pipeline._process_gated_audio(np.full(320, 0.1, dtype=np.float32), (i + 1) * 20_000_000, 20)
    assert len(processed) == 50  # Also adapts while PTT is released.
    assert not calls and not commits  # VAD sees filtered audio, not raw noise.


def test_microphone_processing_failure_mutes_delivery():
    pipeline, _, commits, events = make_pipeline()

    class Processor:
        def process(self, *args):
            raise RuntimeError("DSP failed")

    pipeline.microphone_processor = Processor()
    pipeline._process_gated_audio(np.zeros(320, dtype=np.float32), 20_000_000, 20)
    assert pipeline._routing_muted
    assert not commits
    assert any(e.get("error") == "DSP failed" for e in events)
