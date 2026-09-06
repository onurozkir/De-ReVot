import numpy as np
import pytest

from voice_translator.asr.whisper_backend import WhisperASRAdapter
from voice_translator.audio.devices import DeviceInfo
from voice_translator.audio.resampler import AudioResampler
from voice_translator.config.models import AppConfig, LanguageDefinition
from voice_translator.core.types import Direction
from voice_translator.streaming.pipeline_outgoing import OutgoingPipeline
from voice_translator.streaming.input_control import InputGate
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


def test_fifteen_second_hold_preserves_all_sentences_until_release():
    pipeline, calls, commits, events = make_pipeline()
    # Even eager streaming commit settings must not split a PTT turn.
    pipeline.commit_controller.min_words = 1
    pipeline.request_input("ptt", True, 0)
    captured = []

    def decode(audio, language, **kwargs):
        calls.append((len(audio), kwargs.get("is_final")))
        captured.append(audio.copy())
        return "Birinci cümle. İkinci cümle. Son cümle.", {"avg_logprob": -0.2}

    pipeline.asr_adapter._decode_audio = decode
    # Three sentences separated by pauses longer than the VAD endpoint;
    # also release after a final pause, when the VAD has returned to idle.
    audio = np.concatenate([
        np.full(64000, .1, np.float32), np.zeros(16000, np.float32),
        np.full(64000, .2, np.float32), np.zeros(16000, np.float32),
        np.full(48000, .3, np.float32), np.zeros(32000, np.float32),
    ])
    for offset in range(0, len(audio), 320):
        pipeline._process_gated_audio(audio[offset:offset + 320], (offset + 320) * 62500, 20)
    assert calls == []
    assert commits == []
    assert not any(e["type"] == "asr_partial" for e in events)
    pipeline.request_input("ptt", False, 15_000_000_000)
    pipeline._process_gated_audio(np.empty(0, np.float32), 15_000_000_000, 0)
    assert calls == [(240000, True)]
    np.testing.assert_array_equal(captured[0], audio)
    assert len(commits) == 1
    assert commits[0].text == "Birinci cümle. İkinci cümle. Son cümle."
    assert commits[0].audio_start_ns == 0
    assert commits[0].audio_end_ns == 15_000_000_000


def test_native_resampler_flush_preserves_tail_and_isolates_next_hold():
    pipeline, calls, commits, _ = make_pipeline()
    pipeline.config.audio.sample_rate = 48000
    pipeline.input_gate = InputGate(48000, "ptt", preroll_ms=0)
    pipeline.resampler_in = AudioResampler(48000, 16000, streaming=True)
    recorded = []

    def decode(audio, language, **kwargs):
        recorded.append(audio.copy())
        return "Bir deneme yapıyorum.", {"avg_logprob": -.2}

    pipeline.asr_adapter._decode_audio = decode
    for turn, amplitude in enumerate((.1, .3)):
        start = turn * 2_000_000_000
        pipeline.request_input("ptt", True, start)
        source = np.full(48000, amplitude, np.float32)
        for offset in range(0, len(source), 960):
            pipeline._process_gated_audio(source[offset:offset + 960], start + (offset + 960) * 1_000_000_000 // 48000, 20)
        assert len(recorded) == turn
        pipeline.request_input("ptt", False, start + 1_000_000_000)
        pipeline._process_gated_audio(np.empty(0, np.float32), start + 1_000_000_000, 0)
        expected = AudioResampler(48000, 16000).process(source)
        np.testing.assert_allclose(recorded[-1], expected, atol=2e-5)
        assert commits[-1].audio_start_ns == start
        assert commits[-1].audio_end_ns == start + 1_000_000_000
    assert len(commits) == 2


def test_ptt_overflow_cancels_whole_turn_and_next_hold_recovers():
    pipeline, calls, commits, events = make_pipeline()
    pipeline.input_gate = InputGate(16000, "ptt", preroll_ms=0)
    pipeline.request_input("ptt", True, 0)
    frame = np.full(320, .1, np.float32)
    for i in range(1550):
        pipeline._process_gated_audio(frame, (i + 1) * 20_000_000, 20)
        assert pipeline.asr_session.total_audio_samples <= 480000
    pipeline.request_input("ptt", False, 31_000_000_000)
    pipeline._process_gated_audio(np.empty(0, np.float32), 31_000_000_000, 0)
    assert not calls and not commits
    assert any(e.get("reason") == "ptt_duration_exceeded" for e in events)
    pipeline.request_input("ptt", True, 32_000_000_000)
    pipeline.request_input("ptt", False, 33_000_000_000)
    for i in range(50):
        pipeline._process_gated_audio(frame, 32_000_000_000 + (i + 1) * 20_000_000, 20)
    assert calls == [(16000, True)]
    assert len(commits) == 1


@pytest.mark.parametrize("cancel", ["mute", "pause", "mode", "stale_audio", "language"])
def test_cancelled_ptt_never_translates_only_its_tail(cancel):
    pipeline, calls, commits, _ = make_pipeline()
    pipeline.request_input("ptt", True, 0)
    frame = np.full(320, .1, np.float32)
    for i in range(50):
        pipeline._process_gated_audio(frame, (i + 1) * 20_000_000, 20)
    if cancel == "mute":
        pipeline.set_routing_muted(True)
        pipeline.set_routing_muted(False)
    elif cancel == "stale_audio":
        pipeline._reject_current("stale_audio", "")
    elif cancel == "language":
        pipeline.config.languages.definitions["en"] = LanguageDefinition(
            name="English", whisper_code="en", nllb_code="eng_Latn")
        pipeline.set_source_language("en")
    else:
        pipeline.request_input(cancel, True if cancel == "pause" else "vad", 1_000_000_000)
    pipeline._process_gated_audio(np.empty(0, np.float32), 1_000_000_000, 0)
    if cancel != "mode":
        for i in range(50, 100):
            pipeline._process_gated_audio(frame, (i + 1) * 20_000_000, 20)
    pipeline.request_input("ptt", False, 2_000_000_000)
    pipeline._process_gated_audio(np.empty(0, np.float32), 2_000_000_000, 0)
    assert not calls and not commits


def test_mute_during_final_decode_cancels_even_if_unmuted_before_completion():
    pipeline, calls, commits, _ = make_pipeline()
    def decode(audio, language, **kwargs):
        pipeline.set_routing_muted(True)
        pipeline.set_routing_muted(False)
        return "Bu eski konuşma gönderilmemeli.", {"avg_logprob": -.2}
    pipeline.asr_adapter._decode_audio = decode
    pipeline.request_input("ptt", True, 0)
    pipeline.request_input("ptt", False, 1_000_000_000)
    for i in range(50):
        pipeline._process_gated_audio(np.full(320, .1, np.float32), (i + 1) * 20_000_000, 20)
    assert not commits


def test_closed_gate_backlog_after_final_decode_does_not_report_lost_speech():
    pipeline, calls, commits, events = make_pipeline()
    pipeline._discard_stale_input(2_000_000_000)
    assert not any(e["type"] == "asr_rejected" for e in events)
    assert not calls and not commits


def test_press_inside_discarded_backlog_cannot_translate_only_remaining_tail():
    pipeline, calls, commits, events = make_pipeline()
    pipeline.request_input("ptt", True, 500_000_000)
    pipeline._discard_stale_input(1_000_000_000)
    assert pipeline._ptt_cancelled
    for i in range(50):
        pipeline._process_gated_audio(np.full(320, .1, np.float32), 1_000_000_000 + (i + 1) * 20_000_000, 20)
    pipeline.request_input("ptt", False, 2_000_000_000)
    pipeline._process_gated_audio(np.empty(0, np.float32), 2_000_000_000, 0)
    assert not calls and not commits
    assert any(e.get("reason") == "stale_audio" for e in events)
