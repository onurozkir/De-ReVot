from types import SimpleNamespace
import logging
import threading
import time

import numpy as np
import pytest
import torch

from voice_translator.asr.whisper_backend import WhisperASRAdapter
from voice_translator.asr import whisper_backend
from voice_translator.core.types import Direction


def _run_scheduled_order(requests):
    adapter = WhisperASRAdapter()
    scheduler = adapter._inference_scheduler
    order = []
    threads = []

    def run(direction, audio_end_ns):
        with scheduler.acquire(
            direction=direction,
            is_final=False,
            audio_end_ns=audio_end_ns,
        ):
            order.append(direction)

    now_ns = time.monotonic_ns()
    with scheduler.acquire(direction=None, is_final=True, audio_end_ns=now_ns):
        for direction, audio_end_ns in requests:
            thread = threading.Thread(target=run, args=(direction, audio_end_ns))
            thread.start()
            threads.append(thread)
        deadline = time.monotonic() + 1.0
        while scheduler.pending_count < len(requests) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert scheduler.pending_count == len(requests)

    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    return order


def test_shared_scheduler_prefers_outgoing_for_equal_audio_deadlines():
    deadline_ns = time.monotonic_ns()
    order = _run_scheduled_order([
        (Direction.INCOMING, deadline_ns),
        (Direction.OUTGOING, deadline_ns),
    ])

    assert order == [Direction.OUTGOING, Direction.INCOMING]


def test_shared_scheduler_admission_grace_allows_outgoing_to_precede_new_incoming_partial():
    scheduler = whisper_backend._SharedInferenceScheduler(admission_grace_ms=100.0)
    order = []

    def run(direction):
        with scheduler.acquire(
            direction=direction,
            is_final=False,
            audio_end_ns=time.monotonic_ns(),
        ):
            order.append(direction)

    incoming = threading.Thread(target=run, args=(Direction.INCOMING,))
    incoming.start()
    deadline = time.monotonic() + 1.0
    while scheduler.pending_count < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert scheduler.pending_count == 1

    outgoing = threading.Thread(target=run, args=(Direction.OUTGOING,))
    outgoing.start()
    incoming.join(timeout=1.0)
    outgoing.join(timeout=1.0)

    assert not incoming.is_alive()
    assert not outgoing.is_alive()
    assert order == [Direction.OUTGOING, Direction.INCOMING]


def test_shared_scheduler_prioritizes_older_audio_before_direction():
    now_ns = time.monotonic_ns()
    order = _run_scheduled_order([
        (Direction.OUTGOING, now_ns),
        (Direction.INCOMING, now_ns - 2_000_000_000),
    ])

    assert order == [Direction.INCOMING, Direction.OUTGOING]


def test_adapter_reports_inference_wait_for_each_session():
    samples = []
    adapter = WhisperASRAdapter(
        min_audio_rms=0.0,
        on_inference_wait=lambda session, sample: samples.append((session, sample)),
    )
    adapter.model = object()
    adapter._decode_audio = lambda audio, language, **kwargs: (
        "Merhaba",
        {
            "asr_inference_wait_ms": 12.5,
            "asr_inference_queue_depth": 2,
            "asr_inference_deadline_miss_ms": 3.0,
        },
    )
    session = adapter.create_session("tx", Direction.OUTGOING, "tr")

    event = adapter.process_audio(
        session,
        np.ones(4800, dtype=np.float32),
        time.monotonic_ns(),
    )

    assert event is not None
    assert len(samples) == 1
    assert samples[0][0] is session
    assert samples[0][1] == {
        "wait_ms": 12.5,
        "queue_depth": 2,
        "deadline_miss_ms": 3.0,
        "is_final": False,
    }


@pytest.mark.parametrize("is_final,max_length", [(False, 68), (True, 448)])
@pytest.mark.parametrize("speech_verified", [False, True])
def test_huggingface_decode_uses_one_length_owner_and_current_language(is_final, max_length, speech_verified):
    adapter = WhisperASRAdapter(min_audio_rms=0.0, beam_size=3)

    class FakeProcessor:
        def __call__(self, audio, sampling_rate, return_tensors):
            return SimpleNamespace(input_features=torch.ones((1, 80, 20)))

        def batch_decode(self, ids, **kwargs):
            assert kwargs["clean_up_tokenization_spaces"] is False
            return ["Merhaba nasılsınız?"]

    class FakeModel:
        def __init__(self):
            self.generation_config = SimpleNamespace(
                max_length=448,
                max_new_tokens=None,
                forced_decoder_ids=[[1, None], [2, 50360]],
            )
            self.received = None

        def generate(self, input_features, generation_config):
            self.received = generation_config
            scores = (torch.zeros((1, 8)), torch.zeros((1, 8)))
            return SimpleNamespace(sequences=torch.tensor([[1, 2]]), scores=scores)

    adapter.processor = FakeProcessor()
    adapter.model = FakeModel()
    if speech_verified:
        adapter.min_audio_rms = .003
    text, info = adapter._decode_audio(
        np.ones(4800, dtype=np.float32) * (.001 if speech_verified else .1), "tr",
        is_final=is_final, speech_verified=speech_verified,
    )
    if speech_verified and not is_final:
        assert text == ""
        assert adapter.model.received is None
        return

    assert text == "Merhaba nasılsınız?"
    assert "avg_logprob" in info
    assert adapter.model.generation_config.max_length == 448
    assert adapter.model.received.max_length == max_length
    assert adapter.model.received.max_new_tokens is None
    assert adapter.model.received.forced_decoder_ids is None
    assert adapter.model.received.language == "tr"
    assert adapter.model.received.task == "transcribe"
    assert adapter.model.received.num_beams == 3


def test_initialize_resolves_unavailable_cuda_to_cpu(monkeypatch, tmp_path, caplog):
    loaded = {}

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return object()

    class FakeModel:
        def __init__(self):
            self.generation_config = SimpleNamespace(max_length=448, forced_decoder_ids=[])
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded["dtype"] = kwargs["dtype"]
            loaded["model"] = FakeModel()
            return loaded["model"]

    monkeypatch.setattr(whisper_backend, "AutoProcessor", FakeProcessorLoader)
    monkeypatch.setattr(whisper_backend, "AutoModelForSpeechSeq2Seq", FakeModelLoader)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    adapter = WhisperASRAdapter()
    with caplog.at_level(logging.WARNING):
        adapter.initialize(str(tmp_path), device="cuda", compute_type="float16")

    assert adapter.device == "cpu"
    assert adapter.compute_type == "float32"
    assert loaded["dtype"] == torch.float32
    assert loaded["model"].moved_to == "cpu"
    assert "CUDA is unavailable" in caplog.text


def test_partial_decode_is_coalesced_even_when_first_attempt_returns_empty():
    adapter = WhisperASRAdapter(partial_interval_ms=500, min_audio_rms=0.0)
    adapter.model = object()
    calls = []

    def fake_decode(audio, language):
        calls.append(len(audio))
        return "", {"avg_logprob": -0.2}

    adapter._decode_audio = fake_decode
    session = adapter.create_session("tx", Direction.OUTGOING, "tr")
    frame = np.ones(320, dtype=np.float32) * 0.1

    for _ in range(15):
        adapter.process_audio(session, frame, 1)
    assert len(calls) == 1

    for _ in range(24):
        adapter.process_audio(session, frame, 2)
    assert len(calls) == 1
    adapter.process_audio(session, frame, 3)
    assert len(calls) == 2


def test_flush_redecodes_complete_buffer_instead_of_committing_stale_partial():
    adapter = WhisperASRAdapter(min_audio_rms=0.0)
    adapter.model = object()
    adapter._decode_audio = lambda audio, language: (
        "Merhaba nasılsınız?",
        {"avg_logprob": -0.2, "audio_samples": len(audio)},
    )
    session = adapter.create_session("tx", Direction.OUTGOING, "tr")
    session.audio_buffer = [np.ones(4800, dtype=np.float32), np.ones(3200, dtype=np.float32)]
    session.total_audio_samples = 8000
    session.last_partial_text = "Merhaba"

    event = adapter.flush_session(session)

    assert event is not None
    assert event.text == "Merhaba nasılsınız?"
    assert event.model_info["audio_samples"] == 8000
    assert session.audio_buffer == []
    assert session.total_audio_samples == 0


def test_flush_preserves_real_capture_span_and_resets_it_for_next_utterance():
    adapter = WhisperASRAdapter(min_audio_rms=0.0)
    adapter.model = object()
    adapter._decode_audio = lambda audio, language: (
        "Merhaba",
        {"avg_logprob": -0.2},
    )
    session = adapter.create_session("tx", Direction.OUTGOING, "tr")

    adapter.process_audio(session, np.ones(3200, dtype=np.float32), 1_000_000_000)
    adapter.process_audio(session, np.ones(4800, dtype=np.float32), 1_300_000_000)
    event = adapter.flush_session(session)

    assert event is not None
    assert event.audio_start_ns == 800_000_000
    assert event.audio_end_ns == 1_300_000_000

    adapter.process_audio(session, np.ones(4800, dtype=np.float32), 2_000_000_000)
    next_event = adapter.flush_session(session)

    assert next_event is not None
    assert next_event.audio_start_ns == 1_700_000_000
    assert next_event.audio_end_ns == 2_000_000_000


def test_strip_prompt_prefix():
    from voice_translator.asr.whisper_backend import strip_prompt_prefix

    prompt = "Toplantı, Türkçe, teknik, iş, günlük konuşma."
    # Case 1: Exact prompt repeated by Whisper
    raw1 = "Toplantı, Türkçe, teknik, iş, günlük konuşma. Bugün güne sakin bir başlangıç yaptım."
    assert strip_prompt_prefix(raw1, prompt) == "Bugün güne sakin bir başlangıç yaptım."

    # Case 2: Multi-sentence prompt repeated
    prompt2 = "Merhaba. Türkçe iş, teknik ve günlük konuşma toplantısı."
    raw2 = "Merhaba. Türkçe iş, teknik ve günlük konuşma toplantısı. Kahvemi içerken gün içinde yapacağım işleri düşündüm."
    assert strip_prompt_prefix(raw2, prompt2) == "Kahvemi içerken gün içinde yapacağım işleri düşündüm."

    # Case 3: Prompt not repeated
    raw3 = "Hava oldukça güzeldi."
    assert strip_prompt_prefix(raw3, prompt) == "Hava oldukça güzeldi."


def test_empty_final_decode_does_not_revive_hallucinated_partial():
    adapter = WhisperASRAdapter(min_audio_rms=0)
    adapter._decode_audio = lambda *args, **kwargs: ("", {"no_speech_prob": 0.99})
    session = adapter.create_session("tx", Direction.OUTGOING, "tr")
    session.audio_buffer = [np.ones(4800, dtype=np.float32)]
    session.total_audio_samples = 4800
    session.last_partial_text = "Abone ol"
    session.metadata["last_model_info"] = {"avg_logprob": -0.2}
    assert adapter.flush_session(session) is None
    assert session.total_audio_samples == 0


def test_ct2_load_uses_absolute_local_path_and_requested_compute_type(monkeypatch, tmp_path):
    (tmp_path / "model.bin").write_bytes(b"fixture")
    received = {}
    monkeypatch.setattr(whisper_backend, "WhisperModel", lambda **kwargs: received.update(kwargs) or object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    adapter = WhisperASRAdapter(backend_preference="faster_whisper")
    adapter.initialize(str(tmp_path), compute_type="int8_float16")
    assert received["model_size_or_path"] == str(tmp_path.resolve())
    assert received["local_files_only"] is True
    assert received["compute_type"] == "int8_float16"


def test_ct2_unavailable_never_falls_back_to_huggingface(monkeypatch, tmp_path):
    import pytest
    (tmp_path / "model.bin").write_bytes(b"fixture")
    monkeypatch.setattr(whisper_backend, "WhisperModel", None)
    with pytest.raises(RuntimeError, match="faster-whisper is required"):
        WhisperASRAdapter().initialize(str(tmp_path))


def test_ct2_preserves_real_prompt_and_subtitle_words():
    adapter = WhisperASRAdapter(min_audio_rms=0)
    adapter.backend_type = "faster_whisper"
    adapter.model = SimpleNamespace(transcribe=lambda *args, **kwargs: (
        [SimpleNamespace(text="Altyazı hakkında konuşalım. Teşekkür ederim.")], SimpleNamespace()))
    text, _ = adapter._decode_audio(np.ones(4800), "tr", prompt="Altyazı hakkında konuşalım.")
    assert text == "Altyazı hakkında konuşalım. Teşekkür ederim."
