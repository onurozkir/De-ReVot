from types import SimpleNamespace

import numpy as np
import pytest

from voice_translator.asr.mock_backend import MockASRAdapter
from voice_translator.audio.devices import DeviceInfo
from voice_translator.config.loader import load_config
from voice_translator.core.types import Direction, UtteranceEvent, UtteranceState
from voice_translator.streaming.commit_policy import CommitController
from voice_translator.streaming.pipeline_incoming import IncomingPipeline
from voice_translator.streaming.pipeline_outgoing import OutgoingPipeline
from voice_translator.streaming.vad import VADResult
from voice_translator.translation.mock_backend import MockMTAdapter
from voice_translator.tts.base import VoiceProfile
from voice_translator.tts.mock_backend import MockTTSAdapter


ACTIVE_SPEECH = VADResult(
    active=True,
    probability=0.9,
    phase="speech",
    transition=None,
    rms=0.1,
    peak=0.2,
    dbfs=-20.0,
    utterance_ms=800.0,
    voiced_ms=700.0,
    voiced_ratio=0.875,
    model_backend="test",
)


def _partial_event(session, text, revision):
    return UtteranceEvent(
        meeting_id="test",
        stream_id=session.stream_id,
        direction=session.direction,
        utterance_id=f"{session.stream_id}_{session.sequence_id}",
        sequence_id=session.sequence_id,
        revision=revision,
        state=UtteranceState.PARTIAL,
        source_language=session.language,
        text=text,
        audio_start_ns=1,
        audio_end_ns=2,
        is_final=False,
        model_info={"backend": "test"},
    )


def _build_pipeline(direction):
    config = load_config()
    adapter = MockASRAdapter()
    device = DeviceInfo(1, "Device", 1, "Windows WASAPI", 2, 2, 48000, direction == Direction.INCOMING)
    if direction == Direction.OUTGOING:
        profile = VoiceProfile("test", "Test", "mock", "reference.wav")
        pipeline = OutgoingPipeline(
            "test", device, device, adapter, MockMTAdapter(), MockTTSAdapter(), profile, config
        )
        session = adapter.create_session("tx_test", direction, "tr")
    else:
        pipeline = IncomingPipeline("test", device, adapter, MockMTAdapter(), config)
        session = adapter.create_session("rx_test", direction, "en")

    pipeline.asr_session = session
    pipeline.is_running = True
    pipeline.resampler_in.process = lambda frame: frame
    pipeline.vad = SimpleNamespace(process=lambda frame: ACTIVE_SPEECH)
    pipeline.guard = SimpleNamespace(
        evaluate=lambda text, evidence, model_info: SimpleNamespace(accepted=True, reason="accepted")
    )
    pipeline.commit_controller = CommitController(min_words=2, max_wait_ms=10_000)
    return pipeline, adapter, session


@pytest.mark.parametrize("direction", [Direction.OUTGOING, Direction.INCOMING])
def test_split_commit_is_dispatched_and_remainder_becomes_next_partial(direction):
    pipeline, adapter, session = _build_pipeline(direction)
    emitted = []
    submitted = []
    pipeline.on_event_callback = emitted.append
    pipeline._submit_queue = lambda queue, item, name: submitted.append((name, item))

    texts = iter(["Hello stable clause. trailing words", "more speech"])
    revisions = iter([1, 2])
    adapter.process_audio = lambda current_session, audio, captured_at: _partial_event(
        current_session, next(texts), next(revisions)
    )

    pipeline._process_audio_frame(np.ones(320, dtype=np.float32), 2, 0.0)

    committed = [item for name, item in submitted if name.endswith("_committed")]
    assert [item.text for item in committed] == ["Hello stable clause."]
    assert session.sequence_id == 1
    assert session.last_partial_text == "trailing words"

    pipeline._process_audio_frame(np.ones(320, dtype=np.float32), 3, 0.0)

    if direction == Direction.OUTGOING:
        partial_texts = [item["text"] for item in emitted if item["type"] == "asr_partial"]
    else:
        partial_texts = [item.text for name, item in submitted if name == "incoming_partial"]
    assert partial_texts[-1] == "trailing words more speech"
