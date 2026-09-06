import numpy as np
import pytest

from voice_translator.audio.processing import EchoReferenceBuffer, MicrophoneProcessor
from voice_translator.config.models import AudioConfig


class FakeEngine:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.references = []
        self.inputs = []

    def set_stream_format(self, *args):
        pass

    def set_reverse_stream_format(self, *args):
        pass

    def set_stream_delay(self, delay):
        self.delay = delay

    def process_reverse_stream(self, data):
        self.references.append(np.frombuffer(data, dtype="<i2"))

    def process_stream(self, data):
        self.inputs.append(np.frombuffer(data, dtype="<i2"))
        return data


def test_bounded_reference_alignment_and_expiration():
    reference = EchoReferenceBuffer(16000, 20)
    for i in range(100):
        reference.push(np.full(320, i, dtype=np.float32), (i + 1) * 20_000_000)
    assert len(reference._blocks) == 50
    frame, complete = reference.frame(1_990_000_000, 320)
    assert complete
    np.testing.assert_array_equal(frame[:160], 98)
    np.testing.assert_array_equal(frame[160:], 99)
    assert not reference.frame(500_000_000, 160)[1]
    reference.clear()
    assert not reference.frame(2_000_000_000, 160)[1]


def test_capture_callback_only_copies_independent_echo_reference(monkeypatch):
    from types import SimpleNamespace
    import voice_translator.audio.capture as module
    from voice_translator.audio.devices import DeviceInfo
    callbacks = []
    stream = SimpleNamespace(start_stream=lambda: None, stop_stream=lambda: None, close=lambda: None)

    def open_stream(**kwargs):
        callbacks.append(kwargs["stream_callback"])
        return stream

    monkeypatch.setattr(module, "pyaudio", SimpleNamespace(
        PyAudio=lambda: SimpleNamespace(open=open_stream, terminate=lambda: None), paInt16=8, paContinue=0))
    device = DeviceInfo(1, "Loopback", 1, "Windows WASAPI", 1, 0, 16000, True)
    capture = module.AudioCaptureEngine(device, sample_rate=16000)
    capture.echo_reference = EchoReferenceBuffer(16000, 20)
    capture.start()
    raw = np.full(320, 3200, dtype=np.int16)
    callbacks[0](raw.tobytes(), 320, {}, 0)
    # Incoming consumer may drain its ring; AEC has an independent bounded copy.
    capture.read_samples()
    frame, complete = capture.echo_reference.frame(capture.last_callback_ns, 320)
    assert complete
    np.testing.assert_allclose(frame, 3200 / 32768)
    capture.stop()


def test_10ms_processing_preserves_timestamps_samples_and_double_talk():
    config = AudioConfig(sample_rate=16000, echo_delay_ms=30)
    reference = EchoReferenceBuffer(16000, 20)
    reference.push(np.full(320, 0.2, dtype=np.float32), 20_000_000)
    processor = MicrophoneProcessor(config, reference, FakeEngine)
    assert processor.process(np.full(80, 0.3, dtype=np.float32), 25_000_000) == []
    output = processor.process(np.full(240, 0.3, dtype=np.float32), 40_000_000)
    assert [t for _, t in output] == [30_000_000, 40_000_000]
    np.testing.assert_allclose(np.concatenate([a for a, _ in output]), 0.3, atol=1 / 32768)
    assert processor.reference_misses == 0
    assert all(np.all(a > 0) for a in processor.engine.references)
    assert processor.engine.options["enable_agc"] is False
    assert processor.engine.delay == 10  # Retain one causal block for WebRTC.


def test_missing_reference_never_mutes_local_speech_and_gap_resets():
    processor = MicrophoneProcessor(AudioConfig(sample_rate=16000), factory=FakeEngine)
    audio = np.full(320, 0.2, dtype=np.float32)
    assert len(processor.process(audio, 20_000_000)) == 2
    assert processor.reference_misses == 2
    assert len(processor.process(audio, 200_000_000)) == 2
    assert processor.resets == 1
    assert processor.snapshot()["pending_samples"] == 0


def test_invalid_native_output_and_config_fail_visibly():
    with pytest.raises(ValueError, match="requires"):
        MicrophoneProcessor(AudioConfig(sample_rate=44100), factory=FakeEngine)
    with pytest.raises(ValueError, match="multiple of 10"):
        MicrophoneProcessor(AudioConfig(frame_duration_ms=15), factory=FakeEngine)
    processor = MicrophoneProcessor(AudioConfig(sample_rate=16000), factory=FakeEngine)
    processor.engine.process_stream = lambda data: b""
    with pytest.raises(RuntimeError, match="frame size"):
        processor.process(np.zeros(160, dtype=np.float32), 10_000_000)
    assert "frame size" in processor.snapshot()["last_error"]


@pytest.mark.parametrize("rate", [16000, 48000])
def test_native_webrtc_offline_noise_attenuation(rate):
    # Deterministic negative control, not evidence of outdoor speech quality.
    pytest.importorskip("aec_audio_processing")
    processor = MicrophoneProcessor(AudioConfig(sample_rate=rate, echo_cancellation=False))
    rng = np.random.default_rng(16)
    before, after = [], []
    for i in range(300):
        block = rng.normal(0, 0.02, rate // 100).astype(np.float32)
        output = processor.process(block, (i + 1) * 10_000_000)[0][0]
        if i > 200:
            before.extend(block)
            after.extend(output)
    assert np.sqrt(np.mean(np.square(after))) < np.sqrt(np.mean(np.square(before))) * 0.8


def test_native_echo_filter_retains_causal_reference_lead():
    pytest.importorskip("aec_audio_processing")
    rate = 16000
    reference = EchoReferenceBuffer(rate, 10)
    processor = MicrophoneProcessor(AudioConfig(sample_rate=rate, noise_suppression=False), reference)
    far = np.random.default_rng(16).normal(0, 0.08, rate * 6).astype(np.float32)
    mic = np.zeros_like(far)
    mic[800:] = 0.6 * far[:-800]
    output = []
    for offset in range(0, len(mic), 160):
        at_ns = round((offset + 160) * 1e9 / rate)
        reference.push(far[offset:offset + 160], at_ns)
        output.extend(processor.process(mic[offset:offset + 160], at_ns)[0][0])
    # Keeping the reference exactly coincident with the echo regressed this to
    # ~5dB. Retain a causal block to support the native filter bank (>12dB).
    assert np.std(output[rate * 2:]) < np.std(mic[rate * 2:]) * 0.25
