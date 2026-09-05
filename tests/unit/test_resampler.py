import numpy as np
import pytest
from voice_translator.audio.resampler import AudioResampler


def test_resampler_rate_conversion():
    resampler = AudioResampler(in_rate=48000, out_rate=16000)
    # 1 second of 48kHz audio
    audio_48k = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
    
    out_16k = resampler.process(audio_48k)
    assert len(out_16k) == 16000
    assert out_16k.dtype == np.float32


def test_resampler_stereo_to_mono():
    resampler = AudioResampler(in_rate=16000, out_rate=16000, in_channels=2, out_channels=1)
    stereo = np.ones((1600, 2), dtype=np.float32)
    mono = resampler.process(stereo)
    assert mono.ndim == 1
    assert len(mono) == 1600


def test_resampler_int16_scaling_and_output_contract():
    resampler = AudioResampler(in_rate=48000, out_rate=16000)
    source = np.array([-32768, 0, 32767] * 320, dtype=np.int16)
    output = resampler.process(source)
    assert output.dtype == np.float32
    assert output.ndim == 1
    assert np.max(np.abs(output)) <= 1.0


def test_streaming_soxr_chunk_continuity_and_flush():
    pytest.importorskip("soxr")
    source = (0.2 * np.sin(2 * np.pi * 997 * np.arange(48000) / 48000)).astype(np.float32)
    expected = AudioResampler(48000, 16000).process(source)
    streaming = AudioResampler(48000, 16000, streaming=True)
    parts = [streaming.process(source[offset : offset + 960]) for offset in range(0, len(source), 960)]
    parts.append(streaming.flush())
    actual = np.concatenate(parts)
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual, expected, atol=2e-5)
