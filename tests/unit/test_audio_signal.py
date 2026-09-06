import numpy as np
import asyncio

from voice_translator.audio.devices import DeviceInfo
from voice_translator.audio.render import AudioRenderEngine
from voice_translator.audio.resampler import AudioResampler


def test_long_render_preserves_all_pcm_with_bounded_ring_and_resampler_tail():
    async def run():
        device = DeviceInfo(1, "Fake", 1, "Windows WASAPI", 0, 2, 48000, False)
        render = AudioRenderEngine(device, ring_buffer_sec=.1)
        render.is_running = True  # No real device or callback is opened.
        source = np.linspace(-.2, .2, 24000 * 15, dtype=np.float32)
        output = []
        done = False

        async def consume():
            while not done or render.ring_buffer.available_read:
                if render.ring_buffer.available_read:
                    output.append(render.ring_buffer.read(4800))
                await asyncio.sleep(0)

        consumer = asyncio.create_task(consume())
        async for _ in render.enqueue_pcm(source, 24000, cancelled=lambda: False):
            assert render.ring_buffer.available_read <= render.ring_buffer.capacity
        async for _ in render.enqueue_pcm(np.empty(0, np.float32), 24000, cancelled=lambda: False, final=True):
            pass
        done = True
        await consumer
        np.testing.assert_allclose(np.concatenate(output), AudioResampler(24000, 48000).process(source), atol=2e-5)
        assert render.ring_buffer.overrun_count == 0
    asyncio.run(run())


def test_blocked_render_can_be_muted_without_delivering_remaining_pcm():
    async def run():
        device = DeviceInfo(1, "Fake", 1, "Windows WASAPI", 0, 2, 48000, False)
        render = AudioRenderEngine(device, ring_buffer_sec=.02)
        render.is_running = True
        async for _ in render.enqueue_pcm(np.ones(48000, np.float32), 48000, cancelled=lambda: False):
            render.set_muted(True)
        assert render.ring_buffer.available_read == 0
        assert render.ring_buffer.overrun_count == 0
    asyncio.run(run())

from voice_translator.audio.signal import downmix_to_mono, pcm_to_float32, signal_levels
from voice_translator.audio.diagnostic import dominant_frequency, frame_level_summary


def test_int16_scaling_maps_extremes_without_overflow():
    converted = pcm_to_float32(np.array([-32768, 0, 32767], dtype=np.int16))
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, [-1.0, 0.0, 32767 / 32768], atol=1e-7)
    assert np.all(converted >= -1.0)
    assert np.all(converted <= 1.0)


def test_interleaved_stereo_downmix_preserves_frame_count_and_mean():
    stereo = np.array([32767, -32768, 16384, 16384], dtype=np.int16)
    mono = downmix_to_mono(stereo, channels=2)
    assert len(mono) == 2
    np.testing.assert_allclose(mono, [-1 / 65536, 0.5], atol=2e-5)


def test_signal_levels_distinguish_silence_and_known_tone():
    silence = np.zeros(1600, dtype=np.float32)
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)).astype(np.float32)
    assert signal_levels(silence) == (0.0, 0.0, -120.0)
    rms, peak, dbfs = signal_levels(tone)
    assert 0.06 < rms < 0.08
    assert 0.099 < peak <= 0.101
    assert -24.0 < dbfs < -22.0


def test_diagnostic_frame_levels_require_sustained_signal():
    pcm = np.concatenate(
        [np.zeros(960, dtype=np.float32), np.full(960 * 12, 0.01, dtype=np.float32)]
    )
    summary = frame_level_summary(pcm, active_rms_threshold=0.001)
    assert summary["active_frames"] == 12
    assert summary["active_duration_ms"] == 240
    assert summary["max_frame_rms"] > 0.009


def test_diagnostic_dominant_frequency_finds_known_tone():
    tone = (0.05 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    assert abs(dominant_frequency(tone) - 440.0) < 1.0
