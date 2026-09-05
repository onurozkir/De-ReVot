"""Phase B1: Audio benchmark script for WASAPI device verification."""

import time
import numpy as np
from voice_translator.audio.devices import AudioDeviceManager
from voice_translator.audio.capture import AudioCaptureEngine
from voice_translator.audio.render import AudioRenderEngine

print("=" * 60)
print(" BENCHMARK B1: AUDIO CAPTURE & VB-CABLE RENDER TEST")
print("=" * 60)

mgr = AudioDeviceManager()
mic = mgr.find_default_mic()
render = mgr.find_vbcable_render()

if not mic:
    print("Error: No microphone found.")
    exit(1)

print(f"Testing Microphone: {mic.name}")
capture = AudioCaptureEngine(device_info=mic, sample_rate=48000)
capture.start()

print("Capturing 3 seconds of audio...")
time.sleep(3.0)
samples = capture.read_samples()
capture.stop()

print(f"Captured {len(samples)} samples ({len(samples)/48000:.2f}s). Overruns: {capture.ring_buffer.overrun_count}")

if render:
    print(f"\nTesting VB-CABLE Output Render: {render.name}")
    render_engine = AudioRenderEngine(device_info=render, sample_rate=48000)
    render_engine.start()

    # Generate 1-second 440Hz test tone
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    print("Playing 440Hz test tone into VB-CABLE Input...")
    render_engine.push_pcm(tone)
    time.sleep(1.5)
    render_engine.stop()
    print("Audio playback completed.")
else:
    print("\n[NOTE] VB-CABLE Render device not found. Install VB-CABLE driver to test render endpoint.")

mgr.close()
print("=" * 60)

