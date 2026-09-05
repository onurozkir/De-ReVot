import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_translator.tts.xtts_backend import XTTSv2Adapter
from voice_translator.tts.conditioning import VoiceProfileManager

print("=" * 60)
print(" BENCHMARK B4: XTTS-V2 VOICE CLONING TTS")
print("=" * 60)

model_path = "models/tts/xtts-v2"
if not Path(model_path).exists():
    print(f"XTTS model not found at '{model_path}'. Run 'python scripts/download_models.py xtts' first.")
    exit(1)

tts = XTTSv2Adapter()
tts.initialize(model_path=model_path, device="cuda", sample_rate=24000)
tts.warmup()

mgr = VoiceProfileManager()
prof = mgr.get_default_profile()

if prof:
    print(f"Using voice profile: {prof.display_name}")
    tts.prepare_voice_profile(prof)

    text = "Hello everyone, welcome to our Microsoft Teams engineering meeting."
    print(f"\nSynthesizing: '{text}'")
    
    t0 = time.monotonic_ns()
    pcm_chunks = list(tts.synthesize_committed(text, prof, "en"))
    t1 = time.monotonic_ns()

    total_samples = sum(len(c) for c in pcm_chunks)
    audio_dur_s = total_samples / 24000.0
    synth_time_s = (t1 - t0) / 1e9
    rtf = synth_time_s / audio_dur_s if audio_dur_s > 0 else 0

    print(f"Audio Duration:   {audio_dur_s:.2f} s")
    print(f"Synthesis Time:   {synth_time_s:.2f} s")
    print(f"Real-Time Factor: {rtf:.2f}x (Target: < 1.0x)")

tts.shutdown()
print("=" * 60)
