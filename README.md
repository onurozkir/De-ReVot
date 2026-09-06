# De-ReVot

Speak your language. Be heard in theirs.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OS: Windows 11](https://img.shields.io/badge/OS-Windows%2011%20Native-orange.svg)](https://www.microsoft.com/windows)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A local, private, real-time, full-duplex voice translator for **MS Teams, Zoom, Google Meet (Chrome / Edge), Discord, Slack Huddles, and Games** on native Windows. Application presets provide setup guidance over the same VB-CABLE and WASAPI audio paths. Per-application remote-call validation remains a separate hardware gate.

Translate your speech live between **Turkish**, **English** and **French** while preserving your original identity through **cross-lingual voice cloning**.

---

## Key Features

- **Any supported source ➔ any supported target (Outgoing)**: Speak Turkish, English or French into your physical microphone; cloned speech in the selected meeting language (Turkish, English or French) reaches the target application's microphone through VB-CABLE.
- **Meeting audio ➔ live subtitles in your language (Incoming)**: Speaker loopback feeds subtitles in your selected display language to both the primary WhisperLiveKit Web UI and an automatic floating desktop HUD.
- **Hands-free or Push-to-Talk**: Start Meeting starts both directions, including games. PTT gates only your microphone; incoming subtitles stay active. Default global keys: `V` to talk, `F9` to switch mode, `Ctrl+Shift+T` to pause/resume both directions, `Ctrl+Shift+M` for outgoing emergency mute.
- **Multi-reference voice cloning (XTTS-v2)**: Use six clean 8–10 second WAVs or an existing single reference. Validated, cached speaker conditioning supports every target language in XTTS-v2's synthesis matrix; identity and prosody require listening evaluation. See the [voice recording guide](docs/voice-recording-guide.md).
- **Microphone noise reduction and echo cancellation**: Offline WebRTC processing runs before VAD/ASR. Speaker loopback supplies a separate bounded echo reference, including while incoming subtitles are paused.
- **Live Mid-Meeting Switching**:
  - Switch voice profiles during a session; uncached conditioning must complete before the new profile is used.
  - Switch source and target languages on-the-fly (e.g. **Turkish ➔ English** to **English ➔ French**, or **French ➔ Turkish**) without restarting or interrupting audio streams.
- **Local inference**: ASR and TTS use the GPU; default MT uses CPU INT8. No cloud translation service or recurring API subscription is required.
- **Hardware-Aware Diagnostics**: Real-time dBFS audio meters for physical mic, WASAPI loopback, and VB-CABLE virtual render, along with P50/P95 end-to-end latency telemetry and VRAM monitoring.

---

## System Requirements

### Hardware
- **Operating System**: Windows 11 (64-bit) native execution (required for native WASAPI loopback capture).
- **GPU**: NVIDIA GPU with CUDA support and at least **12 GB VRAM** (16 GB VRAM recommended).
  - Target hardware: NVIDIA GeForce RTX 5060 Ti (16 GB).
- **RAM**: 16 GB minimum (32 GB recommended).
- **Storage**: ~15 GB free NVMe / SSD disk space for offline model checkpoints.

### Software
- **Python**: Version 3.12 (64-bit).
- **NVIDIA Driver & CUDA**: CUDA 12.x compatible driver.
- **Virtual Audio Device**: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (Free virtual audio driver).

---

## Architecture & Pipeline

```
[ Physical Mic ] ──> [ VAD / PTT ] ──> [ Whisper ASR ] ──> [ CTranslate2 MT ] ──> [ XTTS-v2 Voice Cloning ] ──> [ VB-CABLE Input ] ──> [ App Mic ]
                                              │                         │                               │
                                      (Chosen source language)  (Chosen target language text)     (Cloned speech PCM)

[ App Speaker ] ──> [ WASAPI Loopback ] ──> [ Whisper ASR ] ──> [ CTranslate2 MT ] ──> [ Web UI + Desktop HUD ]
                                                    │                         │
                                            (Incoming audio)          (Chosen subtitle language)
```

- **VAD**: Silero VAD with speech envelope hysteresis and adaptive hangover preserving natural Turkish SOV sentence structure when the source language is Turkish.
- **ASR**: `openai/whisper-large-v3-turbo` or `Systran/faster-whisper-large-v3`, with acoustic evidence filtering and no phrase blacklist.
- **MT**: Helsinki-NLP OPUS-MT TC-Big pairs (`tr-en`, `en-tr`, `tr-fr`) with CTranslate2 INT8 / HuggingFace MarianMT execution; NLLB-200 covers every other pair between enabled languages.
- **TTS**: Coqui XTTS-v2 with persistent speaker latent caching.

---

## Step-by-Step Installation

### 1. Install VB-Audio Virtual Cable
1. Download **VB-CABLE Driver** from [VB-Audio](https://vb-audio.com/Cable/).
2. Extract the archive and run `VBCABLE_Setup_x64.exe` as Administrator.
3. Reboot your computer if prompted.

### 2. Clone Repository & Setup Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/onurozkir/De-ReVot.git
cd speech-to-translate-en-tr

# Create Python 3.12 virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install package in editable mode
python -m pip install --upgrade pip
pip install -e .
```

### 3. Download Model Weights
Models are downloaded once and cached offline in `models/`.

> [!WARNING]
> **Do not blindly run `python scripts/download_models.py all`** unless you need every single supported language and model checkpoint. Running `all` downloads over 12+ GB of weights across Whisper, all OPUS-MT checkpoints, NLLB-200 multilingual models, and XTTS-v2.
>
> Instead, download only the models required for your active setup.

**Recommended Minimal Setup (Turkish ↔ English + Voice Cloning):**
```powershell
# 1. Speech Recognition (Whisper Large v3 Turbo)
python scripts/download_models.py whisper

# 2. Real-Time Translation (OPUS-MT with automatic INT8 quantization)
python scripts/download_models.py mt-tr-en --convert-ct2
python scripts/download_models.py mt-en-tr --convert-ct2
python scripts/download_models.py mt-en-fr --convert-ct2

# 3. Voice Cloning (XTTS-v2)
python scripts/download_models.py xtts
```

**Optional Additional Models:**
```powershell
# Optional: Full large-v3 decoder, already in CTranslate2 format
python scripts/download_models.py whisper-large-v3

# Optional: Turkish -> French translation
python scripts/download_models.py mt-tr-fr --convert-ct2

# Optional: Higher-quality multilingual model (Meta NLLB-200 Distilled 600M)
python scripts/download_models.py mt-nllb-200 --convert-ct2
```

For a language enabled in `config/default.toml`, download every pinned pair
involving it with `--lang` (pairs without a pinned OPUS model fall back to
NLLB-200):

```powershell
python scripts/download_models.py --lang fr
```

---

## Translation Engine: OPUS-MT vs. NLLB-200

The translator supports two machine translation backends running on CPU via **CTranslate2 INT8** quantization:

### 1. OPUS-MT TC-Big (Default & Recommended)
- **Why choose OPUS-MT?**
  - **Ultra-Low Latency**: Measured **P50: ~88 ms** (CTranslate2 INT8). Fits comfortably within the strict live meeting real-time budget.
  - **Minimal Memory Footprint**: Only **~238 MB** per language pair.
  - **Low CPU Overhead**: Leaves maximum CPU/GPU headroom for Whisper ASR and XTTS-v2 voice synthesis.
  - **Bilingual Focus**: Dedicated bilateral models specifically trained on English ↔ Turkish conversational datasets.

### 2. Meta NLLB-200 Distilled 600M (Alternative High-Quality Model)
- **Why choose NLLB-200?**
  - **Broad Vocabulary & Idioms**: Meta's state-of-the-art multilingual model excels with complex sentence structures, technical slang, and nuanced idioms.
  - **Universal Multilingual Support**: Single 600M parameter model supports 200+ languages (FLORES-200 tags: `tur_Latn`, `eng_Latn`, `fra_Latn`, etc.).
  - **Costs & Trade-offs**:
    - **Higher Latency**: Measured **P50: ~310 ms** in INT8 (~3.5x slower than OPUS-MT, or ~680 ms in unquantized PyTorch float32).
    - **Larger Memory Footprint**: ~600 MB in INT8 (~2.46 GB in float32).

### Performance & Resource Cost Comparison

| Model | Format / Runtime | Measured P50 Latency | Disk / Memory | Best Use Case |
|---|---|---|---|---|
| **OPUS-MT TC-Big** *(Default)* | CTranslate2 INT8 | **~88 ms** | **~238 MB** | **Live meetings**, low latency, lower-spec PCs |
| **OPUS-MT TC-Big** | HuggingFace Float32 | ~188 ms | ~470 MB | Fallback mode |
| **NLLB-200 Distilled 600M** | CTranslate2 INT8 | **~310 ms** | **~600 MB** | Complex idioms, literary prose, multilingual |
| **NLLB-200 Distilled 600M** | HuggingFace Float32 | ~680 ms | ~2.46 GB | High-spec machines requiring maximal fidelity |

### How to Configure the Active Model

Edit `config/default.toml` (or create an override in `config/local.toml`):

```toml
[translation]
# Options:
#   "auto" : Load installed OPUS pairs and NLLB-200; route per language pair (default)
#   "opus" : Explicitly enforce OPUS-MT pairs only
#   "nllb" : Enforce Meta NLLB-200 Distilled 600M only
model_type = "auto"   # change to "nllb" to activate NLLB-200
```

To manually convert any downloaded HuggingFace model checkpoint to CTranslate2 INT8:
```powershell
python scripts/convert_models_ct2.py --model models/mt/nllb-200-distilled-600M
```

### Domain Glossary & Context Priming

- **Domain Glossary (`[translation.glossary]`)**: Protects technical, enterprise, or project-specific terminology from being mistranslated:
  ```toml
  [translation.glossary]
  "pull request" = "pull request"
  "standup" = "standup"
  "deploy" = "deploy"
  "pipeline" = "pipeline"
  "arka uç" = "backend"
  "ön yüz" = "frontend"
  ```
- **Discourse Context Priming (`enable_context_priming = true`)**: Feeds the previous committed sentence as discourse context to the decoder, resolving Turkish pro-drop ambiguities (e.g. distinguishing *"I made"* vs *"they made"*).

---

## Application Setup

Every preset uses this same routing: translator input = physical microphone;
translator render = **CABLE Input**; target app microphone = **CABLE Output**;
target app speaker = physical headset; translator incoming = that headset's
**[Loopback]** endpoint. The names Input and Output are from the cable driver's
perspective. Presets guide these choices and never change Windows or game settings.

| Application | Setup guidance |
|---|---|
| MS Teams | Devices: CABLE Output microphone, physical headset speaker. Check cloned speech in a test call. |
| Zoom | Audio: CABLE Output microphone, headset speaker. Compare suppression settings if speech is clipped. |
| Google Meet (Chrome / Edge) | Meet Audio settings: CABLE Output microphone, headset speaker. In Windows Volume mixer, route the browser to the same headset. This captures endpoint audio, not an isolated browser tab. |
| Discord | Voice & Video: CABLE Output input, headset output. Use Voice Activity and adjust sensitivity/Krisp if translated speech is clipped. |
| Slack Huddles | Audio & video: CABLE Output microphone, headset speaker. |
| VRChat | Select CABLE Output input and headset output. Keep the game's microphone enabled during translated playback. |
| Dota 2 / PUBG / CS2 | Select CABLE Output as voice input, or Windows default communications input where the game uses the default. Keep game voice input open while translation plays; use translator PTT to gate your physical mic. Use borderless window for the HUD. |

Headset loopback also captures game effects, notifications and other audio on the
same endpoint. The baseline does not isolate voice chat from game effects. A
separate voice-output device can help when the game provides that setting.
Process loopback would isolate incoming processes; it does not replace the
outgoing virtual microphone, and is deferred here. No extra cable or paid
translation service was added.

## Push-to-Talk and Desktop Subtitles

Choose an Application preset and microphone mode, then click **Start Meeting**.
Gaming presets default to PTT; meetings default to hands-free VAD. Both audio
directions remain in the same session. Mode can change live using `F9` or the UI.

PTT retains 150 ms of microphone pre-roll. Releasing the configured key requests
the final decode immediately without an added VAD silence timer. ASR, MT and TTS
still take time. Committed translated speech continues after key release, so using
the same short PTT press in the game would clip that speech; use the game's open
mic/voice-activity mode. When no committed speech is playing, VB-CABLE renders
digital silence. The Web UI also has a hold-to-talk button with release on blur.

The desktop HUD starts with the session by default. It is transparent, topmost,
click-through and does not take keyboard focus. It shows the latest incoming
translation and replaceable partial, then clears after inactivity. Toggle it live
from the UI; Stop Meeting releases both the HUD window and global hooks. Settings
for font size, width and bottom margin are in `[overlay]` in `config/local.toml`.
It targets the primary desktop monitor and borderless/windowed games. Exclusive
fullscreen, protected/anti-cheat surfaces and per-game compatibility are not
guaranteed; the implementation does not inject into games or bypass protection.

Emergency Mute clears outgoing buffered PCM and cancels older pending speech; it
keeps incoming subtitles running. Pause suspends translation in both directions
and resumes from new audio. Original incoming sound still reaches your headset.
There is no raw-microphone fallback that could transmit speech unexpectedly.

## Full large-v3 and Hallucination Filtering

Download explicitly (never during startup or tests):

```powershell
uv sync
uv run python scripts/download_models.py whisper-large-v3
```

Repository: `Systran/faster-whisper-large-v3`; pinned revision:
`edaa852ec7e145841d8ffdb056a99866b5f0a478`; license: MIT; destination:
`models/asr/whisper-large-v3/`. The downloader records provenance and SHA256 file
checksums in `download-manifest.json`. No conversion is needed for this model.

Refresh the Web UI, select **Faster Whisper large-v3**, and start a session. The
model loads and warms before capture starts; the same weights serve both
directions. To select it persistently, add to `config/local.toml`:

```toml
[asr]
backend = "faster_whisper"
model_path = "models/asr/whisper-large-v3"
compute_type = "int8_float16" # CUDA; use int8 for CPU, or float16 for CUDA comparison
```

To return to turbo, select it in the UI or use `backend = "whisper_turbo"`,
`model_path = "models/asr/whisper-large-v3-turbo"`, `compute_type = "float16"`.
Absolute local model paths remain supported. Model changes apply between sessions.

The guard checks voiced duration/ratio, audio age, confidence and letters/digits
per voiced second (`streaming.guard_max_chars_per_second`, default 50). This is a
tunable plausibility check, not a guarantee of zero hallucinations. Real speech
such as “teşekkür ederim”, “abone ol” and “altyazı” is never removed merely because
of its words. The model's larger decoder is a quality option; it can still make
mistakes. The model card documents its [CTranslate2 format and precision options](https://huggingface.co/Systran/faster-whisper-large-v3).

Compare the same local audio excerpt with both models:

```powershell
uv run python scripts/benchmark_asr_acoustics.py --model turbo --audio voices/default/reference.wav --audio-offset 2 --audio-seconds 4 --output benchmarks/asr/results/turbo.json
uv run python scripts/benchmark_asr_acoustics.py --model large_v3 --audio voices/default/reference.wav --audio-offset 2 --audio-seconds 4 --output benchmarks/asr/results/large-v3.json
```

Reports separate raw output and guard acceptance, P50/P95, RTF, shared-ASR wait
and whole-GPU VRAM. `--corpus` accepts a JSON array with `path` (relative to JSON),
`language`, optional reference `text`, and `kind` (`speech` / `non_speech`). Supply
real silence/breath/keyboard clips and genuine short phrases for quality gates.
Synthetic negatives and an ASR-only comparison do not prove a live-call or
30-minute full-duplex result. Without reference text, WER/CER remain UNKNOWN.

---

## Dynamic Languages

The Web UI has two live-switchable selectors:

- **My Language**: ASR language of your microphone and the display language of
  incoming subtitles.
- **Target Language**: language of the cloned speech sent to the meeting
  and the ASR language of the meeting loopback.

Defaults are `tr` (source) and `en` (target). Switching during an active meeting
immediately updates ASR sessions, MT routing and the TTS target language; the
current partial utterance resets so languages never mix mid-sentence. The Web UI
disables target options whose offline models are missing or which XTTS-v2 cannot
synthesize.

### Adding a new language

1. Add its definition and enable it in `config/default.toml`:

   ```toml
   [languages]
   enabled = ["tr", "en", "fr"]

   [languages.definitions.en]
   name = "English"
   whisper_code = "en"
   nllb_code = "en_Latn"
   xtts_supported = true
   asr_prompt = "Meeting, English, technical, business."
   ```

2. Download its MT pairs (pinned OPUS pairs when known, NLLB-200 fallback
   otherwise; never during startup):

   ```powershell
   uv run python scripts/download_models.py --lang en
   ```

3. Refresh the Web UI. The new language appears in both selectors.

Pair routing: a pair uses its pinned OPUS model (e.g. `tr-en` = TC-Big) when the
directory exists and falls back to NLLB-200 (`models/mt/nllb-200-distilled-600M`)
for any other pair. NLLB-200 is CC-BY-NC — fine for personal use. XTTS-v2
synthesizes `en, tr, fr, de, es, it, pt, pl, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi`;
other target languages are rejected before the session starts.

---

## Running the Application

1. Start the server:
   ```powershell
   python src/voice_translator/main.py run
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
3. Configure devices in the Web Dashboard:
   - **Physical Mic**: Your physical microphone.
   - **Incoming Audio (Loopback)**: Your physical headphones/speakers with `[Loopback]`.
   - **VB-CABLE Render**: `CABLE Input (VB-Audio Virtual Cable)`.
   - **Voice Profile**: Choose your personal voice or an avatar voice.
   - **My Language**: Source language you speak (default `tr`).
   - **My Language / Target Language**: Language the meeting hears and you read as subtitles (default `en`).
4. Click **Start Meeting**.

The Python package is now `voice_translator`, and the project skill lives at
`.agents/skills/realtime-voice-translator/SKILL.md`. Use the new launch path above.
`VOICE_TRANSLATOR_*` environment overrides take priority; legacy
`TEAMS_TRANSLATOR_*` overrides and existing saved audio device choices still work.

### CPU / Mock Mode (For testing without GPU)
```powershell
python src/voice_translator/main.py run --mock
```

---

## Adding Custom Voice Profiles

1. Create `voices/default/` and record six clean WAVs with the same microphone,
   speaker and room. Aim for 8–10 seconds each, without aggressive audio filters.
2. Name them `voice_01.wav` through `voice_06.wav`. A manifest is optional; WAVs
   directly inside the directory are discovered in sorted order.
3. Validate offline: `uv run python scripts/validate_voice_profile.py default`.
4. Restart, select **default (6 WAV, xtts_v2)**, then Start Meeting. Preparation
   happens before the voice is used; cached latents are reused during synthesis.

```text
voices/default/
├── profile.json       # optional name, languages, explicit file list
├── voice_01.wav
├── voice_02.wav
├── voice_03.wav
├── voice_04.wav
├── voice_05.wav
├── voice_06.wav
└── cache/             # generated conditioning cache
```

Existing manifests with `"reference_audio_path": "reference.wav"` remain valid.
Explicit multi-file manifests use `reference_audio_paths`; missing files fail
instead of silently reducing the dataset. `/api/profiles` includes
`reference_audio_paths`, `reference_count` and manifest errors.

See the [recording guide](docs/voice-recording-guide.md) for the full manifest,
six Turkish reading scripts, validation thresholds, cache lifecycle and XTTS's
30-second GPT conditioning window. This workflow is conditioning, not fine-tuning.

## Background noise and speaker echo

Run `uv sync` once after updating to install the pinned native Windows dependency
[`aec-audio-processing==1.0.1`](https://pypi.org/project/aec-audio-processing/)
(BSD-3-Clause; WebRTC AudioProcessing, not a claim of AEC3). No new model weights
or GPU memory are required. Both controls are enabled by default and available in
**Audio & Devices** for the next session. With headphones, echo cancellation can
be disabled while keeping noise reduction enabled.

Persistent settings in `config/local.toml`:

```toml
[audio]
noise_suppression = true
echo_cancellation = true
noise_suppression_level = 2 # 0 low, 1 moderate, 2 high, 3 very high
echo_delay_ms = 50         # estimated loopback-to-mic delay; calibrate per device
```

Select the loopback matching your physical speaker. The diagnostics panel reports
missing echo reference or DSP errors; failures mute outgoing delivery visibly.
Noise reduction can reduce steady traffic/engine noise, but transient vehicle
sounds and Whisper hallucinations still need real recordings for verification.
No spoken-word blacklist is used. AEC preserves simultaneous microphone speech
instead of muting the mic whenever someone else speaks. Headphones remain the
most reliable way to avoid speaker bleed.

Offline benchmark (no device recording):

```powershell
uv run python scripts/benchmark_audio_processing.py --speech voices/default/reference.wav --output benchmarks/audio/results/processing.json
```

Synthetic noise/echo tests measure processing time and attenuation. Real outdoor
speech quality, end-to-end queue age and 30-minute full-duplex soak remain separate
gates.

---

## Running Tests

Run the complete test suite:
```powershell
python -m pytest tests/ -v
```

---

## License

- Code: [MIT License](LICENSE)
- XTTS-v2 Weights: Coqui Public Model License (CPML - Non-commercial / Personal use)
- Whisper & MarianMT: MIT / Apache 2.0 / CC-BY-4.0

