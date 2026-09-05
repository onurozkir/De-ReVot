# Realtime Voice Translator for Meetings and Games

> Canonical architecture and delivery plan. This document supersedes architectural
> recommendations in the research history when they conflict with a DEC-Uxxx decision.
> Research documents remain unchanged evidence, not current authority.

Reference notation:

- [DEC: DEC-Uxxx] identifies a direct user decision.
- [REF: SRC-xx:Bxxx:Lx-Ly] identifies a logical block and its 1-based physical
  line range in a historical research source.
- TARGET is an acceptance objective, MEASURED is a result produced on the target
  PC, and UNKNOWN means no valid local measurement exists yet.

## 1. Status and Scope

Current workstream: **runtime stabilization and physical-audio verification across
Phases B-H**. The native Python application, local model adapters, web UI, audio
pipelines, and tests now exist. This does not mean the B-H acceptance gates have
passed: device enumeration is MEASURED, software regressions are TESTED, while raw
microphone and isolated physical-speaker loopback signal gates are MEASURED/PASS.
VB-CABLE-to-Teams, a live Teams test call, translation/clone quality and end-to-end
latency, reconnect behavior, and full-duplex soak remain UNKNOWN until their hardware
procedures are completed.

### 1.1 Runtime findings and corrections — through 2026-09-03

The phrase-denylist findings below are historical: DEC-U013 and §12.4 supersede
their filtering policy. Their recorded measurements remain unchanged.

The following observations were reproduced from the implementation and target PC;
they supersede the earlier empty-skeleton status without changing DEC-U001-U008:

| Observation | Status | Architectural consequence |
|---|---|---|
| PortAudio capture callbacks invoked resampling, VAD, and Whisper synchronously | FIXED/TESTED | Callbacks now only convert/downmix and write bounded PCM. Directional async workers drain live-edge PCM and run VAD/ASR outside callbacks. |
| VAD alternated Silero decisions with a per-frame hard RMS fallback whenever fewer than 512 samples were buffered | FIXED/TESTED | Installed offline Silero is the sole model path; deterministic energy VAD is used only if Silero cannot load. Start/end thresholds, confirmation frames, silence, and hangover are separately configurable. |
| Silence/noise ASR text could commit and reach TTS through a short phrase denylist | FIXED/TESTED | Commit now requires VAD speech duration/ratio, bounded queue age, optional Whisper confidence metadata, and a final normalized denylist. Rejected decoder state is reset. TTS additionally requires a committed event carrying accepted speech evidence. |
| PyAudio integer indexes were persisted and explicit lookup failures silently fell back to another endpoint | FIXED/TESTED | UI/config persist a host/name/format fingerprint. Explicit missing or role-incompatible selectors fail visibly; raw endpoint fields and resolved mapping are exposed. |
| Windows default input was `CABLE Output`, so it was incorrectly selected as the physical microphone | FIXED/MEASURED | `CABLE Output` is classified as VB-CABLE capture, not physical mic. The AB13X WASAPI capture path was then measured near-silent; its DirectSound analogue passed the sustained-signal gate and is the configured physical mic. |
| `[Loopback]` friendly-name normalization did not match its physical speaker endpoint | FIXED/MEASURED | The measured physical speaker/loopback pair is WASAPI 16/21. |
| WebSocket publication used `asyncio.create_task` from model/audio worker threads | FIXED/TESTED | Worker-originated events use a thread-safe handoff to the web event loop. |
| Committed and TTS queue rejection could be ignored | FIXED/TESTED | Queue age/depth/drop policy are visible; committed rejection raises an overload event instead of silent loss. |
| Resampling was restarted independently for every PCM chunk | FIXED/TESTED | ASR/TTS realtime paths use stateful soxr streams; each completed TTS utterance flushes delayed PCM before the resampler resets. |
| XTTS initialization reported that Coqui or PyTorch was absent | FIXED/MEASURED | `coqui-tts==0.27.5` is locked. Local XTTS-v2 initialized on CUDA, warmup completed, and the full orchestrator reached Ready without downloading weights. Import failures now preserve the actual dependency error. |
| XTTS import permanently replaced `torchaudio.load` process-wide to work around Coqui conditioning audio loading | FIXED/TESTED | `torchaudio.load` is no longer modified. The SoundFile compatibility loader temporarily replaces only `TTS.tts.models.xtts.load_audio` during profile conditioning and is restored under a lock in `finally`. |
| HuggingFace Whisper decoded every 20 ms audio frame, allowing inference time to stall capture draining and make later speech appear one utterance late | FIXED/TESTED | Full-window partial decoding is coalesced to a configurable 500 ms cadence. VAD endpoint performs one final decode over the complete buffered utterance instead of committing a stale partial. Split partial remainders are never discarded. |
| Shared-weight outgoing and incoming Whisper calls used one non-prioritized lock and exposed no inference wait telemetry | FIXED/TESTED; contention MEASURED | A bounded scheduler now orders queued work by captured-audio deadline, reserves downstream time for outgoing delivery, lets sufficiently older incoming audio win, and records wait/deadline-miss telemetry per direction. Shared model calls remain serialized for backend safety. |
| A rejected replaceable partial reset the active utterance buffer | FIXED/TESTED | Partial rejection is observable but non-destructive; only final rejection or overload resets utterance state. This lets later hypotheses repair early hallucinations. |
| `İzlediğiniz için teşekkür ederim` escaped the known-hallucination denylist and UI labeled MT completion as cloned audio before any PCM existed | FIXED/MEASURED | Turkish capital dotted-I normalization now matches the denylist; the previously captured noisy mic WAV reproduces and rejects this Whisper hallucination. Outgoing history is appended only after first PCM enters the bounded render path and is labeled `EN Routed`. |
| Transformers repeatedly logged conflicting `max_new_tokens=64` and model `max_length=448` during speech | FIXED/MEASURED | Whisper generation uses one bounded total-length owner compatible with the installed Transformers nested generation path. The exact warning is absent in local CUDA warmup; `HF_TOKEN` is irrelevant because weights load with `local_files_only=True`. |
| Final Whisper flush events replaced the captured audio span with the event-creation time | FIXED/TESTED | Each utterance now retains its first-sample and last-sample monotonic capture timestamps through endpoint flush, then resets them. Commit-latency P50/P95 therefore measures from the real audio edge instead of a near-zero placeholder. |
| Configured ASR/MT beam sizes and XTTS temperature/speed were ignored | FIXED/TESTED | Adapter construction now receives the declared settings. ASR uses its configured beam on both supported Whisper runtimes; committed MT uses its configured beam while replaceable partial MT intentionally remains greedy to protect latency; XTTS warmup and committed synthesis use configured temperature/speed. Their quality/latency effects remain UNKNOWN until B2-B4 measurement. |
| XTTS silently synthesized with zero conditioning tensors when a selected profile reference was missing | FIXED/TESTED | Missing reference audio or unavailable conditioning now fails before Ready with the exact profile error exposed by status/API/UI. A cache miss recomputes from valid reference audio; production synthesis never substitutes zero latents. |
| The reported short Whisper output `Abone ol` escaped the longer subscribe-phrase denylist | FIXED/TESTED; trigger UNKNOWN | The current text producer is `openai/whisper-large-v3-turbo` ASR, and the normalized exact guard now rejects this reported phrase. Whether a particular occurrence was triggered by silence/noise, endpoint contamination, or voiced audio remains UNKNOWN without its captured PCM and model evidence; model replacement still requires the B2 comparison gate. |
| MT models ran HuggingFace MarianMT on CPU float32 and had no domain glossary or context priming | FIXED/TESTED | Converted OPUS-MT models to CTranslate2 INT8 format with automatic directory resolution, added mandatory EOS token handling, added word-boundary domain glossary injection, and wired rolling discourse context priming to disambiguate Turkish pro-drop pronouns. Added conversion utility `scripts/convert_models_ct2.py` and NLLB-200 support. |
| Conservative commit/endpoint timing caused 750–950 ms dead-air latency after Turkish speech | FIXED/TESTED | Added Turkish SOV verb-aware adaptive endpointing (`enable_adaptive_sov=true`, `sov_min_silence_ms=200`). Detects finite Turkish predicate suffixes and immediate copulas to commit at 200 ms silence, while holding on conjunction tails (`ve`, `ama`, `çünkü`) and protecting deverbal nouns (`toplantı`, `alıntı`). Lowered default `commit_max_wait_ms` from 3500 ms to 1800 ms. (Fixes #17, #10) |
| XTTS voice cloning quality suffered from robotic artifacts, clipping, and single-sample variance | FIXED/TESTED | Tuned generation hyperparameters (`temperature=0.65`, `top_p=0.85`, `repetition_penalty=2.0`), added multi-sample reference conditioning support in `VoiceProfileManager` and `XTTSv2Adapter`, and added -1.0 dBFS (0.89125) peak normalization to eliminate digital harmonic clipping before rendering. (Fixes #18) |

Measured endpoint mapping on the target PC (PyAudio indexes are diagnostic only):

| Role | Measured native Windows endpoint | Stable selector | Format |
|---|---|---|---|
| Physical mic | DirectSound `Mikrofon (AB13X USB Audio)` index 8 | `pa:6fdfc0bd1fd45dcd` | Default 44.1 kHz; validated at runtime with 48 kHz, 2 input channels |
| Rejected physical-mic alias | WASAPI `Mikrofon (AB13X USB Audio)` index 18 | `pa:b7eeb3f5873eaa37` | 48 kHz, 2 input channels; near-silent on this driver/path |
| Physical speaker | `Hoparlör (AB13X USB Audio)` index 16 | `pa:836d4aa4b129454f` | 48 kHz, 2 output channels |
| Physical speaker loopback | `Hoparlör (AB13X USB Audio) [Loopback]` index 21 | `pa:77c84e773b1deac5` | 48 kHz, 2 input channels |
| VB-CABLE render | `CABLE Input (VB-Audio Virtual Cable)` index 14 | `pa:53a42370d5a570e0` | 48 kHz, 2 output channels |
| VB-CABLE capture | `CABLE Output (VB-Audio Virtual Cable)` index 17 | `pa:856092c2791e1d30` | 48 kHz, 2 input channels |

The configured DirectSound microphone passed a recorded sustained-signal check
(RMS 0.00766, peak 0.14340, 320 ms active in a 1.96-second uncoordinated sample).
The WASAPI physical-speaker loopback passed an isolated generated 440 Hz tone check
(RMS 0.03061, peak 0.05002, 1.52 seconds active in 2.00 seconds), with zero capture
overrun, discontinuity, or callback error in both checks. These prove the individual
capture paths, not Teams or end-to-end readiness. The VB-CABLE tone through Teams,
live test call, reconnect, and long-run dropout/backlog checks remain mandatory.

The shared-ASR contention microbenchmark used the local Transformers Whisper model
on CUDA, three repetitions, and 2.00-second synthetic audio per direction. Sequential
outgoing decode wall P50/P95 was 94.00/108.40 ms with scheduler wait 0.00/0.00 ms;
sequential incoming was 109.00/109.00 ms with admission wait 15.00/15.90 ms.
Simultaneous outgoing was 109.00/109.90 ms with wait 0.00/0.00 ms; simultaneous
incoming was 203.00/217.40 ms with wait 109.00/109.90 ms. Pair elapsed P50/P95 was
203.00/217.40 ms, maximum scheduler depth was two, and no scheduler deadline miss
was observed. This is MEASURED scheduler contention evidence only: WER/CER, VRAM,
physical audio edges, 30-minute full-duplex soak, and Phase H acceptance remain UNKNOWN.

The product scope is one local, localhost-only, full-duplex Windows application:

- Outgoing: User's microphone speech is transcribed, translated, and synthesized in the user's cloned voice into the meeting platform (Teams, Zoom, Meet, Discord) over VB-CABLE.
- Incoming: Meeting speech is transcribed and translated into live subtitles in the user's preferred language in the WhisperLiveKit web interface.
- Both directions remain active together, while end-to-end perceived latency and
  backlog age are treated as primary system properties. [REF: SRC-FINAL:B002:L29-L59;
  SRC-00:B001:L3-L8]

The first implementation is intentionally personal and single-user. It is not a
Teams bot, cloud service, multi-tenant system, or commercial product. [DEC: DEC-U001]

### 1.2 Target Milestones Driven by Active Issue Roadmap

The following active GitHub issues define the authoritative delivery targets for the system:

| Issue | Title | Architectural Scope & Target |
|---|---|---|
| #27 | Dynamic Source & Target Language Selection (Simultaneous Translation) | Full bidirectional simultaneous interpretation: paired UI selectors ("Konuştuğum Dil", "Toplantı Dili"), dynamic ASR source language, MT matrix (TR, EN, FR, DE, ES), XTTS-v2 multilingual synthesis. |
| #28 | Modular Language Registry & Streamlined Addition | Declarative `[languages]` registry in `config/default.toml`, CLI language downloader (`download_models.py --lang <code>`), dynamic UI language management. |
| #26 | Add Systran/faster-whisper-large-v3 to Eliminate Hallucinations | Replace/augment 4-layer turbo decoder with full 32-layer decoder; implement speech-rate / acoustic duration gate (`chars/sec`) to reject non-speech hallucinations without word blacklists. |
| #25 | Universal Application Support (Zoom, Meet, Discord, Slack, VRChat, Online Games) | Application presets, dynamic audio routing, floating HUD subtitle overlay, global hotkeys, and Dual Input Modes: Voice Activity (Meeting Mode) vs Push-to-Talk (Gaming Mode with 0 ms key-release endpointing and 150 ms pre-roll buffer). |
| #9 | True Streaming TTS Chunking | Move from sentence-level synthesis to incremental generator/stream synthesis in XTTS, reducing TTS TTFA from full utterance time to <300 ms. |
| #16 | Acoustic Echo Cancellation (AEC) | Implement WebRTC AEC3 / SpeexDSP between physical mic and loopback render to eliminate speaker-to-mic bleed without headsets. |
| #12 | Render Gain Staging & Limiting | Soft limiter / AGC / peak compressor in audio render engine to eliminate digital clipping across all volume levels. |
| #1 | Toggle Incoming Audio Translation & Pass-Through | UI toggle to switch between translating incoming audio or passing original sound through directly. |
| #20 | Multi-Reference Voice Cloning Guide & Workflow | Documentation and tooling for recording 6 clean 8–10s WAV reference samples for optimal XTTS speaker similarity. |
| #15 | Benchmark Gates B2–B6 | Systematically run and report P50/P95 latency, RTF, VRAM, and soak evidence on target hardware. |
| #29 | Standalone Windows Desktop App, Inno Setup Installer & First-Run Onboarding Wizard | Lightweight Inno Setup installer (`TeamsTranslator-Setup.exe`), pywebview Edge WebView2 native desktop window, system tray daemon (`pystray`), automated VB-CABLE driver installer helper, interactive model downloader wizard with resume & SHA256 verification, and in-app voice cloning onboarding recorder. |

## 2. User Decisions

These decisions are authoritative until the user explicitly changes them.

| ID | Decision | Binding consequence |
|---|---|---|
| DEC-U001 | Personal/local use only | CPML and non-commercial licenses are acceptable. Every license is still recorded, but a technically stronger model is not penalized merely for being non-commercial. [DEC: DEC-U001] |
| DEC-U002 | Native local Windows | Windows 11 is the canonical runtime. Docker, Docker Compose, containerized model services, and normal-runtime WSL are forbidden. Linux-only candidates are excluded or experimental. [DEC: DEC-U002] |
| DEC-U003 | Python only | Host/orchestration uses Python 3.12 and async where useful. No Node backend or separate Node frontend build is introduced, except assets/tooling already internal to WhisperLiveKit. [DEC: DEC-U003] |
| DEC-U004 | WhisperLiveKit web UI | The existing WLK web page is the primary UI, extended for both pipelines and operations. It binds only to localhost. PySide6 is not the primary UI. [DEC: DEC-U004] |
| DEC-U005 | Piper rejected | Clone-capable TTS is a first-class requirement. XTTS-v2 and Chatterbox are early candidates; CosyVoice is retained only if native Windows installation is realistic. Piper is neither selected nor a planned default. [DEC: DEC-U005] |
| DEC-U006 | Onur voice is selectable/default | Multiple profiles are supported, exactly one may be default, and speaker conditioning is cached rather than recomputed per text chunk. [DEC: DEC-U006] |
| DEC-U007 | SQLite meeting persistence | Local meeting history uses configurable SQLite, default data/meetings.sqlite3. Raw continuous PCM is not stored as database BLOBs. [DEC: DEC-U007] |
| DEC-U008 | Manual model download | Startup never silently downloads model weights. Exact repository, license, revision, role, and deterministic local path are recorded before a manual download. [DEC: DEC-U008] |
| DEC-U009 | Dynamic bidirectional simultaneous languages | The system supports arbitrary user-selected source and target language pairs (TR, EN, FR, DE, ES) rather than being permanently locked to TR->EN. [DEC: DEC-U009] |
| DEC-U010 | Universal meetings and games | Teams, Zoom, Meet (Chrome/Edge), Discord, Slack, VRChat, Dota 2, PUBG and CS2 use guided application presets. Keep one VB-CABLE outgoing and physical speaker loopback incoming. Process loopback remains deferred; no additional driver or paid service is required by this expansion. [REF: ISSUE-25] |
| DEC-U011 | One full-duplex session with selectable outgoing input | Start Meeting runs both directions for meetings and games. Only microphone admission switches between hands-free VAD and PTT with 150 ms pre-roll. Key release requests a final decode without an added silence timer; inference/scheduling latency is nonzero and measured separately. [REF: ISSUE-25] |
| DEC-U012 | Desktop subtitles accompany WLK | A native Win32 transparent, click-through, always-on-top, no-activate HUD starts with local sessions by default. WLK stays the primary interface. HUD supports desktop/borderless use; exclusive-fullscreen and anti-cheat compatibility require application testing. [REF: ISSUE-25] |
| DEC-U013 | Acoustic filtering without word censorship | Never blacklist “teşekkür ederim”, “abone ol”, subtitle phrases or prompt words. Add pinned Systran/faster-whisper-large-v3 as an offline option; compare against turbo on target hardware before changing the latency default. A larger decoder does not guarantee zero hallucinations. [REF: ISSUE-26; HF-LARGE-V3] |
| DEC-U014 | Desktop Distribution & Onboarding Wizard | The system supports packaging as a standalone Windows desktop app via Inno Setup installer, pywebview (Edge WebView2) desktop container, system tray daemon (`pystray`), automated VB-CABLE driver setup helper, model downloader wizard, and in-app voice cloning onboarding. [DEC: DEC-U014] |

## 3. Superseded Research Decisions

The following historical recommendations are explicitly non-canonical. They
remain in the research files for traceability.

| Superseded recommendation | Current ruling |
|---|---|
| Docker deployment, Docker model servers, or Docker Compose | Forbidden baseline; all canonical runtime components are native Windows. [DEC: DEC-U002] [REF: SRC-05:B009:L250-L297] |
| WSL as a normal runtime or required model-service boundary | Forbidden baseline; a Linux-only model can only be documented as excluded/experimental. [DEC: DEC-U002] [REF: SRC-FINAL:B009:L332-L362] |
| PySide6/desktop overlay as the primary UI | Replaced by the existing WLK web UI. [DEC: DEC-U004] [REF: SRC-05:B004:L119-L138] |
| Piper as default/generic Phase-1 TTS | Rejected; the planned TTS path begins with clone-capable models. [DEC: DEC-U005] [REF: SRC-FINAL:B019:L553-L576] |
| Commercial-license-first model filtering | Replaced by personal-use suitability; license is documented but non-commercial terms are acceptable. [DEC: DEC-U001] [REF: SRC-01:B005:L40-L47] |
| SQLite postponed indefinitely or transcript persistence omitted | Replaced by required, user-controlled local meeting persistence. [DEC: DEC-U007] [REF: SRC-FINAL:B009:L332-L362] |

## 4. System Requirements

### 4.1 Target environment

| Item | Canonical requirement |
|---|---|
| OS | Windows 11, native execution |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM |
| RAM | 32 GB |
| Host language | Python 3.12 |
| CUDA target | CUDA 12.8; actual PyTorch/driver/Blackwell compatibility must be verified |
| Audio routing | PyAudioWPatch with WASAPI for speaker loopback/render and a signal-gated native Windows host-API fallback for the physical mic; VB-CABLE |
| UI | Existing WhisperLiveKit web UI, localhost-only |
| Persistence | Python stdlib sqlite3, default data/meetings.sqlite3 |
| Network | No cloud dependency in the realtime pipeline; no non-loopback listener |

[DEC: DEC-U002; DEC: DEC-U003; DEC: DEC-U004; DEC: DEC-U007]

Driver-reported CUDA capability, an installed CUDA toolkit, and the CUDA runtime
bundled with a Python wheel are not assumed equivalent. Phase B/C environment
checks must prove that selected native wheels support the RTX 5060 Ti/Blackwell
architecture before model work is accepted. [REF: SRC-02:B012:L410-L448;
SRC-03:B010:L121-L138]

### 4.2 Functional requirements

- Start and stop a full-duplex meeting session from the WLK page.
- Select the physical microphone, physical/Teams output loopback endpoint, and
  VB-CABLE render endpoint.
- Stream outgoing Turkish through ASR, TR→EN MT, clone TTS, and VB-CABLE.
- Stream incoming English through ASR and EN→TR MT into partial and committed UI
  states.
- Select ASR, MT, TTS, and voice profile without coupling orchestration to one
  implementation.
- Optionally save local meeting history and telemetry, with recording files
  outside SQLite.
- Remain independently functional without Ollama or a general-purpose Qwen LLM.

[DEC: DEC-U004; DEC: DEC-U006; DEC: DEC-U007] [REF: SRC-FINAL:B004:L130-L171]

### 4.3 Non-functional requirements

- Optimize speech-to-useful-output latency, not isolated inference time.
- Maintain live-edge behavior: under normal speech, backlog age converges toward
  zero instead of growing through a long meeting.
- Warm selected models before reporting Ready.
- Preserve ordering and irreversible-output safety.
- Keep all audio, transcripts, profiles, and settings local.
- Make model choice evidence-based on the target PC, using P50/P95 and
  full-duplex soak data. [REF: SRC-00:B004:L422-L433; SRC-01:B004:L31-L38]

Runtime dependencies include WhisperLiveKit, PyAudioWPatch, NumPy, soxr,
torch, torchaudio, transformers, CTranslate2, sentencepiece,
`coqui-tts==0.27.5`, pytest, and pytest-asyncio. Exact resolved revisions are
recorded by `uv.lock`; model weights remain manual and external to dependency sync.
[REF: SRC-05:B008:L207-L246]

## 5. Canonical Architecture

There is one canonical architecture: a **native Windows Python application**
whose latency-sensitive audio edge, orchestration, WLK integration, adapters,
persistence worker, and telemetry run locally. Dependency isolation may use
separate native Python processes/virtual environments behind typed local
interfaces; it may not introduce containers or WSL. [DEC: DEC-U002; DEC: DEC-U003]

~~~text
Windows Native Python Application (localhost only)
│
├── Audio Layer
│   ├── PyAudioWPatch native Windows device discovery (WASAPI preferred)
│   ├── physical microphone capture
│   ├── selected application speaker endpoint loopback
│   ├── VB-CABLE render
│   ├── explicit resampling
│   ├── bounded PCM ring buffers
│   └── device loss/reconnect lifecycle
│
├── Streaming Orchestrator
│   ├── VAD
│   ├── partial / committed state machines
│   ├── stable-prefix + semantic commit
│   ├── bounded queues and cancellation
│   ├── backpressure / overload signaling
│   └── deadline-aware scheduling
│
├── ASR Adapter
│   ├── WhisperLiveKit-compatible backend
│   └── two sessions over shared weights where supported
│
├── Translation Adapter
│   ├── TR → EN
│   └── EN → TR
│
├── TTS Adapter
│   ├── incremental/chunked PCM contract
│   ├── cross-language voice cloning
│   └── speaker-conditioning cache
│
├── Persistence Worker
│   ├── SQLite meeting history
│   └── optional recording paths outside SQLite
│
├── Telemetry
│   ├── stage and end-to-end latency
│   ├── queue age/depth and drops
│   ├── ASR/TTS RTF
│   ├── GPU/VRAM/CPU
│   └── underrun, overrun, reconnect
│
└── WhisperLiveKit Web UI
    ├── devices, models, profile, meeting controls
    ├── outgoing committed English
    ├── incoming partial/committed Turkish
    └── latency, backlog, and resource status
~~~

The TX/RX data-flow separation, adapter boundaries, and bounded orchestration
come from the strongest common research result; concrete model selection remains
behind benchmark gates. [REF: SRC-04:B002:L41-L43; SRC-02:B003:L49-L91;
SRC-FINAL:B003:L61-L128]

### 5.1 Component ownership

| Component | Owns | Must not own |
|---|---|---|
| Audio layer | Device formats, callbacks, ring buffers, timestamps, reconnect | ASR/MT/TTS inference, SQLite, web requests |
| Orchestrator | Event identity, state transition, deadlines, cancellation, queue policy | Model-specific token logic |
| ASR adapter | Audio-to-partial/committed transcript contract | Translation, UI rendering, TTS |
| MT adapter | Directional text translation and model metadata | Audio devices, commit authority |
| TTS adapter | Committed text-to-PCM and conditioning cache | Partial text, cable device policy |
| Persistence worker | Batched local durable writes | Audio callbacks or inference scheduling |
| WLK integration | Primary server/UI, session controls, WebSocket events | Inference or platform-specific audio capture |
| Desktop companions | Win32 HUD projection, native global input hooks, bounded control edges | Primary settings UI, model inference, game injection |
| Telemetry | Monotonic timestamps, aggregates, resource samples | Blocking hot-path exports |

## 6. Outgoing Pipeline

The canonical outgoing flow is:

~~~text
Physical microphone
→ configured native Windows capture (WASAPI preferred; measured fallback allowed)
→ short bounded PCM ring buffer
→ VAD
→ streaming ASR
→ replaceable Turkish partial transcript
→ stable-prefix / semantic commit
→ append-only committed Turkish segment
→ TR → EN translation
→ append-only committed English segment
→ selected voice-cloning TTS
→ first PCM as early as safely possible
→ bounded playback ring/queue
→ VB-CABLE Input render endpoint
→ target application reads VB-CABLE Output as its microphone
~~~

[DEC: DEC-U005; DEC: DEC-U006] [REF: SRC-FINAL:B004:L130-L171;
SRC-02:B002:L28-L47]

Each utterance carries meeting_id, direction, sequence_id, revision, source
language, capture interval, state, and monotonic timestamps. Partial ASR output
may replace the current revision. Only the commit controller creates immutable
segments. MT translates committed segments in sequence, and TTS receives only
committed English. Spoken PCM is irreversible, so unstable text is never
synthesized. [REF: SRC-FINAL:B005:L173-L204]

While PCM for committed segment N is rendered, translation and synthesis for
N+1 may run within the bounded deadline budget. This overlap lowers perceived
latency without relaxing ordering. [REF: SRC-00:B003:L99-L124]

### 6.1 Dual Input Modes: Meeting Mode vs Push-to-Talk Gaming Mode

To serve both professional meetings (hands-free) and competitive online games (Dota 2, PUBG, Discord), the outgoing capture pipeline operates in two user-selectable input modes:

1. **Voice Activity Mode (Toplantı Modu - Hands-Free)**:
   - Microphone capture is continuously analyzed by Silero VAD.
   - Speech endpointing is governed by the Turkish SOV verb-aware commit policy (200 ms silence commit on predicate tails, 750–950 ms on open clauses).
   - Zero physical keystrokes required; optimal for Zoom, Teams, and Google Meet.

2. **Push-to-Talk Mode (Oyun / Bas-Konuş Modu - Low Latency)**:
   - Audio capture is gated by a global key hook (e.g. `Mouse 4`, `Caps Lock`, `V`) registered via low-level Windows keyboard/mouse hooks (`WH_KEYBOARD_LL`).
   - **Pre-Roll Rolling Buffer (150 ms)**: A circular FIFO memory buffer continuously retains the trailing 150 ms of mic audio so that when the user presses PTT slightly after beginning to speak, the initial syllable/consonant is never clipped.
   - **Release endpointing**: Timestamped keyboard/mouse edges split queued PCM at the release sample. The audio worker flushes with `commit_reason="ptt_release"` without waiting for VAD silence. Queue drain, hook polling and final inference still take time; zero is only the added silence delay, not total latency.
   - **Dead-air muting**: Unpressed PTT admits no new microphone audio except the bounded pre-roll of the next press. Already committed cloned speech finishes after release; an empty render ring produces digital silence. A game's own PTT must stay open for translated playback (open-mic voice mode is the guided baseline).

Users toggle modes via `F9` or the Web UI during the same Start Meeting session.
Incoming transcription keeps running while PTT is released. Default PTT is `V`;
`Mouse 4`, `Mouse 5`, `Caps Lock` and modifier chords are configurable. Low-level
Windows hooks pass original input through and never inject into a game. There is
no tray dependency. `Ctrl+Shift+T` pauses/resumes both directions;
`Ctrl+Shift+M` mutes outgoing only, clears pending render PCM and cancels old
committed delivery explicitly. New speech is required after unmute. Bounded
control overflow fails closed with a visible error. [DEC: DEC-U011; REF: ISSUE-25]

## 7. Incoming Pipeline

The canonical incoming flow is:

~~~text
Target application speaker stream
→ physical output endpoint
→ WASAPI loopback capture
→ short bounded PCM ring buffer
→ VAD
→ streaming English ASR
→ English partial / committed transcript
→ EN → TR translation
→ replaceable Turkish partial / append-only Turkish committed state
→ WhisperLiveKit web page
→ native desktop HUD (default enabled)
→ optional SQLite meeting history
~~~

[DEC: DEC-U004; DEC: DEC-U007] [REF: SRC-FINAL:B004:L130-L171;
SRC-05:B005:L142-L157]

Incoming favors the earliest useful Turkish text. A partial translation is a UI
projection keyed by sequence_id and revision; it can be replaced or removed.
Once the English source and Turkish translation are committed, both are
append-only and chronologically persisted when Save Meeting is enabled.

## 8. Audio Routing

### 8.1 Initial topology

The canonical first topology is one VB-CABLE for outgoing audio and physical
headset loopback for incoming audio:

~~~text
Physical mic ──capture──► application ──TTS PCM──► CABLE Input
                                                     │
Application microphone ◄────────────────────── CABLE Output

Meeting/game speaker ──► physical headset ──WASAPI loopback──► translator
~~~

Configuration shared by every application preset:

- Microphone: CABLE Output (VB-Audio Virtual Cable).
- Speaker: the user's physical headset.
- The application renders TTS to CABLE Input and captures loopback from the
  selected physical speaker endpoint.
- A headset is strongly preferred to avoid acoustic feedback.

This is the smallest useful routing surface, but loopback can also capture
notifications, music, and other applications on that endpoint. [REF:
SRC-FINAL:B006:L206-L235; SRC-02:B004:L125-L171]

### 8.2 Audio contracts

- Device callbacks use small frames, initially benchmarked at 10 or 20 ms.
- Callback work is limited to timestamping and bounded ring-buffer copy/read.
- Device-native formats are discovered; ASR and TTS sample-rate conversion is
  explicit, stateful, and measured.
- The likely initial contract is 48 kHz device audio, 16 kHz mono PCM16 at the
  ASR boundary, and backend-native TTS PCM resampled once for VB-CABLE. These
  are starting configurations, not measurements. [REF: SRC-02:B005:L173-L181]
- Stable device identifiers and format capabilities are persisted, while
  display names are only fallback hints.
- Device loss moves the meeting to Degraded/Reconnecting; it never blocks an
  audio callback.

### 8.3 Isolation upgrade path

Issue #25 keeps this upgrade deferred. Process loopback isolates incoming process
audio; it does not replace the outgoing virtual microphone. Shared endpoint
loopback also includes game effects, music and notifications. Preset guidance
states this limitation instead of pretending to isolate a process or browser tab.
[DEC: DEC-U010; REF: MS-PROCESS-LOOPBACK]

If shared-endpoint contamination fails acceptance tests, benchmark a second
virtual cable and then native Windows process-specific ApplicationLoopback.
These are ordered upgrades, not competing baseline architectures. A
process-loopback helper must remain native Windows and requires an explicit
decision before native helper code is added. [REF: SRC-FINAL:B006:L206-L235;
SRC-05:B002:L52-L60]

## 9. WhisperLiveKit Integration

WhisperLiveKit supplies the primary local web page, realtime ASR/server
foundation, partial-result transport, and reusable frontend assets. The project
extends that page and its Python-side event interfaces instead of building a
PySide application or a second Node frontend. [DEC: DEC-U003; DEC: DEC-U004]
[REF: SRC-05:B001:L20-L31; SRC-05:B003:L62-L77]

Required final UI surface:

- application state: Stopped, Starting, Warming, Ready, Running, Degraded, Error;
- application preset, physical microphone, speaker loopback and VB-CABLE selectors;
- input mode, configurable PTT key, desktop HUD toggle, pause and emergency mute;
- local ASR choice for the next session, explicit missing-model download guidance;
- selected ASR, TR→EN MT, EN→TR MT, TTS, and voice profile;
- outgoing committed English text;
- incoming replaceable partial Turkish and append-only committed Turkish;
- Start Meeting, Stop Meeting, and Save Meeting;
- P50/P95 current latency, queue-age/backlog warning, underrun/overrun state;
- GPU utilization/VRAM and model warm state where practical.

The HTTP/WebSocket server binds to 127.0.0.1 by default and rejects non-loopback
binding unless a future explicit user decision changes the security boundary.
The application may open the page in the default browser after the health check,
but the page may show Ready only after required devices and models are warm.
[REF: SRC-FINAL:B013:L434-L445]

WLK is an integration foundation, not an excuse to couple orchestration to a
single ASR. The project-specific ASR adapter normalizes backend partial/commit
semantics, timestamps, cancellation, and session identity.

## 10. Streaming / Commit State Machine

### 10.1 State model

Each direction has an independent session state and monotonic sequence. An
utterance moves through:

~~~text
Idle
  → SpeechDetected
  → Partial(revision 1..n, replaceable)
  → CommitCandidate(stable prefix / semantic boundary)
  → Committed(sequence fixed, append-only)
  → Dispatched(MT; and TTS for outgoing)
  → Delivered(UI committed or cable render)
  → Closed
~~~

A newer partial supersedes the older revision with the same utterance key.
Commit is the only irreversible text transition. Committed segments never move
back to Partial and their sequence never changes. [REF:
SRC-FINAL:B005:L173-L204; SRC-03:B004:L48-L54]

### 10.2 Commit policy

The commit controller combines signals rather than relying on one timer:

1. stable-prefix agreement across consecutive ASR revisions;
2. punctuation or a likely clause boundary;
3. VAD silence/endpoint evidence;
4. a configurable maximum semantic-wait deadline;
5. minimum useful content;
6. current TTS RTF, playback queue age, and downstream health.

Turkish SOV to English SVO means word-by-word irreversible speech can be
grammatically wrong; some semantic lookahead is unavoidable. Clause boundaries,
short pauses, and conjunctions are useful evidence, but historical values such
as 250–350 ms, 600–900 ms, 1.8 seconds, or 4–8 words are only benchmark starting
points. They are not product guarantees. [REF: SRC-04:B004:L124-L135;
SRC-00:B003:L99-L124]

Outgoing TTS consumes only committed English. Incoming partial Turkish remains
visual and replaceable. A final endpoint flushes any remaining meaningful stable
content, subject to sequence and duplicate checks.

### 10.3 Event identity

All inter-stage messages include:

- meeting_id, stream_id, direction, utterance_id, sequence_id, and revision;
- state and source/target language;
- source audio start/end monotonic timestamps;
- created_at monotonic timestamp and optional UTC wall-clock timestamp;
- backend/model/revision identity;
- cancellation/supersession token.

This identity makes stale partial cancellation safe and makes committed ordering
testable.

## 11. Backpressure and Scheduling

### 11.1 Hard latency invariants

These are architecture rules for every implementation phase:

1. Audio callbacks must never block on inference.
2. PCM queues must be bounded.
3. Partial hypotheses may be discarded or replaced.
4. Committed text may not be silently reordered or discarded.
5. TTS may only consume committed text.
6. Models must be warmed before a meeting is marked Ready.
7. No unbounded asyncio.Queue is allowed.
8. Queue age matters more than queue item count.
9. Backlog must converge toward zero during normal speech.
10. Long meetings must not accumulate latency.

[REF: SRC-00:B004:L422-L433; SRC-FINAL:B005:L173-L204]

### 11.2 Queue policy

| Queue | Capacity policy | On pressure |
|---|---|---|
| Mic/loopback PCM ring | Fixed frames and maximum age | Drop oldest unprocessed audio only under explicit overload policy, increment a discontinuity counter, and show Degraded; callbacks never wait |
| ASR partial events | One latest revision per active utterance | Replace stale revision and cancel downstream partial work |
| Incoming partial MT/UI | One latest revision per sequence | Coalesce/replace; UI may skip obsolete revisions |
| Committed MT | Small bounded FIFO | Preserve order; reduce new commit rate/coalesce before commit, then signal overload rather than silently discard |
| Outgoing TTS requests | Small bounded FIFO ordered by sequence | Preserve order; adapt pre-commit chunking, prioritize deadline, warn/fail visibly at hard capacity |
| TTS PCM playback ring | Fixed time capacity with low/high watermarks | Prioritize render fill; count underrun/overrun; never grow without limit |
| Persistence | Bounded event batches | Batch/coalesce telemetry; committed history failure becomes visible Save Degraded, never blocks audio |
| Metrics/UI | Latest-value snapshots plus bounded events | Coalesce samples and drop obsolete refreshes |

A hard committed-work overflow is an explicit fault, not permission to create an
unbounded queue. The system stops accepting new irreversible work, surfaces the
condition, and preserves ordering for accepted commits. Normal-operation
acceptance requires never reaching that state.

### 11.3 Deadline-aware priority

Approximate priority is:

1. audio capture and render deadlines;
2. outgoing audio-delivery work, including the current committed segment's
   ASR/MT/TTS and playback fill;
3. incoming committed subtitle;
4. incoming partial subtitle;
5. persistence;
6. metrics and background work.

TTS is part of outgoing delivery and is never treated as generic low-priority
background work. Priority uses next deadline and queue age, not a permanent
component-name ordering. [REF: SRC-FINAL:B010:L364-L379]

Shared-weight ASR uses a bounded admission scheduler rather than a bare inference
lock. Captured-audio age defines the base deadline. Outgoing work receives a bounded
downstream reserve over incoming partial work; an older incoming request wins after
that reserve is consumed, preventing starvation. A short incoming-partial admission
window lets concurrently arriving outgoing work become visible before selection.
Wait time, queue depth, and deadline miss are recorded per session and direction.
This schedules contention but does not remove the backend's required serialization.

## 12. ASR Architecture

### 12.1 Baseline

The initial bring-up model is openai/whisper-large-v3-turbo through a
WhisperLiveKit-compatible native Windows backend. Whisper is used as a mature
bring-up baseline; windowed/recomputed partial decoding must not be mislabeled
as model-native streaming. [DEC: DEC-U002; DEC: DEC-U004] [REF:
SRC-03:B001:L8-L16; SRC-02:B007:L213-L251]

Where the backend supports it, one warm model-weight instance serves:

- Session A: Turkish physical microphone;
- Session B: English application loopback.

VAD, audio context, partial history, endpointing, cancellation, and sequence
state remain isolated per session. Loading two duplicate GPU weight instances is
allowed only after measurement proves sharing impossible and the VRAM budget
still passes full-duplex gates. [REF: SRC-02:B007:L213-L251]

### 12.2 Adapter contract

An ASR adapter must provide:

- initialize(model_path, device, compute policy) with offline-only path
  validation;
- warmup() and readiness diagnostics;
- create_session(language, stream_id);
- push_audio(frames, timestamps) without blocking the audio callback;
- partial and committed events with revision/sequence identity;
- cancel_superseded(), flush_endpoint(), close_session(), and shutdown();
- telemetry for first partial, first commit, RTF, errors, and resource use.
- per-session shared-inference wait, queue depth, and deadline-miss telemetry.

The orchestrator owns cross-backend semantics. Backend-specific fields remain in
diagnostic metadata.

### 12.3 Benchmark candidates

| Candidate | Status | Native Windows rule |
|---|---|---|
| WLK + whisper-large-v3-turbo | Required bring-up | Must pass Python/CUDA/Blackwell checks |
| Systran/faster-whisper-large-v3 | Supported full-decoder option (Issue #26) | CTranslate2 `int8_float16`/`float16` on CUDA; benchmark against turbo with shared sessions |
| Qwen/Qwen3-ASR-0.6B through WLK/Hugging Face | Benchmark | Only the native Windows path is eligible; bounded recompute is not claimed as native streaming |
| nvidia/nemotron-3.5-asr-streaming-0.6b | Benchmark | Eligible only if its NeMo-Speech/native Windows CUDA route is reproducible |
| Any vLLM/Linux-only ASR route | Excluded/experimental | Cannot become canonical under DEC-U002 |

The final ASR is selected with identical Turkish/English corpora, two concurrent
sessions, first-partial/first-commit latency, WER/CER, terminology accuracy, RTF,
VRAM, and long-session stability. [REF: SRC-FINAL:B014:L447-L464;
SRC-01:B001:L3-L8]

### 12.4 Acoustic guard and offline model selection (Issue #26)

The shared guard requires utterance duration, voiced duration/ratio, bounded audio
queue age, optional Whisper confidence and a configurable letter/digit rate
(`guard_max_chars_per_second=50`). Spaces and punctuation do not inflate that
rate. This conservative ceiling is a tunable heuristic, not proof of speech or
a universal biological limit. Malformed confidence cannot bypass the guard.
There is no phrase blacklist at ASR, commit or TTS. An empty final decode retracts
any obsolete partial instead of reviving it. Partial rejection leaves buffered
audio available for a later corrected hypothesis. [DEC: DEC-U013; REF: ISSUE-26]

The Web UI selects the next session's installed model; changing models first
unloads the previous ASR weights, loads the selected local model and warms it
before Ready. Missing files fail before unloading the current model. A failed
load attempts to restore the previous selection and surfaces the exact error.
Runtime remains offline. Turbo remains the default until B2/B6 gates justify a
change; full large-v3 can be selected explicitly. Configuration also accepts
absolute external model paths. [DEC: DEC-U008; DEC-U013]

Target-hardware partial decode (RTX 5060 Ti, shared-weight scheduler, same
excerpts): large-v3 P50/P95 587/933 ms, turbo P50/P95 243/366 ms. Issue #26's
250 ms partial-chunk target is therefore not met by either model at P95, and
large-v3 misses it outright; the full decoder stays an explicit non-default
option and turbo keeps the latency default. Measured 2026-09-06 with
`scripts/benchmark_asr_acoustics.py`. [DEC: DEC-U013; REF: ISSUE-26]

`scripts/benchmark_asr_acoustics.py` compares the same explicitly selected local excerpt or
corpus, includes synthetic negative controls, P50/P95, RTF, shared-ASR scheduler
wait and whole-GPU VRAM snapshots. It never records a microphone or downloads
weights. Raw model output and guard acceptance are separate. WER/CER require
reference text; real breathing/keyboard recordings, remote client receipt and
30-minute ASR+MT+TTS full-duplex soak remain separate acceptance gates.

## 13. Translation Architecture

Translation remains behind direction-independent adapters. The baseline is two
small dedicated models converted/loaded for CTranslate2, initially benchmarked
on CPU INT8 to protect GPU headroom:

- Helsinki-NLP/opus-mt-tc-big-tr-en for outgoing;
- Helsinki-NLP/opus-mt-tc-big-en-tr for incoming.

[REF: SRC-FINAL:B007:L237-L284; SRC-02:B008:L253-L272]

The adapter contract accepts source text, source/target language, state,
sequence/revision, glossary snapshot, and deadline; it returns translated text,
model identity, timing, and cancellation outcome. It must support replacing
incoming partial translations without confusing them with committed output.

### 13.1 Measured alternatives

| Path | Role | Main trade-off |
|---|---|---|
| Turkish ASR → OPUS TR→EN | Canonical baseline | Preserves Turkish transcript, gives independent quality/stability controls, and isolates ASR from MT |
| Turkish speech → Whisper direct English translation | Outgoing benchmark fast path | May remove one stage but loses an independent Turkish transcript and may change commit/quality behavior |
| NLLB-200 distilled 600M | Non-commercial quality/reference benchmark | Accepted for personal use; larger resource/integration cost must earn its place |
| General LLM/Ollama/Qwen text cleanup | Optional post-meeting only | Not required and excluded from the realtime hot path until evidence proves no latency/backlog regression |

Measure 5/10/20-word or equivalent semantic prefixes, P50/P95, chrF or another
declared automatic metric, terminology accuracy, commit stability, and human
bilingual review. Direct translation is compared rather than assumed faster or
better. [DEC: DEC-U001] [REF: SRC-FINAL:B015:L465-L477;
SRC-01:B002:L14-L17]

## 14. TTS and Voice Cloning

Voice identity and English intelligibility are hard quality gates, not optional
polish. The mandatory benchmark scenario uses Onur's Turkish reference recording
to synthesize English speech. [DEC: DEC-U005; DEC: DEC-U006]

| Candidate | Plan role | Advantages to test | Risks to prove |
|---|---|---|---|
| coqui/XTTS-v2 | Required first clone implementation and quality baseline | Turkish/English multilingual support and cross-language cloning; CPML acceptable | Native Python 3.12/CUDA compatibility, TTFA, sustained RTF, chunk joins |
| ResembleAI/chatterbox | Early benchmark candidate | MIT, multilingual/clone-oriented project, relatively approachable integration | Server stream flags may deliver only after a full text chunk; prove first-PCM behavior |
| CosyVoice, native-compatible release | Conditional benchmark | Research-backed cloning/bi-streaming capability | Retain only if a reproducible native Windows environment exists; otherwise excluded/experimental |
| Qwen3-TTS/vLLM-Omni family | Excluded/experimental under current architecture | Research evidence distinguishes true incremental PCM | Linux/vLLM requirement conflicts with native Windows baseline |
| Piper | Rejected | None needed for the planned architecture | No required voice cloning; must not become a selected/default path |

[REF: SRC-FINAL:B007:L237-L284; SRC-02:B009:L274-L307]

True streaming means PCM becomes consumable while synthesis for the same logical
segment is still progressing. Splitting text and sending a completed waveform in
chunks is pseudo-streaming. Both can reduce perceived latency, but benchmark
reports must label them correctly. A Chatterbox stream option is not evidence of
incremental model decoding by itself. [REF: SRC-FINAL:B007:L237-L284;
SRC-02:B009:L274-L307]

The TTS adapter must expose warmup, prepare_voice_profile, synthesize_committed,
first-PCM timing, PCM chunks with sample format, cancellation before irreversible
render where supported, RTF, and shutdown. It must never accept Partial state.
The playback layer performs at most one explicit resample to the VB-CABLE device
format.

Selection metrics are time_to_first_pcm, sustained RTF, Turkish-reference to
English speaker similarity, English intelligibility, prosody, chunk-boundary
quality, VRAM, long-session stability, and 3/10/30-second reference sensitivity.
All values remain UNKNOWN until locally measured. [REF:
SRC-FINAL:B016:L479-L495; SRC-01:B003:L22-L25]

## 15. Voice Profiles

Canonical profile layout:

~~~text
voices/
└── onur-default/
    ├── reference.wav
    ├── profile.json
    └── cache/
~~~

Future sibling profiles can include Onur - Default, Onur - Formal, Onur -
Energetic, Other Voice 1, and Other Voice 2. [DEC: DEC-U006]

Minimum profile manifest fields:

| Field | Meaning |
|---|---|
| id | Stable filesystem/database identifier |
| display_name | UI label |
| backend | TTS adapter identifier |
| reference_audio | Relative profile path or approved absolute external path |
| reference_text | Exact reference transcript when required |
| reference_language | Normally tr for Onur source audio |
| target_language | Normally en for the outgoing pipeline |
| is_default | Boolean; exactly one profile may be true |
| consent_recorded_at | Local provenance/consent timestamp |
| backend_options | Non-secret adapter-specific JSON |

Speaker embeddings, latents, tokenized prompts, or conditioning tensors are
computed during profile preparation/warmup and cached under a key containing
audio content hash, normalized transcript hash, backend/model revision, and
conditioning parameters. A cache mismatch triggers recomputation before Ready,
never per text chunk. [REF: SRC-FINAL:B008:L286-L330]

The filesystem manifest is the portable profile definition. SQLite mirrors
searchable metadata and selection state; it does not store the reference WAV or
conditioning tensor as a BLOB. A partial unique database index enforces at most
one default profile, and startup requires one valid default before outgoing TTS
can be Ready.

## 16. Existing Local Model Inventory

These assets are recorded exactly as reported. They are not copied, downloaded,
or inserted into the realtime path. Configuration will later accept an absolute
external path. [DEC: DEC-U008]

| Reported asset | Safely inferable from name | UNKNOWN until metadata inspection | Allowed future role |
|---|---|---|---|
| qwen_3_4b_fp8_mixed.safetensors | It is named as a safetensors file; tokens in the filename suggest Qwen, 4B, and mixed FP8 claims, but do not prove them | Absolute path, repo/revision, architecture, parameter count, quantization scheme, tokenizer, config, license, tensor completeness, runtime compatibility, speech capability | Optional offline terminology/cleanup experiment only after identification |
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | It is named as a safetensors file; its many name tokens are labels only and may conflict | Absolute path, source repo, actual architecture/modality, parameter count, quantization, tokenizer/config, license, compatibility | Optional post-meeting analysis only if metadata justifies it |
| qwen3.8-max-16k | Only the reported name is known; file versus directory and format are not inferable | Absolute path, type, repo/revision, architecture, tokenizer, context, quantization, license, runtime and purpose | Optional terminology, summary, cleanup, or glossary generation after identification |

ASR → MT → TTS must work without these assets and without Ollama. A general LLM
may be evaluated only outside the realtime hot path unless a controlled
full-duplex benchmark proves a net benefit without violating latency gates.

## 17. Model Download Manifest

No model is downloaded in Phase A. Startup must run in offline/local-files-only
mode and fail with a precise missing-path message rather than contacting a model
hub. Every actual download must pin a revision and record repository ID, commit,
license, file inventory/checksums where available, destination, and acceptance
status in a future lock manifest. [DEC: DEC-U008]

### 17.1 Required for first model phases

| Exact repository | Purpose | Source | Recorded license | Role/timing | Exact local destination |
|---|---|---|---|---|---|
| openai/whisper-large-v3-turbo | Turkish and English ASR bring-up | https://huggingface.co/openai/whisper-large-v3-turbo | MIT | Required before Phase C | models/asr/whisper-large-v3-turbo/ |
| Systran/faster-whisper-large-v3 | Full-decoder ASR option | https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478 | MIT (pinned card) | Explicit optional download for #26; revision `edaa852ec7e145841d8ffdb056a99866b5f0a478` | models/asr/whisper-large-v3/ |
| Helsinki-NLP/opus-mt-tc-big-tr-en | Outgoing TR→EN MT baseline | https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-tr-en | CC-BY-4.0 | Required before Phase E | models/mt/opus-mt-tc-big-tr-en/ |
| Helsinki-NLP/opus-mt-tc-big-en-tr | Incoming EN→TR MT baseline | https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr | CC-BY-4.0 | Required before Phase D | models/mt/opus-mt-tc-big-en-tr/ |
| coqui/XTTS-v2 | Cross-language voice-clone baseline | https://huggingface.co/coqui/XTTS-v2 | Coqui Public Model License (CPML), personal/non-commercial accepted | Required before Phase F | models/tts/xtts-v2/ |

Licenses are recorded from the current research synthesis and must be rechecked
against the exact pinned model revision before download/use; a later license
change does not silently alter DEC-U001.

### 17.2 Benchmark candidates

| Exact repository | Purpose | Source | Recorded license | Role/timing | Exact local destination |
|---|---|---|---|---|---|
| ResembleAI/chatterbox | Multilingual cloned TTS comparison | https://huggingface.co/ResembleAI/chatterbox | MIT | Early Phase F benchmark | models/tts/chatterbox/ |
| facebook/nllb-200-distilled-600M | Higher-coverage MT quality/reference | https://huggingface.co/facebook/nllb-200-distilled-600M | CC-BY-NC-4.0 | Phase D/E benchmark, not baseline | models/mt/nllb-200-distilled-600m/ |
| Qwen/Qwen3-ASR-0.6B | Alternate ASR | https://huggingface.co/Qwen/Qwen3-ASR-0.6B | Apache-2.0 | Benchmark only through a verified native Windows WLK/HF path | models/asr/qwen3-asr-0.6b/ |
| nvidia/nemotron-3.5-asr-streaming-0.6b | Native-streaming ASR candidate | https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b | NVIDIA Open Model Development and Work License 1.1 (OpenMDW-1.1) | Benchmark only if native NeMo-Speech route is reproducible | models/asr/nemotron-3.5-asr-streaming-0.6b/ |
| FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | Voice-clone/bi-streaming TTS candidate | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | Apache-2.0 | Conditional: only if native Windows compatibility is realistic | models/tts/cosyvoice3/ |

### 17.3 Optional / future / excluded

| Candidate | Classification | Download decision |
|---|---|---|
| Qwen3-TTS with vLLM-Omni | Excluded/experimental because the researched true-streaming route is Linux-oriented | Do not download for canonical implementation; exact repository remains unresolved until a native Windows route and explicit approval exist |
| OpenVoice or another cloning model | Optional research candidate | No download until an exact repo, weights license, Windows feasibility, and benchmark value are documented |
| General local Qwen/Ollama LLM | Optional post-meeting feature | Reuse an identified absolute external path; do not duplicate into models/ or require for realtime |
| Piper | Rejected | No planned download |

### 17.4 Already available locally

The three reported Qwen-named assets in Section 16 are ALREADY AVAILABLE
LOCALLY, but their paths and identities are UNKNOWN. They are inventory entries,
not satisfied dependencies for any required ASR, MT, or TTS model.

Phase B requires no model download. Manual downloads should occur just before
the first phase that needs each model, after revision/license verification, so
large unused assets are not acquired prematurely.

## 18. GPU / VRAM Strategy

The 16 GB VRAM ceiling is a hard shared budget for outgoing ASR, incoming ASR,
TTS, CUDA context, transient tensors, and safety margin. Historical per-model
VRAM estimates are not treated as measured results on this PC. [DEC: DEC-U002]
[REF: SRC-01:B001:L3-L8; SRC-02:B013:L450-L474]

Rules:

1. Prefer one ASR model-weight instance with two independent sessions.
2. Load no duplicate GPU model merely because the two directions have different
   stream IDs.
3. Keep baseline OPUS MT on CPU INT8 unless measurement shows a better total
   schedule with GPU MT.
4. Warm ASR and selected TTS, prepare the selected voice profile, and record
   steady VRAM before Ready.
5. Reserve headroom for concurrent first-PCM generation and ASR decode; a model
   that fits alone but fails full duplex is rejected.
6. Use inference mode, appropriate precision, pinned compatible builds, and
   bounded batch/session settings; precision changes require quality comparison.
7. GPU work is deadline-aware. Audio render and outgoing TTS deadlines may
   preempt replaceable incoming partial work.
8. Sampling telemetry must be non-blocking and low frequency relative to audio.
9. Record cold start, warm steady state, peak allocated/reserved VRAM, utilization,
   RTF, OOM, and recovery behavior.
10. Other GPU-heavy programs such as Ollama or creative workloads are closed or
    explicitly recorded during reproducible benchmarks.

[REF: SRC-FINAL:B010:L364-L379; SRC-FINAL:B013:L434-L445]

If a candidate requires an incompatible dependency set, it may be tested in a
separate **native Windows Python** virtual environment and local worker process.
That does not relax the no-container/no-WSL rule. A candidate unable to meet this
boundary is excluded/experimental.

## 19. SQLite Meeting Persistence

SQLite is local, configurable, and outside all audio callbacks and model
deadlines. Default path: data/meetings.sqlite3. [DEC: DEC-U007]

### 19.1 Initial schema

The following is the proposed first migration contract. Types and constraints
are intentionally concrete; an ORM is not selected yet.

~~~sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE meetings (
    id                   TEXT PRIMARY KEY,
    title                TEXT,
    started_at           TEXT NOT NULL,
    ended_at             TEXT,
    status               TEXT NOT NULL
                         CHECK (status IN ('starting','running','stopped','error')),
    save_enabled         INTEGER NOT NULL DEFAULT 1
                         CHECK (save_enabled IN (0,1)),
    asr_backend          TEXT NOT NULL,
    asr_model            TEXT NOT NULL,
    mt_tr_en_backend     TEXT NOT NULL,
    mt_tr_en_model       TEXT NOT NULL,
    mt_en_tr_backend     TEXT NOT NULL,
    mt_en_tr_model       TEXT NOT NULL,
    tts_backend          TEXT NOT NULL,
    tts_model            TEXT NOT NULL,
    voice_profile_id     TEXT,
    config_snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL,
    FOREIGN KEY (voice_profile_id) REFERENCES voice_profiles(id)
        ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE utterances (
    id                TEXT PRIMARY KEY,
    meeting_id        TEXT NOT NULL,
    direction         TEXT NOT NULL
                      CHECK (direction IN ('outgoing','incoming')),
    sequence          INTEGER NOT NULL CHECK (sequence >= 0),
    source_language   TEXT NOT NULL,
    text              TEXT NOT NULL,
    state             TEXT NOT NULL
                      CHECK (state IN ('partial','committed')),
    started_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    committed_at      TEXT,
    audio_path        TEXT,
    audio_metadata_json TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (meeting_id, direction, sequence)
);

CREATE TABLE translations (
    id                TEXT PRIMARY KEY,
    utterance_id      TEXT NOT NULL,
    target_language   TEXT NOT NULL,
    text              TEXT NOT NULL,
    state             TEXT NOT NULL
                      CHECK (state IN ('partial','committed')),
    backend           TEXT NOT NULL,
    model             TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    committed_at      TEXT,
    FOREIGN KEY (utterance_id) REFERENCES utterances(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (utterance_id, target_language)
);

CREATE TABLE latency_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id        TEXT NOT NULL,
    utterance_id      TEXT,
    direction         TEXT
                      CHECK (direction IN ('outgoing','incoming')),
    event_type        TEXT NOT NULL,
    occurred_at       TEXT NOT NULL,
    monotonic_ns      INTEGER NOT NULL,
    duration_ms       REAL,
    queue_age_ms      REAL,
    metadata_json     TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (utterance_id) REFERENCES utterances(id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE voice_profiles (
    id                       TEXT PRIMARY KEY,
    display_name             TEXT NOT NULL UNIQUE,
    backend                  TEXT NOT NULL,
    reference_audio_path     TEXT NOT NULL,
    reference_text           TEXT,
    reference_language       TEXT NOT NULL,
    target_language          TEXT NOT NULL,
    is_default               INTEGER NOT NULL DEFAULT 0
                             CHECK (is_default IN (0,1)),
    conditioning_cache_path  TEXT,
    metadata_json            TEXT NOT NULL DEFAULT '{}',
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL
);

CREATE TABLE settings (
    key          TEXT PRIMARY KEY,
    value_json   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE meeting_statistics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    aggregation  TEXT NOT NULL,
    value        REAL NOT NULL,
    unit         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (meeting_id, metric_name, aggregation)
);

CREATE INDEX idx_meetings_started_at
    ON meetings(started_at DESC);
CREATE INDEX idx_utterances_meeting_order
    ON utterances(meeting_id, direction, sequence);
CREATE INDEX idx_utterances_meeting_started
    ON utterances(meeting_id, started_at);
CREATE INDEX idx_latency_meeting_time
    ON latency_events(meeting_id, monotonic_ns);
CREATE INDEX idx_latency_utterance_type
    ON latency_events(utterance_id, event_type);
CREATE UNIQUE INDEX uq_voice_profiles_single_default
    ON voice_profiles(is_default) WHERE is_default = 1;
~~~

### 19.2 Persistence behavior

- Save Meeting is chosen at meeting start in the initial contract. When off,
  meeting text/audio/latency history remains in bounded memory only and no
  meeting-history rows are created. Device/model preferences may still persist
  in settings.
- When on, one background writer owns the connection, batches transactions, and
  receives typed events through a bounded queue.
- The current Partial row is UPSERTed by utterance/translation identity; obsolete
  partial revisions do not create an unbounded event history. Commit updates that
  row to append-only final state.
- Continuous PCM is never a SQLite BLOB. If recording is later enabled, files go
  under recordings/<meeting-id>/ as WAV/FLAC; audio_path and metadata are stored
  in utterances.
- A persistence backlog cannot block callbacks. At its hard cap, the UI reports
  Save Degraded/Error; accepted committed history is never silently discarded.
- UTC ISO-8601 values support chronological history; monotonic_ns supports
  intra-session latency and is not interpreted across restarts.
- All content remains local. Deletion/retention policy is an explicit future UI
  feature, not an automatic cloud sync.

## 20. Telemetry and Latency

### 20.1 Required timestamps

At minimum record:

- audio_captured_at
- vad_started_at
- asr_first_partial_at
- asr_committed_at
- mt_started_at
- mt_completed_at
- tts_request_at
- tts_first_pcm_at
- cable_write_at
- ui_partial_at
- ui_committed_at

Use a monotonic clock for duration math and UTC wall time for records. Correlate
timestamps by meeting/stream/utterance/sequence/revision. [REF:
SRC-00:B005:L437-L474; SRC-FINAL:B011:L381-L417]

Derived telemetry:

- outgoing speech-start/capture → first cable PCM and committed segment delivery;
- incoming capture → first useful Turkish partial and committed Turkish;
- ASR first partial/commit, MT, TTS TTFA, TTS RTF, and stage distributions;
- outgoing/incoming ASR scheduler wait and deadline-miss distributions;
- queue age, depth, capacity, high-water mark, stale partial count, cancellation;
- callback duration, audio underrun/overrun/discontinuity, reconnect;
- CPU, process memory, GPU utilization, allocated/reserved/peak VRAM, and OOM;
- meeting-duration drift and backlog convergence.

P50 and P95 are mandatory; P99 is retained for soak diagnostics where sample
count supports it. Metrics windows and sample counts must be shown.

### 20.2 Initial targets

These are TARGET values inherited as starting acceptance hypotheses, not
MEASURED results:

| End-to-end measure | P50 TARGET | P95 TARGET | Current result |
|---|---:|---:|---|
| Incoming English audio → first useful Turkish partial | ≤ 0.8 s | ≤ 1.5 s | UNKNOWN |
| Incoming English audio → committed Turkish | ≤ 1.5 s | ≤ 2.5 s | UNKNOWN |
| Outgoing Turkish audio → first English PCM written toward cable | ≤ 1.5 s | ≤ 2.5 s | UNKNOWN |
| 30-minute full-duplex backlog | Converges to zero | No sustained growth | UNKNOWN |
| 30-minute audio rendering | No underrun caused by pipeline | No committed reordering | UNKNOWN |

[REF: SRC-FINAL:B011:L381-L417; SRC-02:B003:L49-L91]

No component-card time is presented as end-to-end latency. Targets can change
only after a recorded benchmark and an architectural decision update.

## 21. Configuration Strategy

Proposed configuration files:

~~~text
config/
├── default.toml       # versioned safe defaults, no machine-specific IDs
└── local.toml         # local paths/devices; should remain untracked
~~~

Precedence, lowest to highest:

1. validated default.toml;
2. local.toml;
3. `TEAMS_TRANSLATOR_*` legacy environment overrides, then `VOICE_TRANSLATOR_*`;
4. command-line one-session overrides;
5. current WLK UI session choices.

Persisted UI preferences are stored as settings and materialized as the next
session's local configuration; they never rewrite default.toml automatically.

Configuration groups:

- server: host fixed to 127.0.0.1 by default, port, browser-open option;
- audio: stable device IDs, formats, frame size, ring duration, resampler;
- streaming: VAD, commit starting parameters, queue capacities/age limits;
- asr, translation, tts: adapter, exact local model path, revision, device,
  precision, warmup;
- voice: profile root and selected/default profile;
- persistence: enabled default, data/meetings.sqlite3, recordings root, batch;
- telemetry: windows, sampling interval, warning thresholds;
- external_models: optional named absolute paths for already-local assets.

Relative model paths resolve from the repository root. Absolute external paths
are allowed and are never copied. Every path is validated locally; missing
models produce actionable offline errors, never downloads. [DEC: DEC-U008]

## 22. Repository Structure

Future implementation structure:

~~~text
/
├── AGENTS.md
├── .agents/
│   └── skills/
│       └── realtime-voice-translator/
│           └── SKILL.md
├── Plan.md                         # canonical architecture and runtime findings
├── docs/
│   └── research/                  # immutable historical research
├── src/
│   └── voice_translator/
│       ├── audio/
│       ├── asr/
│       ├── translation/
│       ├── tts/
│       ├── streaming/
│       ├── persistence/
│       ├── telemetry/
│       ├── web/
│       ├── config/
│       └── core/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── audio/
│   ├── benchmarks/
│   └── fixtures/
├── models/
│   ├── asr/
│   ├── mt/
│   ├── tts/
│   ├── llm/
│   └── cache/
├── voices/
│   └── onur-default/
│       └── cache/
├── recordings/
├── data/
├── scripts/
└── config/
~~~

Phase A is historical and complete. Runtime code is present because later user
tasks explicitly authorized implementation. A component's presence is not evidence
that its hardware or latency exit gate has passed.

## 23. Testing Strategy

Tests begin with implementation, not Phase A.

### 23.1 Unit tests

- ring-buffer capacity, wraparound, age, overflow, and callback time budget;
- resampler continuity and format contracts;
- partial replacement, stable-prefix detection, commit immutability, endpoint
  flush, and sequence ordering;
- bounded queue construction and overload policy;
- cancellation of superseded partial ASR/MT work;
- deadline priority, starvation prevention, and TTS ordering;
- adapter contract tests with deterministic fake ASR/MT/TTS;
- voice profile validation, exactly-one-default, cache-key invalidation;
- configuration precedence, offline path validation, and localhost binding;
- schema migration, foreign keys, chronological queries, Partial UPSERT, WAL
  writer batching, and Save Meeting off behavior;
- latency calculation with monotonic timestamps.

### 23.2 Integration tests

- prerecorded PCM through fake or real single-stream WLK ASR;
- incoming ASR→MT→WLK partial/committed events;
- outgoing ASR→MT→fake streaming TTS→fake cable sink;
- persistence worker failure without callback blockage;
- model/device warmup gating Ready;
- two streams sharing model weights where the backend supports it;
- intentional slow-stage injection to prove bounded queues and convergence;
- browser UI contract against localhost APIs/WebSockets.

### 23.3 Hardware and benchmark tests

Tests requiring a physical device, VB-CABLE, Teams test call, CUDA model, or the
user's voice sample are explicitly marked and never run as ordinary fast unit
tests. Their reports capture hardware, drivers, package/model revisions, config,
corpus hash, warmup, and measurement status.

No test may silently contact the network or download a model. Quality snapshots
are not substitutes for WER/CER, bilingual review, speaker similarity, and
long-session gates.

## 24. Benchmark Strategy

Selection follows progressive, reproducible gates rather than model-card claims.
[REF: SRC-00:B006:L478-L542; SRC-01:B005:L40-L47]

| Gate | Experiment | Mandatory evidence |
|---|---|---|
| B0 Environment | Python, WLK revision, PyTorch/CUDA/Blackwell, package imports, model-path/license manifest | Pass/fail matrix; no inference claim before compatibility passes |
| B1 Audio | Device discovery, mic capture, loopback, test PCM→VB-CABLE, resampling, reconnect | Callback duration, discontinuity, underrun/overrun, device formats |
| B2 ASR | Same TR/EN corpus, one and two simultaneous sessions, each native candidate | First partial/commit and scheduler-wait P50/P95/P99, deadline miss, WER/CER, terminology, RTF, RAM/VRAM |
| B3 MT | OPUS directions, beams/settings, NLLB reference, outgoing direct Whisper | Prefix/semantic-chunk latency, chrF/declared metric, stability, bilingual review, RAM/VRAM |
| B4 TTS | XTTS-v2, Chatterbox, approved native clone candidates, Onur TR reference→EN | First PCM, RTF, identity, intelligibility, prosody, joins, VRAM, reference-length effect |
| B5 Directional E2E | Incoming and outgoing separately through real audio edges | Perceived first output, committed delivery, queue age, quality |
| B6 Full duplex soak | Simultaneous prerecorded/real TX, RX, TTS playback, WLK UI for ≥30 min | No growing backlog, no pipeline-caused underrun, no reorder, P95 targets, peak resources |

Corpus coverage includes short/long utterances, normal/fast Turkish, Turkish SOV
and clause boundaries, technical terminology, English accents, background noise,
Teams-processed audio, overlapping directions, and silence/bursts. [REF:
SRC-02:B015:L523-L582]

Benchmark discipline:

- cold start is recorded separately from warm steady state;
- warmups and repetitions are fixed and reported;
- candidates use the same audio/text corpus and declared quality settings;
- other GPU load is controlled and recorded;
- every result is labeled TARGET, MEASURED, or UNKNOWN;
- raw observations are retained locally, with P50/P95 and sample count;
- exact model and package revisions are included;
- non-commercial licensing is documented, not scored as a technical penalty for
  this personal project. [DEC: DEC-U001]

After hard gates, the provisional weighted comparison is:

| Dimension | Weight |
|---|---:|
| End-to-end P95, queue age, and backlog behavior | 35% |
| ASR/translation quality and TTS identity/intelligibility | 25% |
| Native Windows stability and long-session reliability | 20% |
| Peak VRAM/RAM and shared-session efficiency | 10% |
| Integration/maintenance complexity | 10% |

Any unbounded backlog, committed reordering, failed cross-language clone quality,
non-native required runtime, or unstable 30-minute run is a hard rejection
regardless of weighted score. [REF: SRC-01:B004:L31-L38]

## 25. Implementation Phases

| Phase | Scope | Exit gate | Model timing |
|---|---|---|---|
| A — complete | Plan, AGENTS, project skill, initial skeleton | Required artifacts validated; research unchanged | None |
| B — current verification | Device discovery, native mic/WASAPI loopback, VB-CABLE playback, resampler, bounded rings | Stable formats/IDs, callback non-blocking, test PCM heard by Teams, reconnect test | None |
| C — WLK + one ASR stream | Integrate existing WLK server/UI and ASR adapter | Local page receives measured partial/commit; Ready waits for warmup | Manually acquire Whisper model first |
| D — incoming subtitles | English loopback ASR→EN/TR MT→WLK partial/committed UI | Earliest useful partial measured, commits append-only, no backlog | Manually acquire EN→TR OPUS first |
| E — outgoing text | Turkish mic ASR→commit→TR/EN MT, no TTS | Ordered committed English appears in WLK with measured latency | Manually acquire TR→EN OPUS first |
| F — clone TTS benchmark | XTTS-v2 vs Chatterbox and approved native candidate with Onur TR reference→EN | Hard identity/intelligibility gate plus first-PCM, RTF, VRAM, stability report | Manually acquire XTTS-v2 and Chatterbox |
| G — outgoing audio | Selected clone TTS→bounded playback→VB-CABLE→Teams | First PCM measured; Teams hears ordered cloned English before long utterance ends | Selected TTS already local |
| H — full duplex | Concurrent sessions, shared weights, deadline scheduler, backpressure | Two directions plus TTS pass bounded 30-minute soak | Benchmark candidates only as approved |
| I — meeting persistence | SQLite migrations/writer, Save Meeting, history/browser UI, file-path recording metadata | Writes never stall realtime path; chronological history and off mode pass | None |
| J — optimization | Tune commit, chunking, isolation, memory, reconnect, packaging | P50/P95 gates, ≥30-minute soak, no accumulated latency | Only evidence-backed alternatives |
| K — simultaneous languages (Issues #27, #28) | Dynamic Source & Target language pairing, modular language registry, multi-language NLLB/OPUS & XTTS-v2 | Bidirectional & cross-language switching verified during active call without restart | Supported language weights local |
| L — anti-hallucination ASR (Issue #26) | Faster-Whisper-Large-v3 (32 decoder layers), acoustic duration / speech-rate validation gate | Non-speech / breath / click audio yields 0% hallucinations; WER improvement measured | Acquired and benchmarked 2026-09-06; turbo kept as latency default, large-v3 selectable |
| M — universal application support (Issue #25) | Presets for Teams, Zoom, Meet, Discord, Slack, VRChat, Dota 2, PUBG, CS2; floating HUD subtitle overlay, global hotkeys, Dual Input Modes (Voice Activity vs PTT with 150 ms pre-roll and 0 ms release flush) | Clean audio routing verified on non-Teams clients; PTT 0 ms release and pre-roll verified; HUD click-through passes | Local HUD/PTT smoke passed; per-application remote-call routing remains a separate hardware gate |
| N — true streaming TTS (Issue #9) | Chunked / generator streaming synthesis in XTTS adapter | First PCM TTFA drops from full sentence time to <300 ms | Existing XTTS-v2 |
| O — acoustic conditioning & AEC (Issues #16, #12, #1) | WebRTC AEC3 / SpeexDSP acoustic echo cancellation, render gain staging & soft limiter, pass-through toggle | Speaker bleed eliminated under open mic; zero digital clipping on loud utterances | None |
| P — voice dataset tooling (Issue #20) | Multi-reference 6-sample WAV recording guide, automated clipping/RMS validator | High speaker similarity score across multiple reference clips | Local voice profiles |
| Q — desktop packaging & onboarding (Issue #29) | Inno Setup installer (`TeamsTranslator-Setup.exe`), pywebview Edge native desktop window, pystray system tray daemon, VB-CABLE auto-installer helper, interactive model downloader wizard with resume & SHA256, in-app voice cloning onboarding recorder | Single installer builds cleanly; VB-CABLE installs with 1 click; models download with resume/progress; voice sample records & clones in-app | None (wizard handles acquisition) |

This ordering isolates device, ASR, MT, clone, routing, concurrency, and
persistence failures. [REF: SRC-00:B007:L573-L624; SRC-FINAL:B018:L512-L551]
No phase begins merely because it appears here; the current user task must
explicitly authorize it.

## 26. Acceptance Criteria

### 26.1 Global hard gates

- No inference, disk/database operation, network request, or blocking wait in an
  audio callback.
- No unbounded realtime queue.
- No sustained queue-age growth during normal speech.
- No unstable/Partial text reaches TTS.
- No silently lost or reordered committed segment.
- No Ready state before devices, selected models, and voice conditioning are warm.
- No non-loopback server listener by default.
- No startup model download or hidden WSL/container dependency.
- ASR, MT, and TTS remain replaceable and independently testable.
- The system runs without Ollama/general LLM processing.

### 26.2 Phase gates and status

| Criterion | Phase | Required status before exit |
|---|---|---|
| Native device enumeration and stable selection | B | MEASURED/PASS |
| Configured native Windows mic and WASAPI loopback continuous capture | B | MEASURED/PASS |
| Test PCM rendered through VB-CABLE into Teams | B | MEASURED/PASS |
| First ASR partial and first commit | C | MEASURED |
| Incoming Turkish partial/commit and ordering | D | MEASURED/PASS |
| Outgoing committed English text before TTS | E | MEASURED/PASS |
| Turkish-reference→English clone quality | F | MEASURED/PASS by declared listener/metric gate |
| TTS first PCM and sustained RTF < 1 under chosen workload | F/G | MEASURED/PASS |
| Full outgoing sound reaches Teams before a 10-second utterance ends | G | MEASURED/PASS |
| Two concurrent directions and shared resource policy | H | MEASURED/PASS |
| 30-minute full-duplex no-backlog/no-reorder soak | H/J | MEASURED/PASS |
| SQLite writes do not stall callback/realtime scheduling | I | MEASURED/PASS |
| Save Meeting off produces no meeting-history rows | I | TESTED/PASS |

Initial numerical targets are in Section 20 and remain UNKNOWN until benchmarked.

## 27. Risks and Mitigations

| Risk | Impact | Mitigation / decision gate |
|---|---|---|
| Turkish SOV requires later verb/context | Incorrect or delayed English speech | Stable-prefix + semantic clause commit, bounded max wait, adaptive pre-commit sizing; never speak partial |
| TTS RTF approaches/exceeds 1 | Growing outgoing backlog | Warm model, benchmark smaller/alternate clone backend, deadline priority, bounded queue, reject failing candidate |
| Turkish reference does not preserve identity in English | Core product goal fails | Test clean 3/10/30-second Turkish references, exact transcripts/conditioning modes, optional user-owned English reference, XTTS quality baseline |
| Two ASR directions duplicate weights | VRAM exhaustion/contention | Shared weights with isolated sessions; smaller incoming alternative only after comparison |
| RTX 50/Blackwell Python wheel mismatch | Installation or kernels fail | B0 compatibility matrix, pinned revisions/wheels, native smoke tests before model integration |
| Physical loopback captures notifications/music | False incoming transcript | Dedicated endpoint, second cable, then native process-loopback experiment |
| Teams AGC/noise suppression damages TTS | Robotic/cut speech | Fixed levels and Teams processing A/B test; telemetry/test call |
| Chatterbox or another backend is pseudo-streaming | TTFA worse than expected | Instrument synth request→first usable PCM; label true vs pseudo accurately |
| Persistence stalls or grows | Realtime regression/data loss | One background writer, WAL/batches/bounded queue, explicit Save Degraded state |
| Partial event explosion in SQLite | Database growth | UPSERT current partial row; commit final row state; aggregate telemetry |
| Model license/revision drift | Reproducibility or permitted-use ambiguity | Pin commit, store license snapshot/checksum; DEC-U001 governs use scope |
| Private transcript/voice exposure | Privacy harm | Localhost-only, local paths, Save Meeting control, no cloud, future retention/delete controls |
| Device disappears mid-call | Pipeline interruption | Stable IDs, reconnect state machine, visible Degraded/Error, no callback block |
| General LLM inserted into hot path | Latency/hallucination/backlog | Keep optional post-meeting adapter disabled; require full-duplex proof before reconsideration |

[REF: SRC-FINAL:B020:L578-L615; SRC-05:B012:L358-L364]

## 28. Open Questions

These do not reopen DEC-U001…DEC-U008 and do not block Phase A:

1. Does the measured stable endpoint mapping survive reboot/device reorder, and
   does VB-CABLE render reach a Teams test call without clipping or processing loss?
   Resolve the remaining Phase B hardware gates.
2. Which WhisperLiveKit revision and native ASR runtime combination passes
   Python 3.12, CUDA 12.8, and Blackwell tests? Resolve in B0/Phase C.
3. Can the selected WLK backend share one warm ASR weight instance across both
   language sessions with acceptable scheduling? Measure in Phase H.
4. Does one-cable physical loopback remain clean enough, or is a second cable or
   process-specific loopback needed? Decide after recorded contamination tests.
5. What are the exact paths, model configs, hashes, licenses, and tokenizers for
   the three already-local Qwen-named assets? Inspect only when the user supplies
   locations; do not search/copy blindly.
   The user updated the following information:
   - qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors: C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders
   - qwen_3_4b_fp8_mixed.safetensors: C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders
   - Qwen3.8-27B-Uncensored: C:\Users\Onur\.ollama\models\manifests\registry.ollama.ai\orcarouter\Qwen3.8-27B-Uncensored\q3_k_l
6. Which user-owned Turkish reference recordings and transcripts will represent
   Default, Formal, and Energetic profiles? Capture/approve before Phase F.
7. What listener protocol and similarity metric constitute the voice-identity
   hard gate? Define before comparing TTS candidates.
8. Should Save Meeting be immutable for a session, as initially proposed, or may
   it be toggled mid-meeting with prospective-only persistence?
9. What transcript/audio retention and manual deletion policy should the history
   UI expose?
   -No, no historical data will be kept for each meeting.
10. After the first local benchmark, should the provisional latency targets be
    tightened or relaxed? Any change must retain TARGET/MEASURED history.
    -No, no historical data will be kept for each meeting.

## 29. Source Reference Index

Block IDs below are deterministic within each source: ascending by physical
start line. A block is indexed without modifying the historical file. Qualified
or superseded content is cited only with the explicit correction in this Plan.

| Source ID | File | Block | Lines | Heading / Description |
|---|---|---|---|---|
| ISSUE-25 | https://github.com/onurozkir/speech-to-translate-en-tr/issues/25 | current | 2026-09-05 user clarification; resolved 2026-09-06 | Universal apps/games, same-session PTT, automatic secondary HUD, retain VB-CABLE; process loopback deferred (DEC-U010) |
| ISSUE-26 | https://github.com/onurozkir/speech-to-translate-en-tr/issues/26 | current | 2026-09-05 user clarification; resolved 2026-09-06 | Full large-v3 option; preserve genuine spoken phrases. Large-v3 partial P50/P95 587/933 ms vs turbo 243/366 ms; turbo keeps latency default |
| HF-LARGE-V3 | https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478 | pinned | model card | CTranslate2 large-v3 format, MIT, FP16 stored weights; compute type configurable |
| MS-PROCESS-LOOPBACK | https://learn.microsoft.com/en-us/samples/microsoft/windows-classic-samples/applicationloopbackaudio-sample/ | official | accessed 2026-09-05 | Process capture isolates incoming audio; deferred baseline upgrade |
| MS-HOOKS | https://learn.microsoft.com/en-us/windows/win32/winmsg/lowlevelkeyboardproc | official | accessed 2026-09-05 | Dedicated hook thread and prompt non-blocking handoff |
| SRC-00 | docs/research/00_RESEARCH_PLAN.md | B001 | L3-L8 | Status, target hardware, latency priority, benchmark decision gate |
| SRC-00 | docs/research/00_RESEARCH_PLAN.md | B002 | L12-L28 | Dual incremental outgoing/incoming product goal |
| SRC-00 | docs/research/00_RESEARCH_PLAN.md | B003 | L99-L124 | Partial versus committed, stable prefix, overlapping TTS chunks |
| SRC-00 | docs/research/00_RESEARCH_PLAN.md | B004 | L422-L433 | Bounded queues and no-backlog rule |
| SRC-00 | docs/research/00_RESEARCH_PLAN.md | B005 | L437-L474 | Latency instrumentation, distributions, linguistic floor |
| SRC-00 | docs/research/00_RESEARCH_PLAN.md | B006 | L478-L542 | Progressive ASR/MT/TTS/full-duplex benchmark matrix |
| SRC-00 | docs/research/00_RESEARCH_PLAN.md | B007 | L573-L624 | Phased POC and exit criteria; model names are historical |
| SRC-01 | docs/research/01_DECISION_MATRIX.md | B001 | L3-L8 | Blank-before-measurement rule and ASR metrics |
| SRC-01 | docs/research/01_DECISION_MATRIX.md | B002 | L14-L17 | MT benchmark columns |
| SRC-01 | docs/research/01_DECISION_MATRIX.md | B003 | L22-L25 | TTS/voice-clone benchmark columns |
| SRC-01 | docs/research/01_DECISION_MATRIX.md | B004 | L31-L38 | End-to-end P50/P95/P99, drop, backlog comparison |
| SRC-01 | docs/research/01_DECISION_MATRIX.md | B005 | L40-L47 | Final selection and hard-gate rules |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B001 | L7-L26 | Feasibility, VB-CABLE role, GPU inference, benchmark-before-choice |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B002 | L28-L47 | Exact outgoing and incoming user flows |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B003 | L49-L91 | Scope, initial targets, and dual-pipeline architecture |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B004 | L125-L171 | Audio topology alternatives and Teams settings |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B005 | L173-L181 | Audio framing, resampling, callback boundary |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B006 | L183-L211 | Partial/committed state, semantic commit, backpressure |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B007 | L213-L251 | ASR candidates, streaming distinction, shared weights/sessions |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B008 | L253-L272 | OPUS/CTranslate2 MT and Turkish→English linguistic latency |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B009 | L274-L307 | TTS true/pseudo streaming and cross-language clone risk |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B010 | L309-L368 | Voice profiles, reference repositories, adapter/fork decision |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B011 | L370-L408 | Windows tools and isolated model adapters |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B012 | L410-L448 | Environment verification and telemetry |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B013 | L450-L474 | Target-hardware benchmark decision gate |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B014 | L476-L522 | Audio, incoming, outgoing, voice, and optimization phases |
| SRC-02 | docs/research/02-teams-realtime-translation-plan.md | B015 | L523-L582 | Corpus, quality metrics, risks, and final decision matrix |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B001 | L8-L16 | Current WLK Qwen/native Windows and Nemotron paths |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B002 | L17-L24 | Current MT/TTS/license/runtime compatibility notes |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B003 | L26-L46 | Implementation-oriented outgoing/incoming summary |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B004 | L48-L54 | Hot-path partial/commit and bounded-queue invariants |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B005 | L56-L70 | Native bridge and adapter-process architecture |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B006 | L72-L83 | Environment isolation and hot-path persistence boundary |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B007 | L85-L99 | Repository roles including WLK and NeMo-Speech.cpp |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B008 | L101-L110 | Closed/open research decision gates |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B009 | L112-L119 | Phase-0 local benchmark summary |
| SRC-03 | docs/research/03-teams-realtime-translation-research.md | B010 | L121-L138 | Pre-implementation environment-to-soak sequence |
| SRC-04 | docs/research/04-teams-research-with-google.md | B001 | L10-L14 | Superseded fixed product promises |
| SRC-04 | docs/research/04-teams-research-with-google.md | B002 | L41-L43 | Parallel TX/RX decomposition |
| SRC-04 | docs/research/04-teams-research-with-google.md | B003 | L45-L95 | Qualified TX/RX data-flow diagram; model/timing details not canonical |
| SRC-04 | docs/research/04-teams-research-with-google.md | B004 | L124-L135 | Qualified semantic clause chunking and SOV/SVO constraint |
| SRC-04 | docs/research/04-teams-research-with-google.md | B005 | L137-L148 | Superseded fixed end-to-end latency arithmetic |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B001 | L20-L31 | Qualified WLK-based high-level flow |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B002 | L52-L60 | Windows audio tools and device directions |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B003 | L62-L77 | WLK capabilities; historical Docker boundary superseded |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B004 | L119-L138 | WLK web UI and native Windows boundary; old overlay/Docker advice qualified |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B005 | L142-L157 | Qualified outgoing/incoming WLK client flows |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B006 | L161-L187 | Superseded unmeasured VRAM and latency estimates |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B007 | L191-L205 | Historical license notes; personal-use decision now governs |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B008 | L207-L246 | Native stack and dependency categories |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B009 | L250-L297 | Superseded Docker Compose design |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B010 | L301-L328 | Reference repositories and models |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B011 | L330-L354 | Qualified bring-up sequence and superseded open decisions |
| SRC-05 | docs/research/05-ms-teams-realtime-ceviri-arastirma-ve-mimari.md | B012 | L358-L364 | Operational risk notes |
| SRC-FINAL | docs/research/final.md | B001 | L7-L27 | Qualified research synthesis |
| SRC-FINAL | docs/research/final.md | B002 | L29-L59 | Full-duplex immutable goals |
| SRC-FINAL | docs/research/final.md | B003 | L61-L128 | Research comparison and qualified layered architecture |
| SRC-FINAL | docs/research/final.md | B004 | L130-L171 | Outgoing/incoming flows and independent MT rationale |
| SRC-FINAL | docs/research/final.md | B005 | L173-L204 | Partial/committed, semantic commit, backpressure |
| SRC-FINAL | docs/research/final.md | B006 | L206-L235 | Single-cable topology and isolation upgrades |
| SRC-FINAL | docs/research/final.md | B007 | L237-L284 | Component candidates and true/pseudo TTS correction |
| SRC-FINAL | docs/research/final.md | B008 | L286-L330 | Voice profile and conditioning-cache design |
| SRC-FINAL | docs/research/final.md | B009 | L332-L362 | Native responsibilities and hot-path boundaries; WSL/SQLite advice qualified |
| SRC-FINAL | docs/research/final.md | B010 | L364-L379 | Deadline scheduler and shared ASR sessions |
| SRC-FINAL | docs/research/final.md | B011 | L381-L417 | Target latency and telemetry |
| SRC-FINAL | docs/research/final.md | B012 | L419-L432 | Integration difficulty ranking |
| SRC-FINAL | docs/research/final.md | B013 | L434-L445 | Environment/revision readiness gate |
| SRC-FINAL | docs/research/final.md | B014 | L447-L464 | ASR benchmark |
| SRC-FINAL | docs/research/final.md | B015 | L465-L477 | MT benchmark |
| SRC-FINAL | docs/research/final.md | B016 | L479-L495 | TTS benchmark candidates and metrics; Piper item superseded |
| SRC-FINAL | docs/research/final.md | B017 | L497-L510 | Full-duplex 30-minute soak hard gate |
| SRC-FINAL | docs/research/final.md | B018 | L512-L551 | Qualified phased implementation |
| SRC-FINAL | docs/research/final.md | B019 | L553-L576 | Superseded Piper/PySide/WSL bring-up stack |
| SRC-FINAL | docs/research/final.md | B020 | L578-L615 | Risks and qualified final selection order |
