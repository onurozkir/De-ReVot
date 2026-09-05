---
name: realtime-voice-translator
description: Use for planning, implementing, debugging, benchmarking, or reviewing the local Windows Python realtime voice translator for meetings and games. Prioritize end-to-end latency, full-duplex streaming, WhisperLiveKit UI, desktop subtitles, push-to-talk, voice cloning, bounded queues, native Windows execution, and canonical root Plan.md. Do not use for unrelated repositories.
---

# Realtime Voice Translator

Apply this workflow whenever the skill matches:

1. Read root Plan.md completely before architectural work.
2. Treat DEC-Uxxx user decisions in that Plan as authoritative.
3. Preserve the native Windows and Python-only architecture.
4. Never introduce Docker, containers, or Compose unless the user explicitly
   reverses DEC-U002.
5. Never silently introduce WSL; report Linux-only dependencies as
   excluded/experimental.
6. Reuse and extend WhisperLiveKit's existing web UI as the primary interface.
7. Do not introduce Piper as a selected, default, or planned TTS.
8. Keep ASR, MT, and TTS behind narrow, testable adapters.
9. Keep audio callbacks non-blocking and free of inference, disk, database, and
   network work.
10. Use bounded queues/ring buffers only; never add an unbounded asyncio.Queue.
11. Preserve replaceable Partial versus append-only Committed semantics.
12. Never send unstable or Partial text to TTS.
13. Measure actual speech-to-output end-to-end latency, not only model inference.
14. Treat the RTX 5060 Ti 16 GB VRAM budget as a hard shared constraint.
15. Share model weights across sessions when supported; avoid duplicate GPU loads.
16. Keep SQLite writes and reads outside latency-critical audio callbacks.
17. Never silently download model weights at startup, tests, or runtime.
18. Respect the deterministic models/ layout and pinned model manifest in the Plan.
19. Reuse approved externally stored models via configurable absolute paths;
    never copy them merely to fit the repository layout.
20. Add task-relevant unit, integration, hardware, and benchmark coverage when
    implementation begins.
21. Update root Plan.md when an architectural decision or proven runtime finding changes.
22. Add or update DEC/REF traceability when modifying architectural decisions.
23. Never silently contradict a DEC-Uxxx decision, even if historical research
    recommends otherwise.
24. Stop implementation and report the exact conflict when work would require
    violating a hard architecture decision.
25. Prefer WASAPI for Windows audio endpoints, especially speaker loopback and
    render. Permit another native Windows host API for a specific physical mic
    only after recorded format/signal evidence shows its WASAPI path is unusable;
    persist the measured endpoint fingerprint and retain WASAPI loopback.
26. Diagnose capture with saved WAV evidence plus frame-level RMS/peak, active
    duration, callback counts, discontinuities, and errors. An opened stream or
    endpoint name alone is not proof of signal.
27. For isolated speaker-loopback verification, render a known tone to the
    matching physical output while capturing its WASAPI loopback, then verify
    duration and dominant frequency. Keep this distinct from application routing
    and full-duplex acceptance gates.
28. A full-window ASR backend must not decode on every audio frame. Coalesce
    replaceable partial inference to a measured bounded cadence, then run one
    final decode over the complete buffered utterance at the VAD endpoint.
29. Never erase current utterance audio merely because one replaceable partial
    fails confidence or hallucination checks. Record the rejection and allow a
    later hypothesis to repair it; reset only on final rejection or overload.
30. Never claim cloned/routed speech in UI history at MT completion. Append the
    durable outgoing history item only after first TTS PCM has entered the
    bounded render path; this still does not claim remote application receipt.
31. Reject unsupported ASR hypotheses using acoustic duration, voiced ratio,
    character rate and model confidence. Never blacklist spoken phrases or strip
    subtitle/context prefixes. Preserve genuine Turkish dotted-I speech. An empty
    final decode must not revive an obsolete partial. See Plan.md §12.4.
32. Preserve the real first-sample and last-sample monotonic capture timestamps
    through ASR endpoint flush. Event creation time is separate and must never
    replace the captured audio span used by end-to-end telemetry.
33. Every exposed ASR, MT, or TTS configuration knob must be wired to its
    adapter or removed and documented as fixed. Keep latency-protecting partial
    settings explicit, and do not claim a quality/latency effect without the
    corresponding benchmark.
34. Never use dummy or zero conditioning tensors for production voice-clone
    synthesis. Prepare the selected profile before Ready or meeting start and
    fail visibly with the exact profile error if its reference or conditioning
    is unavailable.
35. Shared-weight ASR inference must use a bounded, captured-audio-deadline-aware
    scheduler rather than a bare lock. Reserve bounded priority for outgoing
    delivery, prevent starvation by queue age, and record wait/deadline-miss
    telemetry per session and direction.
36. MT execution must use CTranslate2 INT8 rather than unquantized HuggingFace float32.
    Always ensure `</s>` EOS token is present before CTranslate2 batch translation,
    decode with `skip_special_tokens=True`, and apply domain glossary replacements
    with word-boundary awareness.
37. Always consult `FILE_MAP.md` before searching, reading, or editing files.
    Identify the responsible subsystem and inspect only the 1–2 targeted source
    files and their paired unit tests. Never run broad recursive repository searches.
38. Start Meeting owns both directions for every application preset. PTT gates
    only outgoing microphone admission; incoming subtitles remain active. Keep
    key edges timestamped, queues bounded and 150 ms pre-roll sample-accurate.
    Release requests immediate endpoint flush; it does not promise zero inference
    latency or cut the committed cloned playback that follows the release.
39. The Win32 desktop HUD is a secondary, click-through, no-activate surface.
    Keep WLK primary, start/stop native workers with the session, and expose native
    failures in status. Do not inject into games or claim exclusive-fullscreen
    support. Unit tests must not open real HUD windows or register global hooks;
    run the separate explicit native smoke check when validating desktop changes.
40. Keep VB-CABLE outgoing and physical endpoint loopback incoming as baseline.
    Presets guide setup, never silently change Windows/game settings or fall back
    from an explicit missing endpoint. Record per-application hardware acceptance.
41. Use `voice_translator` imports and `VOICE_TRANSLATOR_*` configuration names.
    Legacy `TEAMS_TRANSLATOR_*` environment values remain compatible with lower
    precedence. Keep root Plan.md and this skill tracked through naming migrations.
42. Language support is registry-driven via `[languages]` in config and
    `voice_translator.config.languages`. Never hardcode language codes in
    pipelines, routes, or the Web UI. Pair routing prefers a pinned OPUS model
    and falls back to NLLB-200; validate pairs and the XTTS matrix before
    starting or switching. New languages require a registry definition,
    an `enabled` entry, and explicit model downloads (`download_models.py --lang`).

Before editing, inspect the current implementation and dirty worktree. During
review, reject growing backlog, committed reordering/loss, false Ready states,
non-localhost exposure, or unmeasured claims labeled as measured. Do not begin a
later phase unless the current user task authorizes it.
