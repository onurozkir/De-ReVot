# Project

Local, full-duplex voice translator for meetings and games: Turkish microphone
to cloned English speech over VB-CABLE, and incoming English audio to live
Turkish subtitles in the Web UI and desktop HUD.

# Source of Truth

- `Plan.md` is canonical.
- `docs/research/` contains historical inputs, not current authority.
- Direct user requests override the Plan; update the Plan and its DEC/REF
  traceability when an architectural decision changes.

# Communication
- Final reports and explanations to the user must be Turkish.
- Code, identifiers, protocols and filenames remain English.

# Agent File Navigation Invariant (FILE_MAP)

- **ALWAYS** consult `FILE_MAP.md` before searching, reading, or editing files.
- **NEVER** perform broad recursive codebase scans, blind searches, or whole-repo directory dumps.
- Use the **Subsystem Routing Matrix** in `FILE_MAP.md` to pinpoint the exact 1–2 files responsible for the task domain and open ONLY those files.
- Inspect paired unit tests in `tests/unit/` for that specific subsystem first before running the full test suite.

## At the beginning of every task

Before making changes:

1. Consult `FILE_MAP.md` to identify the responsible subsystem, source files, and test files.
2. Read the relevant skill under `.agents/skills/`.
3. Inspect the existing implementation of only the targeted files before proposing changes.

# Skills

If a task changes the reusable contract or workflow of a subsystem,
update the corresponding skill.

- Common Project agent workflow changed
  → update `.agents/skills/realtime-voice-translator/SKILL.md`

Read and apply this skill for planning,
implementation, debugging, benchmarking, or review. Inspect the current
implementation and dirty worktree before editing.


Do not copy the same information into multiple files unless necessary.
Prefer references to the canonical source.


# SQLite Database
- NEVER create temporary `.py`, `.js`, or shell scripts to query, inspect, or mutate the SQLite database (`/data/speech-to-translate.sqlite`).
- ALWAYS use the `speech-to-translate_db` MCP server tools:
  - `get_schema`: Discover tables, columns, foreign keys, and indexes.
  - `read_query`: Execute read-only SELECT/PRAGMA queries (auto-parses JSON columns).
  - `write_query`: Execute parameterized INSERT/UPDATE/DELETE queries with transaction safety.

# Environment

- Windows 11, native execution.
- Python 3.12.
- CUDA 12.8 target on RTX 5060 Ti 16 GB; 32 GB RAM.
- No Docker, Docker Compose, containerized model service, or normal-runtime WSL.
- No Node backend or separate frontend build beyond WhisperLiveKit's own assets.
- Localhost-only runtime.

# Architecture Invariants

- WhisperLiveKit web UI is primary; do not replace it with PySide.
- Voice cloning is required; Piper is rejected.
- ASR, MT, and TTS stay behind adapters.
- Audio callbacks never run inference or I/O.
- Realtime queues are bounded; no unbounded asyncio.Queue.
- Partial text is replaceable; committed text is ordered and append-only.
- TTS receives committed text only.
- Backlog must not accumulate in long meetings.
- Share GPU model weights where supported and warm models before Ready.
- SQLite meeting persistence runs outside the realtime callback path.

# Model Policy

- Models are downloaded explicitly into the paths defined by root `Plan.md`.
- Never silently download weights.
- Record exact repository, revision, license, and destination.
- External local models use configurable absolute paths and are not copied.
- Non-commercial licenses are acceptable for this personal project.

# Testing Policy

Add phase-relevant unit/integration tests and target-hardware benchmarks when
implementation begins. Report latency as TARGET, MEASURED, or UNKNOWN; include
P50/P95, queue age, RTF, VRAM, and full-duplex soak evidence. No test may
silently use the network or download a model.

# Documentation Policy

Keep architecture in root `Plan.md` rather than duplicating it here. Preserve
historical research files. Update the project skill if a reusable workflow
contract changes. Final reports and explanations to the user are Turkish; code,
identifiers, protocols, and filenames remain English.

For data inspection or mutation, do not create temporary Python, JavaScript, or
shell scripts against data/meetings.sqlite3. Use the configured
speech-to-translate_db MCP tools when available; otherwise add and use the
project's reviewed persistence/test interfaces once that phase exists.

Use the installed Caveman skill in lite mode for agent responses unless the user
requests normal mode. Compression must not omit technical or validation details.

# Current Phase

Implementation active (Phases B through J completed; Phases K through P tracked via Plan.md and GitHub issues).
All production adapters, streaming pipelines, persistence, telemetry, and Web UI are operational.
Consult `FILE_MAP.md` and `Plan.md` for subsystem architecture and active issue scope.

# Caveman

Use the installed Caveman skill in lite mode by default for agent responses in this repository.

Preserve all technical substance, code, commands, paths, errors, identifiers, and architectural details exactly.

Do not let compression reduce implementation completeness or omit validation steps.

If the user says "stop caveman" or "normal mode", disable Caveman for that session.
