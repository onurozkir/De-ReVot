# Contributing to De-ReVot

De-ReVot welcomes bug reports, feature requests, documentation improvements, and code contributions.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

## Development Setup

1. **Prerequisites**:
   - Windows 11 (64-bit)
   - Python 3.12 (64-bit)
   - NVIDIA GPU with CUDA 12.x support (or CPU for mock development)
   - Git

2. **Clone and Install**:
   ```powershell
   git clone https://github.com/onurozkir/speech-to-translate-en-tr.git
   cd speech-to-translate-en-tr

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -e .
   ```

3. **Running the Test Suite**:
   All PRs must pass the test suite:
   ```powershell
   python -m pytest tests/ -v
   ```

4. **Running with Mock Adapters** (No GPU required):
   ```powershell
   python src/voice_translator/main.py run --mock
   ```

---

## Architectural Guidelines

To maintain low latency, reliability, and local privacy, all contributions must respect the following core invariants:

- **Offline-First**: Never trigger silent network requests or weight downloads during runtime or unit tests. Models must be explicitly managed via `scripts/download_models.py`.
- **Decoupled Adapters**: ASR (Whisper), MT (MarianMT / CTranslate2), and TTS (XTTS-v2) must remain behind their respective abstract adapter interfaces.
- **Non-blocking Audio Callbacks**: PortAudio callbacks must never perform inference, disk I/O, or blocking synchronization. Audio data flows through bounded lock-free ring buffers.
- **Bounded Real-time Queues**: Always use `BoundedQueue` to prevent unbounded latency accumulation during long meetings.
- **Two-Phase Text Commitment**: Partial text remains replaceable; only finalized punctuation/silence-committed text enters MT and TTS queues.

---

## Pull Request Workflow

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Make your changes with concise, descriptive commit messages:
   ```bash
   git commit -m "feat(mt): add Ukrainian NLLB translation adapter"
   ```
3. Ensure all tests pass:
   ```bash
   python -m pytest tests/ -v
   ```
4. Push to your fork and submit a Pull Request to `main`.
5. Describe your changes clearly in the PR description, referencing any related issues.

---

## Reporting Issues & Feature Requests

- Use GitHub Issues to report bugs or propose new features.
- Provide step-by-step reproduction steps, logs, and your hardware configuration (GPU model, VRAM, Windows version) when filing bug reports.
