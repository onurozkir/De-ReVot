"""Deterministic profile discovery, reference validation and content hashing."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_translator.tts.base import VoiceProfile

logger = logging.getLogger(__name__)


class VoiceProfileManager:
    def __init__(self, profiles_root: str = "voices"):
        self.profiles_root = Path(profiles_root)
        self.profiles: dict[str, VoiceProfile] = {}
        self.errors: dict[str, str] = {}
        self.load_profiles()

    def load_profiles(self):
        self.profiles.clear()
        self.errors.clear()
        if not self.profiles_root.exists():
            return
        for directory in sorted(self.profiles_root.iterdir()):
            if not directory.is_dir() or directory.name == "cache":
                continue
            manifest = directory / "profile.json"
            try:
                data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
                if not isinstance(data, dict):
                    raise ValueError("profile.json must contain an object")
                if "reference_audio_paths" in data:
                    refs = data["reference_audio_paths"]
                    if not isinstance(refs, list) or not refs or not all(isinstance(p, str) and p for p in refs):
                        raise ValueError("reference_audio_paths must be a nonempty list of paths")
                elif "reference_audio_path" in data:
                    refs = [data["reference_audio_path"]]
                    if not isinstance(refs[0], str) or not refs[0]:
                        raise ValueError("reference_audio_path must be a nonempty path")
                else:
                    # Immediate WAV children only; explicit manifests stay curated.
                    refs = [p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
                    if not refs:
                        if manifest.exists():
                            raise ValueError("No reference WAV files found")
                        continue
                paths = sorted({str((directory / p).resolve()) for p in refs})
                profile_id = str(data.get("id", directory.name))
                if profile_id in self.profiles:
                    raise ValueError(f"Duplicate profile id: {profile_id}")
                targets = data.get("target_languages") or [data.get("target_language", "en")]
                if isinstance(targets, str):
                    targets = [targets]
                self.profiles[profile_id] = VoiceProfile(
                    id=profile_id, display_name=data.get("display_name", profile_id),
                    backend=data.get("backend", "xtts_v2"), reference_audio_paths=paths,
                    reference_text=data.get("reference_text"),
                    reference_language=data.get("reference_language", "tr"),
                    target_language=data.get("target_language", targets[0]), target_languages=targets,
                    is_default=bool(data.get("is_default", False)),
                    conditioning_cache_path=str((directory / "cache").resolve()),
                    metadata=data.get("metadata", {}),
                )
            except (ValueError, TypeError, OSError) as exc:
                self.errors[directory.name] = f"{manifest}: {exc}"
                logger.error("Failed to load voice profile: %s", self.errors[directory.name])

    def get_default_profile(self) -> VoiceProfile | None:
        return next((p for p in self.profiles.values() if p.is_default),
                    next(iter(self.profiles.values()), None))

    def get_profile(self, profile_id: str) -> VoiceProfile | None:
        return self.profiles.get(profile_id)

    def list_profiles(self) -> list[VoiceProfile]:
        return list(self.profiles.values())

    @staticmethod
    def compute_audio_hash(audio_paths: str | list[str] | Path) -> str:
        paths = [audio_paths] if isinstance(audio_paths, (str, Path)) else audio_paths
        if not paths:
            raise RuntimeError("Voice reference audio not found: no paths supplied")
        hasher = hashlib.sha256(b"multi-reference-v2\0")
        for path in sorted({Path(p).resolve() for p in paths}):
            try:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").digest()
            except OSError as exc:
                raise RuntimeError(f"Voice reference audio '{path}' not found or unreadable: {exc}") from exc
            hasher.update(path.name.encode("utf-8") + b"\0" + digest)
        return hasher.hexdigest()

    @staticmethod
    def validate_reference_audio(audio_paths: list[str]) -> list[dict]:
        """Validate before cache lookup with bounded reads and no file conversion."""
        if not audio_paths:
            raise RuntimeError("Voice reference audio not found: no paths supplied")
        reports = []
        for path in audio_paths:
            try:
                if not Path(path).is_file():
                    raise ValueError("not found")
                if Path(path).stat().st_size <= 4096:
                    raise ValueError("empty or too small (must exceed 4 KB)")
                with sf.SoundFile(path) as audio:
                    duration = len(audio) / audio.samplerate
                    if audio.format not in {"WAV", "WAVEX", "RF64"}:
                        raise ValueError("expected WAV audio")
                    energy, count, clipped, peak = 0.0, 0, 0, 0.0
                    for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
                        if not np.isfinite(block).all():
                            raise ValueError("audio contains NaN or infinity")
                        mono = block.mean(axis=1, dtype=np.float64)
                        energy += float(np.dot(mono, mono))
                        count += len(mono)
                        clipped += int(np.count_nonzero(np.abs(block) >= 0.999))
                        peak = max(peak, float(np.max(np.abs(block))))
                    rms = float(np.sqrt(energy / max(count, 1)))
                    if rms < 0.005:
                        raise ValueError(f"near-silent audio after mono downmix (RMS {rms:.6f} < 0.005)")
                    warnings = []
                    if duration < 3 or duration > 30:
                        warnings.append(f"duration {duration:.2f}s outside recommended 3-30s; aim for 8-10s")
                    if clipped:
                        warnings.append(f"{clipped} clipped samples; re-record at lower gain")
                    if audio.channels != 1:
                        warnings.append(f"{audio.channels} channels; XTTS will downmix to mono")
                    reports.append(dict(path=path, duration_sec=duration, sample_rate=audio.samplerate,
                                        channels=audio.channels, rms=rms, peak=peak, warnings=warnings))
            except (OSError, ValueError, sf.LibsndfileError) as exc:
                raise RuntimeError(f"Voice reference audio '{path}': {exc}") from exc
        if len({r["sample_rate"] for r in reports}) > 1:
            for report in reports:
                report["warnings"].append("mixed sample rates; XTTS resamples internally")
        for report in reports:
            for warning in report["warnings"]:
                logger.warning("Voice reference '%s': %s", report["path"], warning)
        return reports
