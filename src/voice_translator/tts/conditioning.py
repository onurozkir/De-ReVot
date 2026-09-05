"""Speaker conditioning cache and voice profile manager."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from voice_translator.tts.base import VoiceProfile

logger = logging.getLogger(__name__)


class VoiceProfileManager:
    """Discovers and caches voice cloning profiles on disk."""

    def __init__(self, profiles_root: str = "voices"):
        self.profiles_root = Path(profiles_root)
        self.profiles: Dict[str, VoiceProfile] = {}
        self.load_profiles()

    def load_profiles(self):
        self.profiles.clear()
        if not self.profiles_root.exists():
            self.profiles_root.mkdir(parents=True, exist_ok=True)
            return

        for profile_dir in self.profiles_root.iterdir():
            if profile_dir.is_dir():
                manifest_file = profile_dir / "profile.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        profile_id = data.get("id", profile_dir.name)
                        ref_audio = data.get("reference_audio_path", "reference.wav")
                        # Resolve audio path relative to profile dir if not absolute
                        ref_audio_path = Path(ref_audio)
                        if not ref_audio_path.is_absolute():
                            ref_audio_path = profile_dir / ref_audio

                        # Multi-sample reference audio discovery
                        multi_refs = data.get("reference_audio_paths", [])
                        resolved_multi: List[str] = []
                        if isinstance(multi_refs, list):
                            for m in multi_refs:
                                mp = Path(m)
                                if not mp.is_absolute():
                                    mp = profile_dir / m
                                if mp.exists():
                                    resolved_multi.append(str(mp.resolve()))

                        # Auto-discover reference_*.wav, ref_*.wav, sample_*.wav in profile_dir
                        for pattern in ["reference_*.wav", "ref_*.wav", "sample_*.wav"]:
                            for found_path in sorted(profile_dir.glob(pattern)):
                                abs_p = str(found_path.resolve())
                                if abs_p not in resolved_multi and abs_p != str(ref_audio_path.resolve()):
                                    resolved_multi.append(abs_p)

                        cache_dir = profile_dir / "cache"
                        cache_dir.mkdir(exist_ok=True)

                        target_langs = data.get("target_languages")
                        if not target_langs:
                            target_langs = [data.get("target_language", "en")]
                        elif isinstance(target_langs, str):
                            target_langs = [target_langs]

                        prof = VoiceProfile(
                            id=profile_id,
                            display_name=data.get("display_name", profile_id),
                            backend=data.get("backend", "xtts_v2"),
                            reference_audio_path=str(ref_audio_path.resolve()),
                            reference_audio_paths=resolved_multi,
                            reference_text=data.get("reference_text"),
                            reference_language=data.get("reference_language", "tr"),
                            target_language=data.get("target_language", target_langs[0] if target_langs else "en"),
                            target_languages=target_langs,
                            is_default=bool(data.get("is_default", False)),
                            conditioning_cache_path=str(cache_dir.resolve()),
                            metadata=data.get("metadata", {}),
                        )
                        self.profiles[profile_id] = prof
                    except Exception as e:
                        logger.error(f"Failed to load profile manifest '{manifest_file}': {e}")

    def get_default_profile(self) -> Optional[VoiceProfile]:
        for prof in self.profiles.values():
            if prof.is_default:
                return prof
        if self.profiles:
            return next(iter(self.profiles.values()))
        return None

    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        return self.profiles.get(profile_id)

    def list_profiles(self) -> List[VoiceProfile]:
        return list(self.profiles.values())

    @staticmethod
    def compute_audio_hash(audio_paths: str | List[str] | Path) -> str:
        """Compute SHA256 of reference audio file(s) for cache keying."""
        if isinstance(audio_paths, (str, Path)):
            paths = [Path(audio_paths)]
        elif isinstance(audio_paths, list):
            paths = [Path(p) for p in audio_paths]
        else:
            paths = []

        existing_paths = [p for p in paths if p.exists()]
        if not existing_paths:
            return ""

        hasher = hashlib.sha256()
        # Sort by filename to ensure deterministic hashing
        for p in sorted(existing_paths, key=lambda x: str(x)):
            hasher.update(p.name.encode("utf-8"))
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        return hasher.hexdigest()

