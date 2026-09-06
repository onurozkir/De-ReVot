"""Validate a local voice dataset without loading models or downloading weights."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_translator.tts.conditioning import VoiceProfileManager


def main():
    parser = argparse.ArgumentParser(description="De-ReVot reference WAV validator")
    parser.add_argument("profile_id")
    parser.add_argument("--profiles-root", default="voices")
    args = parser.parse_args()
    manager = VoiceProfileManager(args.profiles_root)
    profile = manager.get_profile(args.profile_id)
    if profile is None:
        parser.exit(1, f"Profile '{args.profile_id}' not found. {manager.errors}\n")
    try:
        reports = manager.validate_reference_audio(profile.all_reference_paths)
        print(json.dumps({"profile_id": profile.id, "reference_count": len(reports),
                          "audio_hash": manager.compute_audio_hash(profile.all_reference_paths),
                          "references": reports}, ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
