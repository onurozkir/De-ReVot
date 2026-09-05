"""Manual offline model downloader for pinned Hugging Face models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MODELS = {
    "whisper-large-v3": {
        "repo_id": "Systran/faster-whisper-large-v3",
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "destination": "models/asr/whisper-large-v3",
        "license": "MIT",
        "purpose": "Full large-v3 CTranslate2 ASR; acoustic quality/latency comparison",
    },
    "whisper": {
        "repo_id": "openai/whisper-large-v3-turbo",
        "destination": "models/asr/whisper-large-v3-turbo",
        "license": "MIT",
        "purpose": "ASR bring-up (TR & EN)",
    },
    "mt-tr-en": {
        "repo_id": "Helsinki-NLP/opus-mt-tc-big-tr-en",
        "destination": "models/mt/opus-mt-tc-big-tr-en",
        "license": "CC-BY-4.0",
        "purpose": "Outgoing TR->EN Translation",
    },
    "mt-en-tr": {
        "repo_id": "Helsinki-NLP/opus-mt-tc-big-en-tr",
        "destination": "models/mt/opus-mt-tc-big-en-tr",
        "license": "CC-BY-4.0",
        "purpose": "Incoming EN->TR Translation",
    },
    "mt-tr-fr": {
        "repo_id": "Helsinki-NLP/opus-mt-tr-fr",
        "destination": "models/mt/opus-mt-tr-fr",
        "license": "CC-BY-4.0",
        "purpose": "Outgoing TR->FR Translation",
    },
    "mt-nllb-200": {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "destination": "models/mt/nllb-200-distilled-600M",
        "license": "CC-BY-NC-4.0",
        "purpose": "High-Quality Multilingual MT Benchmark (TR <-> EN & FR)",
    },
    "xtts": {
        "repo_id": "coqui/XTTS-v2",
        "destination": "models/tts/xtts-v2",
        "license": "CPML (Personal / Non-commercial)",
        "purpose": "Cross-language Voice Cloning TTS",
    },
}


def download_model(key: str, info: dict, revision: str | None = None):
    print(f"\n--- Downloading: {key} ---")
    print(f"Repo ID:     {info['repo_id']}")
    print(f"Destination: {info['destination']}")
    print(f"License:     {info['license']}")
    print(f"Purpose:     {info['purpose']}")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    # Resolve metadata before any weights transfer. Every download uses a commit,
    # never a moving branch, and records the license from that revision.
    metadata = HfApi().model_info(info["repo_id"], revision=revision or info.get("revision") or "main")
    commit = metadata.sha
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Model hub did not return an immutable commit revision.")
    card = metadata.card_data or {}
    license_id = card.get("license") or info["license"]
    print(f"Revision:    {commit}")
    print(f"Model license at revision: {license_id}")
    root = Path(__file__).resolve().parents[1]
    dest_path = root / info["destination"]
    dest_path.mkdir(parents=True, exist_ok=True)

    print("Starting download...")
    snapshot_download(
        repo_id=info["repo_id"],
        revision=commit,
        local_dir=str(dest_path.resolve()),
    )
    inventory = []
    for sibling in metadata.siblings or []:
        local_file = dest_path / sibling.rfilename
        if local_file.is_file():
            with local_file.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            inventory.append({"path": sibling.rfilename, "bytes": local_file.stat().st_size, "sha256": checksum})
    manifest = {"repo_id": info["repo_id"], "revision": commit, "license": license_id,
                "destination": str(dest_path.resolve()), "purpose": info["purpose"], "files": inventory}
    (dest_path / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Successfully downloaded to: {dest_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Download pinned models for Teams Translator")
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()) + ["all"],
        help="Model to download (or 'all')",
    )
    parser.add_argument(
        "--convert-ct2",
        action="store_true",
        help="Automatically convert downloaded MT models to CTranslate2 INT8 format",
    )
    args = parser.parse_args()

    if args.model == "all":
        for k, v in MODELS.items():
            download_model(k, v)
    else:
        download_model(args.model, MODELS[args.model])

    if args.convert_ct2:
        print("\n--- Converting MT models to CTranslate2 INT8 ---")
        from convert_models_ct2 import convert_model
        mt_root = Path("models/mt")
        if mt_root.exists():
            for d in sorted(mt_root.iterdir()):
                if d.is_dir() and not d.name.endswith("-ct2"):
                    convert_model(d, quantization="int8")


if __name__ == "__main__":
    main()

