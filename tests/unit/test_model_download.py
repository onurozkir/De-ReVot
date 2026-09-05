import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def load_downloader():
    spec = importlib.util.spec_from_file_location("model_downloader", Path(__file__).parents[2] / "scripts/download_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_v3_download_is_pinned_and_records_provenance_without_network(monkeypatch, tmp_path):
    downloader = load_downloader()
    info = {**downloader.MODELS["whisper-large-v3"], "destination": str(tmp_path / "model")}
    received = {}

    class FakeApi:
        def model_info(self, repo_id, revision):
            assert repo_id == "Systran/faster-whisper-large-v3"
            assert revision == info["revision"]
            return SimpleNamespace(sha=revision, card_data={"license": "mit"}, siblings=[SimpleNamespace(rfilename="model.bin")])

    def download(**kwargs):
        received.update(kwargs)
        Path(kwargs["local_dir"], "model.bin").write_bytes(b"model fixture")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi, snapshot_download=download))
    downloader.download_model("whisper-large-v3", info)
    assert received["revision"] == info["revision"]
    manifest = json.loads(Path(info["destination"], "download-manifest.json").read_text())
    assert manifest["repo_id"] == info["repo_id"]
    assert manifest["license"] == "mit"
    assert len(manifest["files"][0]["sha256"]) == 64
