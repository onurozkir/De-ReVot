"""Coqui XTTS-v2 Voice Cloning TTS Adapter."""

from __future__ import annotations

import logging
import inspect
import time
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from types import MethodType
import numpy as np

from voice_translator.core.errors import ModelNotFoundError, WarmupError
from voice_translator.tts.base import TTSAdapter, VoiceProfile
from voice_translator.tts.conditioning import VoiceProfileManager

logger = logging.getLogger(__name__)


def _stream_cache_compat(self, sequence_length, device, model_kwargs):
    """Coqui 0.27.5 calls a helper removed in Transformers 5.3+.

    No legacy cache_position kwarg is needed. Input slicing is restored separately
    by _configure_streaming_compat. Bound only to this XTTS inference instance.
    """
    return model_kwargs


def _configure_streaming_compat(inference_model):
    """Bridge Coqui's legacy streaming loop to the installed generation API."""
    if not hasattr(inference_model, "_get_initial_cache_position"):
        inference_model._get_initial_cache_position = MethodType(_stream_cache_compat, inference_model)
    prepare = inference_model.prepare_inputs_for_generation
    if "next_sequence_length" not in inspect.signature(prepare).parameters:
        return

    @wraps(prepare)
    def prepare_stream_inputs(input_ids, **kwargs):
        cache = kwargs.get("past_key_values")
        # Coqui does not pass the new slicing argument. Refeeding the entire
        # prefix into a populated KV cache grows memory and corrupts decoding.
        # Preserve explicit lengths from native generate() and uncached inputs.
        if ("next_sequence_length" not in kwargs and kwargs.get("use_cache", True)
                and cache is not None and cache.get_seq_length() > 0):
            kwargs["next_sequence_length"] = 1
        return prepare(input_ids, **kwargs)

    inference_model.prepare_inputs_for_generation = prepare_stream_inputs

_tts_import_error: Optional[Exception] = None
try:
    import torch
    import transformers.utils.import_utils
    import transformers.pytorch_utils

    # Compatibility shim for PyTorch 2.10+ and Transformers 5.x with Coqui TTS
    transformers.utils.import_utils.is_torchcodec_available = lambda: True
    if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
        transformers.pytorch_utils.isin_mps_friendly = torch.isin

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import TTS.tts.models.xtts as xtts_model_module
    import torchaudio
    import soundfile as sf

    def _soundfile_load(filepath, sampling_rate):
        """Load XTTS conditioning audio without replacing torchaudio.load globally."""
        data, source_rate = sf.read(filepath, dtype="float32")
        audio = torch.from_numpy(data)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if source_rate != sampling_rate:
            audio = torchaudio.functional.resample(audio, source_rate, sampling_rate)
        return audio.clamp_(-1, 1)

    _xtts_loader_lock = threading.RLock()

    @contextmanager
    def _scoped_xtts_audio_loader():
        """Apply the Coqui loader workaround only while preparing conditioning."""
        with _xtts_loader_lock:
            original_loader = xtts_model_module.load_audio
            xtts_model_module.load_audio = _soundfile_load
            try:
                yield
            finally:
                xtts_model_module.load_audio = original_loader
except Exception as _tts_err:
    _tts_import_error = _tts_err
    if "torch" not in globals():
        torch = None  # type: ignore
    XttsConfig = None  # type: ignore
    Xtts = None  # type: ignore
    xtts_model_module = None  # type: ignore


class XTTSv2Adapter(TTSAdapter):
    """XTTS-v2 Voice Cloning Adapter with cross-language synthesis and conditioning caching."""

    synthesis_mode = "true_streaming"

    def __init__(
        self,
        temperature: float = 0.65,
        speed: float = 1.0,
        top_p: float = 0.85,
        repetition_penalty: float = 2.0,
        peak_normalization: bool = True,
        stream_chunk_size: int = 8,
    ):
        self.model: Optional[Any] = None
        self.config: Optional[Any] = None
        self.model_path: str = ""
        self.device: str = "cuda"
        self.sample_rate: int = 24000
        self.temperature = float(temperature)
        self.speed = float(speed)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.peak_normalization = bool(peak_normalization)
        if not 2 <= stream_chunk_size <= 40:
            raise ValueError("XTTS stream_chunk_size must be between 2 and 40 acoustic tokens.")
        self.stream_chunk_size = int(stream_chunk_size)
        if self.temperature <= 0:
            raise ValueError("XTTS temperature must be greater than zero.")
        if self.speed <= 0:
            raise ValueError("XTTS speed must be greater than zero.")
        if self.top_p <= 0 or self.top_p > 1.0:
            raise ValueError("XTTS top_p must be between 0 and 1.0.")
        if self.repetition_penalty <= 0:
            raise ValueError("XTTS repetition_penalty must be greater than zero.")
        self._latents_cache: Dict[str, Tuple[Any, Any]] = {}
        self._prepared_profiles: dict[tuple, str] = {}
        self._model_identity = "uninitialized"
        self._is_warm = False

    def initialize(self, model_path: str, device: str = "cuda", sample_rate: int = 24000):
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self._latents_cache.clear()
        self._prepared_profiles.clear()

        p = Path(model_path)
        if not p.exists():
            raise ModelNotFoundError(
                f"XTTS-v2 model not found at '{model_path}'. "
                "Download model weights manually into models/tts/xtts-v2."
            )

        if Xtts is None or torch is None:
            detail = (
                f"{type(_tts_import_error).__name__}: {_tts_import_error}"
                if _tts_import_error is not None else "unknown import failure"
            )
            raise RuntimeError(
                "XTTS runtime unavailable. Required package: coqui-tts==0.27.5. "
                f"Import failure: {detail}"
            ) from _tts_import_error

        logger.info(f"Loading XTTS-v2 model from '{model_path}' on {device}...")
        self.config = XttsConfig()
        self.config.load_json(str((p / "config.json").resolve()))
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir=str(p.resolve()),
            eval=True,
            use_deepspeed=False,
        )
        if device == "cuda" and torch.cuda.is_available():
            self.model.cuda()
        inference_model = self.model.gpt.gpt_inference
        _configure_streaming_compat(inference_model)
        # Local checkpoint metadata plus config/provenance content identifies the
        # offline model without rehashing gigabytes on every profile switch.
        identity = {"path": str(p.resolve()), "runtime": "coqui-tts==0.27.5"}
        for name in ("model.pth", "config.json", "vocab.json", "download-manifest.json"):
            file = p / name
            if file.is_file():
                stat = file.stat()
                identity[name] = [stat.st_size, stat.st_mtime_ns]
                if file.suffix == ".json":
                    identity[name].append(hashlib.sha256(file.read_bytes()).hexdigest())
        self._model_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        logger.info("XTTS-v2 model loaded successfully.")

    def warmup(self):
        self._is_warm = False
        if self.model is None:
            raise WarmupError("XTTS model not initialized before warmup.")
        logger.info("Warming up XTTS-v2 model...")
        started = time.perf_counter()
        try:
            # Create dummy latents with correct dimensions
            gpt_cond_latent = torch.zeros((1, 30, 1024), device=self.device)
            speaker_embedding = torch.zeros((1, 512, 1), device=self.device)
            
            # Exercise the same incremental decoder/vocoder path before Ready.
            stream = self.model.inference_stream(
                text="Test warmup.",
                language="en",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.temperature,
                speed=self.speed,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                enable_text_splitting=False,
                stream_chunk_size=self.stream_chunk_size,
                overlap_wav_len=1024,
                # Zero conditioning need not emit EOS promptly. Exercise the
                # decoder and vocoder without generating a full dummy utterance.
                max_new_tokens=2 * self.stream_chunk_size,
            )
            chunks = 0
            try:
                for chunk in stream:
                    if torch.is_tensor(chunk):
                        chunk = chunk.detach().float().cpu().numpy()
                    pcm = np.asarray(chunk)
                    if not np.isfinite(pcm).all():
                        raise ValueError("XTTS warmup produced non-finite PCM")
                    if pcm.size:
                        chunks += 1
            finally:
                stream.close()
            if not chunks:
                raise ValueError("XTTS warmup produced no PCM")
            self._is_warm = True
            logger.info("XTTS-v2 warmup completed in %.2fs (%d chunks).", time.perf_counter() - started, chunks)
        except Exception as e:
            raise WarmupError(f"XTTS warmup failed: {e}") from e

    def prepare_voice_profile(self, profile: VoiceProfile):
        """Compute and cache speaker latents for voice profile."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before preparing voice profiles.")

        signature = (profile.id, tuple(profile.all_reference_paths))
        self._prepared_profiles.pop(signature, None)
        valid_paths = profile.all_reference_paths
        VoiceProfileManager.validate_reference_audio(valid_paths)
        audio_hash = VoiceProfileManager.compute_audio_hash(valid_paths)
        identity = [audio_hash, self._model_identity, "gpt_cond_len=30", "max_ref_length=60",
                    "sound_norm_refs=false", "load_sr=22050", "conditioning-v2"]
        cache_key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()

        if cache_key in self._latents_cache:
            self._prepared_profiles[signature] = cache_key
            return

        # Check disk cache
        cache_dir = Path(profile.conditioning_cache_path or "voices/cache")
        cache_file = cache_dir / f"latents_{cache_key}.pt"

        if cache_file.exists():
            try:
                latents = torch.load(str(cache_file), map_location=self.device, weights_only=True)
                if (not isinstance(latents, (tuple, list)) or len(latents) != 2
                        or not all(torch.is_tensor(t) and t.numel() and torch.isfinite(t).all() for t in latents)):
                    raise ValueError("Invalid conditioning tensors")
                self._latents_cache[cache_key] = latents
                self._prepared_profiles[signature] = cache_key
                logger.info(f"Loaded voice conditioning cache for profile '{profile.display_name}' from disk.")
                return
            except Exception as e:
                logger.warning(f"Could not load cache file: {e}")

        logger.info(
            f"Computing speaker conditioning latents for profile '{profile.display_name}' "
            f"from {len(valid_paths)} audio file(s)..."
        )
        with _scoped_xtts_audio_loader():
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=valid_paths,
                gpt_cond_len=30,
                max_ref_length=60,
                sound_norm_refs=False,
            )

        latents = (gpt_cond_latent, speaker_embedding)
        self._latents_cache[cache_key] = latents
        self._prepared_profiles[signature] = cache_key

        # Save to disk
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(latents, str(cache_file))
            logger.info(f"Saved voice conditioning cache to '{cache_file}'.")
        except Exception as e:
            logger.warning(f"Could not save latents cache to disk: {e}")

    def synthesize_committed(
        self,
        text: str,
        profile: VoiceProfile,
        target_language: str = "en",
    ) -> Iterator[np.ndarray]:
        if self.model is None or not text.strip():
            return

        signature = (profile.id, tuple(profile.all_reference_paths))
        cache_key = self._prepared_profiles.get(signature)
        if cache_key is None:
            self.prepare_voice_profile(profile)
            cache_key = self._prepared_profiles.get(signature)

        if cache_key in self._latents_cache:
            gpt_cond_latent, speaker_embedding = self._latents_cache[cache_key]
        else:
            raise RuntimeError(
                f"Voice conditioning is unavailable for profile '{profile.display_name}'."
            )

        stream = None
        try:
            stream = self.model.inference_stream(
                text=text,
                language=target_language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.temperature,
                speed=self.speed,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                enable_text_splitting=False,
                stream_chunk_size=self.stream_chunk_size,
                overlap_wav_len=1024,
            )
            for chunk in stream:
                if torch is not None and torch.is_tensor(chunk):
                    chunk = chunk.detach().float().cpu().numpy()
                pcm_data = np.asarray(chunk, dtype=np.float32).reshape(-1).copy()
                if not pcm_data.size:
                    continue
                if not np.isfinite(pcm_data).all():
                    raise RuntimeError("XTTS produced non-finite PCM")
                if self.peak_normalization:
                    # A full-utterance peak needs future samples. Keep a fixed
                    # -1 dBFS ceiling without chunk-wise gain jumps or buffering.
                    np.clip(pcm_data, -0.89125, 0.89125, out=pcm_data)
                yield pcm_data
        except Exception as e:
            raise RuntimeError(f"XTTS streaming synthesis failed: {e}") from e
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()

    def shutdown(self):
        self.model = None
        self._latents_cache.clear()
        self._prepared_profiles.clear()
        self._is_warm = False
