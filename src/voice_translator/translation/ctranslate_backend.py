"""Translation Backend supporting CTranslate2 and HuggingFace MarianMT/NLLB."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from voice_translator.config.languages import NLLB_FLORES, resolve_model_dir
from voice_translator.core.errors import ModelNotFoundError, WarmupError
from voice_translator.translation.base import MTAdapter

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None  # type: ignore

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


def _apply_glossary(text: str, glossary: Optional[Dict[str, str]]) -> str:
    """Apply dictionary replacements with word-boundary awareness."""
    if not glossary or not text:
        return text
    result = text
    for term, replacement in glossary.items():
        if not term.strip():
            continue
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        result = pattern.sub(replacement, result)
    return result


def _load_tokenizer(path: Path):
    """Load tokenizer, falling back to MarianTokenizer if auto-detection fails on ct2 configs."""
    try:
        return AutoTokenizer.from_pretrained(str(path.resolve()), local_files_only=True)
    except Exception:
        from transformers import MarianTokenizer
        return MarianTokenizer.from_pretrained(str(path.resolve()), local_files_only=True)


def _load_ct2_translator(path: Path, device: str, compute_type: str):
    if ctranslate2 is None:
        raise RuntimeError("ctranslate2 is required for this CTranslate2 MT model. Run uv sync.")
    return ctranslate2.Translator(str(path.resolve()), device=device, compute_type=compute_type)


class CTranslate2MTAdapter(MTAdapter):
    """MT Adapter routing language pairs to dedicated OPUS models with NLLB fallback."""

    def __init__(self, beam_size: int = 2):
        self.opus_pairs: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
        self.opus_backends: Dict[Tuple[str, str], str] = {}
        self.unified_translator: Optional[Any] = None  # NLLB-200 multilingual model
        self.unified_tokenizer: Optional[Any] = None
        self.unified_backend: str = "transformers"
        self.backend_type: str = "transformers"  # dominant backend for telemetry
        self.model_family: str = "opus"  # "opus", "nllb", or "hybrid"
        self.device = "cpu"
        self.compute_type = "int8"
        self.beam_size = max(1, int(beam_size))
        self._is_warm = False

    def initialize(
        self,
        *,
        pair_paths: Optional[Dict[str, str]] = None,
        nllb_model_path: Optional[str] = None,
        languages: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.device = device
        self.compute_type = compute_type
        loaded_opus = False

        for pair_key, path_str in (pair_paths or {}).items():
            parts = pair_key.split("-", 1)
            if len(parts) != 2:
                logger.warning("Skipping malformed MT pair key '%s'", pair_key)
                continue
            pair = (parts[0].strip().lower(), parts[1].strip().lower())
            p = resolve_model_dir(path_str)
            if not p.exists():
                continue
            self._load_opus_pair(pair, p)
            loaded_opus = True

        if nllb_model_path:
            p_nllb = resolve_model_dir(nllb_model_path)
            if p_nllb.exists():
                self._load_nllb(p_nllb)

        if not loaded_opus and self.unified_translator is None:
            raise ModelNotFoundError(
                "No MT model directories found. Download at least one bilingual "
                "OPUS pair or NLLB-200: uv run python scripts/download_models.py --lang tr "
                "or mt-nllb-200."
            )
        self.model_family = "nllb" if (not loaded_opus and self.unified_translator is not None) else (
            "hybrid" if (loaded_opus and self.unified_translator is not None) else "opus"
        )
        logger.info(
            "MT models loaded (family: %s, opus pairs: %s, nllb: %s).",
            self.model_family, sorted(f"{s}-{t}" for s, t in self.opus_pairs),
            "yes" if self.unified_translator is not None else "no",
        )

    def _load_opus_pair(self, pair: Tuple[str, str], p: Path) -> None:
        has_ct2 = (
            (p / "model.bin").exists()
            and ((p / "shared_vocabulary.json").exists() or (p / "source_vocabulary.json").exists())
        )
        if has_ct2 and ctranslate2 is not None and AutoTokenizer is not None:
            logger.info("Loading CTranslate2 OPUS %s->%s from '%s' (%s, %s)...",
                        pair[0], pair[1], p, self.device, self.compute_type)
            self.opus_pairs[pair] = (_load_ct2_translator(p, self.device, self.compute_type), _load_tokenizer(p))
            self.opus_backends[pair] = "ctranslate2"
        elif AutoModelForSeq2SeqLM is not None and AutoTokenizer is not None:
            logger.info("Loading HuggingFace MarianMT %s->%s from '%s'...", pair[0], pair[1], p)
            tokenizer = _load_tokenizer(p)
            translator = AutoModelForSeq2SeqLM.from_pretrained(str(p.resolve()), local_files_only=True)
            if self.device == "cuda" and torch is not None and torch.cuda.is_available():
                translator.to("cuda")
            self.opus_pairs[pair] = (translator, tokenizer)
            self.opus_backends[pair] = "transformers"
        else:
            logger.warning("MT model at '%s' cannot be loaded (missing runtime).", p)

    def _load_nllb(self, p_nllb: Path) -> None:
        has_ct2 = (p_nllb / "model.bin").exists()
        if has_ct2 and ctranslate2 is not None and AutoTokenizer is not None:
            self.unified_backend = "ctranslate2"
            logger.info("Loading CTranslate2 NLLB-200 from '%s' (%s, %s)...", p_nllb, self.device, self.compute_type)
            self.unified_translator = _load_ct2_translator(p_nllb, self.device, self.compute_type)
            self.unified_tokenizer = _load_tokenizer(p_nllb)
        elif AutoModelForSeq2SeqLM is not None and AutoTokenizer is not None:
            self.unified_backend = "transformers"
            logger.info("Loading HuggingFace NLLB-200 from '%s'...", p_nllb)
            self.unified_tokenizer = _load_tokenizer(p_nllb)
            self.unified_translator = AutoModelForSeq2SeqLM.from_pretrained(str(p_nllb.resolve()), local_files_only=True)
            if self.device == "cuda" and torch is not None and torch.cuda.is_available():
                self.unified_translator.to("cuda")
        else:
            logger.warning("NLLB model at '%s' cannot be loaded (missing runtime).", p_nllb)
        if self.unified_translator is not None:
            self.backend_type = self.unified_backend

    def warmup(self):
        if not self.opus_pairs and self.unified_translator is None:
            raise WarmupError("No MT models initialized before warmup.")
        try:
            logger.info("Warming up MT models...")
            _ = self.translate("Merhaba dünya", "tr", "en")
            _ = self.translate("Hello world", "en", "tr")
            self._is_warm = True
            logger.info("MT models warmup completed.")
        except Exception as e:
            raise WarmupError(f"MT warmup failed: {e}") from e

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool = False,
        context: Optional[str] = None,
        glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        text = text.strip()
        if not text:
            return ""

        source_lang = source_lang.lower().split("-")[0]
        target_lang = target_lang.lower().split("-")[0]

        working_text = _apply_glossary(text, glossary)

        try:
            pair = (source_lang, target_lang)
            if pair in self.opus_pairs:
                result = self._translate_opus(working_text, pair, is_partial)
            elif self.unified_translator is not None:
                result = self._translate_nllb(working_text, source_lang, target_lang, is_partial, context)
            else:
                logger.warning(
                    "No offline MT model for %s->%s. Download the pair or NLLB-200.",
                    source_lang, target_lang,
                )
                return text

            if glossary:
                result = _apply_glossary(result, glossary)
            return result
        except Exception as e:
            logger.error("MT translation error (%s->%s): %s", source_lang, target_lang, e)
            return text

    def _translate_opus(self, text: str, pair: Tuple[str, str], is_partial: bool) -> str:
        translator, tokenizer = self.opus_pairs[pair]
        backend = self.opus_backends[pair]
        beam_size = 1 if is_partial else self.beam_size

        if backend == "ctranslate2":
            if hasattr(tokenizer, "tokenize"):
                tokens = tokenizer.tokenize(text)
            elif hasattr(tokenizer, "encode") and hasattr(tokenizer, "convert_ids_to_tokens"):
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            else:
                tokens = text.split()

            if not tokens:
                return ""
            eos = getattr(tokenizer, "eos_token", "</s>") or "</s>"
            if tokens[-1] != eos:
                tokens.append(eos)

            try:
                results = translator.translate_batch([tokens], beam_size=beam_size)
            except TypeError:
                results = translator.translate_batch([tokens], beam_size)

            out_tokens = results[0].hypotheses[0]
            token_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            try:
                out_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            except TypeError:
                out_text = tokenizer.decode(token_ids)
            return out_text.strip()

        # HuggingFace MarianMT inference
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        if torch is not None:
            with torch.inference_mode():
                translated_tokens = translator.generate(
                    **inputs,
                    max_length=128 if is_partial else 512,
                    num_beams=beam_size,
                )
        else:
            translated_tokens = translator.generate(
                **inputs,
                max_length=128 if is_partial else 512,
                num_beams=beam_size,
            )

        out_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return out_text[0].strip() if out_text else text

    def _translate_nllb(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool,
        context: Optional[str] = None,
    ) -> str:
        translator = self.unified_translator
        tokenizer = self.unified_tokenizer
        if translator is None or tokenizer is None:
            return text

        src_code = NLLB_FLORES.get(source_lang, source_lang)
        tgt_code = NLLB_FLORES.get(target_lang, target_lang)
        beam_size = 1 if is_partial else self.beam_size

        # NLLB has no separate discourse-context channel or output alignment.
        # Prepending a previous turn and keeping the last sentence deletes real
        # sentences from this turn. Translate the complete current turn only.
        input_text = text

        if self.unified_backend == "ctranslate2":
            tokenizer.src_lang = src_code
            tokens = tokenizer.tokenize(input_text)
            if not tokens:
                return ""
            eos = getattr(tokenizer, "eos_token", "</s>") or "</s>"
            if tokens[-1] != eos:
                tokens.append(eos)

            results = translator.translate_batch(
                [tokens],
                target_prefix=[[tgt_code]],
                beam_size=beam_size,
                max_decoding_length=128 if is_partial else 512,
            )
            out_tokens = results[0].hypotheses[0]
            out_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True)
            return out_text.strip()

        tokenizer.src_lang = src_code
        inputs = tokenizer(input_text, return_tensors="pt")
        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        if torch is not None:
            with torch.inference_mode():
                translated_tokens = translator.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=128 if is_partial else 512,
                    num_beams=beam_size,
                )
        else:
            translated_tokens = translator.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128 if is_partial else 512,
                num_beams=beam_size,
            )

        out_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return out_text[0].strip() if out_text else text

    def shutdown(self):
        self.opus_pairs = {}
        self.opus_backends = {}
        self.unified_translator = None
        self.unified_tokenizer = None
        self._is_warm = False
