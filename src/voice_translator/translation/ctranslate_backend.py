"""Translation Backend supporting CTranslate2 and HuggingFace MarianMT/NLLB."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

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

FLORES_LANG_MAP: Dict[str, str] = {
    "tr": "tur_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ja": "jpn_Jpan",
    "uk": "ukr_Cyrl",
}


def _resolve_model_dir(path_str: str) -> Path:
    """Resolve model path, preferring -ct2 converted directory if present."""
    p = Path(path_str)
    if (p / "model.bin").exists():
        return p
    ct2_p = Path(f"{path_str}-ct2")
    if (ct2_p / "model.bin").exists():
        return ct2_p
    return p


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


class CTranslate2MTAdapter(MTAdapter):
    """MT Adapter supporting CTranslate2 INT8 or Hugging Face MarianMT/NLLB."""

    def __init__(self, beam_size: int = 2):
        self.tr_en_translator: Optional[Any] = None
        self.en_tr_translator: Optional[Any] = None
        self.tr_fr_translator: Optional[Any] = None
        self.unified_translator: Optional[Any] = None  # For multilingual models like NLLB
        self.tr_en_tokenizer: Optional[Any] = None
        self.en_tr_tokenizer: Optional[Any] = None
        self.tr_fr_tokenizer: Optional[Any] = None
        self.unified_tokenizer: Optional[Any] = None
        self.backend_type: str = "transformers"  # "ctranslate2" or "transformers"
        self.model_family: str = "opus"  # "opus" or "nllb"
        self.device = "cpu"
        self.compute_type = "int8"
        self.beam_size = max(1, int(beam_size))
        self._is_warm = False

    def initialize(
        self,
        tr_en_model_path: str,
        en_tr_model_path: str,
        tr_fr_model_path: Optional[str] = "models/mt/opus-mt-tr-fr",
        nllb_model_path: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.device = device
        self.compute_type = compute_type

        # Check if NLLB multilingual model is requested or available
        if nllb_model_path:
            p_nllb = _resolve_model_dir(nllb_model_path)
            if p_nllb.exists():
                self._initialize_nllb(p_nllb, device, compute_type)
                return

        # Otherwise initialize dedicated bilingual OPUS-MT models
        self._initialize_opus(tr_en_model_path, en_tr_model_path, tr_fr_model_path, device, compute_type)

    def _initialize_nllb(self, p_nllb: Path, device: str, compute_type: str):
        self.model_family = "nllb"
        has_ct2 = (p_nllb / "model.bin").exists()

        if has_ct2 and ctranslate2 is not None and AutoTokenizer is not None:
            self.backend_type = "ctranslate2"
            logger.info("Loading CTranslate2 NLLB-200 model from '%s' (%s, %s)...", p_nllb, device, compute_type)
            self.unified_translator = ctranslate2.Translator(
                str(p_nllb.resolve()),
                device=device,
                compute_type=compute_type,
            )
            self.unified_tokenizer = _load_tokenizer(p_nllb)
        else:
            self.backend_type = "transformers"
            if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
                raise RuntimeError("transformers is required for HuggingFace MT models.")
            logger.info("Loading HuggingFace NLLB-200 model from '%s'...", p_nllb)
            self.unified_tokenizer = _load_tokenizer(p_nllb)
            self.unified_translator = AutoModelForSeq2SeqLM.from_pretrained(str(p_nllb.resolve()), local_files_only=True)
            if device == "cuda" and torch is not None and torch.cuda.is_available():
                self.unified_translator.to("cuda")

        logger.info("NLLB-200 MT loaded successfully (backend: %s).", self.backend_type)

    def _initialize_opus(
        self,
        tr_en_model_path: str,
        en_tr_model_path: str,
        tr_fr_model_path: Optional[str],
        device: str,
        compute_type: str,
    ):
        self.model_family = "opus"
        p_tr_en = _resolve_model_dir(tr_en_model_path)
        p_en_tr = _resolve_model_dir(en_tr_model_path)

        if not p_tr_en.exists():
            raise ModelNotFoundError(
                f"TR->EN MT model path '{tr_en_model_path}' not found. "
                "Models must be downloaded manually."
            )
        if not p_en_tr.exists():
            raise ModelNotFoundError(
                f"EN->TR MT model path '{en_tr_model_path}' not found. "
                "Models must be downloaded manually."
            )

        has_ct2 = (
            (p_tr_en / "model.bin").exists()
            and ((p_tr_en / "shared_vocabulary.json").exists() or (p_tr_en / "source_vocabulary.json").exists())
        )

        if has_ct2 and ctranslate2 is not None and AutoTokenizer is not None:
            self.backend_type = "ctranslate2"
            logger.info("Loading CTranslate2 TR->EN model from '%s' (%s, %s)...", p_tr_en, device, compute_type)
            self.tr_en_translator = ctranslate2.Translator(
                str(p_tr_en.resolve()),
                device=device,
                compute_type=compute_type,
            )
            self.tr_en_tokenizer = _load_tokenizer(p_tr_en)

            logger.info("Loading CTranslate2 EN->TR model from '%s' (%s, %s)...", p_en_tr, device, compute_type)
            self.en_tr_translator = ctranslate2.Translator(
                str(p_en_tr.resolve()),
                device=device,
                compute_type=compute_type,
            )
            self.en_tr_tokenizer = _load_tokenizer(p_en_tr)
        else:
            self.backend_type = "transformers"
            if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
                raise RuntimeError("transformers is required for HuggingFace MT models.")

            logger.info("Loading HuggingFace MarianMT TR->EN from '%s'...", p_tr_en)
            self.tr_en_tokenizer = _load_tokenizer(p_tr_en)
            self.tr_en_translator = AutoModelForSeq2SeqLM.from_pretrained(str(p_tr_en.resolve()), local_files_only=True)

            logger.info("Loading HuggingFace MarianMT EN->TR from '%s'...", p_en_tr)
            self.en_tr_tokenizer = _load_tokenizer(p_en_tr)
            self.en_tr_translator = AutoModelForSeq2SeqLM.from_pretrained(str(p_en_tr.resolve()), local_files_only=True)

            if device == "cuda" and torch is not None and torch.cuda.is_available():
                self.tr_en_translator.to("cuda")
                self.en_tr_translator.to("cuda")

        # Optional TR->FR model loading
        if tr_fr_model_path:
            p_tr_fr = _resolve_model_dir(tr_fr_model_path)
            if p_tr_fr.exists():
                logger.info("Loading TR->FR translation model from '%s'...", p_tr_fr)
                has_ct2_fr = (p_tr_fr / "model.bin").exists() and (
                    (p_tr_fr / "shared_vocabulary.json").exists() or (p_tr_fr / "source_vocabulary.json").exists()
                )
                if has_ct2_fr and ctranslate2 is not None and AutoTokenizer is not None:
                    self.tr_fr_translator = ctranslate2.Translator(
                        str(p_tr_fr.resolve()),
                        device=device,
                        compute_type=compute_type,
                    )
                    self.tr_fr_tokenizer = _load_tokenizer(p_tr_fr)
                elif AutoModelForSeq2SeqLM is not None and AutoTokenizer is not None:
                    self.tr_fr_tokenizer = _load_tokenizer(p_tr_fr)
                    self.tr_fr_translator = AutoModelForSeq2SeqLM.from_pretrained(str(p_tr_fr.resolve()), local_files_only=True)
                    if device == "cuda" and torch is not None and torch.cuda.is_available():
                        self.tr_fr_translator.to("cuda")
                logger.info("TR->FR translation model loaded successfully.")
            else:
                logger.info(
                    "TR->FR model '%s' not found. Run 'python scripts/download_models.py mt-tr-fr' to enable French translation.",
                    tr_fr_model_path,
                )

        logger.info("MT models loaded successfully (family: %s, backend: %s).", self.model_family, self.backend_type)

    def warmup(self):
        if self.model_family == "nllb" and self.unified_translator is None:
            raise WarmupError("NLLB model not initialized before warmup.")
        if self.model_family == "opus" and (self.tr_en_translator is None or self.en_tr_translator is None):
            raise WarmupError("OPUS models not initialized before warmup.")
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

        # Apply pre-translation glossary terms if available
        working_text = _apply_glossary(text, glossary)

        try:
            if self.model_family == "nllb":
                result = self._translate_nllb(working_text, source_lang, target_lang, is_partial, context)
            else:
                result = self._translate_opus(working_text, source_lang, target_lang, is_partial, context)

            # Apply post-translation glossary alignment
            if glossary:
                result = _apply_glossary(result, glossary)
            return result
        except Exception as e:
            logger.error("MT translation error (%s->%s): %s", source_lang, target_lang, e)
            return text

    def _translate_opus(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool,
        context: Optional[str] = None,
    ) -> str:
        if source_lang.startswith("tr") and target_lang.startswith("en"):
            translator = self.tr_en_translator
            tokenizer = self.tr_en_tokenizer
        elif source_lang.startswith("tr") and target_lang.startswith("fr"):
            translator = self.tr_fr_translator
            tokenizer = self.tr_fr_tokenizer
            if translator is None:
                logger.warning("TR->FR model not loaded yet. Falling back to original text.")
                return text
        elif source_lang.startswith("en") and target_lang.startswith("tr"):
            translator = self.en_tr_translator
            tokenizer = self.en_tr_tokenizer
        else:
            logger.warning("Unsupported OPUS language pair: %s->%s", source_lang, target_lang)
            return text

        if translator is None or tokenizer is None:
            return text

        beam_size = 1 if is_partial else self.beam_size

        if self.backend_type == "ctranslate2":
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
        else:
            # HuggingFace MarianMT inference
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            if self.device == "cuda" and torch is not None and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            if torch is not None:
                with torch.inference_mode():
                    translated_tokens = translator.generate(
                        **inputs,
                        max_length=128,
                        num_beams=beam_size,
                    )
            else:
                translated_tokens = translator.generate(
                    **inputs,
                    max_length=128,
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

        src_code = FLORES_LANG_MAP.get(source_lang, "tur_Latn")
        tgt_code = FLORES_LANG_MAP.get(target_lang, "eng_Latn")
        beam_size = 1 if is_partial else self.beam_size

        # If discourse context exists, prime input
        input_text = f"{context} {text}" if context and not is_partial else text

        if self.backend_type == "ctranslate2":
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
                max_decoding_length=128,
            )
            out_tokens = results[0].hypotheses[0]
            out_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True)
            # If context was primed, take only the translation of the second sentence if applicable
            if context and not is_partial and "." in out_text:
                parts = [p.strip() for p in out_text.split(".") if p.strip()]
                if len(parts) >= 2:
                    out_text = parts[-1]
            return out_text.strip()
        else:
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
                        max_length=128,
                        num_beams=beam_size,
                    )
            else:
                translated_tokens = translator.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=128,
                    num_beams=beam_size,
                )

            out_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            return out_text[0].strip() if out_text else text

    def shutdown(self):
        self.tr_en_translator = None
        self.en_tr_translator = None
        self.tr_fr_translator = None
        self.unified_translator = None
        self.tr_en_tokenizer = None
        self.en_tr_tokenizer = None
        self.tr_fr_tokenizer = None
        self.unified_tokenizer = None
        self._is_warm = False
