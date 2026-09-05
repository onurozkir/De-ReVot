"""Benchmark B3: CTranslate2 INT8 MT vs Transformers, Glossary, and Context Priming."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_translator.translation.ctranslate_backend import CTranslate2MTAdapter

print("=" * 65)
print(" BENCHMARK B3: CTRANSLATE2 INT8 MT (OPUS / NLLB) TR <-> EN")
print("=" * 65)

tr_en_path = "models/mt/opus-mt-tc-big-tr-en"
en_tr_path = "models/mt/opus-mt-tc-big-en-tr"

if not Path(tr_en_path).exists() or not Path(en_tr_path).exists():
    print("MT models not found. Run 'python scripts/download_models.py mt-tr-en' and 'mt-en-tr'.")
    sys.exit(1)

mt = CTranslate2MTAdapter(beam_size=2)
mt.initialize(
    tr_en_model_path=tr_en_path,
    en_tr_model_path=en_tr_path,
    device="cpu",
    compute_type="int8",
)
mt.warmup()

print(f"\nActive Backend: {mt.backend_type.upper()} ({mt.model_family.upper()}) | Device: {mt.device} | Compute: {mt.compute_type}")

sentences_tr = [
    "Merhaba, bugünkü toplantımıza katıldığınız için teşekkür ederim.",
    "Proje mimarisi hakkında ne düşünüyorsunuz?",
    "Ses gecikmesi bir saniyenin altında kalmalı.",
    "Yeni özellikleri ana branch'e ekledik ve test ettik.",
    "Bunu bir pull request olarak açıp incelemeye gönderdim.",
]

print("\n--- 1. Testing TR -> EN Translation (Committed, Beam=2) ---")
latencies = []
for s in sentences_tr:
    t0 = time.monotonic_ns()
    res = mt.translate(s, "tr", "en", is_partial=False)
    t1 = time.monotonic_ns()
    dur_ms = (t1 - t0) / 1e6
    latencies.append(dur_ms)
    print(f"TR:  {s}")
    print(f"EN:  {res} [{dur_ms:.2f} ms]\n")

latencies.sort()
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
print(f"Latency Summary: P50 = {p50:.2f} ms | P95 = {p95:.2f} ms")

print("\n--- 2. Testing Domain Glossary Injection ---")
sample_glossary = {
    "pull request": "pull request",
    "branch'e": "to branch",
    "Ses gecikmesi": "Audio latency",
    "Sound delay": "Audio latency",
}
glossary_sentence = "Ses gecikmesi bir saniyenin altında kalmalı."
raw_res = mt.translate(glossary_sentence, "tr", "en")
glossary_res = mt.translate(glossary_sentence, "tr", "en", glossary=sample_glossary)
print(f"Original:        {glossary_sentence}")
print(f"Without glossary: {raw_res}")
print(f"With glossary:    {glossary_res}")

print("\n--- 3. Testing Context Priming for Turkish Pro-Drop ---")
context = "Bugün yeni bir özellik yazdım."
prodrop_sentence = "Test ettim ve onayladım."
no_ctx = mt.translate(prodrop_sentence, "tr", "en")
with_ctx = mt.translate(prodrop_sentence, "tr", "en", context=context)
print(f"Context:      {context}")
print(f"Target:       {prodrop_sentence}")
print(f"No context:   {no_ctx}")
print(f"With context: {with_ctx}")

mt.shutdown()
print("\n" + "=" * 65)
