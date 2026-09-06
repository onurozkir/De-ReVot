# De-ReVot Voice Recording Guide

De-ReVot accepts multiple local WAV references for XTTS-v2. This is zero-shot
speaker conditioning, not checkpoint training or fine-tuning. Six varied clips
can provide broader voice evidence; better identity, pronunciation or emotion
is a listening/benchmark outcome, not a guarantee. Reference intonation does not
give the synthesizer explicit per-sentence emotion control.

## Record six clean clips

Use the same speaker, microphone, gain, distance and quiet room for every clip.
Keep the microphone roughly 15–20 cm away, slightly off-axis to limit plosives.
Avoid clipping, music, traffic, room echo and long leading/trailing silence. Use
a natural pace and consistent loudness; do not rush a script to meet a duration.
Aim for 8–10 seconds per clip, adjusting the text naturally if necessary.

Record unprocessed PCM WAV, preferably mono, 16-bit or 24-bit, at the device's
native rate (44.1 or 48 kHz is fine). Disable OBS filters, aggressive noise
suppression, gates and automatic gain processing for these reference recordings.
The application's live microphone noise reduction is a separate path and never
rewrites your reference recordings. XTTS downmixes/resamples internally.

**Note: You can record your voice by choosing any of the audio options below.**

| File | Purpose | Target |
|---|---|---|
| `voice_01.wav` | Neutral identity | 8–10 s |
| `voice_02.wav` | Confident, conversational identity | 8–10 s |
| `voice_03.wav` | Long sentence and natural rhythm | 8–10 s |
| `voice_04.wav` | Questions and pauses | 8–10 s |
| `voice_05.wav` | Technical terms and names | 8–10 s |
| `voice_06.wav` | phonemes and numbers | 9–10 s |

### voice_01.wav

TR
Bugün üzerinde çalıştığım projeyi tamamlamak için birkaç farklı yöntemi karşılaştıracağım. Önce mevcut sonuçlara bakıp daha sonra nasıl devam edeceğime karar vereceğim.

EN
Today I'll compare several different methods for completing the project I'm working on. First, I'll look at the current results and then decide how to proceed.

FR
Aujourd'hui, je vais comparer différentes méthodes pour mener à bien le projet sur lequel je travaille. Je commencerai par examiner les résultats actuels, puis je déciderai de la marche à suivre.


### voice_02.wav

TR
Aslında mesele düşündüğümüz kadar karmaşık değil. Sorunun nereden kaynaklandığını doğru şekilde belirlersek geri kalan kısmını çok daha kolay çözebiliriz.

EN
Actually, the issue isn't as complicated as we think. If we correctly identify the source of the problem, we can solve the rest much more easily.

FR
En réalité, le problème n'est pas aussi compliqué qu'il n'y paraît. Si nous identifions correctement la source du problème, nous pourrons résoudre le reste beaucoup plus facilement.

### voice_03.wav

TR
Sistemin düzgün ve kararlı çalışabilmesi için bütün bileşenlerin birbiriyle doğru şekilde iletişim kurması, verilerin zamanında işlenmesi ve sonuçların dikkatlice değerlendirilmesi gerekiyor.

EN
Actually, the issue isn't as complicated as we think. If we correctly identify the source of the problem, we can solve the rest much more easily.

FR
En réalité, le problème n'est pas aussi compliqué qu'il n'y paraît. Si nous identifions correctement la source du problème, nous pourrons résoudre le reste beaucoup plus facilement.

### voice_04.wav

TR
Tamam, bunu deneyebiliriz değil mi? Önce bir test yapalım. Sonuç doğru mu? Emin miyiz? Bir problem varsa ayarları değiştirip tekrar deneyebilir miyiz? Bu bizim için ne ifade ediyor?

EN
Okay, we can try this, right? Let's do a test first. Is the result correct? Are we sure? If there's a problem, can we change the settings and try again? What does this mean for us?

FR
D'accord, on peut essayer, non ? Faisons d'abord un test. Le résultat est-il correct ? On en est sûrs ? S'il y a un problème, on peut modifier les paramètres et réessayer ? Qu'est-ce que cela signifie pour nous ?

### voice_05.wav

TR
OpenAI, NVIDIA, Microsoft Teams, Python üzerinde çalışan sistemde GPU kullanımı, inference latency, model cache ve speaker embedding değerlerini birlikte kontrol edeceğiz.

EN
In a system running OpenAI, NVIDIA, Microsoft Teams, and Python, we will simultaneously monitor GPU usage, inference latency, model cache, and speaker embedding values.

FR
Dans un système exécutant OpenAI, NVIDIA, Microsoft Teams et Python, nous surveillerons simultanément l'utilisation du GPU, la latence d'inférence, le cache du modèle et les valeurs d'intégration du locuteur.

### voice_06.wav

TR
Beşinci ayın ikisinde saat dokuz buçukta görüşeceğiz. Üç farklı çözümü karşılaştırıp yirmi dört dosyayı inceleyeceğiz; işimiz yaklaşık kırk beş dakika sürerse öğleden önce bitirebiliriz ya da bir gün daha bekleriz.

EN
We will meet on May Second at Nine-fifty. We will compare three different solutions and review twenty-four files; if our work takes approximately forty-five minutes, we can finish before noon, or we can wait one day.

FR
Nous nous réunirons le 2 mai à 9h30. Nous comparerons trois solutions différentes et examinerons vingt-quatre fichiers ; si notre travail prend environ quarante-cinq minutes, nous pourrons terminer avant midi, sinon nous pourrons attendre un autre jour.

## Install and validate the profile

Create `voices/default/` with the six WAV files and optional `profile.json`:

```json
{
  "id": "default",
  "display_name": "Default — Six References",
  "backend": "xtts_v2",
  "reference_audio_paths": [
    "voice_01.wav", "voice_02.wav", "voice_03.wav",
    "voice_04.wav", "voice_05.wav", "voice_06.wav"
  ],
  "reference_language": "tr", // your-choosed language
  "target_languages": ["en", "fr"], // another language
  "is_default": false
}
```

If the manifest or audio path fields are omitted, the immediate WAV children are
discovered and sorted. Subdirectories, including `cache/`, are excluded. Explicit
`reference_audio_paths` is authoritative; additional files are not mixed in. The
legacy `reference_audio_path` field still selects one file. Absolute paths to
external local references work without copying. Do not specify both formats.

```powershell
uv run python scripts/validate_voice_profile.py default
```

This command uses SoundFile only; it loads no model and makes no network request.
Missing, unreadable, corrupt, non-finite, ≤4 KB or near-silent WAVs fail with their
path. RMS below 0.005 after mono downmix is rejected. Clips outside 3–30 seconds,
clipping, multiple channels and mixed sample rates generate warnings. Warnings
are advice to inspect/re-record; no lossy conversion or gain change is saved.

Restart De-ReVot after adding or editing files, then select the profile in the Web
UI. The selector displays its WAV count. For a persistent default, set
`tts.voice_profile_id` and `voice.default_profile` to `default` in
`config/local.toml`, and keep at most one manifest marked `is_default=true`.

The profile is validated and conditioned before use. Sorted audio-content hashes,
local model identity and conditioning parameters key the disk cache under
`voices/default/cache/`. An audio edit invalidates the cache on the next prepare
(restart/session start/profile switch). During synthesis, the prepared snapshot
is reused without audio file reads or latent extraction. Do not edit active
recordings and expect them to change an already prepared voice mid-sentence.

XTTS receives all references for speaker embedding; its GPT conditioner uses up
to the first 30 seconds of concatenated audio (`gpt_cond_len=30`,
`max_ref_length=60` per file). All six clips therefore contribute to speaker
identity, while later clips may fall outside the GPT conditioning window.

Compare single-reference and six-reference output on identical Turkish, English
and French texts. Listen for identity, intelligibility, pauses and distortion;
measure TTFA/RTF separately. Neither a cache hit nor a successful tensor extraction
proves higher voice quality.
