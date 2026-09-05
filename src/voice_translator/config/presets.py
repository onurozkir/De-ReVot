"""Application guidance and local ASR choices; no process injection or downloads."""

from pathlib import Path


APP_PRESETS = {
    "teams": ("MS Teams", "vad", "Devices: microphone = CABLE Output; speaker = selected physical headset."),
    "zoom": ("Zoom", "vad", "Audio: microphone = CABLE Output; speaker = selected physical headset. Check suppression if cloned speech is clipped."),
    "google_meet": ("Google Meet (Chrome / Edge)", "vad", "Meet audio settings: microphone = CABLE Output; speaker = selected headset. Route browser output to that headset in Windows Volume mixer."),
    "discord": ("Discord", "vad", "Voice & Video: input = CABLE Output; output = selected headset. Use Voice Activity; adjust sensitivity/Krisp if translated speech is clipped."),
    "slack": ("Slack Huddles", "vad", "Audio & video: microphone = CABLE Output; speaker = selected headset."),
    "vrchat": ("VRChat", "ptt", "Input = CABLE Output; output = selected headset. Keep the application's mic enabled while translated speech plays."),
    "gaming": ("Online Games", "ptt", "Game voice input = CABLE Output; output = selected headset. Use open mic in game: translator PTT releases before translated playback finishes."),
    "dota2": ("Dota 2", "ptt", "Voice input = CABLE Output (or Windows default communications input). Enable open mic for translated playback; headset output. Use borderless window for HUD."),
    "pubg": ("PUBG", "ptt", "Voice input = CABLE Output (or Windows default communications input). Enable open mic for translated playback; headset output. Use borderless window for HUD."),
    "cs2": ("Counter-Strike 2", "ptt", "Voice input = CABLE Output; enable the game's microphone during translated playback. Use borderless window for HUD."),
    "custom": ("Custom", "vad", "Choose physical microphone, physical speaker loopback and CABLE Input explicitly; target app microphone = CABLE Output."),
}

ASR_PRESETS = {
    "turbo": {"label": "Whisper large-v3-turbo", "backend": "whisper_turbo", "model_path": "models/asr/whisper-large-v3-turbo", "compute_type": "float16", "download": "whisper"},
    "large_v3": {"label": "Faster Whisper large-v3", "backend": "faster_whisper", "model_path": "models/asr/whisper-large-v3", "compute_type": "int8_float16", "download": "whisper-large-v3"},
}


def resolve_input_mode(preset: str, mode: str) -> str:
    if preset not in APP_PRESETS:
        raise ValueError(f"Unknown application preset: {preset}")
    if mode not in ("auto", "vad", "ptt"):
        raise ValueError(f"Unknown input mode: {mode}")
    return APP_PRESETS[preset][1] if mode == "auto" else mode


def asr_choices(current_path: str) -> list[dict]:
    choices = []
    for key, preset in ASR_PRESETS.items():
        path = Path(preset["model_path"])
        required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json") if key == "large_v3" else ("config.json",)
        choices.append({"id": key, **preset, "available": all((path / name).is_file() for name in required)})
    choices.append({"id": "configured", "label": "Configured local model", "model_path": current_path, "available": Path(current_path).is_dir()})
    return choices
