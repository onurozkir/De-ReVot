"""REST API routes for Voice Translator."""

from __future__ import annotations

from typing import Literal, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from voice_translator.streaming.orchestrator import MeetingOrchestrator
from voice_translator.telemetry.system import SystemResourceMonitor


class StartMeetingRequest(BaseModel):
    mic_id: Optional[str] = None
    loopback_id: Optional[str] = None
    render_id: Optional[str] = None
    voice_profile_id: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    save_meeting: bool = False
    prompt: Optional[str] = None
    app_preset: Optional[str] = None
    input_mode: Optional[Literal["auto", "vad", "ptt"]] = None
    ptt_key: Optional[str] = None
    overlay_enabled: Optional[bool] = None
    asr_model: Literal["configured", "turbo", "large_v3"] = "configured"


class SessionControlsRequest(BaseModel):
    input_mode: Optional[Literal["vad", "ptt"]] = None
    ptt_pressed: Optional[bool] = None
    paused: Optional[bool] = None
    muted: Optional[bool] = None
    overlay_enabled: Optional[bool] = None


class SwitchVoiceRequest(BaseModel):
    profile_id: str


class SwitchLanguageRequest(BaseModel):
    target_language: str


class SwitchLanguagesRequest(BaseModel):
    source_language: str
    target_language: str


def create_routes(orchestrator: MeetingOrchestrator) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/status")
    async def get_status():
        return {
            "status": orchestrator.status.value,
            "meeting_id": orchestrator.current_meeting_id,
            "error": orchestrator.last_start_error,
            "controls": orchestrator.controls_snapshot(),
            "system": SystemResourceMonitor.get_stats(),
        }

    @router.get("/devices")
    async def get_devices():
        orchestrator.device_manager.refresh()
        devices = orchestrator.device_manager.list_devices(wasapi_only=False)
        configured_mic = orchestrator.device_manager.find_by_identifier(orchestrator.config.audio.mic_device_id)
        configured_loopback = orchestrator.device_manager.find_by_identifier(orchestrator.config.audio.loopback_device_id)
        configured_render = orchestrator.device_manager.find_by_identifier(orchestrator.config.audio.render_device_id)

        default_mic = configured_mic if (configured_mic and "physical_mic" in configured_mic.roles) else orchestrator.device_manager.find_default_mic()
        default_loop = configured_loopback if (configured_loopback and configured_loopback.is_loopback) else orchestrator.device_manager.find_default_loopback()
        default_render = configured_render if (configured_render and configured_render.is_output) else orchestrator.device_manager.find_vbcable_render()

        return {
            "devices": [d.to_dict() for d in devices],
            "defaults": {
                "mic": getattr(default_mic, "stable_id", None),
                "loopback": getattr(default_loop, "stable_id", None),
                "render": getattr(default_render, "stable_id", None),
                "vb_capture": getattr(orchestrator.device_manager.find_vbcable_capture(), "stable_id", None),
            }
        }

    @router.get("/audio/diagnostics")
    async def get_audio_diagnostics():
        return orchestrator.get_audio_diagnostics()

    @router.get("/profiles")
    async def get_profiles():
        profiles = orchestrator.profile_manager.list_profiles()
        return {
            "profiles": [
                {
                    "id": p.id,
                    "display_name": p.display_name,
                    "backend": p.backend,
                    "is_default": p.is_default,
                    "reference_language": p.reference_language,
                    "target_language": p.target_language,
                    "target_languages": getattr(p, "target_languages", [p.target_language]),
                }
                for p in profiles
            ]
        }

    @router.get("/session/options")
    async def get_session_options():
        return orchestrator.session_options()

    @router.get("/languages")
    async def get_languages():
        source, target = orchestrator.source_language, orchestrator.target_language
        return {
            "languages": orchestrator.session_options()["languages"],
            "defaults": {"source": source, "target": target},
        }

    @router.post("/session/controls")
    async def set_session_controls(req: SessionControlsRequest):
        try:
            return await orchestrator.update_controls(**req.model_dump(exclude_none=True))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/meeting/start")
    async def start_meeting(req: StartMeetingRequest):
        try:
            await orchestrator.start_meeting(
                mic_id=req.mic_id,
                loopback_id=req.loopback_id,
                render_id=req.render_id,
                voice_profile_id=req.voice_profile_id,
                source_language=req.source_language,
                target_language=req.target_language,
                save_meeting=req.save_meeting,
                context_prompt=req.prompt,
                app_preset=req.app_preset,
                input_mode=req.input_mode,
                ptt_key=req.ptt_key,
                overlay_enabled=req.overlay_enabled,
                asr_model=req.asr_model,
            )
            return {"status": "ok", "meeting_id": orchestrator.current_meeting_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/meeting/switch_voice")
    async def switch_voice(req: SwitchVoiceRequest):
        try:
            orchestrator.switch_voice_profile(req.profile_id)
            return {"status": "ok", "profile_id": req.profile_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/meeting/switch_languages")
    async def switch_languages(req: SwitchLanguagesRequest):
        try:
            orchestrator.switch_languages(req.source_language, req.target_language)
            return {"status": "ok", "source_language": req.source_language, "target_language": req.target_language}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/meeting/switch_language")
    async def switch_language(req: SwitchLanguageRequest):
        try:
            orchestrator.switch_target_language(req.target_language)
            return {"status": "ok", "target_language": req.target_language}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/meeting/stop")
    async def stop_meeting():
        try:
            await orchestrator.stop_meeting()
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/telemetry")
    async def get_telemetry():
        return orchestrator.telemetry.get_snapshot()

    return router
