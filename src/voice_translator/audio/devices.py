"""Audio device enumeration and stable identification using WASAPI / PyAudioWPatch."""

from __future__ import annotations

import logging
import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    try:
        import pyaudio
    except ImportError:
        pyaudio = None  # type: ignore


@dataclass(slots=True)
class DeviceInfo:
    index: int
    name: str
    host_api: int
    host_api_name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int
    is_loopback: bool = False

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0 and not self.is_loopback

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    @property
    def stable_id(self) -> str:
        """Best available PyAudio endpoint fingerprint; unlike index it survives reordering."""
        raw = "|".join(
            [
                self.host_api_name.casefold(),
                _normalized_name(self.name),
                str(self.max_input_channels),
                str(self.max_output_channels),
                str(self.default_sample_rate),
                "loopback" if self.is_loopback else "endpoint",
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"pa:{digest}"

    @property
    def roles(self) -> List[str]:
        roles: List[str] = []
        name = self.name.casefold()
        if self.is_input and "cable output" not in name and "vb-audio" not in name:
            roles.append("physical_mic")
        if self.is_output:
            roles.append("physical_render")
        if self.is_loopback:
            roles.append("speaker_loopback")
        if self.is_output and ("cable input" in name or "vb-audio" in name):
            roles.append("vb_cable_render")
        if (self.max_input_channels > 0 or self.is_loopback) and "cable output" in name:
            roles.append("vb_cable_capture")
        return roles

    def to_dict(self) -> dict:
        return {
            "stable_id": self.stable_id,
            "index": self.index,
            "name": self.name,
            "host_api": self.host_api,
            "host_api_name": self.host_api_name,
            "max_input_channels": self.max_input_channels,
            "max_output_channels": self.max_output_channels,
            "default_sample_rate": self.default_sample_rate,
            "is_input": self.is_input,
            "is_output": self.is_output,
            "is_loopback": self.is_loopback,
            "roles": self.roles,
        }


def _normalized_name(name: str) -> str:
    normalized = name.casefold().replace("(loopback)", "").replace("[loopback]", "")
    return re.sub(r"\s+", " ", normalized).strip()


class AudioDeviceManager:
    """Manages audio device discovery and selection via WASAPI."""

    def __init__(self):
        self._pa = None

    def _get_pyaudio(self):
        if self._pa is None:
            if pyaudio is None:
                raise RuntimeError("PyAudioWPatch or PyAudio is not installed.")
            self._pa = pyaudio.PyAudio()
        return self._pa

    def refresh(self):
        """Terminate and reset cached PyAudio instance so new USB devices are enumerated."""
        if getattr(self, "_pa", None) is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def list_devices(self, wasapi_only: bool = True, refresh: bool = False) -> List[DeviceInfo]:
        if refresh:
            self.refresh()
        if pyaudio is None:
            logger.warning("PyAudioWPatch or PyAudio is not installed.")
            return []
        pa = self._get_pyaudio()
        devices: List[DeviceInfo] = []

        device_count = pa.get_device_count()
        for idx in range(device_count):
            try:
                info = pa.get_device_info_by_index(idx)
                host_api = info.get("hostApi", -1)
                host_api_name = pa.get_host_api_info_by_index(host_api).get("name", "Unknown") if host_api >= 0 else "Unknown"
                is_loopback = bool(info.get("isLoopbackDevice", False))

                if wasapi_only and "WASAPI" not in host_api_name.upper():
                    continue

                dev = DeviceInfo(
                    index=idx,
                    name=str(info.get("name", f"Device {idx}")),
                    host_api=host_api,
                    host_api_name=host_api_name,
                    max_input_channels=int(info.get("maxInputChannels", 0)),
                    max_output_channels=int(info.get("maxOutputChannels", 0)),
                    default_sample_rate=int(info.get("defaultSampleRate", 48000)),
                    is_loopback=is_loopback,
                )
                devices.append(dev)
            except Exception as e:
                logger.debug(f"Error inspecting device {idx}: {e}")

        if not devices and wasapi_only:
            return self.list_devices(wasapi_only=False)

        return devices

    def find_default_mic(self) -> Optional[DeviceInfo]:
        devices = self.list_devices()
        if not devices:
            return None
        pa = self._get_pyaudio()
        try:
            default_index = int(pa.get_default_input_device_info()["index"])
            exact = next((dev for dev in devices if dev.index == default_index and "physical_mic" in dev.roles), None)
            if exact:
                return exact
        except Exception:
            pass
        for dev in devices:
            if "WASAPI" in dev.host_api_name.upper() and "physical_mic" in dev.roles:
                return dev
        # Fallback to any input
        for dev in devices:
            if "physical_mic" in dev.roles:
                return dev
        return None

    def find_default_loopback(self) -> Optional[DeviceInfo]:
        devices = self.list_devices()
        if not devices:
            return None
        pa = self._get_pyaudio()
        try:
            if hasattr(pa, "get_default_wasapi_loopback"):
                info = pa.get_default_wasapi_loopback()
                default_index = int(info["index"])
                exact = next((dev for dev in devices if dev.index == default_index), None)
                if exact and exact.is_loopback:
                    return exact
                default_name = _normalized_name(str(info.get("name", "")))
                exact = next(
                    (dev for dev in devices if dev.is_loopback and _normalized_name(dev.name) == default_name),
                    None,
                )
                if exact:
                    return exact
        except Exception:
            pass
        for dev in devices:
            if dev.is_loopback:
                return dev
        return None

    def find_vbcable_render(self) -> Optional[DeviceInfo]:
        devices = self.list_devices()
        for dev in devices:
            name_upper = dev.name.upper()
            if ("CABLE INPUT" in name_upper or "VB-AUDIO" in name_upper) and dev.is_output:
                return dev
        return None

    def find_vbcable_capture(self) -> Optional[DeviceInfo]:
        for dev in self.list_devices():
            if "CABLE OUTPUT" in dev.name.upper() and (dev.max_input_channels > 0 or dev.is_loopback):
                return dev
        return None

    def find_render_for_loopback(self, loopback: Optional[DeviceInfo]) -> Optional[DeviceInfo]:
        if loopback is None:
            return None
        target_name = _normalized_name(loopback.name)
        candidates = [
            dev for dev in self.list_devices()
            if dev.is_output and not dev.is_loopback and _normalized_name(dev.name) == target_name
        ]
        return candidates[0] if len(candidates) == 1 else None

    def find_by_identifier(self, identifier: str) -> Optional[DeviceInfo]:
        if not identifier:
            return None
        res = self._find_by_identifier_internal(identifier)
        if res is not None:
            return res
        # Re-enumerate audio endpoints in case device was plugged in after startup
        self.refresh()
        return self._find_by_identifier_internal(identifier)

    def _find_by_identifier_internal(self, identifier: str) -> Optional[DeviceInfo]:
        # Explicit selectors may use a measured native Windows fallback (for example
        # DirectSound mic) when the WASAPI analogue is present but unusably quiet.
        devices = self.list_devices(wasapi_only=False)
        for dev in devices:
            if identifier == dev.stable_id:
                return dev
        if identifier.isdigit():
            idx = int(identifier)
            for dev in devices:
                if dev.index == idx:
                    return dev
        normalized = _normalized_name(identifier)
        exact = [dev for dev in devices if _normalized_name(dev.name) == normalized]
        if len(exact) == 1:
            return exact[0]
        partial = [dev for dev in devices if normalized in _normalized_name(dev.name)]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            logger.warning("Ambiguous audio device selector '%s': %s", identifier, [d.name for d in partial])
        return None

    def resolve_required(self, identifier: str, role: str) -> DeviceInfo:
        device = self.find_by_identifier(identifier)
        if device is None:
            raise ValueError(f"Audio endpoint '{identifier}' was not found for role '{role}'.")
        valid = {
            "mic": "physical_mic" in device.roles,
            "loopback": device.is_loopback,
            "render": device.is_output,
            "vb_capture": "vb_cable_capture" in device.roles,
        }.get(role, True)
        if not valid:
            raise ValueError(
                f"Audio endpoint '{device.name}' (index {device.index}) is incompatible with role '{role}'."
            )
        return device

    def close(self):
        if getattr(self, "_pa", None) is not None:
            self._pa.terminate()
            self._pa = None
