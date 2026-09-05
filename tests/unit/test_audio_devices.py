from voice_translator.audio.devices import AudioDeviceManager, DeviceInfo


def make_device(index: int, name: str, inputs: int, outputs: int, loopback: bool = False) -> DeviceInfo:
    return DeviceInfo(
        index=index, name=name, host_api=2, host_api_name="Windows WASAPI",
        max_input_channels=inputs, max_output_channels=outputs,
        default_sample_rate=48000, is_loopback=loopback,
    )


def test_stable_id_survives_pyaudio_index_reordering():
    first = make_device(7, "Microphone (AB13X USB Audio)", 1, 0)
    reordered = make_device(19, "Microphone (AB13X USB Audio)", 1, 0)
    assert first.stable_id == reordered.stable_id


def test_role_mapping_and_stable_resolution(monkeypatch):
    devices = [
        make_device(1, "Microphone (AB13X USB Audio)", 1, 0),
        make_device(2, "Speakers (AB13X USB Audio)", 0, 2),
        make_device(3, "Speakers (AB13X USB Audio) (Loopback)", 2, 0, True),
        make_device(4, "CABLE Input (VB-Audio Virtual Cable)", 0, 2),
        make_device(5, "CABLE Output (VB-Audio Virtual Cable) (Loopback)", 2, 0, True),
    ]
    manager = AudioDeviceManager()
    monkeypatch.setattr(manager, "list_devices", lambda wasapi_only=True: devices)
    assert manager.find_by_identifier(devices[0].stable_id) is devices[0]
    assert manager.resolve_required(devices[2].stable_id, "loopback") is devices[2]
    assert manager.find_vbcable_render() is devices[3]
    assert manager.find_vbcable_capture() is devices[4]
    assert manager.find_render_for_loopback(devices[2]) is devices[1]
    assert "vb_cable_render" in devices[3].roles
    assert "vb_cable_capture" in devices[4].roles


def test_ambiguous_partial_name_is_not_silently_selected(monkeypatch):
    devices = [make_device(1, "Microphone One", 1, 0), make_device(2, "Microphone Two", 1, 0)]
    manager = AudioDeviceManager()
    monkeypatch.setattr(manager, "list_devices", lambda wasapi_only=True: devices)
    assert manager.find_by_identifier("Microphone") is None

