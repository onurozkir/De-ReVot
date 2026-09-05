// Realtime Voice Translator UI Client

document.addEventListener("DOMContentLoaded", () => {
  const statusBadge = document.getElementById("statusBadge");
  const meetingIdLabel = document.getElementById("meetingIdLabel");
  const micSelect = document.getElementById("micSelect");
  const loopbackSelect = document.getElementById("loopbackSelect");
  const renderSelect = document.getElementById("renderSelect");
  const profileSelect = document.getElementById("profileSelect");
  const targetLanguageSelect = document.getElementById("targetLanguageSelect");
  const promptInput = document.getElementById("promptInput");
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const appPresetSelect = document.getElementById("appPresetSelect");
  const inputModeSelect = document.getElementById("inputModeSelect");
  const pttKeyInput = document.getElementById("pttKeyInput");
  const overlayEnabled = document.getElementById("overlayEnabled");
  const asrModelSelect = document.getElementById("asrModelSelect");
  const btnPTT = document.getElementById("btnPTT");
  const btnPause = document.getElementById("btnPause");
  const btnMute = document.getElementById("btnMute");
  const controlsStatus = document.getElementById("controlsStatus");
  const micDeviceInfo = document.getElementById("micDeviceInfo");
  const loopbackDeviceInfo = document.getElementById("loopbackDeviceInfo");
  const renderDeviceInfo = document.getElementById("renderDeviceInfo");
  const outgoingAsrState = document.getElementById("outgoingAsrState");
  const incomingAsrState = document.getElementById("incomingAsrState");
  const queueState = document.getElementById("queueState");
  const resolvedDeviceMap = document.getElementById("resolvedDeviceMap");

  const outgoingBox = document.getElementById("outgoingBox");
  const outgoingPartial = document.getElementById("outgoingPartial");
  const incomingBox = document.getElementById("incomingBox");
  const incomingPartial = document.getElementById("incomingPartial");

  const latIncomingPartial = document.getElementById("latIncomingPartial");
  const latIncomingCommit = document.getElementById("latIncomingCommit");
  const latOutgoingPcm = document.getElementById("latOutgoingPcm");
  const latMT = document.getElementById("latMT");
  const gpuVram = document.getElementById("gpuVram");

  let ws = null;
  let devicesById = new Map();
  let currentMeetingStatus = "STOPPED";
  let liveControls = {};
  let controlRequests = Promise.resolve();
  let uiPttPressed = false;
  let presets = [];

  function applyControls(controls) {
    if (!controls) return;
    liveControls = controls;
    inputModeSelect.value = controls.input_mode;
    overlayEnabled.checked = controls.overlay_enabled;
    btnPTT.hidden = controls.input_mode !== "ptt";
    btnPTT.disabled = !!(controls.paused || controls.muted);
    btnPTT.textContent = controls.ptt_pressed ? "Listening — release to translate" : `Hold to Talk (${controls.ptt_key})`;
    btnPTT.setAttribute("aria-pressed", String(!!controls.ptt_pressed));
    btnPause.textContent = controls.paused ? "Resume Translation" : "Pause Translation";
    btnMute.textContent = controls.muted ? "Unmute" : "Emergency Mute";
    btnPause.setAttribute("aria-pressed", String(!!controls.paused));
    btnMute.setAttribute("aria-pressed", String(!!controls.muted));
    controlsStatus.textContent = controls.desktop_error ||
      `${controls.toggle_mode_key}: mode · ${controls.pause_key}: pause · ${controls.mute_key}: mute` +
      (currentMeetingStatus === "running" ? ` · HUD ${controls.overlay_running ? "on" : "off"} · Global keys ${controls.hotkeys_running ? "on" : "off"}` : "");
  }

  function sendControls(change) {
    // Preserve press/release order even when HTTP requests have different delays.
    controlRequests = controlRequests.then(async () => {
      const res = await fetch("/api/session/controls", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(change),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      applyControls(data);
    }).catch(error => { controlsStatus.textContent = error.message; });
    return controlRequests;
  }

  function showPreset() {
    const preset = presets.find(p => p.id === appPresetSelect.value);
    document.getElementById("setupGuidance").textContent = (preset?.guidance || "") +
      " Incoming loopback includes every app, notification and game sound on the selected speaker.";
  }

  async function loadSessionOptions() {
    const response = await fetch("/api/session/options");
    const data = await response.json();
    presets = data.presets;
    appPresetSelect.replaceChildren(...presets.map(p => new Option(p.label, p.id)));
    appPresetSelect.value = localStorage.getItem("voice_trans_app_preset") || data.controls.app_preset;
    if (!appPresetSelect.value) appPresetSelect.value = data.controls.app_preset;
    pttKeyInput.value = data.controls.ptt_key;
    asrModelSelect.replaceChildren(...data.asr_models.map(model => {
      const option = new Option(model.label + (model.available ? "" : " — download required"), model.id);
      option.disabled = !model.available && model.id !== "configured";
      return option;
    }));
    asrModelSelect.value = "configured";
    const large = data.asr_models.find(m => m.id === "large_v3");
    document.getElementById("asrModelHelp").textContent = large.available
      ? "Large-v3 is installed. Model changes load and warm before the next session."
      : "To install large-v3 explicitly: uv run python scripts/download_models.py whisper-large-v3. Then refresh this page.";
    applyControls(data.controls);
    inputModeSelect.value = presets.find(p => p.id === appPresetSelect.value)?.input_mode || "vad";
    showPreset();
  }

  async function loadInitialData() {
    try {
      await loadSessionOptions();
      // 1. Fetch devices
      const devRes = await fetch("/api/devices");
      const devData = await devRes.json();
      devicesById = new Map(devData.devices.map(d => [d.stable_id, d]));
      
      micSelect.innerHTML = "";
      loopbackSelect.innerHTML = "";
      renderSelect.innerHTML = "";

      const sortedDevices = [...devData.devices].sort((a, b) => {
        const aWasapi = a.host_api_name.toUpperCase().includes("WASAPI") ? 1 : 0;
        const bWasapi = b.host_api_name.toUpperCase().includes("WASAPI") ? 1 : 0;
        return bWasapi - aWasapi;
      });

      sortedDevices.forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.stable_id;
        opt.textContent = `[${d.index}] ${d.name} (${d.host_api_name}, ${d.default_sample_rate} Hz)`;

        if (d.roles.includes("physical_mic")) micSelect.appendChild(opt.cloneNode(true));
        if (d.is_loopback) loopbackSelect.appendChild(opt.cloneNode(true));
        if (d.is_output) renderSelect.appendChild(opt.cloneNode(true));
      });

      // Apply defaults or saved localStorage
      const savedMic = localStorage.getItem("teams_trans_mic");
      const savedLoop = localStorage.getItem("teams_trans_loop");
      const savedRender = localStorage.getItem("teams_trans_render");
      const savedProfile = localStorage.getItem("teams_trans_profile");

      if (savedMic && micSelect.querySelector(`option[value="${savedMic}"]`)) {
        micSelect.value = savedMic;
      } else if (devData.defaults.mic !== null) {
        micSelect.value = devData.defaults.mic;
      }

      if (savedLoop && loopbackSelect.querySelector(`option[value="${savedLoop}"]`)) {
        loopbackSelect.value = savedLoop;
      } else if (devData.defaults.loopback !== null) {
        loopbackSelect.value = devData.defaults.loopback;
      }

      if (savedRender && renderSelect.querySelector(`option[value="${savedRender}"]`)) {
        renderSelect.value = savedRender;
      } else if (devData.defaults.render !== null) {
        renderSelect.value = devData.defaults.render;
      }

      // 2. Fetch profiles
      const profRes = await fetch("/api/profiles");
      const profData = await profRes.json();
      profileSelect.innerHTML = "";
      profData.profiles.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = `${p.display_name} (${p.backend})`;
        if (savedProfile === p.id || (!savedProfile && p.is_default)) opt.selected = true;
        profileSelect.appendChild(opt);
      });

      const savedTargetLang = localStorage.getItem("teams_trans_target_lang") || "en";
      if (targetLanguageSelect) {
        targetLanguageSelect.value = savedTargetLang;
      }

      // Save choices automatically & switch dynamically during meetings
      micSelect.onchange = () => { localStorage.setItem("teams_trans_mic", micSelect.value); updateSelectedDeviceDetails(); };
      loopbackSelect.onchange = () => { localStorage.setItem("teams_trans_loop", loopbackSelect.value); updateSelectedDeviceDetails(); };
      renderSelect.onchange = () => { localStorage.setItem("teams_trans_render", renderSelect.value); updateSelectedDeviceDetails(); };
      
      profileSelect.onchange = async () => {
        localStorage.setItem("teams_trans_profile", profileSelect.value);
        if (currentMeetingStatus.toLowerCase() === "running") {
          try {
            console.log("Live switching voice profile to:", profileSelect.value);
            const res = await fetch("/api/meeting/switch_voice", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ profile_id: profileSelect.value }),
            });
            if (!res.ok) {
              const err = await res.json().catch(() => ({}));
              console.error("Failed to switch voice live:", err);
            } else {
              console.log("Voice switched successfully.");
            }
          } catch (err) {
            console.error("Network error switching voice:", err);
          }
        }
      };

      if (targetLanguageSelect) {
        targetLanguageSelect.onchange = async () => {
          localStorage.setItem("teams_trans_target_lang", targetLanguageSelect.value);
          if (currentMeetingStatus.toLowerCase() === "running") {
            try {
              console.log("Live switching target language to:", targetLanguageSelect.value);
              const res = await fetch("/api/meeting/switch_language", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ target_language: targetLanguageSelect.value }),
              });
              if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                alert("Failed to switch language:\n" + (err.detail || res.statusText));
              } else {
                console.log("Target language switched successfully.");
              }
            } catch (err) {
              console.error("Network error switching language:", err);
            }
          }
        };
      }
      updateSelectedDeviceDetails();

      // 3. Fetch status
      const statRes = await fetch("/api/status");
      const statData = await statRes.json();
      updateStatus(statData.status, statData.meeting_id, statData.error);
      if (statData.status.toLowerCase() === "running") applyControls(statData.controls);
      if (statData.system) updateSystemStats(statData.system);

    } catch (e) {
      console.error("Failed to load initial metadata:", e);
    }
  }

  function updateSelectedDeviceDetails() {
    const format = (id) => {
      const d = devicesById.get(id);
      return d ? `${d.stable_id} | hostApi=${d.host_api} | in=${d.max_input_channels} out=${d.max_output_channels}` : "Not resolved";
    };
    micDeviceInfo.textContent = format(micSelect.value);
    loopbackDeviceInfo.textContent = format(loopbackSelect.value);
    renderDeviceInfo.textContent = format(renderSelect.value);
  }

  function updateLevel(prefix, signal) {
    const textNode = document.getElementById(`${prefix}LevelText`);
    const bar = document.getElementById(`${prefix}LevelBar`);
    const dbfs = signal && Number.isFinite(signal.dbfs) ? signal.dbfs : -120;
    textNode.textContent = `${dbfs.toFixed(1)} dBFS`;
    bar.style.width = `${Math.max(0, Math.min(100, (dbfs + 60) / 60 * 100))}%`;
  }

  async function loadDiagnostics() {
    try {
      const res = await fetch("/api/audio/diagnostics", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      const outgoing = data.outgoing;
      const incoming = data.incoming;
      updateLevel("mic", outgoing?.capture?.signal);
      updateLevel("loop", incoming?.capture?.signal);
      updateLevel("cable", outgoing?.render?.signal);
      outgoingAsrState.textContent = outgoing?.asr_state || "idle";
      incomingAsrState.textContent = incoming?.asr_state || "idle";
      const queues = [...(outgoing?.queues || []), ...(incoming?.queues || [])];
      queueState.textContent = queues.length
        ? queues.map(q => `${q.name}:${q.current}/${q.max} ${q.oldest_age_ms.toFixed(0)}ms`).join(" | ")
        : "--";
      const resolved = data.resolved || {};
      resolvedDeviceMap.textContent = [
        `Mic: ${resolved.physical_mic?.name || "unresolved"}`,
        `Speaker: ${resolved.physical_speaker?.name || "unresolved"}`,
        `Loopback: ${resolved.speaker_loopback?.name || "unresolved"}`,
        `VB render: ${resolved.vb_cable_render?.name || "unresolved"}`,
        `VB capture: ${resolved.vb_cable_capture?.name || "unresolved"}`,
      ].join("\n");
    } catch (e) {
      console.debug("Audio diagnostics unavailable", e);
    }
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleServerEvent(data);
      } catch (e) {
        console.error("WS Parse error:", e);
      }
    };

    ws.onclose = () => {
      setTimeout(connectWebSocket, 2000);
    };
  }

  function handleServerEvent(data) {
    switch (data.type) {
      case "status_change":
        updateStatus(data.status, data.meeting_id, data.error);
        if (data.controls && currentMeetingStatus === "running") applyControls(data.controls);
        break;

      case "controls_change":
        applyControls(data.controls);
        break;

      case "asr_partial":
        if (data.direction === "outgoing") {
          outgoingPartial.textContent = `🎙️ [TR Partial] ${data.text}`;
        }
        break;

      case "mt_committed":
        if (data.direction === "outgoing") {
          outgoingPartial.textContent = `Synthesizing: ${data.translated_text}`;
        }
        break;

      case "tts_started":
        outgoingPartial.textContent = "Listening...";
        addUtterance(outgoingBox, data.source_text, `🔊 [EN Routed] ${data.translated_text}`, false);
        break;

      case "tts_rejected":
        if (data.direction === "outgoing") outgoingPartial.textContent = "Listening...";
        break;

      case "incoming_partial":
        incomingPartial.textContent = `⚡ [EN] ${data.source_text} ➔ [TR] ${data.translated_text}`;
        break;

      case "incoming_committed":
        incomingPartial.textContent = "Waiting for incoming audio...";
        addUtterance(incomingBox, data.source_text, `🇹🇷 [TR Subtitle] ${data.translated_text}`, true);
        break;

      case "latency_update":
        if (data.metrics) updateMetrics(data.metrics);
        break;
    }
  }

  function updateStatus(status, meetingId, error) {
    const norm = (status || "").toLowerCase();
    currentMeetingStatus = norm;
    document.getElementById("liveControls").hidden = norm !== "running";
    for (const control of [appPresetSelect, asrModelSelect, pttKeyInput, micSelect, loopbackSelect, renderSelect]) {
      control.disabled = norm === "running" || norm === "warming" || norm === "starting";
    }
    if (norm !== "running") uiPttPressed = false;
    statusBadge.className = `status-badge status-${norm}`;
    statusBadge.textContent = (status || "").toUpperCase();

    if (norm === "running") {
      btnStart.style.display = "none";
      btnStop.style.display = "block";
      meetingIdLabel.textContent = `Meeting: ${meetingId || "Active"}`;
    } else if (norm === "error" && error) {
      btnStart.style.display = "block";
      btnStop.style.display = "none";
      meetingIdLabel.textContent = `Error: ${error}`;
    } else {
      btnStart.style.display = "block";
      btnStop.style.display = "none";
      meetingIdLabel.textContent = "No active meeting";
    }
  }

  function addUtterance(container, original, translated, isIncoming) {
    const item = document.createElement("div");
    item.className = `utterance-item ${isIncoming ? "incoming" : ""}`;
    const source = document.createElement("div");
    source.className = "utterance-source";
    source.textContent = original;
    const target = document.createElement("div");
    target.className = `sub-text ${isIncoming ? "tr" : ""}`;
    target.textContent = translated;
    item.append(source, target);
    container.insertBefore(item, container.lastElementChild);
    while (container.children.length > 201) container.firstElementChild.remove();
    container.scrollTop = container.scrollHeight;
  }

  function updateMetrics(m) {
    if (m.incoming_partial) {
      latIncomingPartial.textContent = `${m.incoming_partial.p50.toFixed(0)} / ${m.incoming_partial.p95.toFixed(0)} ms`;
    }
    if (m.incoming_committed) {
      latIncomingCommit.textContent = `${m.incoming_committed.p50.toFixed(0)} / ${m.incoming_committed.p95.toFixed(0)} ms`;
    }
    if (m.outgoing_pcm) {
      latOutgoingPcm.textContent = `${m.outgoing_pcm.p50.toFixed(0)} / ${m.outgoing_pcm.p95.toFixed(0)} ms`;
    }
    if (m.mt_duration) {
      latMT.textContent = `${m.mt_duration.p50.toFixed(0)} ms`;
    }
  }

  function updateSystemStats(s) {
    if (s.gpu_available) {
      gpuVram.textContent = `${s.gpu_allocated_mb} / ${s.gpu_total_mb} MB (${s.gpu_name || "GPU"})`;
    } else {
      gpuVram.textContent = "CPU Mode";
    }
  }

  if (promptInput) {
    promptInput.value = localStorage.getItem("voice_translator_prompt") || localStorage.getItem("teams_translator_prompt") || "";
    promptInput.addEventListener("input", () => {
      localStorage.setItem("voice_translator_prompt", promptInput.value);
    });
  }

  btnStart.addEventListener("click", async () => {
    try {
      btnStart.disabled = true;
      const res = await fetch("/api/meeting/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mic_id: micSelect.value,
          loopback_id: loopbackSelect.value,
          render_id: renderSelect.value,
          voice_profile_id: profileSelect.value,
          target_language: targetLanguageSelect ? targetLanguageSelect.value : "en",
          save_meeting: false,
          app_preset: appPresetSelect.value,
          input_mode: inputModeSelect.value,
          ptt_key: pttKeyInput.value.trim(),
          overlay_enabled: overlayEnabled.checked,
          asr_model: asrModelSelect.value,
          prompt: promptInput ? promptInput.value.trim() : undefined,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: res.statusText }));
        alert("Failed to start meeting:\n" + (data.detail || res.statusText));
      }
    } catch (e) {
      alert("Connection Error: " + e.message);
    } finally {
      btnStart.disabled = false;
    }
  });

  btnStop.addEventListener("click", async () => {
    try {
      btnStop.disabled = true;
      await fetch("/api/meeting/stop", { method: "POST" });
    } catch (e) {
      alert("Failed to stop meeting: " + e.message);
    } finally {
      btnStop.disabled = false;
    }
  });

  appPresetSelect.addEventListener("change", () => {
    localStorage.setItem("voice_trans_app_preset", appPresetSelect.value);
    inputModeSelect.value = presets.find(p => p.id === appPresetSelect.value)?.input_mode || "vad";
    showPreset();
  });
  inputModeSelect.addEventListener("change", () => {
    if (currentMeetingStatus === "running") sendControls({ input_mode: inputModeSelect.value });
  });
  overlayEnabled.addEventListener("change", () => {
    if (currentMeetingStatus === "running") sendControls({ overlay_enabled: overlayEnabled.checked });
  });
  btnPause.addEventListener("click", () => sendControls({ paused: !liveControls.paused }));
  btnMute.addEventListener("click", () => sendControls({ muted: !liveControls.muted }));
  btnPTT.addEventListener("pointerdown", event => {
    if (event.button !== 0 || uiPttPressed) return;
    event.preventDefault();
    btnPTT.setPointerCapture(event.pointerId);
    uiPttPressed = true;
    sendControls({ ptt_pressed: true });
  });
  function releasePTT() {
    if (uiPttPressed) {
      uiPttPressed = false;
      sendControls({ ptt_pressed: false });
    }
  }
  btnPTT.addEventListener("pointerup", releasePTT);
  btnPTT.addEventListener("pointercancel", releasePTT);
  btnPTT.addEventListener("lostpointercapture", releasePTT);
  window.addEventListener("blur", releasePTT);
  document.addEventListener("visibilitychange", () => { if (document.hidden) releasePTT(); });

  loadInitialData();
  loadDiagnostics();
  setInterval(loadDiagnostics, 750);
  connectWebSocket();
});
