/* Kokoro Studio — Interactive Frontend Controller */

const SLOT_COLORS = ["#f5a623", "#60a5fa", "#34d399", "#c084fc"];

const HUMANIZE_PRESETS = {
  off: { enabled: false, breath: 0.0, airiness: 0.0, smile: 0.0, jitter: 0.0, fry: 0.0, roomFloor: 0.0, rhythm: 0.0 },
  natural: { enabled: true, breath: 0.25, airiness: 0.15, smile: 0.15, jitter: 0.20, fry: 0.12, roomFloor: 0.20, rhythm: 0.30 },
  radio: { enabled: true, breath: 0.34, airiness: 0.22, smile: 0.45, jitter: 0.24, fry: 0.10, roomFloor: 0.26, rhythm: 0.45 },
  intimate: { enabled: true, breath: 0.42, airiness: 0.38, smile: 0.05, jitter: 0.28, fry: 0.34, roomFloor: 0.30, rhythm: 0.40 },
};

const CHATTERBOX_PRESETS = [
  {
    id: "cb-natural",
    name: "Natural Conversational",
    tag: "Balanced",
    description: "Warm, balanced Chatterbox expressiveness with natural pacing and light studio polish.",
    cb: { exaggeration: 0.50, cfg_weight: 0.50, temperature: 0.80, repetition_penalty: 1.20, sentence_pause: 0.28, paragraph_pause: 0.58 },
    mastering: { enabled: true, highpass_hz: 80, warmth_db: 1.8, presence_db: 1.4, air_db: 1.5, tube_warmth: 0.14, formant_shift: 0.0, deesser: 0.36, compression: 0.34, room_ambience: 0.10, pitch_semitones: 0.0, target_lufs: -15.5 },
  },
  {
    id: "cb-radio-clone",
    name: "Warm UK Radio Clone",
    tag: "Broadcast",
    description: "Upbeat, warm radio-presenter energy — lower CFG preserves regional accent & lively inflection, paired with broadcast preamp polish.",
    cb: { exaggeration: 0.65, cfg_weight: 0.40, temperature: 0.84, repetition_penalty: 1.20, sentence_pause: 0.24, paragraph_pause: 0.48 },
    mastering: { enabled: true, highpass_hz: 85, warmth_db: 2.6, presence_db: 2.0, air_db: 1.2, tube_warmth: 0.30, formant_shift: 0.0, deesser: 0.38, compression: 0.50, room_ambience: 0.08, pitch_semitones: 0.0, target_lufs: -14.5 },
  },
  {
    id: "cb-storyteller",
    name: "Expressive Storyteller",
    tag: "Acting",
    description: "Rich dynamic acting and emotional range — ideal for fiction, narration, and character reads.",
    cb: { exaggeration: 0.82, cfg_weight: 0.34, temperature: 0.88, repetition_penalty: 1.22, sentence_pause: 0.38, paragraph_pause: 0.74 },
    mastering: { enabled: true, highpass_hz: 75, warmth_db: 2.4, presence_db: 1.2, air_db: 1.8, tube_warmth: 0.18, formant_shift: 0.0, deesser: 0.40, compression: 0.36, room_ambience: 0.14, pitch_semitones: 0.0, target_lufs: -16.0 },
  },
  {
    id: "cb-documentary",
    name: "Calm Documentary",
    tag: "Steady",
    description: "Composed, tightly controlled adherence to the reference speaker with minimal vocal variance.",
    cb: { exaggeration: 0.28, cfg_weight: 0.65, temperature: 0.65, repetition_penalty: 1.18, sentence_pause: 0.42, paragraph_pause: 0.78 },
    mastering: { enabled: true, highpass_hz: 70, warmth_db: 2.2, presence_db: 1.6, air_db: 1.4, tube_warmth: 0.12, formant_shift: 0.0, deesser: 0.35, compression: 0.38, room_ambience: 0.14, pitch_semitones: 0.0, target_lufs: -16.0 },
  },
  {
    id: "cb-high-drama",
    name: "High Emotion",
    tag: "Dramatic",
    description: "Pushed exaggeration and expressive phrasing for trailers, punchy hooks, or theatrical reads.",
    cb: { exaggeration: 1.05, cfg_weight: 0.28, temperature: 0.92, repetition_penalty: 1.24, sentence_pause: 0.32, paragraph_pause: 0.65 },
    mastering: { enabled: true, highpass_hz: 75, warmth_db: 2.8, presence_db: 2.2, air_db: 1.6, tube_warmth: 0.26, formant_shift: 0.0, deesser: 0.42, compression: 0.48, room_ambience: 0.15, pitch_semitones: 0.0, target_lufs: -14.5 },
  },
];

const SAMPLE_SCRIPTS = {
  podcast:
    "Hey, welcome back. Today we're testing how close a tiny local model can get to a real studio voice — warm, unhurried, and genuinely conversational.\n\nListen closely to the phrasing, the subtle breath between thoughts, and the softness on the high frequencies. When you dial in the blend and mastering chain just right, you'll hear the difference.",
  conversational:
    "Honestly? I wasn't expecting it to sound this natural on a laptop. Once you slow the pacing down just a touch and blend two complementary voices together, all of that robotic sharpness kind of melts away.",
  story:
    "The rain hadn't stopped since midnight. Down along the harbor, amber streetlamps reflected off wet cobblestones, and somewhere out beyond the fog, a single ferry horn echoed across the water.\n\nShe pulled her coat tighter, stepped off the curb, and finally opened the letter.",
  explainer:
    "Most text-to-speech systems sound artificial for three simple reasons: rushed transitions between sentences, overly dry studio acoustics, and harsh sibilance around seven kilohertz.\n\nBy blending style vectors, adding natural clause pauses, and running a lightweight mastering chain, we can fix all three in milliseconds.",
};

const state = {
  engine: "kokoro", // 'kokoro' | 'chatterbox'
  kokoroHealthText: "Connecting…",
  chatterboxStatus: "not_loaded", // 'not_loaded' | 'loading' | 'ready' | 'error'
  references: [],
  activeRefId: null, // null -> Chatterbox built-in default
  activeCbPresetId: "cb-natural",
  exportedVoices: [],
  selectedExportSlug: null,
  exportCodeTab: "curl",
  voices: [],
  voiceMap: {},
  presets: [],
  activePresetId: "eleven-warm-host",
  blend: [
    { voice: "af_heart", weight: 0.65 },
    { voice: "af_nicole", weight: 0.25 },
    { voice: "af_bella", weight: 0.10 },
  ],
  pronunciationRules: [
    { find: "Kokoro", replace: "Ko-ko-ro" },
  ],
  activeMode: "tuned", // 'tuned' | 'raw'
  currentTake: null,
  history: [],
  isPlaying: false,
  remasterTimer: null,
  liveSynthTimer: null,
  synthAbort: null,
  remasterAbort: null,
  synthSeq: 0,
  remasterSeq: 0,
  lastRawScript: "",
  lastRawPrimary: "",
  rafId: null,
};

// DOM Shortcuts
const $ = (id) => document.getElementById(id);

const audioTuned = $("audioTuned");
const audioRaw = $("audioRaw");
const audioRefPreview = $("audioRefPreview");

function getActiveAudio() {
  return state.activeMode === "raw" ? audioRaw : audioTuned;
}

function showToast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 2600);
}

function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) return "0:00.0";
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

// ---------- Read / Write Config ----------

function readMasteringConfig() {
  return {
    enabled: $("masterEnabledCheck").checked,
    highpass_hz: parseFloat($("highpassSlider").value),
    warmth_db: parseFloat($("warmthSlider").value),
    presence_db: parseFloat($("presenceSlider").value),
    air_db: parseFloat($("airSlider").value),
    tube_warmth: parseFloat($("tubeWarmthSlider").value),
    formant_shift: parseFloat($("formantSlider").value),
    deesser: parseFloat($("deesserSlider").value),
    compression: parseFloat($("compressionSlider").value),
    room_ambience: parseFloat($("roomSlider").value),
    pitch_semitones: parseFloat($("pitchSlider").value),
    target_lufs: parseFloat($("lufsSlider").value),
    humanize_enabled: $("humanizeEnabledCheck").checked,
    breath_amount: parseFloat($("breathSlider").value),
    airiness: parseFloat($("airinessSlider").value),
    smile: parseFloat($("smileSlider").value),
    organic_jitter: parseFloat($("jitterSlider").value),
    vocal_fry: parseFloat($("vocalFrySlider").value),
    room_floor: parseFloat($("roomFloorSlider").value),
  };
}

function readFullConfig() {
  return {
    lang: $("langSelect").value,
    blend: state.blend.map((b) => ({ voice: b.voice, weight: Number(b.weight.toFixed(3)) })),
    split_mode: $("splitModeCheck").checked,
    prosody_voice: $("prosodyVoiceSelect").value || "af_heart",
    prosody_intensity: parseFloat($("prosodyIntensity").value),
    style_offset: parseInt($("styleOffset").value, 10),
    speed: parseFloat($("speedSlider").value),
    sentence_pause: parseFloat($("sentencePauseSlider").value),
    clause_pause: parseFloat($("clausePauseSlider").value),
    paragraph_pause: parseFloat($("paragraphPauseSlider").value),
    rhythm_humanize: parseFloat($("rhythmHumanizeSlider").value),
    continuous: $("continuousCheck").checked,
    smart_punctuation: $("smartPuncCheck").checked,
    mastering: readMasteringConfig(),
  };
}

function applyConfig(cfg, triggerRemaster = false) {
  if (cfg.lang) $("langSelect").value = cfg.lang;
  if (Array.isArray(cfg.blend) && cfg.blend.length) {
    state.blend = cfg.blend.map((b) => ({ voice: b.voice, weight: b.weight }));
  }
  $("splitModeCheck").checked = Boolean(cfg.split_mode);
  $("splitPickerWrap").classList.toggle("hidden", !cfg.split_mode);
  if (cfg.prosody_voice && state.voiceMap[cfg.prosody_voice]) {
    $("prosodyVoiceSelect").value = cfg.prosody_voice;
  }

  $("prosodyIntensity").value = cfg.prosody_intensity ?? 1.0;
  $("styleOffset").value = cfg.style_offset ?? 0;
  $("speedSlider").value = cfg.speed ?? 0.92;
  $("sentencePauseSlider").value = cfg.sentence_pause ?? 0.35;
  $("clausePauseSlider").value = cfg.clause_pause ?? 0.14;
  $("paragraphPauseSlider").value = cfg.paragraph_pause ?? 0.65;
  $("continuousCheck").checked = cfg.continuous ?? true;
  $("smartPuncCheck").checked = cfg.smart_punctuation ?? true;

  const m = cfg.mastering || {};
  $("masterEnabledCheck").checked = m.enabled ?? true;
  $("highpassSlider").value = m.highpass_hz ?? 75;
  $("warmthSlider").value = m.warmth_db ?? 2.0;
  $("presenceSlider").value = m.presence_db ?? 1.5;
  $("airSlider").value = m.air_db ?? 1.8;
  $("tubeWarmthSlider").value = m.tube_warmth ?? 0.0;
  $("formantSlider").value = m.formant_shift ?? 0.0;
  $("deesserSlider").value = m.deesser ?? 0.35;
  $("compressionSlider").value = m.compression ?? 0.35;
  $("roomSlider").value = m.room_ambience ?? 0.12;
  $("pitchSlider").value = m.pitch_semitones ?? 0.0;
  $("lufsSlider").value = m.target_lufs ?? -16.0;

  syncMasteringEnabledUI();
  renderBlendUI();
  syncSliderReadouts();

  if (triggerRemaster && state.currentTake) {
    scheduleRemaster();
  }
}

function syncMasteringEnabledUI() {
  const on = $("masterEnabledCheck").checked;
  $("masterControlsWrap").classList.toggle("disabled", !on);
  $("masterLiveDot").style.opacity = on ? "0.9" : "0.2";
}

function syncHumanizeEnabledUI() {
  const on = $("humanizeEnabledCheck").checked;
  $("humanizeControlsWrap").classList.toggle("disabled", !on);
  $("humanizeLiveDot").style.opacity = on ? "0.9" : "0.2";
}

function applyHumanizePreset(key, triggerLive = true) {
  const hp = HUMANIZE_PRESETS[key];
  if (!hp) return;
  const prevRhythm = parseFloat($("rhythmHumanizeSlider").value);
  $("humanizeEnabledCheck").checked = hp.enabled;
  $("breathSlider").value = hp.breath;
  $("airinessSlider").value = hp.airiness;
  $("smileSlider").value = hp.smile;
  $("jitterSlider").value = hp.jitter;
  $("vocalFrySlider").value = hp.fry;
  $("roomFloorSlider").value = hp.roomFloor;
  $("rhythmHumanizeSlider").value = hp.rhythm;

  document.querySelectorAll("#humanizePresets .chip").forEach((c) => {
    c.classList.toggle("active-chip", c.dataset.hpreset === key);
  });

  syncHumanizeEnabledUI();
  syncSliderReadouts();

  if (triggerLive) {
    if (Math.abs(hp.rhythm - prevRhythm) > 0.01 && !state.currentTake) {
      scheduleLiveSynthesize(240);
    } else {
      scheduleRemaster();
    }
  }
}

function syncSliderReadouts() {
  $("valProsodyIntensity").textContent = `${parseFloat($("prosodyIntensity").value).toFixed(2)}×`;
  const off = parseInt($("styleOffset").value, 10);
  $("valStyleOffset").textContent = off > 0 ? `+${off}` : `${off}`;
  $("valSpeed").textContent = `${parseFloat($("speedSlider").value).toFixed(2)}×`;
  $("valSentencePause").textContent = `${parseFloat($("sentencePauseSlider").value).toFixed(2)}s`;
  $("valClausePause").textContent = `${parseFloat($("clausePauseSlider").value).toFixed(2)}s`;
  $("valParagraphPause").textContent = `${parseFloat($("paragraphPauseSlider").value).toFixed(2)}s`;

  const fmtDb = (v) => {
    const n = parseFloat(v);
    return `${n >= 0 ? "+" : ""}${n.toFixed(1)} dB`;
  };
  $("valWarmth").textContent = fmtDb($("warmthSlider").value);
  $("valPresence").textContent = fmtDb($("presenceSlider").value);
  $("valAir").textContent = fmtDb($("airSlider").value);
  $("valTubeWarmth").textContent = `${Math.round(parseFloat($("tubeWarmthSlider").value) * 100)}%`;
  const fm = parseFloat($("formantSlider").value);
  $("valFormant").textContent = `${fm > 0 ? "+" : ""}${fm.toFixed(1)}%`;
  $("valDeesser").textContent = `${Math.round(parseFloat($("deesserSlider").value) * 100)}%`;
  $("valCompression").textContent = `${Math.round(parseFloat($("compressionSlider").value) * 100)}%`;
  $("valRoom").textContent = `${Math.round(parseFloat($("roomSlider").value) * 100)}%`;
  const pt = parseFloat($("pitchSlider").value);
  $("valPitch").textContent = `${pt > 0 ? "+" : ""}${pt.toFixed(2)} st`;
  $("valHighpass").textContent = `${Math.round(parseFloat($("highpassSlider").value))} Hz`;
  $("valLufs").textContent = `${parseFloat($("lufsSlider").value).toFixed(1)} LUFS`;

  // Chatterbox readouts
  if ($("cbExagSlider")) {
    $("valCbExag").textContent = parseFloat($("cbExagSlider").value).toFixed(2);
    $("valCbCfg").textContent = parseFloat($("cbCfgSlider").value).toFixed(2);
    $("valCbTemp").textContent = parseFloat($("cbTempSlider").value).toFixed(2);
    $("valCbRep").textContent = parseFloat($("cbRepSlider").value).toFixed(2);
  }

  // Humanizer readouts
  if ($("breathSlider")) {
    $("valBreath").textContent = `${Math.round(parseFloat($("breathSlider").value) * 100)}%`;
    $("valAiriness").textContent = `${Math.round(parseFloat($("airinessSlider").value) * 100)}%`;
    const sm = Math.round(parseFloat($("smileSlider").value) * 100);
    $("valSmile").textContent = `${sm > 0 ? "+" : ""}${sm}%`;
    $("valJitter").textContent = `${Math.round(parseFloat($("jitterSlider").value) * 100)}%`;
    $("valVocalFry").textContent = `${Math.round(parseFloat($("vocalFrySlider").value) * 100)}%`;
    $("valRoomFloor").textContent = `${Math.round(parseFloat($("roomFloorSlider").value) * 100)}%`;
    $("valRhythmHumanize").textContent = `${Math.round(parseFloat($("rhythmHumanizeSlider").value) * 100)}%`;
  }
  updateLiveSummary();
}

function isLiveSyncEnabled() {
  const el = $("liveSyncCheck");
  return !el || el.checked;
}

function setSyncStatus(syncing, text) {
  const pill = document.querySelector(".live-toggle-pill");
  const wrap = $("waveformWrap");
  const st = $("liveSyncState");
  if (pill) pill.classList.toggle("syncing", syncing);
  if (wrap) wrap.classList.toggle("syncing", syncing);
  if (st) {
    st.textContent = text;
    st.classList.toggle("syncing", syncing);
  }
}

function updateLiveSummary() {
  const el = $("liveSummaryText");
  if (!el) return;
  const mOn = $("masterEnabledCheck").checked;
  const mStr = mOn
    ? ` · Warm ${parseFloat($("warmthSlider").value) >= 0 ? "+" : ""}${parseFloat($("warmthSlider").value).toFixed(1)}dB · Tube ${Math.round(parseFloat($("tubeWarmthSlider").value) * 100)}%`
    : " · Master Bypassed";

  if (state.engine === "chatterbox") {
    const refObj = state.references.find((r) => r.id === state.activeRefId);
    const refName = refObj ? `Clone: ${refObj.name}` : "Chatterbox Default";
    const exag = parseFloat($("cbExagSlider").value).toFixed(2);
    const cfg = parseFloat($("cbCfgSlider").value).toFixed(2);
    el.textContent = `${refName} · Exag ${exag} · CFG ${cfg}${mStr}`;
    return;
  }

  const total = state.blend.reduce((a, b) => a + Math.max(0, b.weight), 0) || 1;
  const mix = state.blend
    .map((b) => {
      const name = state.voiceMap[b.voice]?.name || b.voice;
      return `${name} ${Math.round((Math.max(0, b.weight) / total) * 100)}%`;
    })
    .join(" + ");

  const speed = `${parseFloat($("speedSlider").value).toFixed(2)}×`;
  const split = $("splitModeCheck").checked
    ? ` · Prosody:${state.voiceMap[$("prosodyVoiceSelect").value]?.name || $("prosodyVoiceSelect").value}`
    : "";
  el.textContent = `${mix}${split} · ${speed}${mStr}`;
}

function updateEngineStatusPill() {
  const pill = $("engineStatus");
  const txt = $("engineStatusText");
  if (state.engine === "kokoro") {
    pill.classList.add("ready");
    txt.textContent = state.kokoroHealthText;
  } else {
    if (state.chatterboxStatus === "ready") {
      pill.classList.add("ready");
      txt.textContent = "Chatterbox MLX · Metal Ready";
    } else if (state.chatterboxStatus === "loading") {
      pill.classList.remove("ready");
      txt.textContent = "Loading Chatterbox MLX…";
    } else if (state.chatterboxStatus === "error") {
      pill.classList.remove("ready");
      txt.textContent = "Chatterbox Load Error";
    } else {
      pill.classList.remove("ready");
      txt.textContent = "Chatterbox MLX · Standby";
    }
  }
}

async function ensureChatterboxLoaded() {
  if (state.chatterboxStatus === "ready" || state.chatterboxStatus === "loading") return;
  state.chatterboxStatus = "loading";
  updateEngineStatusPill();
  try {
    const res = await fetch("/api/chatterbox/load", { method: "POST" });
    if (!res.ok) throw new Error("Load failed");
    state.chatterboxStatus = "ready";
    updateEngineStatusPill();
    showToast("Chatterbox MLX loaded on Apple Silicon Metal");
  } catch (err) {
    state.chatterboxStatus = "error";
    updateEngineStatusPill();
    showToast("Could not load Chatterbox MLX");
  }
}

function setEngine(engine) {
  state.engine = engine;
  const isCb = engine === "chatterbox";

  $("btnEngineKokoro").classList.toggle("active", !isCb);
  $("btnEngineChatterbox").classList.toggle("active", isCb);

  $("tab01Label").textContent = isCb ? "Voice Clone" : "Voice Blend";
  $("kokoroTab01").classList.toggle("hidden", isCb);
  $("chatterboxTab01").classList.toggle("hidden", !isCb);
  document.querySelectorAll(".kokoro-only").forEach((el) => el.classList.toggle("hidden", isCb));

  $("btnTogglePhonemes").classList.toggle("hidden", isCb);
  if (isCb) $("phonemeDrawer").classList.add("hidden");
  $("langSelect").classList.toggle("hidden", isCb);
  $("btnMatchRef").classList.toggle("hidden", isCb);
  $("abRawLabel").textContent = isCb ? "Raw Chatterbox" : "Raw Kokoro";
  $("abTunedLabel").textContent = isCb ? "Mastered" : "Tuned + Mastered";

  renderPresets();
  if (isCb) {
    renderReferences();
    ensureChatterboxLoaded();
  }
  updateEngineStatusPill();
  updateLiveSummary();
}

// ---------- Chatterbox Reference Voice Library ----------

async function loadReferences() {
  try {
    const res = await fetch("/api/references");
    if (res.ok) {
      const data = await res.json();
      state.references = data.references || [];
      renderReferences();
    }
  } catch (_) {}
}

function renderReferences() {
  const list = $("cbRefList");
  if (!list) return;
  list.innerHTML = "";

  // Built-in default option
  const defItem = document.createElement("div");
  defItem.className = `ref-item ${state.activeRefId === null ? "active" : ""}`;
  defItem.innerHTML = `
    <div class="ref-item-left">
      <span class="ref-radio"></span>
      <div>
        <div class="ref-name">Chatterbox Built-in Default</div>
        <div class="ref-meta">Pre-computed voice · No clip required</div>
      </div>
    </div>
  `;
  defItem.addEventListener("click", () => {
    state.activeRefId = null;
    renderReferences();
    updateLiveSummary();
    scheduleLiveSynthesize(400);
  });
  makePressable(defItem, !state.activeRefId);
  list.appendChild(defItem);

  state.references.forEach((ref) => {
    const row = document.createElement("div");
    row.className = `ref-item ${state.activeRefId === ref.id ? "active" : ""}`;
    row.innerHTML = `
      <div class="ref-item-left">
        <span class="ref-radio"></span>
        <div style="min-width:0;">
          <div class="ref-name">${escapeAttr(ref.name)}</div>
          <div class="ref-meta">${ref.duration_s}s · ${ref.f0_hz} Hz</div>
        </div>
      </div>
      <div class="ref-item-actions">
        <button class="btn btn-xs btn-ghost ref-play-btn" title="Preview reference audio">▶</button>
        <button class="slot-remove ref-del-btn" title="Delete reference">&times;</button>
      </div>
    `;
    row.addEventListener("click", (e) => {
      if (e.target.closest(".ref-play-btn") || e.target.closest(".ref-del-btn")) return;
      state.activeRefId = ref.id;
      renderReferences();
      updateLiveSummary();
      scheduleLiveSynthesize(400);
    });
    makePressable(row, state.activeRefId === ref.id);
    row.setAttribute("aria-label", `Use reference voice ${ref.name}`);
    row.querySelector(".ref-play-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      audioRefPreview.src = ref.audio_url;
      audioRefPreview.play().catch(() => {});
    });
    row.querySelector(".ref-del-btn").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/references/${ref.id}`, { method: "DELETE" });
      state.references = state.references.filter((r) => r.id !== ref.id);
      if (state.activeRefId === ref.id) state.activeRefId = null;
      renderReferences();
      updateLiveSummary();
      showToast("Reference clip removed");
    });
    list.appendChild(row);
  });
}

async function uploadReferenceFile(file) {
  if (!file) return;
  showToast(`Uploading "${file.name}"…`);
  try {
    const res = await fetch(`/api/references?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Upload failed");
    }
    const { reference } = await res.json();
    state.references.unshift(reference);
    state.activeRefId = reference.id;
    renderReferences();
    updateLiveSummary();
    showToast(`Added "${reference.name}" (${reference.duration_s}s) — ready to clone!`);
    if (isLiveSyncEnabled()) {
      scheduleLiveSynthesize(350, true);
    }
  } catch (err) {
    showToast(`Upload error: ${err.message}`);
  }
}

async function saveCurrentTakeAsRef() {
  if (!state.currentTake) {
    showToast("Synthesize a take first to save it as a reference.");
    return;
  }
  const res = await fetch("/api/references/from-take", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      take_id: state.currentTake.take_id,
      name: state.currentTake.label || "Studio Voice Ref",
      use_raw: false,
    }),
  });
  if (!res.ok) {
    showToast("Could not save current take as reference.");
    return;
  }
  const { reference } = await res.json();
  state.references.unshift(reference);
  state.activeRefId = reference.id;
  renderReferences();
  updateLiveSummary();
  showToast(`Saved "${reference.name}" to Reference Library`);
}

// ---------- Voice Blend UI ----------

function buildVoiceOptionsHtml(selectedId) {
  const groups = {};
  for (const v of state.voices) {
    const key = `${v.language} · ${v.gender}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(v);
  }
  return Object.entries(groups)
    .map(([label, list]) => {
      const opts = list
        .map(
          (v) =>
            `<option value="${v.id}" ${v.id === selectedId ? "selected" : ""}>${v.name} (${v.id}) · ${v.grade}</option>`
        )
        .join("");
      return `<optgroup label="${label}">${opts}</optgroup>`;
    })
    .join("");
}

function renderProsodyDonorSelect() {
  const cur = $("prosodyVoiceSelect").value || "af_heart";
  $("prosodyVoiceSelect").innerHTML = buildVoiceOptionsHtml(cur);
}

function renderBlendUI() {
  const total = state.blend.reduce((acc, s) => acc + Math.max(0, s.weight), 0) || 1;
  const bar = $("blendBar");
  bar.innerHTML = state.blend
    .map((slot, idx) => {
      const pct = (Math.max(0, slot.weight) / total) * 100;
      const color = SLOT_COLORS[idx % SLOT_COLORS.length];
      return `<div class="blend-seg" style="width:${pct.toFixed(2)}%;background:${color};"></div>`;
    })
    .join("");

  const container = $("blendSlots");
  container.innerHTML = "";

  state.blend.forEach((slot, idx) => {
    const pct = Math.round((Math.max(0, slot.weight) / total) * 100);
    const color = SLOT_COLORS[idx % SLOT_COLORS.length];
    const meta = state.voiceMap[slot.voice];

    const div = document.createElement("div");
    div.className = "blend-slot";
    div.innerHTML = `
      <div class="slot-top">
        <span class="slot-color" style="background:${color}"></span>
        <select class="slot-select" aria-label="Blend voice ${idx + 1}">${buildVoiceOptionsHtml(slot.voice)}</select>
        <span class="slot-pct">${pct}%</span>
        ${state.blend.length > 1 ? `<button class="slot-remove" title="Remove voice" aria-label="Remove blend voice ${idx + 1}">&times;</button>` : ""}
      </div>
      <input type="range" class="slot-weight" aria-label="Blend voice ${idx + 1} weight" min="0.05" max="1" step="0.05" value="${slot.weight}" />
      <div class="slot-desc">${meta ? `${meta.accent} ${meta.gender} — ${meta.description}` : ""}</div>
    `;

    div.querySelector(".slot-select").addEventListener("change", (e) => {
      state.blend[idx].voice = e.target.value;
      // Auto-sync language if primary slot changes between US and UK
      if (idx === 0 && state.voiceMap[e.target.value]) {
        const lc = state.voiceMap[e.target.value].lang_code;
        if (lc) $("langSelect").value = lc;
      }
      state.activePresetId = null;
      renderPresets();
      renderBlendUI();
      scheduleLiveSynthesize(220);
    });

    div.querySelector(".slot-weight").addEventListener("input", (e) => {
      state.blend[idx].weight = parseFloat(e.target.value);
      state.activePresetId = null;
      // Update percentages & bar smoothly without destroying slider focus
      const tot = state.blend.reduce((acc, s) => acc + Math.max(0, s.weight), 0) || 1;
      const pctEls = container.querySelectorAll(".slot-pct");
      const segs = bar.querySelectorAll(".blend-seg");
      state.blend.forEach((s, i) => {
        const p = (Math.max(0, s.weight) / tot) * 100;
        if (pctEls[i]) pctEls[i].textContent = `${Math.round(p)}%`;
        if (segs[i]) segs[i].style.width = `${p.toFixed(2)}%`;
      });
      updateLiveSummary();
      scheduleLiveSynthesize(340);
    });

    const rm = div.querySelector(".slot-remove");
    if (rm) {
      rm.addEventListener("click", () => {
        state.blend.splice(idx, 1);
        state.activePresetId = null;
        renderPresets();
        renderBlendUI();
        scheduleLiveSynthesize(220);
      });
    }

    container.appendChild(div);
  });

  $("btnAddVoice").disabled = state.blend.length >= 4;
  updateLiveSummary();
}

// ---------- Pronunciation Rules UI ----------

function renderRulesUI() {
  const list = $("rulesList");
  list.innerHTML = "";
  if (state.pronunciationRules.length === 0) {
    list.innerHTML = `<div class="section-desc">No rules yet. Click "+ Add Rule" to customize pronunciation.</div>`;
    return;
  }
  state.pronunciationRules.forEach((rule, idx) => {
    const row = document.createElement("div");
    row.className = "rule-row";
    row.innerHTML = `
      <input type="text" class="text-input rule-find" aria-label="Word to replace" placeholder="Word (e.g. Kokoro)" value="${escapeAttr(rule.find)}" />
      <span class="rule-arrow">→</span>
      <input type="text" class="text-input rule-replace" aria-label="Say it as" placeholder="Say as (e.g. Ko-ko-ro)" value="${escapeAttr(rule.replace)}" />
      <button class="slot-remove" title="Delete rule">&times;</button>
    `;
    row.querySelector(".rule-find").addEventListener("input", (e) => {
      state.pronunciationRules[idx].find = e.target.value;
      scheduleLiveSynthesize(550);
    });
    row.querySelector(".rule-replace").addEventListener("input", (e) => {
      state.pronunciationRules[idx].replace = e.target.value;
      scheduleLiveSynthesize(550);
    });
    row.querySelector(".slot-remove").addEventListener("click", () => {
      state.pronunciationRules.splice(idx, 1);
      renderRulesUI();
      scheduleLiveSynthesize(250);
    });
    list.appendChild(row);
  });
}

function escapeAttr(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------- Presets UI ----------

function applyChatterboxPreset(p) {
  const c = p.cb || {};
  $("cbExagSlider").value = c.exaggeration ?? 0.50;
  $("cbCfgSlider").value = c.cfg_weight ?? 0.50;
  $("cbTempSlider").value = c.temperature ?? 0.80;
  $("cbRepSlider").value = c.repetition_penalty ?? 1.20;
  $("sentencePauseSlider").value = c.sentence_pause ?? 0.30;
  $("paragraphPauseSlider").value = c.paragraph_pause ?? 0.60;

  if (p.mastering) {
    const m = p.mastering;
    $("masterEnabledCheck").checked = m.enabled ?? true;
    $("highpassSlider").value = m.highpass_hz ?? 80;
    $("warmthSlider").value = m.warmth_db ?? 2.0;
    $("presenceSlider").value = m.presence_db ?? 1.4;
    $("airSlider").value = m.air_db ?? 1.5;
    $("tubeWarmthSlider").value = m.tube_warmth ?? 0.14;
    $("formantSlider").value = m.formant_shift ?? 0.0;
    $("deesserSlider").value = m.deesser ?? 0.36;
    $("compressionSlider").value = m.compression ?? 0.34;
    $("roomSlider").value = m.room_ambience ?? 0.10;
    $("pitchSlider").value = m.pitch_semitones ?? 0.0;
    $("lufsSlider").value = m.target_lufs ?? -15.5;
    syncMasteringEnabledUI();
  }
  syncSliderReadouts();
}

// Make a non-button element behave like a button for keyboard users.
function makePressable(el, pressed) {
  el.tabIndex = 0;
  el.setAttribute("role", "button");
  if (pressed !== undefined) el.setAttribute("aria-pressed", pressed ? "true" : "false");
  el.addEventListener("keydown", (e) => {
    if (e.target !== el) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      el.click();
    }
  });
}

// Fade the preset ribbon's edges when presets are scrolled out of view.
function updatePresetFades() {
  const list = $("presetsList");
  const max = list.scrollWidth - list.clientWidth;
  list.classList.toggle("fade-left", max > 1 && list.scrollLeft > 1);
  list.classList.toggle("fade-right", max > 1 && list.scrollLeft < max - 1);
}

function renderPresets() {
  const list = $("presetsList");
  const refocusId = list.contains(document.activeElement) ? document.activeElement.dataset.presetId : null;
  list.innerHTML = "";
  const finish = () => {
    if (refocusId) {
      const again = [...list.querySelectorAll(".preset-pill")].find((el) => el.dataset.presetId === refocusId);
      if (again) again.focus();
    }
    updatePresetFades();
  };

  if (state.engine === "chatterbox") {
    CHATTERBOX_PRESETS.forEach((p) => {
      const btn = document.createElement("div");
      btn.className = `preset-pill ${p.id === state.activeCbPresetId ? "active" : ""}`;
      btn.title = p.description || p.name;
      btn.innerHTML = `<span>${escapeAttr(p.name)}</span><span class="preset-tag">${escapeAttr(p.tag)}</span>`;
      btn.dataset.presetId = p.id;
      makePressable(btn, p.id === state.activeCbPresetId);
      btn.addEventListener("click", () => {
        state.activeCbPresetId = p.id;
        applyChatterboxPreset(p);
        renderPresets();
        if (isLiveSyncEnabled()) {
          scheduleLiveSynthesize(120, true);
        } else {
          showToast(`Loaded "${p.name}"`);
        }
      });
      list.appendChild(btn);
    });
    finish();
    return;
  }

  state.presets.forEach((p) => {
    const btn = document.createElement("div");
    btn.className = `preset-pill ${p.id === state.activePresetId ? "active" : ""}`;
    btn.title = p.description || p.name;
    btn.innerHTML = `
      <span>${escapeAttr(p.name)}</span>
      <span class="preset-tag">${escapeAttr(p.tag || "Preset")}</span>
      ${!p.builtin ? `<button class="preset-del" title="Delete preset" aria-label="Delete preset ${escapeAttr(p.name)}">&times;</button>` : ""}
    `;
    btn.dataset.presetId = p.id;
    makePressable(btn, p.id === state.activePresetId);
    btn.addEventListener("click", (e) => {
      if (e.target.classList.contains("preset-del")) return;
      state.activePresetId = p.id;
      applyConfig(p.config, false);
      renderPresets();
      if (isLiveSyncEnabled()) {
        scheduleLiveSynthesize(80, true);
      } else {
        showToast(`Loaded "${p.name}" — press Synthesize to hear it`);
      }
    });
    const del = btn.querySelector(".preset-del");
    if (del) {
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/presets/${p.id}`, { method: "DELETE" });
        state.presets = state.presets.filter((x) => x.id !== p.id);
        if (state.activePresetId === p.id) state.activePresetId = null;
        renderPresets();
        showToast("Preset deleted");
      });
    }
    list.appendChild(btn);
  });
  finish();
}

// ---------- Play-as-it's-made preview ----------
// While a take is being synthesized, the server streams each finished sentence and we
// play it straight away through Web Audio. When the finished take arrives it is loaded
// into the normal player in the background; the preview keeps playing to the end, and
// pausing, seeking or switching to Raw hands over to the player at the same spot.

const preview = {
  ctx: null,
  active: false,
  final: false, // the finished take has arrived and is loaded in the player
  items: [], // { src, at (context time), dur, tl (position in the take) }
  pending: [], // AudioBuffers received but not yet scheduled (see PREVIEW_START_BUFFER_S)
  nextAt: 0,
  timeline: 0,
  rafId: 0,
  startedAt: 0, // performance.now() when the request went out
};
let waveformEmptyHtml = null;

function setPreviewNote(text) {
  const el = $("waveformEmpty");
  if (waveformEmptyHtml === null) waveformEmptyHtml = el.innerHTML;
  if (text) {
    el.textContent = text;
    el.classList.remove("hidden");
  } else {
    el.innerHTML = waveformEmptyHtml;
    if (state.currentTake) el.classList.add("hidden");
  }
}

function startPreview() {
  stopPreview();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return false;
  // Created/resumed inside the click handler so the browser allows sound.
  if (!preview.ctx) preview.ctx = new Ctx();
  if (preview.ctx.state === "suspended") preview.ctx.resume().catch(() => {});
  preview.active = true;
  preview.final = false;
  preview.items = [];
  preview.pending = [];
  preview.nextAt = 0;
  preview.timeline = 0;
  preview.startedAt = performance.now();
  return true;
}

function stopPreview() {
  if (!preview.active) return;
  preview.active = false;
  preview.final = false;
  cancelAnimationFrame(preview.rafId);
  preview.items.forEach((it) => {
    try {
      it.src.stop();
    } catch (_) {}
    it.src.disconnect();
  });
  preview.items = [];
  preview.pending = [];
  setPreviewNote(null);
}

function decodePcm16(b64) {
  const bin = atob(b64);
  const n = bin.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    if (v >= 32768) v -= 65536;
    out[i] = v / 32768;
  }
  return out;
}

// A short opening line ("Hey, welcome back.") can finish playing before a long second
// sentence is ready, leaving a gap. Hold playback until this much audio is in hand
// (or the take is complete).
const PREVIEW_START_BUFFER_S = 2.5;

function queuePreviewChunk(chunk) {
  if (!preview.active) return;
  const ctx = preview.ctx;
  const samples = decodePcm16(chunk.pcm16);
  if (!samples.length) return;
  const buf = ctx.createBuffer(1, samples.length, chunk.sample_rate);
  buf.copyToChannel(samples, 0);
  preview.pending.push(buf);
  const buffered = preview.pending.reduce((s, b) => s + b.duration, 0);
  if (
    preview.items.length ||
    (buffered >= PREVIEW_START_BUFFER_S && canPlayThrough(buffered, chunk.progress))
  ) {
    flushPreviewPending();
  } else {
    setPreviewNote(`Making the take… ${Math.floor(buffered)} s ready, starts once it can play without pauses`);
  }
}

// An engine slower than real time (Chatterbox on a busy Mac) would run dry partway and
// leave long silences. From how fast audio has come so far and how much of the text is
// done ("progress"), wait until what's in hand lasts until the rest is made.
function canPlayThrough(buffered, progress) {
  if (progress == null || progress >= 1) return true;
  const elapsed = (performance.now() - preview.startedAt) / 1000;
  if (elapsed <= 0 || progress <= 0) return false;
  const rate = buffered / elapsed; // seconds of audio made per second
  if (rate >= 1.1) return true;
  const remainingMake = (elapsed * (1 - progress)) / progress;
  // Audio arrives a sentence at a time, so keep a margin.
  return buffered >= remainingMake * (1 - rate) * 1.15 + 1.5;
}

function flushPreviewPending() {
  const bufs = preview.pending;
  preview.pending = [];
  bufs.forEach(schedulePreviewBuffer);
}

function schedulePreviewBuffer(buf) {
  const ctx = preview.ctx;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  // Back to back; if a sentence arrives late, start it as soon as possible.
  const at = Math.max(preview.nextAt, ctx.currentTime + 0.06);
  src.start(at);
  preview.items.push({ src, at, dur: buf.duration, tl: preview.timeline });
  preview.nextAt = at + buf.duration;
  preview.timeline += buf.duration;

  if (!preview.final) $("timeTotal").textContent = fmtTime(preview.timeline);
  if (preview.items.length === 1) {
    audioTuned.pause();
    audioRaw.pause();
    $("btnPlayPause").disabled = false;
    updatePlayButtonUI(ctx.state === "running");
    const canvas = $("waveformCanvas");
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    setPreviewNote("Playing as it's made…");
    cancelAnimationFrame(state.rafId);
    cancelAnimationFrame(preview.rafId);
    preview.rafId = requestAnimationFrame(previewTick);
  }
}

// Position in the take being heard, or null once everything queued has played.
function previewPosition() {
  const now = preview.ctx.currentTime;
  for (const it of preview.items) {
    if (now < it.at) return it.tl; // waiting for a late sentence
    if (now < it.at + it.dur) return it.tl + (now - it.at);
  }
  return null;
}

function previewTick() {
  if (!preview.active) return;
  const pos = previewPosition();
  if (pos === null) {
    if (preview.final) {
      finishPreview();
      return;
    }
    setPreviewNote("Still making the rest…");
  } else {
    $("timeCurrent").textContent = fmtTime(pos);
    if (preview.final) drawWaveform();
    else if (preview.items.length) setPreviewNote("Playing as it's made…");
  }
  preview.rafId = requestAnimationFrame(previewTick);
}

function finishPreview() {
  stopPreview();
  audioTuned.currentTime = 0;
  $("timeCurrent").textContent = fmtTime(0);
  updatePlayButtonUI(false);
  drawWaveform();
}

// Called once the finished take is loaded in the player.
function previewTakeArrived() {
  preview.final = true;
  setPreviewNote(null);
  if (preview.ctx.state !== "running") {
    handPreviewToPlayer(false); // paused: leave the player paused at the same spot
  } else if (previewPosition() === null) {
    finishPreview();
  } else {
    updatePlayButtonUI(true);
  }
}

// Stop the preview and continue (or pause) in the normal player at the same spot.
function handPreviewToPlayer(keepPlaying) {
  if (!preview.active || !preview.final) return;
  const pos = previewPosition();
  const playing = keepPlaying && preview.ctx.state === "running" && pos !== null;
  stopPreview();
  audioTuned.currentTime = pos ?? 0;
  $("timeCurrent").textContent = fmtTime(pos ?? 0);
  if (playing) {
    audioTuned.play().then(() => {
      updatePlayButtonUI(true);
      startPlaybackLoop();
    }).catch(() => {});
  } else {
    updatePlayButtonUI(false);
    drawWaveform();
  }
}

// POST to a *-stream endpoint: play sentences as they come, resolve with the finished take.
async function streamSynthesis(url, payload, signal, seq) {
  const playable = startPreview();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Synthesis failed");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line || seq !== state.synthSeq) continue;
      const ev = JSON.parse(line);
      if (ev.type === "chunk") {
        if (playable) queuePreviewChunk(ev);
      } else if (ev.type === "done") {
        // A short script may finish before anything was played; then the finished take
        // simply plays in the normal player.
        delete ev.type;
        return ev;
      } else if (ev.type === "error") {
        throw new Error(ev.detail || "Synthesis failed");
      }
    }
  }
  throw new Error("Synthesis stopped before the take was finished");
}

// ---------- Waveform Canvas & Playback ----------

function drawWaveform() {
  const canvas = $("waveformCanvas");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(10, Math.floor(rect.width * dpr));
  const h = Math.max(10, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  const take = state.currentTake;
  if (!take) return;
  // While a new take is still streaming, the old take's waveform would be misleading.
  if (preview.active && !preview.final) return;

  const isRaw = state.activeMode === "raw";
  const peaks = isRaw && take.raw_peaks ? take.raw_peaks : take.tuned_peaks;
  if (!peaks || !peaks.length) return;

  const audio = getActiveAudio();
  const dur = audio.duration || (isRaw ? take.raw_duration : take.tuned_duration) || 1;
  const now = preview.active ? previewPosition() ?? dur : audio.currentTime || 0;
  const progress = Math.min(1, Math.max(0, now / dur));

  const barCount = peaks.length;
  const gap = Math.max(1.5 * dpr, (w / barCount) * 0.3);
  const barW = Math.max(1.5 * dpr, (w - gap * (barCount - 1)) / barCount);
  const midY = h / 2;
  const maxBarH = h * 0.82;

  const activeColor = isRaw ? "#60a5fa" : "#f5a623";
  const inactiveColor = isRaw ? "rgba(96, 165, 250, 0.24)" : "rgba(245, 166, 35, 0.22)";

  // Subtle center line
  ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
  ctx.fillRect(0, midY - 0.5 * dpr, w, 1 * dpr);

  for (let i = 0; i < barCount; i++) {
    const x = i * (barW + gap);
    const barH = Math.max(3 * dpr, peaks[i] * maxBarH);
    const ratio = i / barCount;
    ctx.fillStyle = ratio <= progress ? activeColor : inactiveColor;
    const y = midY - barH / 2;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, barW / 2);
    ctx.fill();
  }

  // Playhead indicator
  if (progress > 0 && progress < 1) {
    const px = progress * w;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(px - 1 * dpr, 4 * dpr, 2 * dpr, h - 8 * dpr);
  }
}

function startPlaybackLoop() {
  cancelAnimationFrame(state.rafId);
  const tick = () => {
    const audio = getActiveAudio();
    $("timeCurrent").textContent = fmtTime(audio.currentTime);
    drawWaveform();
    if (!audio.paused && !audio.ended) {
      state.rafId = requestAnimationFrame(tick);
    }
  };
  state.rafId = requestAnimationFrame(tick);
}

function updatePlayButtonUI(playing) {
  state.isPlaying = playing;
  $("iconPlay").classList.toggle("hidden", playing);
  $("iconPause").classList.toggle("hidden", !playing);
  $("btnPlayPause").classList.toggle("mode-raw", state.activeMode === "raw");
}

function setActiveMode(mode, preservePlayhead = true) {
  if (!state.currentTake) return;
  if (mode === "raw" && !state.currentTake.raw_url) return;

  const prevAudio = getActiveAudio();
  const wasPlaying = !prevAudio.paused && !prevAudio.ended;
  const frac =
    preservePlayhead && prevAudio.duration > 0 ? prevAudio.currentTime / prevAudio.duration : 0;

  prevAudio.pause();
  state.activeMode = mode;

  $("btnModeRaw").classList.toggle("active", mode === "raw");
  $("btnModeTuned").classList.toggle("active", mode === "tuned");
  $("btnPlayPause").classList.toggle("mode-raw", mode === "raw");
  $("telMode").textContent = mode === "raw" ? `Raw (${state.currentTake.primary_voice})` : "Tuned + Mastered";

  const nextAudio = getActiveAudio();
  const nextDur =
    nextAudio.duration ||
    (mode === "raw" ? state.currentTake.raw_duration : state.currentTake.tuned_duration) ||
    0;
  $("timeTotal").textContent = fmtTime(nextDur);

  const seekAndPlay = () => {
    if (preservePlayhead && nextAudio.duration > 0) {
      nextAudio.currentTime = Math.min(nextAudio.duration - 0.02, frac * nextAudio.duration);
    }
    $("timeCurrent").textContent = fmtTime(nextAudio.currentTime);
    drawWaveform();
    if (wasPlaying) {
      nextAudio.play().catch(() => {});
      updatePlayButtonUI(true);
      startPlaybackLoop();
    } else if (!preview.active) {
      updatePlayButtonUI(false);
    }
  };

  if (nextAudio.readyState >= 1) {
    seekAndPlay();
  } else {
    nextAudio.addEventListener("loadedmetadata", seekAndPlay, { once: true });
  }
}

function loadTake(take, autoplay = false) {
  audioTuned.pause();
  audioRaw.pause();
  state.currentTake = take;

  audioTuned.src = take.tuned_url;
  audioTuned.load();

  if (take.raw_url) {
    audioRaw.src = take.raw_url;
    audioRaw.load();
    $("btnModeRaw").disabled = false;
  } else {
    audioRaw.removeAttribute("src");
    $("btnModeRaw").disabled = true;
    state.activeMode = "tuned";
  }

  $("btnModeTuned").disabled = false;
  $("btnPlayPause").disabled = false;
  $("btnDownload").disabled = false;
  $("waveformEmpty").classList.add("hidden");

  $("telInfer").textContent = `${take.infer_ms} ms`;
  $("telDsp").textContent = `${take.dsp_ms} ms`;
  $("telRtf").textContent = `${take.rtf}×`;
  $("phonemeInput").value = take.phonemes || "";

  setActiveMode("tuned", false);
  renderHistory();

  if (autoplay) {
    audioTuned.play().then(() => {
      updatePlayButtonUI(true);
      startPlaybackLoop();
    }).catch(() => {});
  }
}

function renderHistory() {
  const strip = $("historyStrip");
  const pills = $("historyPills");
  if (state.history.length <= 1) {
    strip.classList.add("hidden");
    return;
  }
  strip.classList.remove("hidden");
  pills.innerHTML = "";
  state.history.forEach((item, idx) => {
    const btn = document.createElement("button");
    btn.className = `history-pill ${state.currentTake && state.currentTake.take_id === item.take_id ? "active" : ""}`;
    btn.textContent = `#${state.history.length - idx} · ${item.label} (${item.tuned_duration.toFixed(1)}s)`;
    btn.addEventListener("click", () => {
      stopPreview();
      loadTake(item, false);
    });
    pills.appendChild(btn);
  });
}

// ---------- API Actions ----------

function scheduleLiveSynthesize(delayMs = 360, autoplayIfPaused = false) {
  if (!isLiveSyncEnabled()) return;
  const text = $("scriptInput").value.trim();
  const useCustomPh = $("useCustomPhonemesCheck").checked;
  if (!useCustomPh && !text) return;

  clearTimeout(state.liveSynthTimer);
  clearTimeout(state.remasterTimer);
  setSyncStatus(true, "Syncing voice…");
  state.liveSynthTimer = setTimeout(() => {
    synthesize({ isLive: true, autoplayIfPaused });
  }, delayMs);
}

async function synthesize(opts = {}) {
  const isLive = Boolean(opts && opts.isLive);
  const autoplayIfPaused = Boolean(opts && opts.autoplayIfPaused);

  const text = $("scriptInput").value.trim();
  const useCustomPh = $("useCustomPhonemesCheck").checked;
  const customPh = $("phonemeInput").value.trim();

  if (!useCustomPh && !text) {
    if (!isLive) showToast("Enter some script text first.");
    return;
  }

  // Abort any older in-flight synthesis so rapid slider moves never clash
  if (state.synthAbort) {
    state.synthAbort.abort();
  }
  const controller = new AbortController();
  state.synthAbort = controller;
  const seq = ++state.synthSeq;
  stopPreview();

  const btn = $("btnSynthesize");
  if (!isLive) {
    btn.disabled = true;
    $("synthSpinner").classList.remove("hidden");
    $("synthIcon").classList.add("hidden");
    $("synthLabel").textContent = "Synthesizing…";
  }
  setSyncStatus(true, isLive ? "Live syncing…" : "Synthesizing…");

  const wasPlaying = !audioTuned.paused && !audioTuned.ended;
  const prevFrac =
    audioTuned.duration > 0 ? Math.min(0.98, audioTuned.currentTime / audioTuned.duration) : 0;

  try {
    const isCb = state.engine === "chatterbox";
    let res;
    let data = null;
    let canReuseRaw = false;

    if (isCb) {
      const cbPayload = {
        text,
        ref_id: state.activeRefId,
        exaggeration: parseFloat($("cbExagSlider").value),
        cfg_weight: parseFloat($("cbCfgSlider").value),
        temperature: parseFloat($("cbTempSlider").value),
        repetition_penalty: parseFloat($("cbRepSlider").value),
        sentence_pause: parseFloat($("sentencePauseSlider").value),
        paragraph_pause: parseFloat($("paragraphPauseSlider").value),
        rhythm_humanize: parseFloat($("rhythmHumanizeSlider").value),
        smart_punctuation: $("smartPuncCheck").checked,
        pronunciation_rules: state.pronunciationRules.filter((r) => r.find.trim()),
        mastering: readMasteringConfig(),
      };
      if (isLive) {
        res = await fetch("/api/chatterbox/synthesize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(cbPayload),
          signal: controller.signal,
        });
      } else {
        data = await streamSynthesis(
          "/api/chatterbox/synthesize-stream", cbPayload, controller.signal, seq
        );
      }
    } else {
      const cfg = readFullConfig();
      const sortedBlend = [...cfg.blend].sort((a, b) => b.weight - a.weight);
      const expectedPrimary = sortedBlend[0]?.voice || "af_heart";

      // Always synthesize tuned audio first without blocking on Raw A/B!
      // If Raw A/B is checked, we fetch it silently right after tuned audio renders.
      canReuseRaw =
        state.currentTake &&
        state.currentTake.raw_url &&
        state.currentTake.engine !== "chatterbox" &&
        state.lastRawScript === text &&
        state.lastRawPrimary === expectedPrimary;

      const payload = {
        text,
        ...cfg,
        pronunciation_rules: state.pronunciationRules.filter((r) => r.find.trim()),
        use_custom_phonemes: useCustomPh,
        custom_phonemes: customPh,
        include_raw: false,
      };

      if (isLive) {
        // Slider-driven re-syncs swap the take in place at the same spot, so no preview.
        res = await fetch("/api/synthesize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
      } else {
        data = await streamSynthesis("/api/synthesize-stream", payload, controller.signal, seq);
      }
    }

    if (seq !== state.synthSeq) return;

    if (!data) {
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Synthesis failed");
      }
      data = await res.json();
      if (seq !== state.synthSeq) return;
    }

    if (isCb) {
      state.chatterboxStatus = "ready";
      updateEngineStatusPill();
    }

    if (canReuseRaw && state.currentTake) {
      data.raw_url = state.currentTake.raw_url;
      data.raw_peaks = state.currentTake.raw_peaks;
      data.raw_duration = state.currentTake.raw_duration;
    } else if (!isCb && $("includeRawCheck").checked) {
      // Kick off non-blocking background Raw A/B fetch so playback starts 2x sooner!
      const rawTakeId = data.take_id;
      const rawLang = $("langSelect").value;
      const rawVoice = data.primary_voice;
      fetch("/api/synthesize-raw", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ take_id: rawTakeId, text, voice: rawVoice, lang: rawLang }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((rawRes) => {
          if (!rawRes) return;
          state.lastRawScript = text;
          state.lastRawPrimary = rawVoice;
          data.raw_url = rawRes.raw_url;
          data.raw_peaks = rawRes.raw_peaks;
          data.raw_duration = rawRes.raw_duration;
          if (state.currentTake && state.currentTake.take_id === rawTakeId) {
            state.currentTake.raw_url = rawRes.raw_url;
            state.currentTake.raw_peaks = rawRes.raw_peaks;
            state.currentTake.raw_duration = rawRes.raw_duration;
            audioRaw.src = rawRes.raw_url;
            audioRaw.load();
            $("btnModeRaw").disabled = false;
          }
        })
        .catch(() => {});
    }

    if (isCb) {
      const refObj = state.references.find((r) => r.id === state.activeRefId);
      data.label = refObj ? `CB · ${refObj.name}` : "Chatterbox";
    } else {
      const presetObj = state.presets.find((p) => p.id === state.activePresetId);
      data.label = presetObj ? presetObj.name : data.primary_voice;
    }

    if (isLive && state.currentTake) {
      // Update current take in place smoothly without flooding history
      state.currentTake = data;
      if (state.history.length > 0) {
        state.history[0] = data;
      } else {
        state.history.unshift(data);
      }

      audioTuned.src = data.tuned_url;
      audioTuned.load();
      if (data.raw_url && !canReuseRaw) {
        audioRaw.src = data.raw_url;
        audioRaw.load();
        $("btnModeRaw").disabled = false;
      }
      $("btnModeTuned").disabled = false;
      $("btnPlayPause").disabled = false;
      $("btnDownload").disabled = false;
      $("waveformEmpty").classList.add("hidden");

      $("telInfer").textContent = `${data.infer_ms} ms (live)`;
      $("telDsp").textContent = `${data.dsp_ms} ms`;
      $("telRtf").textContent = `${data.rtf}×`;
      $("phonemeInput").value = data.phonemes || "";
      renderHistory();

      audioTuned.addEventListener(
        "loadedmetadata",
        () => {
          if (state.activeMode === "tuned") {
            $("timeTotal").textContent = fmtTime(audioTuned.duration);
            if (wasPlaying) {
              audioTuned.currentTime = Math.min(
                Math.max(0, audioTuned.duration - 0.05),
                prevFrac * audioTuned.duration
              );
              audioTuned.play().catch(() => {});
              updatePlayButtonUI(true);
              startPlaybackLoop();
            } else if (autoplayIfPaused) {
              audioTuned.currentTime = 0;
              audioTuned.play().then(() => {
                updatePlayButtonUI(true);
                startPlaybackLoop();
              }).catch(() => {});
            } else {
              $("timeCurrent").textContent = fmtTime(0);
              drawWaveform();
            }
          }
        },
        { once: true }
      );
      drawWaveform();
    } else {
      state.history.unshift(data);
      if (state.history.length > 8) state.history.pop();
      if (preview.active && preview.items.length) {
        loadTake(data, false);
        previewTakeArrived();
      } else {
        stopPreview();
        loadTake(data, true);
      }
    }

    setSyncStatus(false, `Live · ${(data.infer_ms / 1000).toFixed(2)}s`);
  } catch (err) {
    if (err.name === "AbortError") return;
    if (seq === state.synthSeq) stopPreview();
    setSyncStatus(false, "Error");
    showToast(`Error: ${err.message}`);
  } finally {
    if (seq === state.synthSeq) {
      btn.disabled = false;
      $("synthSpinner").classList.add("hidden");
      $("synthIcon").classList.remove("hidden");
      $("synthLabel").textContent = "Synthesize";
    }
  }
}

function scheduleRemaster() {
  if (!isLiveSyncEnabled()) return;
  if (!state.currentTake) {
    scheduleLiveSynthesize(260);
    return;
  }
  clearTimeout(state.remasterTimer);
  setSyncStatus(true, "Re-mastering…");
  state.remasterTimer = setTimeout(runRemaster, 140);
}

async function runRemaster() {
  if (!state.currentTake) return;
  const takeId = state.currentTake.take_id;

  if (state.remasterAbort) {
    state.remasterAbort.abort();
  }
  const controller = new AbortController();
  state.remasterAbort = controller;
  const seq = ++state.remasterSeq;

  try {
    const res = await fetch("/api/remaster", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        take_id: takeId,
        mastering: readMasteringConfig(),
      }),
      signal: controller.signal,
    });
    if (seq !== state.remasterSeq || !res.ok) return;
    const data = await res.json();
    if (seq !== state.remasterSeq || !state.currentTake || state.currentTake.take_id !== takeId) return;

    handPreviewToPlayer(true); // so the re-mastered audio carries on from the same spot
    const wasPlaying = state.activeMode === "tuned" && !audioTuned.paused && !audioTuned.ended;
    const pos = audioTuned.currentTime || 0;

    state.currentTake.tuned_url = data.tuned_url;
    state.currentTake.tuned_peaks = data.tuned_peaks;
    state.currentTake.tuned_duration = data.tuned_duration;
    state.currentTake.dsp_ms = data.dsp_ms;

    $("telDsp").textContent = `${data.dsp_ms} ms (live)`;
    setSyncStatus(false, `Mastered · ${Math.round(data.dsp_ms)}ms`);
    audioTuned.src = data.tuned_url;
    audioTuned.load();

    audioTuned.addEventListener(
      "loadedmetadata",
      () => {
        if (state.activeMode === "tuned") {
          $("timeTotal").textContent = fmtTime(audioTuned.duration);
          audioTuned.currentTime = Math.min(pos, Math.max(0, audioTuned.duration - 0.05));
          drawWaveform();
          if (wasPlaying) {
            audioTuned.play().catch(() => {});
            startPlaybackLoop();
          }
        }
      },
      { once: true }
    );
  } catch (err) {
    if (err.name !== "AbortError") {
      setSyncStatus(false, "Ready");
    }
  }
}

async function refreshPhonemes() {
  const text = $("scriptInput").value.trim();
  if (!text) return;
  const res = await fetch("/api/phonemize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      lang: $("langSelect").value,
      smart_punctuation: $("smartPuncCheck").checked,
      pronunciation_rules: state.pronunciationRules.filter((r) => r.find.trim()),
    }),
  });
  if (res.ok) {
    const data = await res.json();
    $("phonemeInput").value = data.phonemes || "";
  }
}

// ---------- Export Voice for Apps ----------

function readCurrentEngineConfigForExport() {
  if (state.engine === "chatterbox") {
    return {
      ref_id: state.activeRefId,
      exaggeration: parseFloat($("cbExagSlider").value),
      cfg_weight: parseFloat($("cbCfgSlider").value),
      temperature: parseFloat($("cbTempSlider").value),
      repetition_penalty: parseFloat($("cbRepSlider").value),
      sentence_pause: parseFloat($("sentencePauseSlider").value),
      paragraph_pause: parseFloat($("paragraphPauseSlider").value),
      rhythm_humanize: parseFloat($("rhythmHumanizeSlider").value),
      smart_punctuation: $("smartPuncCheck").checked,
      mastering: readMasteringConfig(),
    };
  }
  return readFullConfig();
}

async function loadExportedVoices() {
  try {
    const res = await fetch("/api/exported-voices");
    if (!res.ok) return;
    const data = await res.json();
    state.exportedVoices = data.voices || [];
    if (!state.selectedExportSlug && state.exportedVoices.length > 0) {
      state.selectedExportSlug = state.exportedVoices[0].slug;
    }
    renderExportedVoices();
    renderExportSnippet();
  } catch (_) {}
}

function getActiveExportVoiceSlug() {
  if (state.selectedExportSlug) return state.selectedExportSlug;
  if (state.activePresetId) return state.activePresetId;
  return "eleven-warm-host";
}

function renderExportedVoices() {
  const list = $("exportedVoicesList");
  if (!list) return;
  list.innerHTML = "";

  if (state.exportedVoices.length === 0) {
    list.innerHTML = `<div class="section-desc" style="padding:10px 4px;">No exported voices yet. Name your current voice above and click <strong>Export &amp; Register Voice</strong> — it immediately goes live on <code>/v1/audio/speech</code> and generates a downloadable ZIP app bundle.</div>`;
    return;
  }

  state.exportedVoices.forEach((v) => {
    const row = document.createElement("div");
    row.className = `exported-row ${v.slug === state.selectedExportSlug ? "active" : ""}`;
    row.innerHTML = `
      <div class="exported-info">
        <div class="exported-title-line">
          <strong>${escapeAttr(v.name)}</strong>
          <span class="exported-slug">${escapeAttr(v.slug)}</span>
          <span class="preset-tag">${v.engine === "chatterbox" ? "Chatterbox" : "Kokoro"}</span>
        </div>
      </div>
      <div class="exported-actions">
        <a class="btn btn-xs btn-accent-outline" href="/api/exported-voices/${encodeURIComponent(v.slug)}/bundle.zip" download title="Download self-contained ZIP with narrate.py, dsp.py, and voice files">ZIP Pack</a>
        <a class="btn btn-xs btn-ghost" href="/api/exported-voices/${encodeURIComponent(v.slug)}/json" download title="Download portable .timbra.json">.timbra.json</a>
        ${
          v.has_style_npy
            ? `<a class="btn btn-xs btn-ghost" href="/api/exported-voices/${encodeURIComponent(v.slug)}/style.npy" download title="Download baked (510, 1, 256) Kokoro style tensor">.npy</a>`
            : ""
        }
        <button class="btn btn-xs btn-ghost exp-load-btn" title="Load this exported voice back into the Studio">Load</button>
        <button class="slot-remove exp-del-btn" title="Delete exported voice">&times;</button>
      </div>
    `;

    row.addEventListener("click", (e) => {
      if (e.target.closest("a") || e.target.closest("button")) return;
      state.selectedExportSlug = v.slug;
      renderExportedVoices();
      renderExportSnippet();
    });
    makePressable(row, v.slug === state.selectedExportSlug);
    row.setAttribute("aria-label", `Show API snippet for ${v.name}`);

    row.querySelector(".exp-load-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      loadExportedVoiceIntoStudio(v);
    });

    row.querySelector(".exp-del-btn").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/exported-voices/${encodeURIComponent(v.slug)}`, { method: "DELETE" });
      state.exportedVoices = state.exportedVoices.filter((x) => x.slug !== v.slug);
      if (state.selectedExportSlug === v.slug) {
        state.selectedExportSlug = state.exportedVoices[0]?.slug || null;
      }
      renderExportedVoices();
      renderExportSnippet();
      showToast(`Removed "${v.name}"`);
    });

    list.appendChild(row);
  });
}

function loadExportedVoiceIntoStudio(v) {
  if (v.engine === "chatterbox") {
    setEngine("chatterbox");
    const c = v.config || {};
    $("cbExagSlider").value = c.exaggeration ?? 0.55;
    $("cbCfgSlider").value = c.cfg_weight ?? 0.50;
    $("cbTempSlider").value = c.temperature ?? 0.80;
    $("cbRepSlider").value = c.repetition_penalty ?? 1.20;
    $("sentencePauseSlider").value = c.sentence_pause ?? 0.30;
    $("paragraphPauseSlider").value = c.paragraph_pause ?? 0.60;
    if (c.rhythm_humanize !== undefined) $("rhythmHumanizeSlider").value = c.rhythm_humanize;
    if (c.mastering) applyConfig({ mastering: c.mastering }, false);
    else syncSliderReadouts();
  } else {
    setEngine("kokoro");
    applyConfig(v.config || {}, false);
  }
  if (Array.isArray(v.pronunciation_rules)) {
    state.pronunciationRules = v.pronunciation_rules.map((r) => ({ find: r.find, replace: r.replace }));
    renderRulesUI();
  }
  $("codeModal").classList.add("hidden");
  showToast(`Loaded "${v.name}" into Studio`);
  if (isLiveSyncEnabled()) scheduleLiveSynthesize(150, true);
}

function renderExportSnippet() {
  const block = $("exportCodeBlock");
  if (!block) return;
  const voiceId = getActiveExportVoiceSlug();
  const sampleText = ($("scriptInput").value.trim().split("\n")[0] || "Hello from my app!").slice(0, 120);
  const tab = state.exportCodeTab || "curl";

  if (tab === "curl") {
    block.textContent = `# Works with Open WebUI, SillyTavern, Obsidian, Raycast, or any OpenAI-compatible app:
# Base URL:  http://127.0.0.1:8765/v1
# Voice ID:  ${voiceId}
# Formats:   wav | mp3 | flac | ogg | pcm

curl http://127.0.0.1:8765/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify({
    model: "timbra",
    voice: voiceId,
    input: sampleText,
    response_format: "mp3",
    speed: 1.0,
  })}' \\
  --output narration.mp3

# List all available voices programmatically:
curl http://127.0.0.1:8765/v1/audio/voices`;
    return;
  }

  if (tab === "js") {
    block.textContent = `// Drop-in narration helper for any Web, Electron, or React app
export async function narrate(text, voice = "${voiceId}", format = "wav") {
  const res = await fetch("http://127.0.0.1:8765/v1/audio/speech", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "timbra",
      voice,
      input: text,
      response_format: format,
      speed: 1.0,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  const url = URL.createObjectURL(await res.blob());
  const audio = new Audio(url);
  audio.addEventListener("ended", () => URL.revokeObjectURL(url));
  await audio.play();
  return audio;
}

// Usage:
narrate(${JSON.stringify(sampleText)});`;
    return;
  }

  if (tab === "swift") {
    block.textContent = `import Foundation
import AVFoundation

final class TimbraNarrator {
    private var player: AVAudioPlayer?

    func speak(_ text: String, voice: String = "${voiceId}") async throws {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:8765/v1/audio/speech")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "timbra",
            "voice": voice,
            "input": text,
            "response_format": "wav",
            "speed": 1.0
        ])
        let (data, _) = try await URLSession.shared.data(for: req)
        await MainActor.run {
            self.player = try? AVAudioPlayer(data: data)
            self.player?.play()
        }
    }
}

// Usage:
// let narrator = TimbraNarrator()
// try await narrator.speak(${JSON.stringify(sampleText)})`;
    return;
  }

  if (tab === "python") {
    block.textContent = `# Works with the official OpenAI Python SDK (no API key needed locally)
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="timbra-local")

with client.audio.speech.with_streaming_response.create(
    model="timbra",
    voice="${voiceId}",
    input=${JSON.stringify(sampleText)},
    response_format="mp3",
) as response:
    response.stream_to_file("narration.mp3")

print("Saved narration.mp3")`;
    return;
  }

  block.textContent = generatePythonCode();
}

function generatePythonCode() {
  const m = readMasteringConfig();
  if (state.engine === "chatterbox") {
    const refObj = state.references.find((r) => r.id === state.activeRefId);
    const refLine = refObj
      ? `ref_audio="references/${refObj.id}.wav",  # ${refObj.name}`
      : `# ref_audio="path/to/reference.wav",  # optional 5-15s clip for zero-shot cloning`;
    return `import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model
from dsp import MasteringParams, apply_mastering

model = load_model("mlx-community/chatterbox-8bit")
text = ${JSON.stringify($("scriptInput").value.trim())}

chunks = [
    np.asarray(r.audio, dtype=np.float32).ravel()
    for r in model.generate(
        text=text,
        ${refLine}
        exaggeration=${parseFloat($("cbExagSlider").value).toFixed(2)},
        cfg_weight=${parseFloat($("cbCfgSlider").value).toFixed(2)},
        temperature=${parseFloat($("cbTempSlider").value).toFixed(2)},
        repetition_penalty=${parseFloat($("cbRepSlider").value).toFixed(2)},
    )
]
audio = np.concatenate(chunks)
sr = model.sample_rate

mastered = apply_mastering(
    audio,
    sr,
    MasteringParams(
        enabled=${m.enabled ? "True" : "False"},
        highpass_hz=${m.highpass_hz},
        warmth_db=${m.warmth_db},
        presence_db=${m.presence_db},
        air_db=${m.air_db},
        tube_warmth=${m.tube_warmth},
        formant_shift=${m.formant_shift},
        deesser=${m.deesser},
        compression=${m.compression},
        room_ambience=${m.room_ambience},
        pitch_semitones=${m.pitch_semitones},
        target_lufs=${m.target_lufs},
        humanize_enabled=${m.humanize_enabled ? "True" : "False"},
        breath_amount=${m.breath_amount},
        airiness=${m.airiness},
        smile=${m.smile},
        organic_jitter=${m.organic_jitter},
        vocal_fry=${m.vocal_fry},
        room_floor=${m.room_floor},
    ),
)

sf.write("chatterbox_mastered.wav", mastered, sr, subtype="PCM_16")
print("Saved chatterbox_mastered.wav")`;
  }

  const cfg = readFullConfig();
  const totalW = cfg.blend.reduce((a, b) => a + b.weight, 0) || 1;
  const blendLines = cfg.blend
    .map((b) => `    ("${b.voice}", ${(b.weight / totalW).toFixed(3)}),`)
    .join("\n");

  return `import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from dsp import MasteringParams, apply_mastering

kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

# 1. Build blended voice style vector
blend = [
${blendLines}
]
style = sum(w * kokoro.get_voice_style(name) for name, w in blend).astype(np.float32)
${
  cfg.split_mode
    ? `\n# Split Timbre / Prosody: borrow cadence from ${cfg.prosody_voice}\nstyle[:, :, 128:] = kokoro.get_voice_style("${cfg.prosody_voice}")[:, :, 128:]\n`
    : ""
}${
  cfg.style_offset !== 0
    ? `\n# Phrasing cadence offset (${cfg.style_offset})\nidx = np.clip(np.arange(style.shape[0]) + (${cfg.style_offset}), 0, style.shape[0] - 1)\nstyle = style[idx]\n`
    : ""
}
# 2. Synthesize with tuned pacing
text = ${JSON.stringify($("scriptInput").value.trim())}
audio, sr = kokoro.create(
    text,
    voice=style,
    speed=${cfg.speed},
    lang="${cfg.lang}",
    sentence_pause=${cfg.sentence_pause},
    clause_pause=${cfg.clause_pause},
)

# 3. Apply Studio Mastering chain
mastered = apply_mastering(
    audio,
    sr,
    MasteringParams(
        enabled=${cfg.mastering.enabled ? "True" : "False"},
        highpass_hz=${cfg.mastering.highpass_hz},
        warmth_db=${cfg.mastering.warmth_db},
        presence_db=${cfg.mastering.presence_db},
        air_db=${cfg.mastering.air_db},
        tube_warmth=${cfg.mastering.tube_warmth},
        formant_shift=${cfg.mastering.formant_shift},
        deesser=${cfg.mastering.deesser},
        compression=${cfg.mastering.compression},
        room_ambience=${cfg.mastering.room_ambience},
        pitch_semitones=${cfg.mastering.pitch_semitones},
        target_lufs=${cfg.mastering.target_lufs},
        humanize_enabled=${cfg.mastering.humanize_enabled ? "True" : "False"},
        breath_amount=${cfg.mastering.breath_amount},
        airiness=${cfg.mastering.airiness},
        smile=${cfg.mastering.smile},
        organic_jitter=${cfg.mastering.organic_jitter},
        vocal_fry=${cfg.mastering.vocal_fry},
        room_floor=${cfg.mastering.room_floor},
    ),
)

sf.write("kokoro_tuned.wav", mastered, sr, subtype="PCM_16")
print("Saved kokoro_tuned.wav")`;
}

// ---------- Wire Events & Boot ----------

function wireEvents() {
  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`panel-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // Sample script chips
  document.querySelectorAll("#sampleChips .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const key = chip.dataset.sample;
      if (SAMPLE_SCRIPTS[key]) {
        $("scriptInput").value = SAMPLE_SCRIPTS[key];
        updateCharCount();
      }
    });
  });

  const updateCharCount = () => {
    const t = $("scriptInput").value;
    const words = t.trim() ? t.trim().split(/\s+/).length : 0;
    $("charCount").textContent = `${t.length} chars · ${words} words`;
  };
  $("scriptInput").addEventListener("input", updateCharCount);
  updateCharCount();

  // Phoneme drawer
  $("btnTogglePhonemes").addEventListener("click", () => {
    const drawer = $("phonemeDrawer");
    drawer.classList.toggle("hidden");
    if (!drawer.classList.contains("hidden") && !$("phonemeInput").value.trim()) {
      refreshPhonemes();
    }
  });
  $("btnRefreshPhonemes").addEventListener("click", refreshPhonemes);

  // Synthesize button + Cmd/Ctrl+Enter shortcut
  $("btnSynthesize").addEventListener("click", synthesize);
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      synthesize();
    } else if (
      e.code === "Space" &&
      !document.activeElement?.closest(
        "button, [role='button'], a[href], input, textarea, select, [contenteditable='true']"
      )
    ) {
      if (state.currentTake || preview.active) {
        e.preventDefault();
        togglePlayPause();
      }
    }
  });

  // Playback controls
  const togglePlayPause = () => {
    if (preview.active) {
      if (preview.final) {
        handPreviewToPlayer(false); // pause in the real player; Play resumes from here
      } else if (preview.ctx.state === "running") {
        preview.ctx.suspend().catch(() => {});
        updatePlayButtonUI(false);
      } else {
        preview.ctx.resume().catch(() => {});
        updatePlayButtonUI(true);
      }
      return;
    }
    if (!state.currentTake) return;
    const audio = getActiveAudio();
    if (audio.paused || audio.ended) {
      audio.play().then(() => {
        updatePlayButtonUI(true);
        startPlaybackLoop();
      }).catch(() => {});
    } else {
      audio.pause();
      updatePlayButtonUI(false);
    }
  };

  $("btnPlayPause").addEventListener("click", togglePlayPause);
  const switchMode = (mode) => {
    if (preview.active && !preview.final) return; // the new take isn't ready yet
    handPreviewToPlayer(true);
    setActiveMode(mode, true);
  };
  $("btnModeRaw").addEventListener("click", () => switchMode("raw"));
  $("btnModeTuned").addEventListener("click", () => switchMode("tuned"));

  [audioTuned, audioRaw].forEach((el) => {
    el.addEventListener("ended", () => {
      updatePlayButtonUI(false);
      drawWaveform();
    });
  });

  // Waveform scrub click
  $("waveformWrap").addEventListener("click", (e) => {
    if (!state.currentTake) return;
    if (preview.active && !preview.final) return;
    handPreviewToPlayer(true); // seeking continues in the real player
    const rect = $("waveformWrap").getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const audio = getActiveAudio();
    if (audio.duration > 0) {
      audio.currentTime = frac * audio.duration;
      $("timeCurrent").textContent = fmtTime(audio.currentTime);
      drawWaveform();
    }
  });

  window.addEventListener("resize", drawWaveform);

  // Download active WAV
  $("btnDownload").addEventListener("click", () => {
    if (!state.currentTake) return;
    const isRaw = state.activeMode === "raw";
    const url = isRaw ? state.currentTake.raw_url : state.currentTake.tuned_url;
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `kokoro_${isRaw ? "raw" : "tuned"}_${state.currentTake.take_id}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  // Live Sync toggle pill
  const liveSyncCheck = $("liveSyncCheck");
  const livePill = document.querySelector(".live-toggle-pill");
  if (liveSyncCheck && livePill) {
    liveSyncCheck.addEventListener("change", () => {
      const on = liveSyncCheck.checked;
      livePill.classList.toggle("off", !on);
      $("liveSyncLabel").textContent = on ? "Live Sync" : "Manual";
      setSyncStatus(false, on ? "Ready" : "Manual mode");
    });
  }

  // Add voice to blend
  $("btnAddVoice").addEventListener("click", () => {
    if (state.blend.length >= 4) return;
    const used = new Set(state.blend.map((b) => b.voice));
    const nextVoice =
      state.voices.find((v) => !used.has(v.id))?.id || state.voices[0]?.id || "af_bella";
    state.blend.push({ voice: nextVoice, weight: 0.25 });
    state.activePresetId = null;
    renderPresets();
    renderBlendUI();
    scheduleLiveSynthesize(240);
  });

  // Split Timbre / Prosody toggle & donor voice
  $("splitModeCheck").addEventListener("change", () => {
    $("splitPickerWrap").classList.toggle("hidden", !$("splitModeCheck").checked);
    updateLiveSummary();
    scheduleLiveSynthesize(220);
  });
  $("prosodyVoiceSelect").addEventListener("change", () => {
    updateLiveSummary();
    scheduleLiveSynthesize(220);
  });

  // Delivery toggles & language selector
  $("continuousCheck").addEventListener("change", () => scheduleLiveSynthesize(220));
  $("smartPuncCheck").addEventListener("change", () => scheduleLiveSynthesize(220));
  $("langSelect").addEventListener("change", () => scheduleLiveSynthesize(220));

  // Pronunciation rule add
  $("btnAddRule").addEventListener("click", () => {
    state.pronunciationRules.push({ find: "", replace: "" });
    renderRulesUI();
  });

  // Voice style & pacing sliders -> live sync left panel!
  [
    "prosodyIntensity",
    "styleOffset",
    "speedSlider",
    "sentencePauseSlider",
    "clausePauseSlider",
    "paragraphPauseSlider",
  ].forEach((id) => {
    $(id).addEventListener("input", () => {
      syncSliderReadouts();
      state.activePresetId = null;
      scheduleLiveSynthesize(360);
    });
  });

  // Live mastering sliders -> instant re-master!
  [
    "warmthSlider",
    "presenceSlider",
    "airSlider",
    "tubeWarmthSlider",
    "formantSlider",
    "deesserSlider",
    "compressionSlider",
    "roomSlider",
    "pitchSlider",
    "highpassSlider",
    "lufsSlider",
  ].forEach((id) => {
    $(id).addEventListener("input", () => {
      syncSliderReadouts();
      scheduleRemaster();
    });
  });

  $("masterEnabledCheck").addEventListener("change", () => {
    syncMasteringEnabledUI();
    scheduleRemaster();
  });

  // Live Humanizer DSP sliders -> instant live re-master!
  [
    "breathSlider",
    "airinessSlider",
    "smileSlider",
    "jitterSlider",
    "vocalFrySlider",
    "roomFloorSlider",
  ].forEach((id) => {
    $(id).addEventListener("input", () => {
      document.querySelectorAll("#humanizePresets .chip").forEach((c) => c.classList.remove("active-chip"));
      syncSliderReadouts();
      scheduleRemaster();
    });
  });

  $("humanizeEnabledCheck").addEventListener("change", () => {
    syncHumanizeEnabledUI();
    scheduleRemaster();
  });

  $("rhythmHumanizeSlider").addEventListener("input", () => {
    document.querySelectorAll("#humanizePresets .chip").forEach((c) => c.classList.remove("active-chip"));
    syncSliderReadouts();
    scheduleLiveSynthesize(380);
  });

  document.querySelectorAll("#humanizePresets .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      applyHumanizePreset(chip.dataset.hpreset, true);
    });
  });

  $("btnResetMastering").addEventListener("click", () => {
    $("warmthSlider").value = 0;
    $("presenceSlider").value = 0;
    $("airSlider").value = 0;
    $("tubeWarmthSlider").value = 0;
    $("formantSlider").value = 0;
    $("deesserSlider").value = 0;
    $("compressionSlider").value = 0;
    $("roomSlider").value = 0;
    $("pitchSlider").value = 0;
    $("highpassSlider").value = 20;
    $("lufsSlider").value = -16;
    syncSliderReadouts();
    scheduleRemaster();
    showToast("Mastering chain reset to flat");
  });

  // Match Reference Voice Modal
  let pendingMatchFile = null;
  const setMatchFile = (file) => {
    if (!file) return;
    pendingMatchFile = file;
    $("matchFileLabel").textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
    $("btnRunMatch").disabled = false;
  };

  $("btnMatchRef").addEventListener("click", () => {
    $("matchModal").classList.remove("hidden");
  });

  $("matchFileInput").addEventListener("change", (e) => {
    setMatchFile(e.target.files?.[0]);
  });

  const dz = $("matchDropzone");
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("dragover");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    setMatchFile(e.dataTransfer.files?.[0]);
  });

  $("btnRunMatch").addEventListener("click", async () => {
    if (!pendingMatchFile) return;
    const btn = $("btnRunMatch");
    btn.disabled = true;
    btn.textContent = "Analyzing voice…";
    try {
      const accent = $("matchAccentSelect").value;
      const gender = $("matchGenderSelect").value;
      const url = `/api/match-reference?accent=${encodeURIComponent(accent)}&gender=${encodeURIComponent(gender)}&filename=${encodeURIComponent(pendingMatchFile.name)}`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: pendingMatchFile,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Reference analysis failed");
      }
      const { analysis, config } = await res.json();

      $("matchMetricsGrid").innerHTML = `
        <div class="match-stat"><span class="t-label">Median Pitch</span><span class="t-val">${analysis.f0_hz} Hz</span></div>
        <div class="match-stat"><span class="t-label">Expressiveness</span><span class="t-val">${analysis.f0_std_st} st</span></div>
        <div class="match-stat"><span class="t-label">Pace</span><span class="t-val">${analysis.syllables_per_sec} syl/s</span></div>
        <div class="match-stat"><span class="t-label">Detected</span><span class="t-val">${analysis.inferred_gender}</span></div>
      `;
      $("matchBlendSummary").textContent =
        "Matched Blend: " +
        analysis.closest_voices
          .map((v) => `${v.name} (${Math.round(v.weight * 100)}%)`)
          .join(" + ");
      $("matchResultBox").classList.remove("hidden");

      state.activePresetId = null;
      applyConfig(config, false);
      renderPresets();
      showToast("Auto-tuned to reference clip! Synthesizing preview…");
      setTimeout(() => {
        $("matchModal").classList.add("hidden");
        synthesize();
      }, 900);
    } catch (err) {
      showToast(`Match error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Analyze & Auto-Tune";
    }
  });

  // Engine Switcher
  $("btnEngineKokoro").addEventListener("click", () => setEngine("kokoro"));
  $("btnEngineChatterbox").addEventListener("click", () => setEngine("chatterbox"));

  // Chatterbox Reference Upload & Sliders
  $("cbRefUploadInput").addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) uploadReferenceFile(f);
    e.target.value = "";
  });
  const cbDz = $("cbRefDropzone");
  cbDz.addEventListener("dragover", (e) => {
    e.preventDefault();
    cbDz.classList.add("dragover");
  });
  cbDz.addEventListener("dragleave", () => cbDz.classList.remove("dragover"));
  cbDz.addEventListener("drop", (e) => {
    e.preventDefault();
    cbDz.classList.remove("dragover");
    const f = e.dataTransfer.files?.[0];
    if (f) uploadReferenceFile(f);
  });
  $("btnUseCurrentTakeRef").addEventListener("click", saveCurrentTakeAsRef);

  ["cbExagSlider", "cbCfgSlider", "cbTempSlider", "cbRepSlider"].forEach((id) => {
    $(id).addEventListener("input", () => {
      syncSliderReadouts();
      state.activeCbPresetId = null;
      scheduleLiveSynthesize(650);
    });
  });

  // Modals
  document.querySelectorAll("[data-close-modal]").forEach((btn) => {
    btn.addEventListener("click", () => $(btn.dataset.closeModal).classList.add("hidden"));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const open = [...document.querySelectorAll(".modal-backdrop")].filter((m) => !m.classList.contains("hidden"));
    if (open.length) {
      e.preventDefault();
      open[open.length - 1].classList.add("hidden");
    }
  });

  // Preset ribbon: edge fades + vertical wheel scrolls it sideways
  const presetRow = $("presetsList");
  presetRow.addEventListener("scroll", updatePresetFades, { passive: true });
  window.addEventListener("resize", updatePresetFades);
  presetRow.addEventListener(
    "wheel",
    (e) => {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
      if (presetRow.scrollWidth <= presetRow.clientWidth) return;
      e.preventDefault();
      presetRow.scrollLeft += e.deltaY;
    },
    { passive: false }
  );

  $("btnSavePreset").addEventListener("click", () => {
    $("presetNameInput").value = "";
    $("presetDescInput").value = "";
    $("presetModal").classList.remove("hidden");
    $("presetNameInput").focus();
  });

  $("btnConfirmSavePreset").addEventListener("click", async () => {
    const name = $("presetNameInput").value.trim();
    if (!name) {
      showToast("Give your preset a name first.");
      return;
    }
    const res = await fetch("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        tag: $("presetTagInput").value.trim() || "Custom",
        description: $("presetDescInput").value.trim(),
        config: readFullConfig(),
      }),
    });
    if (res.ok) {
      const { preset } = await res.json();
      state.presets.push(preset);
      state.activePresetId = preset.id;
      renderPresets();
      $("presetModal").classList.add("hidden");
      showToast(`Saved preset "${preset.name}"`);
    }
  });

  $("btnExportCode").addEventListener("click", () => {
    const presetObj = state.presets.find((p) => p.id === state.activePresetId);
    const defaultName =
      state.engine === "chatterbox"
        ? (state.references.find((r) => r.id === state.activeRefId)?.name || "Chatterbox Voice")
        : (presetObj ? presetObj.name : "Custom Narrator");
    if (!$("exportVoiceName").value.trim()) {
      $("exportVoiceName").value = defaultName;
    }
    loadExportedVoices();
    renderExportSnippet();
    $("codeModal").classList.remove("hidden");
  });

  $("btnRegisterExportVoice").addEventListener("click", async () => {
    const name = $("exportVoiceName").value.trim();
    if (!name) {
      showToast("Enter a voice name first.");
      return;
    }
    const btn = $("btnRegisterExportVoice");
    btn.disabled = true;
    btn.textContent = "Exporting…";
    try {
      const res = await fetch("/api/exported-voices", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          engine: state.engine,
          config: readCurrentEngineConfigForExport(),
          pronunciation_rules: state.pronunciationRules.filter((r) => r.find.trim()),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Export failed");
      }
      const { voice } = await res.json();
      state.selectedExportSlug = voice.slug;
      await loadExportedVoices();
      showToast(`Exported "${voice.slug}" — live at /v1/audio/speech!`);
    } catch (err) {
      showToast(`Export error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Export & Register Voice";
    }
  });

  $("importTimbraInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const res = await fetch("/api/exported-voices/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: file,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Import failed");
      }
      const { voice } = await res.json();
      state.selectedExportSlug = voice.slug;
      await loadExportedVoices();
      loadExportedVoiceIntoStudio(voice);
    } catch (err) {
      showToast(`Import error: ${err.message}`);
    }
  });

  document.querySelectorAll("#exportCodeTabs .code-tab").forEach((tabBtn) => {
    tabBtn.addEventListener("click", () => {
      document.querySelectorAll("#exportCodeTabs .code-tab").forEach((b) => b.classList.remove("active"));
      tabBtn.classList.add("active");
      state.exportCodeTab = tabBtn.dataset.lang;
      renderExportSnippet();
    });
  });

  $("btnCopyCode").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("exportCodeBlock").textContent);
    showToast("Copied snippet to clipboard");
  });
}

async function boot() {
  wireEvents();
  renderRulesUI();

  try {
    const [healthRes, voicesRes, presetsRes] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/voices"),
      fetch("/api/presets"),
    ]);

    const health = await healthRes.json();
    const { voices } = await voicesRes.json();
    const { presets } = await presetsRes.json();

    state.voices = voices;
    state.voiceMap = Object.fromEntries(voices.map((v) => [v.id, v]));
    state.presets = presets;
    const backendTag = health.kokoro_backend ? ` · ${health.kokoro_backend}` : "";
    state.kokoroHealthText = `${health.voices_count} voices${backendTag}`;
    state.chatterboxStatus = health.chatterbox_status || "not_loaded";
    updateEngineStatusPill();

    renderProsodyDonorSelect();
    renderPresets();
    loadReferences();

    if (presets.length > 0) {
      state.activePresetId = presets[0].id;
      applyConfig(presets[0].config, false);
      renderPresets();
    } else {
      renderBlendUI();
      syncSliderReadouts();
    }
  } catch (err) {
    $("engineStatusText").textContent = "Backend offline";
    showToast("Could not connect to Kokoro Studio server.");
  }
}

boot();
