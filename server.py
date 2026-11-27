"""Kokoro Studio — FastAPI backend for real-time Kokoro-82M voice tuning & mastering."""

from __future__ import annotations

import io
import json
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from kokoro_onnx import Kokoro
from pydantic import BaseModel, Field
from scipy import signal

from dsp import MasteringParams, analyze_voice_profile, apply_mastering, compute_waveform_peaks
from voice_export import (
    build_zip_bundle,
    delete_exported_voice,
    list_exported_voices,
    load_exported_voice,
    save_exported_voice,
)
from voices_meta import DEFAULT_PRESETS, build_voice_catalog

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REFERENCES_DIR = BASE_DIR / "references"
STATIC_DIR = BASE_DIR / "static"
PRESETS_FILE = BASE_DIR / "presets.json"
REFERENCES_INDEX = REFERENCES_DIR / "index.json"
FINGERPRINTS_FILE = MODELS_DIR / "voice_fingerprints.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 24000
MAX_CACHED_TAKES = 25
CHATTERBOX_MODEL_ID = "mlx-community/chatterbox-8bit"

app = FastAPI(title="Kokoro & Chatterbox Studio", version="2.0.0")

# The UI is served from this same server, so no cross-origin access is needed.
# Only accept requests addressed to a loopback host (blocks DNS rebinding) and,
# when the browser sends an Origin header, only from that same origin (blocks
# other websites from driving the API while the studio is running).
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    host = request.headers.get("host", "")
    hostname = host.rsplit(":", 1)[0] if not host.endswith("]") else host
    if hostname not in LOOPBACK_HOSTS:
        return JSONResponse({"detail": "Forbidden host."}, status_code=403)
    origin = request.headers.get("origin")
    if origin is not None and origin != f"{request.url.scheme}://{host}":
        return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
    return await call_next(request)

# Global engine state
kokoro: Kokoro | None = None
mlx_kokoro: Any = None
kokoro_backend: str = "ONNX CPU"
mlx_kokoro_lock = threading.Lock()
phoneme_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
voice_catalog: list[dict[str, Any]] = []
mean_prosody_vec: np.ndarray | None = None
take_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

# Lazy Chatterbox MLX engine state
chatterbox_model: Any = None
chatterbox_status: str = "not_loaded"  # not_loaded | loading | ready | error
chatterbox_error: str | None = None
chatterbox_lock = threading.Lock()
chatterbox_conds_cache: OrderedDict[str, Any] = OrderedDict()


def _init_engine() -> None:
    global kokoro, mlx_kokoro, kokoro_backend, voice_catalog, mean_prosody_vec
    model_path = MODELS_DIR / "kokoro-v1.0.onnx"
    voices_path = MODELS_DIR / "voices-v1.0.bin"
    if not model_path.exists() or not voices_path.exists():
        raise RuntimeError(
            f"Missing model files in {MODELS_DIR}. Expected kokoro-v1.0.onnx and voices-v1.0.bin."
        )
    kokoro = Kokoro(str(model_path), str(voices_path))
    voice_names = list(kokoro.voices.files)
    voice_catalog = build_voice_catalog(voice_names)

    # Precompute mean prosody half [:, :, 128:] across English voices for safe expressiveness scaling
    en_styles = [
        kokoro.get_voice_style(v)
        for v in voice_names
        if v.startswith(("af_", "am_", "bf_", "bm_"))
    ]
    if en_styles:
        stacked = np.stack(en_styles, axis=0)
        mean_prosody_vec = np.mean(stacked[:, :, :, 128:], axis=0).astype(np.float32)

    # Load native Apple Silicon MLX Metal GPU Kokoro-82M for ~3.5x faster inference
    mlx_dir = MODELS_DIR / "kokoro-mlx"
    if (mlx_dir / "config.json").exists() and (mlx_dir / "kokoro-v1_0.safetensors").exists():
        try:
            import mlx.core as mx
            from mlx_audio.tts.utils import load_model

            mlx_kokoro = load_model(str(mlx_dir))
            # Warm up Metal shaders with a tiny dummy utterance so first click is instant
            warm_ref = mx.array(kokoro.get_voice_style("af_heart")[8], dtype=mx.float32)
            warm_out = mlx_kokoro("həlˈoʊ.", warm_ref, speed=1.0)
            mx.eval(warm_out)
            kokoro_backend = "MLX Metal GPU"
        except Exception as exc:
            print(f"[Kokoro Studio] MLX Kokoro unavailable, using ONNX CPU: {exc}")
            mlx_kokoro = None
            kokoro_backend = "ONNX CPU"


@app.on_event("startup")
def on_startup() -> None:
    _init_engine()


# ---------- Pydantic Models ----------

class VoiceWeight(BaseModel):
    voice: str
    weight: float = Field(ge=0.0, le=1.0)


class MasteringRequest(BaseModel):
    enabled: bool = True
    highpass_hz: float = 75.0
    warmth_db: float = 2.0
    presence_db: float = 1.5
    air_db: float = 1.8
    tube_warmth: float = 0.0
    formant_shift: float = 0.0
    deesser: float = 0.35
    compression: float = 0.35
    room_ambience: float = 0.12
    pitch_semitones: float = 0.0
    target_lufs: float = -16.0
    # Humanizer tuners
    humanize_enabled: bool = True
    breath_amount: float = 0.25
    airiness: float = 0.15
    smile: float = 0.15
    organic_jitter: float = 0.20
    vocal_fry: float = 0.12
    room_floor: float = 0.20


class PronunciationRule(BaseModel):
    find: str
    replace: str


class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "en-us"
    blend: list[VoiceWeight] = Field(default_factory=lambda: [VoiceWeight(voice="af_heart", weight=1.0)])
    split_mode: bool = False
    prosody_voice: str = "af_heart"
    prosody_intensity: float = Field(default=1.0, ge=0.4, le=1.6)
    style_offset: int = Field(default=0, ge=-40, le=80)
    speed: float = Field(default=0.92, ge=0.55, le=1.6)
    sentence_pause: float = Field(default=0.35, ge=0.0, le=1.5)
    clause_pause: float = Field(default=0.14, ge=0.0, le=0.8)
    paragraph_pause: float = Field(default=0.65, ge=0.0, le=2.0)
    rhythm_humanize: float = Field(default=0.30, ge=0.0, le=1.0)
    continuous: bool = True
    smart_punctuation: bool = True
    pronunciation_rules: list[PronunciationRule] = Field(default_factory=list)
    use_custom_phonemes: bool = False
    custom_phonemes: str = ""
    include_raw: bool = True
    mastering: MasteringRequest = Field(default_factory=MasteringRequest)


class RemasterRequest(BaseModel):
    take_id: str
    mastering: MasteringRequest


class PhonemizeRequest(BaseModel):
    text: str
    lang: str = "en-us"
    smart_punctuation: bool = True
    pronunciation_rules: list[PronunciationRule] = Field(default_factory=list)


class PresetCreateRequest(BaseModel):
    name: str
    description: str = ""
    tag: str = "Custom"
    config: dict[str, Any]


# ---------- Helpers ----------

def _load_custom_presets() -> list[dict[str, Any]]:
    if not PRESETS_FILE.exists():
        return []
    try:
        return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_custom_presets(presets: list[dict[str, Any]]) -> None:
    PRESETS_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def _preprocess_text(
    text: str,
    smart_punctuation: bool,
    rules: list[PronunciationRule],
) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    # Apply custom pronunciation word/phrase replacements (case-insensitive whole-word when alphanumeric)
    for rule in rules:
        src = rule.find.strip()
        dst = rule.replace.strip()
        if not src:
            continue
        if re.match(r"^[A-Za-z0-9_'\-]+$", src):
            pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
            cleaned = pattern.sub(dst, cleaned)
        else:
            cleaned = cleaned.replace(src, dst)

    if smart_punctuation:
        # Normalize typographic quotes & spacing
        cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        # Em-dash and double-dash -> slight conversational comma/pause
        cleaned = re.sub(r"\s*[—–]\s*|\s*--\s*", ", ", cleaned)
        # Ellipsis -> clean ellipsis so espeak/kokoro drops cadence slightly
        cleaned = re.sub(r"\.{3,}", "...", cleaned)
        # Semicolon -> comma for smoother breath instead of full stop
        cleaned = cleaned.replace(";", ",")
        # Clean stray double punctuation
        cleaned = re.sub(r",\s*,+", ",", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

    return cleaned


def _build_style_tensor(req: SynthesizeRequest) -> tuple[np.ndarray, str]:
    assert kokoro is not None
    valid_items = [
        item for item in req.blend if item.weight > 0 and item.voice in kokoro.voices.files
    ]
    if not valid_items:
        fallback = "af_heart" if "af_heart" in kokoro.voices.files else kokoro.voices.files[0]
        valid_items = [VoiceWeight(voice=fallback, weight=1.0)]

    total_w = sum(i.weight for i in valid_items)
    primary_voice = max(valid_items, key=lambda x: x.weight).voice

    style = np.zeros_like(kokoro.get_voice_style(valid_items[0].voice), dtype=np.float32)
    for item in valid_items:
        w = item.weight / total_w
        style += w * kokoro.get_voice_style(item.voice)

    # Optional Split Timbre / Prosody: borrow prosody half [:, :, 128:] from another voice
    if req.split_mode and req.prosody_voice in kokoro.voices.files:
        pros_src = kokoro.get_voice_style(req.prosody_voice)
        style = style.copy()
        style[:, :, 128:] = pros_src[:, :, 128:]

    # Prosody expressiveness scaling around mean prosody vector
    if abs(req.prosody_intensity - 1.0) > 0.005 and mean_prosody_vec is not None:
        style = style.copy()
        pros = style[:, :, 128:]
        style[:, :, 128:] = mean_prosody_vec + (pros - mean_prosody_vec) * float(req.prosody_intensity)

    # Phrasing length-index bias: shift style lookup index across [0..509]
    if req.style_offset != 0:
        n_rows = style.shape[0]
        idx = np.clip(np.arange(n_rows) + int(req.style_offset), 0, n_rows - 1)
        style = style[idx]

    return style.astype(np.float32), primary_voice


def _fade_edges(seg: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    n = min(int(sr * ms / 1000.0), len(seg) // 4)
    if n <= 1:
        return seg
    out = seg.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _trim_silence(
    seg: np.ndarray, sr: int, floor_db: float = -42.0, lead_ms: float = 40.0, tail_ms: float = 80.0
) -> np.ndarray:
    """Trim the model's own leading/trailing silence from a generated sentence.

    Each sentence is synthesized separately and the engines pad both ends with
    ~0.3 s of near-silence. Without trimming, that padding stacks with the
    explicit sentence pause and leaves ~1 s gaps between sentences. A short
    lead-in and a slightly longer tail are kept so onsets and decays stay natural.
    """
    if len(seg) < sr // 10:
        return seg
    frame = max(1, int(sr * 0.01))
    n_frames = len(seg) // frame
    if n_frames < 3:
        return seg
    rms = np.sqrt(np.mean(seg[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1)) + 1e-9
    level_db = 20.0 * np.log10(rms / (rms.max() + 1e-9))
    voiced = np.flatnonzero(level_db > floor_db)
    if len(voiced) == 0:
        return seg
    start = max(0, voiced[0] * frame - int(sr * lead_ms / 1000.0))
    end = min(len(seg), (voiced[-1] + 1) * frame + int(sr * tail_ms / 1000.0))
    return seg[start:end]


def _phonemize_cached(text: str, lang: str) -> str:
    """Cached espeak phonemization so slider tweaks spend 0 ms re-phonemizing."""
    assert kokoro is not None
    key = (text, lang)
    cached = phoneme_cache.get(key)
    if cached is not None:
        phoneme_cache.move_to_end(key)
        return cached
    ph = kokoro.tokenizer.phonemize(text, lang=lang)
    phoneme_cache[key] = ph
    while len(phoneme_cache) > 512:
        phoneme_cache.popitem(last=False)
    return ph


def _split_phonemes(phonemes: str, max_len: int = 440) -> list[str]:
    """Split long phoneme strings cleanly within Kokoro's 510-token context window."""
    p = phonemes.strip()
    if len(p) <= max_len:
        return [p] if p else []
    chunks: list[str] = []
    while len(p) > max_len:
        cut = -1
        for sep in (". ", "! ", "? ", ", ", " "):
            idx = p.rfind(sep, 0, max_len)
            if idx > max_len // 3:
                cut = idx + len(sep)
                break
        if cut <= 0:
            cut = max_len
        chunks.append(p[:cut].strip())
        p = p[cut:].strip()
    if p:
        chunks.append(p)
    return chunks


def _infer_kokoro_from_phonemes(
    phonemes: str,
    style: str | np.ndarray,
    speed: float,
    lang: str,
    sentence_pause: float = 0.25,
    clause_pause: float = 0.10,
) -> tuple[np.ndarray, int]:
    """Run Kokoro inference on MLX Metal GPU when available (3.5x faster), else ONNX."""
    assert kokoro is not None
    if mlx_kokoro is not None:
        import mlx.core as mx

        style_arr = kokoro.get_voice_style(style) if isinstance(style, str) else style
        parts: list[np.ndarray] = []
        with mlx_kokoro_lock:
            for chunk in _split_phonemes(phonemes, max_len=440):
                valid_len = sum(1 for ch in chunk if ch in mlx_kokoro.vocab)
                if valid_len == 0:
                    continue
                ref_row = style_arr[min(valid_len, style_arr.shape[0] - 1)]
                ref_mx = mx.array(ref_row, dtype=mx.float32)
                out = mlx_kokoro(chunk, ref_mx, speed=float(speed))
                mx.eval(out)
                seg = np.asarray(out, dtype=np.float32).ravel()
                if len(seg) > 0:
                    parts.append(_fade_edges(seg, SAMPLE_RATE, ms=4.0))
        return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), SAMPLE_RATE

    audio, sr = kokoro.create(
        phonemes,
        voice=style,
        speed=speed,
        lang=lang,
        is_phonemes=True,
        sentence_pause=sentence_pause,
        clause_pause=clause_pause,
        continuous=False,
    )
    return np.asarray(audio, dtype=np.float32), sr


def _synthesize_paragraphs(
    text: str,
    style: str | np.ndarray,
    speed: float,
    lang: str,
    sentence_pause: float,
    clause_pause: float,
    paragraph_pause: float,
    continuous: bool,
    is_phonemes: bool = False,
    rhythm_humanize: float = 0.0,
) -> tuple[np.ndarray, int, str]:
    assert kokoro is not None

    if is_phonemes:
        audio, sr = _infer_kokoro_from_phonemes(
            text, style=style, speed=speed, lang=lang, sentence_pause=sentence_pause, clause_pause=clause_pause
        )
        return audio, sr, text

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return np.zeros(0, dtype=np.float32), SAMPLE_RATE, ""

    segments: list[np.ndarray] = []
    phoneme_paragraphs: list[str] = []
    rh = float(np.clip(rhythm_humanize, 0.0, 1.0))
    rng = np.random.default_rng(abs(hash(text[:64])) % (2**31 - 1))

    for i, para in enumerate(paragraphs):
        para_clean = re.sub(r"\s*\n\s*", " ", para)

        if continuous or rh > 0.02:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?…])\s+(?=[A-Z0-9\"'“‘(])|(?<=[.!?…])\s+", para_clean)
                if s.strip()
            ]
        else:
            sentences = [para_clean]

        para_phs: list[str] = []
        for j, sent in enumerate(sentences):
            sent_speed = speed
            if rh > 0.01 and len(sentences) > 1:
                len_bias = -0.02 * rh if len(sent) > 110 else (0.02 * rh if len(sent) < 42 else 0.0)
                jitter = float(rng.uniform(-0.045, 0.045)) * rh
                sent_speed = float(np.clip(speed * (1.0 + len_bias + jitter), 0.62, 1.50))

            ph = _phonemize_cached(sent, lang)
            para_phs.append(ph)
            seg, _ = _infer_kokoro_from_phonemes(
                ph,
                style=style,
                speed=sent_speed,
                lang=lang,
                sentence_pause=sentence_pause,
                clause_pause=clause_pause,
            )
            if len(seg) > 0:
                segments.append(_fade_edges(_trim_silence(seg, SAMPLE_RATE), SAMPLE_RATE, ms=5.0))
            if j < len(sentences) - 1 and sentence_pause > 0:
                pause_scale = 1.0 + float(rng.uniform(-0.36, 0.42)) * rh
                if sent.endswith("?"):
                    pause_scale += 0.14 * rh
                elif sent.endswith(("...", "…")):
                    pause_scale += 0.25 * rh
                sp_n = int(round(max(0.06, sentence_pause * pause_scale) * SAMPLE_RATE))
                segments.append(np.zeros(sp_n, dtype=np.float32))

        phoneme_paragraphs.append(" ".join(para_phs))
        if i < len(paragraphs) - 1 and paragraph_pause > 0:
            pp_scale = 1.0 + float(rng.uniform(-0.22, 0.28)) * rh
            pp_n = int(round(max(0.12, paragraph_pause * pp_scale) * SAMPLE_RATE))
            segments.append(np.zeros(pp_n, dtype=np.float32))

    audio_out = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    return audio_out, SAMPLE_RATE, "\n\n".join(phoneme_paragraphs)


def _to_mastering_params(m: MasteringRequest) -> MasteringParams:
    return MasteringParams(
        enabled=m.enabled,
        highpass_hz=m.highpass_hz,
        warmth_db=m.warmth_db,
        presence_db=m.presence_db,
        air_db=m.air_db,
        tube_warmth=m.tube_warmth,
        formant_shift=m.formant_shift,
        deesser=m.deesser,
        compression=m.compression,
        room_ambience=m.room_ambience,
        pitch_semitones=m.pitch_semitones,
        target_lufs=m.target_lufs,
        humanize_enabled=m.humanize_enabled,
        breath_amount=m.breath_amount,
        airiness=m.airiness,
        smile=m.smile,
        organic_jitter=m.organic_jitter,
        vocal_fry=m.vocal_fry,
        room_floor=m.room_floor,
    )


def _evict_old_takes() -> None:
    while len(take_cache) > MAX_CACHED_TAKES:
        old_id, _ = take_cache.popitem(last=False)
        for suffix in ("_tuned.wav", "_raw.wav"):
            p = OUTPUTS_DIR / f"{old_id}{suffix}"
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


# ---------- API Endpoints ----------

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if kokoro is not None else "loading",
        "voices_count": len(voice_catalog),
        "sample_rate": SAMPLE_RATE,
        "has_timings": bool(kokoro.has_timings) if kokoro else False,
        "kokoro_backend": kokoro_backend,
        "chatterbox_status": chatterbox_status,
        "chatterbox_model": CHATTERBOX_MODEL_ID,
        "chatterbox_error": chatterbox_error,
    }


@app.get("/api/voices")
def get_voices() -> dict[str, Any]:
    return {"voices": voice_catalog}


@app.get("/api/presets")
def get_presets() -> dict[str, Any]:
    return {"presets": DEFAULT_PRESETS + _load_custom_presets()}


@app.post("/api/presets")
def create_preset(req: PresetCreateRequest) -> dict[str, Any]:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Preset name cannot be empty.")
    custom = _load_custom_presets()
    preset = {
        "id": f"custom-{uuid.uuid4().hex[:8]}",
        "name": name,
        "tag": req.tag.strip() or "Custom",
        "description": req.description.strip() or "Custom saved voice preset.",
        "builtin": False,
        "config": req.config,
    }
    custom.append(preset)
    _save_custom_presets(custom)
    return {"preset": preset}


@app.delete("/api/presets/{preset_id}")
def delete_preset(preset_id: str) -> dict[str, Any]:
    custom = _load_custom_presets()
    filtered = [p for p in custom if p.get("id") != preset_id]
    if len(filtered) == len(custom):
        raise HTTPException(status_code=404, detail="Custom preset not found.")
    _save_custom_presets(filtered)
    return {"deleted": preset_id}


@app.post("/api/phonemize")
def phonemize_text(req: PhonemizeRequest) -> dict[str, Any]:
    if kokoro is None:
        raise HTTPException(status_code=503, detail="Kokoro engine not ready.")
    processed = _preprocess_text(req.text, req.smart_punctuation, req.pronunciation_rules)
    if not processed:
        return {"processed_text": "", "phonemes": ""}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", processed) if p.strip()]
    phoneme_parts = [
        _phonemize_cached(re.sub(r"\s*\n\s*", " ", p), req.lang)
        for p in paragraphs
    ]
    return {
        "processed_text": processed,
        "phonemes": "\n\n".join(phoneme_parts),
    }


class SynthesizeRawRequest(BaseModel):
    take_id: str
    text: str
    voice: str
    lang: str = "en-us"


@app.post("/api/synthesize-raw")
def synthesize_raw(req: SynthesizeRawRequest) -> dict[str, Any]:
    """Non-blocking background endpoint to generate the Raw Kokoro A/B baseline."""
    if kokoro is None:
        raise HTTPException(status_code=503, detail="Kokoro engine not ready.")
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    voice_name = req.voice if req.voice in kokoro.voices.files else "af_heart"
    raw_audio, sr, _ = _synthesize_paragraphs(
        raw_text,
        style=voice_name,
        speed=1.0,
        lang=req.lang,
        sentence_pause=0.25,
        clause_pause=0.10,
        paragraph_pause=0.35,
        continuous=False,
        is_phonemes=False,
        rhythm_humanize=0.0,
    )
    safe_id = Path(req.take_id).name
    raw_file = f"{safe_id}_raw.wav"
    sf.write(str(OUTPUTS_DIR / raw_file), raw_audio, sr, subtype="PCM_16")
    if safe_id in take_cache:
        take_cache[safe_id]["raw_file"] = raw_file

    return {
        "take_id": safe_id,
        "raw_url": f"/api/audio/{raw_file}",
        "raw_peaks": compute_waveform_peaks(raw_audio),
        "raw_duration": round(len(raw_audio) / sr, 3),
    }


@app.post("/api/synthesize")
def synthesize(req: SynthesizeRequest) -> dict[str, Any]:
    if kokoro is None:
        raise HTTPException(status_code=503, detail="Kokoro engine not ready.")

    if req.use_custom_phonemes:
        synth_input = req.custom_phonemes.strip()
        if not synth_input:
            raise HTTPException(status_code=400, detail="Custom phonemes are empty.")
        processed_text = req.text.strip()
    else:
        processed_text = _preprocess_text(req.text, req.smart_punctuation, req.pronunciation_rules)
        if not processed_text:
            raise HTTPException(status_code=400, detail="Please enter some text to synthesize.")
        synth_input = processed_text

    style_tensor, primary_voice = _build_style_tensor(req)

    t0 = time.perf_counter()
    dry_tuned, sr, phonemes_used = _synthesize_paragraphs(
        synth_input,
        style=style_tensor,
        speed=req.speed,
        lang=req.lang,
        sentence_pause=req.sentence_pause,
        clause_pause=req.clause_pause,
        paragraph_pause=req.paragraph_pause,
        continuous=req.continuous,
        is_phonemes=req.use_custom_phonemes,
        rhythm_humanize=req.rhythm_humanize if req.mastering.humanize_enabled else 0.0,
    )
    infer_s = time.perf_counter() - t0

    t_dsp = time.perf_counter()
    mastered = apply_mastering(dry_tuned, sr, _to_mastering_params(req.mastering))
    dsp_ms = (time.perf_counter() - t_dsp) * 1000.0

    take_id = uuid.uuid4().hex[:10]
    tuned_file = f"{take_id}_tuned.wav"
    sf.write(str(OUTPUTS_DIR / tuned_file), mastered, sr, subtype="PCM_16")

    raw_file = None
    raw_peaks = None
    raw_duration = None
    if req.include_raw:
        raw_text = req.text.strip() or processed_text
        raw_audio, _, _ = _synthesize_paragraphs(
            raw_text,
            style=primary_voice,
            speed=1.0,
            lang=req.lang,
            sentence_pause=0.25,
            clause_pause=0.10,
            paragraph_pause=0.35,
            continuous=False,
            is_phonemes=False,
            rhythm_humanize=0.0,
        )
        raw_file = f"{take_id}_raw.wav"
        sf.write(str(OUTPUTS_DIR / raw_file), raw_audio, sr, subtype="PCM_16")
        raw_peaks = compute_waveform_peaks(raw_audio)
        raw_duration = round(len(raw_audio) / sr, 3)

    tuned_duration = round(len(mastered) / sr, 3)
    rtf = round(infer_s / max(tuned_duration, 0.01), 3)

    take_cache[take_id] = {
        "dry_audio": dry_tuned,
        "sr": sr,
        "primary_voice": primary_voice,
        "raw_file": raw_file,
    }
    _evict_old_takes()

    return {
        "take_id": take_id,
        "tuned_url": f"/api/audio/{tuned_file}",
        "raw_url": f"/api/audio/{raw_file}" if raw_file else None,
        "tuned_peaks": compute_waveform_peaks(mastered),
        "raw_peaks": raw_peaks,
        "tuned_duration": tuned_duration,
        "raw_duration": raw_duration,
        "sample_rate": sr,
        "infer_ms": round(infer_s * 1000.0, 1),
        "dsp_ms": round(dsp_ms, 1),
        "rtf": rtf,
        "primary_voice": primary_voice,
        "processed_text": processed_text,
        "phonemes": phonemes_used,
    }


@app.post("/api/remaster")
def remaster(req: RemasterRequest) -> dict[str, Any]:
    entry = take_cache.get(req.take_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Take expired or not found. Synthesize again.")

    dry_audio = entry["dry_audio"]
    sr = entry["sr"]

    t0 = time.perf_counter()
    mastered = apply_mastering(dry_audio, sr, _to_mastering_params(req.mastering))
    dsp_ms = (time.perf_counter() - t0) * 1000.0

    rev = uuid.uuid4().hex[:6]
    tuned_file = f"{req.take_id}_tuned.wav"
    sf.write(str(OUTPUTS_DIR / tuned_file), mastered, sr, subtype="PCM_16")

    return {
        "take_id": req.take_id,
        "tuned_url": f"/api/audio/{tuned_file}?v={rev}",
        "tuned_peaks": compute_waveform_peaks(mastered),
        "tuned_duration": round(len(mastered) / sr, 3),
        "dsp_ms": round(dsp_ms, 1),
    }


def _decode_uploaded_audio(raw_bytes: bytes, filename: str = "clip.wav") -> tuple[np.ndarray, int]:
    """Decode WAV/MP3/FLAC/OGG via soundfile, falling back to macOS afconvert for M4A/AAC."""
    try:
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        return np.asarray(audio, dtype=np.float32), int(sr)
    except Exception:
        suffix = Path(filename).suffix or ".m4a"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"in{suffix}"
            dst = Path(tmp) / "out.wav"
            src.write_bytes(raw_bytes)
            proc = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", str(src), str(dst)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode != 0 or not dst.exists():
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported audio file. Please upload a WAV, MP3, FLAC, OGG, or M4A clip.",
                )
            audio, sr = sf.read(str(dst), dtype="float32")
            return np.asarray(audio, dtype=np.float32), int(sr)


def _get_voice_fingerprints() -> dict[str, dict[str, Any]]:
    """Lazily compute & cache acoustic fingerprints for all English Kokoro voices."""
    assert kokoro is not None
    if FINGERPRINTS_FILE.exists():
        try:
            cached = json.loads(FINGERPRINTS_FILE.read_text(encoding="utf-8"))
            if len(cached) >= 20:
                return cached
        except Exception:
            pass

    probe_text = "Morning everyone, lovely to have you with us today. Let's take a breath and jump straight in."
    fps: dict[str, dict[str, Any]] = {}
    for v in voice_catalog:
        vid = v["id"]
        if not v["lang_code"].startswith("en"):
            continue
        seg, sr = kokoro.create(probe_text, voice=vid, speed=1.0, lang=v["lang_code"])
        prof = analyze_voice_profile(np.asarray(seg, dtype=np.float32), sr)
        fps[vid] = {
            "f0_hz": prof["f0_hz"],
            "f0_std_st": prof["f0_std_st"],
            "warmth_rel": prof["warmth_rel"],
            "presence_rel": prof["presence_rel"],
            "air_rel": prof["air_rel"],
            "sib_rel": prof["sib_rel"],
            "syllables_per_sec": prof["syllables_per_sec"],
            "fingerprint": prof["fingerprint"],
        }

    FINGERPRINTS_FILE.write_text(json.dumps(fps, indent=2), encoding="utf-8")
    return fps


@app.post("/api/match-reference")
async def match_reference(
    request: Request,
    accent: str = Query(default="auto"),
    gender: str = Query(default="auto"),
    filename: str = Query(default="reference.wav"),
) -> dict[str, Any]:
    """Analyze an uploaded reference voice clip and solve for the closest Kokoro blend + DSP settings."""
    if kokoro is None:
        raise HTTPException(status_code=503, detail="Kokoro engine not ready.")

    raw_bytes = await request.body()
    if not raw_bytes or len(raw_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Uploaded audio clip is empty or too short.")
    if len(raw_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Please upload a clip under 25 MB (5–30 seconds is ideal).")

    audio, sr = _decode_uploaded_audio(raw_bytes, filename=filename)
    ref = analyze_voice_profile(audio, sr)
    voice_fps = _get_voice_fingerprints()

    # Auto-detect gender from median F0 if requested
    inferred_gender = gender
    if gender == "auto":
        inferred_gender = "Female" if ref["f0_hz"] >= 152.0 else "Male"

    candidates = [
        v
        for v in voice_catalog
        if v["id"] in voice_fps
        and (accent == "auto" or accent == "any" or v["lang_code"] == accent)
        and (gender == "any" or v["gender"] == inferred_gender)
    ]
    if not candidates:
        candidates = [v for v in voice_catalog if v["id"] in voice_fps]

    ref_fp = np.array(ref["fingerprint"], dtype=np.float64)
    ref_fp_norm = ref_fp / (np.linalg.norm(ref_fp) + 1e-8)

    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for v in candidates:
        vf = voice_fps[v["id"]]
        v_fp = np.array(vf["fingerprint"], dtype=np.float64)
        v_fp_norm = v_fp / (np.linalg.norm(v_fp) + 1e-8)
        cos_sim = float(np.dot(ref_fp_norm, v_fp_norm))
        # Pitch distance in semitones
        st_diff = abs(12.0 * np.log2(max(ref["f0_hz"], 60.0) / max(vf["f0_hz"], 60.0)))
        pitch_score = float(np.exp(-((st_diff / 4.5) ** 2)))
        total_score = 0.62 * cos_sim + 0.38 * pitch_score
        scored.append((total_score, v, vf))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: min(3, len(scored))]

    # Convert top similarities into smooth blend weights
    scores_arr = np.array([s[0] for s in top], dtype=np.float64)
    exps = np.exp((scores_arr - np.max(scores_arr)) * 8.0)
    weights = exps / np.sum(exps)

    blend = [
        {"voice": top[i][1]["id"], "weight": round(float(weights[i]), 2)}
        for i in range(len(top))
    ]

    # Blended baseline acoustics
    base_f0 = sum(float(weights[i]) * top[i][2]["f0_hz"] for i in range(len(top)))
    base_warmth = sum(float(weights[i]) * top[i][2]["warmth_rel"] for i in range(len(top)))
    base_presence = sum(float(weights[i]) * top[i][2]["presence_rel"] for i in range(len(top)))
    base_air = sum(float(weights[i]) * top[i][2]["air_rel"] for i in range(len(top)))
    base_syl = sum(float(weights[i]) * top[i][2]["syllables_per_sec"] for i in range(len(top)))

    # Solve pitch & formant gently (keep within natural artifact-free bounds)
    raw_st = 12.0 * np.log2(max(ref["f0_hz"], 65.0) / max(base_f0, 65.0))
    pitch_st = round(float(np.clip(raw_st * 0.65, -1.75, 1.75)), 2)
    formant_shift = round(float(np.clip(raw_st * 0.9, -6.0, 6.0)), 1)

    # Speaking rate & pause cadence
    speed = round(float(np.clip(ref["syllables_per_sec"] / max(base_syl, 2.5), 0.80, 1.15)), 2)
    silence_ratio = 1.0 - ref["speech_ratio"]
    sentence_pause = round(float(np.clip(0.18 + silence_ratio * 0.65, 0.20, 0.65)), 2)
    clause_pause = round(float(np.clip(sentence_pause * 0.38, 0.08, 0.24)), 2)
    prosody_intensity = round(float(np.clip(0.85 + (ref["f0_std_st"] / 5.5) * 0.42, 0.85, 1.35)), 2)

    # Spectral delta -> EQ + tube warmth + compression
    warmth_db = round(float(np.clip((ref["warmth_rel"] - base_warmth) * 0.45 + 1.4, -3.5, 5.0)), 1)
    presence_db = round(float(np.clip((ref["presence_rel"] - base_presence) * 0.40 + 1.2, -3.5, 4.5)), 1)
    air_db = round(float(np.clip((ref["air_rel"] - base_air) * 0.35 + 1.2, -3.5, 4.5)), 1)
    # Tighter dynamic range in reference (e.g. radio broadcast) -> more compression & tube warmth
    compression = round(float(np.clip(0.65 - (ref["dynamic_range_db"] - 8.0) * 0.03, 0.22, 0.68)), 2)
    tube_warmth = round(float(np.clip(0.18 + max(0.0, warmth_db) * 0.07, 0.10, 0.48)), 2)
    deesser = 0.42 if ref["sib_rel"] < -14.0 else 0.32

    chosen_lang = accent if accent in ("en-us", "en-gb") else top[0][1]["lang_code"]

    matched_config = {
        "lang": chosen_lang,
        "blend": blend,
        "split_mode": False,
        "prosody_voice": blend[0]["voice"],
        "prosody_intensity": prosody_intensity,
        "style_offset": -4 if speed >= 0.98 else 12,
        "speed": speed,
        "sentence_pause": sentence_pause,
        "clause_pause": clause_pause,
        "paragraph_pause": round(min(0.9, sentence_pause * 1.8), 2),
        "continuous": True,
        "smart_punctuation": True,
        "mastering": {
            "enabled": True,
            "highpass_hz": 80,
            "warmth_db": warmth_db,
            "presence_db": presence_db,
            "air_db": air_db,
            "tube_warmth": tube_warmth,
            "formant_shift": formant_shift,
            "deesser": deesser,
            "compression": compression,
            "room_ambience": 0.10,
            "pitch_semitones": pitch_st,
            "target_lufs": -15.0,
        },
    }

    return {
        "analysis": {
            "duration_s": ref["duration_s"],
            "f0_hz": ref["f0_hz"],
            "f0_std_st": ref["f0_std_st"],
            "inferred_gender": inferred_gender,
            "syllables_per_sec": ref["syllables_per_sec"],
            "dynamic_range_db": ref["dynamic_range_db"],
            "closest_voices": [
                {"id": top[i][1]["id"], "name": top[i][1]["name"], "weight": blend[i]["weight"]}
                for i in range(len(top))
            ],
        },
        "config": matched_config,
    }


# ---------- Chatterbox MLX Engine & Reference Voice Library ----------

class ChatterboxSynthesizeRequest(BaseModel):
    text: str
    ref_id: str | None = None  # None -> built-in Chatterbox default voice
    exaggeration: float = Field(default=0.55, ge=0.0, le=1.5)
    cfg_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    temperature: float = Field(default=0.80, ge=0.15, le=1.5)
    repetition_penalty: float = Field(default=1.20, ge=1.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.2, le=1.0)
    min_p: float = Field(default=0.05, ge=0.0, le=0.5)
    sentence_pause: float = Field(default=0.30, ge=0.0, le=1.5)
    paragraph_pause: float = Field(default=0.60, ge=0.0, le=2.0)
    rhythm_humanize: float = Field(default=0.30, ge=0.0, le=1.0)
    smart_punctuation: bool = True
    pronunciation_rules: list[PronunciationRule] = Field(default_factory=list)
    mastering: MasteringRequest = Field(default_factory=MasteringRequest)


class SaveTakeAsRefRequest(BaseModel):
    take_id: str
    name: str = "Kokoro Sculpted Voice"
    use_raw: bool = False


def _load_references_index() -> list[dict[str, Any]]:
    if not REFERENCES_INDEX.exists():
        return []
    try:
        items = json.loads(REFERENCES_INDEX.read_text(encoding="utf-8"))
        # Filter out missing files
        return [it for it in items if (REFERENCES_DIR / f"{it['id']}.wav").exists()]
    except Exception:
        return []


def _save_references_index(items: list[dict[str, Any]]) -> None:
    REFERENCES_INDEX.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _get_chatterbox() -> Any:
    global chatterbox_model, chatterbox_status, chatterbox_error
    if chatterbox_model is not None:
        return chatterbox_model
    with chatterbox_lock:
        if chatterbox_model is not None:
            return chatterbox_model
        chatterbox_status = "loading"
        chatterbox_error = None
        try:
            from mlx_audio.tts.utils import load_model

            chatterbox_model = load_model(CHATTERBOX_MODEL_ID)
            chatterbox_status = "ready"
            return chatterbox_model
        except Exception as exc:
            chatterbox_status = "error"
            chatterbox_error = str(exc)
            raise HTTPException(status_code=500, detail=f"Failed to load Chatterbox MLX: {exc}")


@app.post("/api/chatterbox/load")
def load_chatterbox() -> dict[str, Any]:
    model = _get_chatterbox()
    return {
        "status": chatterbox_status,
        "model": CHATTERBOX_MODEL_ID,
        "sample_rate": int(getattr(model, "sample_rate", SAMPLE_RATE)),
    }


@app.get("/api/references")
def list_references() -> dict[str, Any]:
    return {"references": _load_references_index()}


@app.post("/api/references")
async def upload_reference(
    request: Request,
    name: str = Query(default=""),
    filename: str = Query(default="reference.wav"),
) -> dict[str, Any]:
    raw_bytes = await request.body()
    if not raw_bytes or len(raw_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Uploaded reference clip is empty or too short.")
    if len(raw_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Reference clip must be under 25 MB.")

    audio, sr = _decode_uploaded_audio(raw_bytes, filename=filename)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if sr != SAMPLE_RATE and len(x) > 0:
        target_n = int(round(len(x) * SAMPLE_RATE / sr))
        x = signal.resample(x, max(1, target_n)).astype(np.float32)
        sr = SAMPLE_RATE

    # Trim silence gently & cap to 25 seconds (Chatterbox uses up to 10-15s cleanly)
    x = x - np.mean(x)
    peak = float(np.max(np.abs(x)) + 1e-8)
    x = (x / peak) * 0.92
    if len(x) > sr * 25:
        x = x[: sr * 25]

    dur = round(len(x) / sr, 2)
    if dur < 1.2:
        raise HTTPException(status_code=400, detail="Reference clip should be at least 1.5 seconds of speech.")

    ref_id = f"ref-{uuid.uuid4().hex[:8]}"
    out_path = REFERENCES_DIR / f"{ref_id}.wav"
    sf.write(str(out_path), x, sr, subtype="PCM_16")

    label = name.strip() or Path(filename).stem.replace("_", " ").replace("-", " ").strip()[:40] or "Voice Clip"
    prof = analyze_voice_profile(x, sr)
    entry = {
        "id": ref_id,
        "name": label,
        "duration_s": dur,
        "f0_hz": prof["f0_hz"],
        "audio_url": f"/api/references/{ref_id}/audio",
        "created_at": int(time.time()),
    }
    items = _load_references_index()
    items.insert(0, entry)
    _save_references_index(items)
    return {"reference": entry}


@app.post("/api/references/from-take")
def save_take_as_reference(req: SaveTakeAsRefRequest) -> dict[str, Any]:
    entry_cache = take_cache.get(req.take_id)
    if entry_cache is None:
        raise HTTPException(status_code=404, detail="Take not found or expired.")
    suffix = "_raw.wav" if req.use_raw else "_tuned.wav"
    src_path = OUTPUTS_DIR / f"{req.take_id}{suffix}"
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Audio file for take not found.")

    audio, sr = sf.read(str(src_path), dtype="float32")
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if len(x) > sr * 18:
        x = x[: sr * 18]

    ref_id = f"ref-{uuid.uuid4().hex[:8]}"
    sf.write(str(REFERENCES_DIR / f"{ref_id}.wav"), x, sr, subtype="PCM_16")
    prof = analyze_voice_profile(x, sr)
    ref_entry = {
        "id": ref_id,
        "name": req.name.strip() or "Kokoro Voice Reference",
        "duration_s": round(len(x) / sr, 2),
        "f0_hz": prof["f0_hz"],
        "audio_url": f"/api/references/{ref_id}/audio",
        "created_at": int(time.time()),
    }
    items = _load_references_index()
    items.insert(0, ref_entry)
    _save_references_index(items)
    return {"reference": ref_entry}


@app.get("/api/references/{ref_id}/audio")
def get_reference_audio(ref_id: str) -> FileResponse:
    safe_id = Path(ref_id).name
    path = REFERENCES_DIR / f"{safe_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Reference clip not found.")
    return FileResponse(str(path), media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.delete("/api/references/{ref_id}")
def delete_reference(ref_id: str) -> dict[str, Any]:
    safe_id = Path(ref_id).name
    items = _load_references_index()
    filtered = [it for it in items if it.get("id") != safe_id]
    if len(filtered) == len(items):
        raise HTTPException(status_code=404, detail="Reference not found.")
    _save_references_index(filtered)
    p = REFERENCES_DIR / f"{safe_id}.wav"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass
    # Evict any cached conditionals for this ref
    for k in [k for k in chatterbox_conds_cache if k.startswith(f"{safe_id}:")]:
        chatterbox_conds_cache.pop(k, None)
    return {"deleted": safe_id}


def _get_chatterbox_conds(model: Any, ref_id: str | None, exaggeration: float) -> tuple[Any, str]:
    """Return cached Chatterbox Conditionals for ref_id (or built-in default)."""
    import mlx.core as mx
    from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals, T3Cond

    exag_key = round(float(exaggeration), 2)
    if not ref_id:
        base_conds = getattr(model, "_conds", None)
        if base_conds is None:
            return None, "Chatterbox Default"
        # Adjust emotion_adv on default conds without mutating original
        t3c = T3Cond(
            speaker_emb=base_conds.t3.speaker_emb,
            cond_prompt_speech_tokens=base_conds.t3.cond_prompt_speech_tokens,
            emotion_adv=mx.ones((1, 1, 1)) * exag_key,
        )
        return Conditionals(t3c, base_conds.gen), "Chatterbox Default"

    safe_id = Path(ref_id).name
    ref_path = REFERENCES_DIR / f"{safe_id}.wav"
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="Selected reference voice clip was not found.")

    cache_key = f"{safe_id}:base"
    base_conds = chatterbox_conds_cache.get(cache_key)
    if base_conds is None:
        wav, sr = sf.read(str(ref_path), dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        base_conds = model.prepare_conditionals(
            mx.array(wav.astype(np.float32)), int(sr), exaggeration=exag_key
        )
        chatterbox_conds_cache[cache_key] = base_conds
        while len(chatterbox_conds_cache) > 8:
            chatterbox_conds_cache.popitem(last=False)

    t3c = T3Cond(
        speaker_emb=base_conds.t3.speaker_emb,
        cond_prompt_speech_tokens=base_conds.t3.cond_prompt_speech_tokens,
        emotion_adv=mx.ones((1, 1, 1)) * exag_key,
    )
    ref_meta = next((r for r in _load_references_index() if r["id"] == safe_id), None)
    ref_label = ref_meta["name"] if ref_meta else safe_id
    return Conditionals(t3c, base_conds.gen), ref_label


def _chunk_text_for_chatterbox(text: str, max_chars: int = 230) -> list[list[str]]:
    """Split text into paragraphs -> cleanly bounded sentence chunks for stable autoregressive synthesis."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    out: list[list[str]] = []
    for para in paragraphs:
        para_clean = re.sub(r"\s*\n\s*", " ", para)
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?…])\s+(?=[A-Z0-9\"'“‘(])|(?<=[.!?…])\s+", para_clean)
            if s.strip()
        ]
        grouped: list[str] = []
        buf = ""
        for s in sents:
            if buf and len(buf) + 1 + len(s) > max_chars:
                grouped.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip() if buf else s
        if buf:
            grouped.append(buf)
        if grouped:
            out.append(grouped)
    return out


@app.post("/api/chatterbox/synthesize")
def synthesize_chatterbox(req: ChatterboxSynthesizeRequest) -> dict[str, Any]:
    processed_text = _preprocess_text(req.text, req.smart_punctuation, req.pronunciation_rules)
    if not processed_text:
        raise HTTPException(status_code=400, detail="Please enter some text to synthesize.")

    model = _get_chatterbox()
    sr = int(getattr(model, "sample_rate", SAMPLE_RATE))

    t0 = time.perf_counter()
    with chatterbox_lock:
        conds, ref_label = _get_chatterbox_conds(model, req.ref_id, req.exaggeration)
        para_chunks = _chunk_text_for_chatterbox(processed_text)
        segments: list[np.ndarray] = []
        rh = float(np.clip(req.rhythm_humanize, 0.0, 1.0)) if req.mastering.humanize_enabled else 0.0
        rng = np.random.default_rng(abs(hash(processed_text[:64])) % (2**31 - 1))

        for p_idx, sentences in enumerate(para_chunks):
            for s_idx, chunk_text in enumerate(sentences):
                chunk_temp = float(
                    np.clip(req.temperature + float(rng.uniform(-0.05, 0.05)) * rh, 0.15, 1.45)
                )
                gen_parts = [
                    np.asarray(res.audio, dtype=np.float32).ravel()
                    for res in model.generate(
                        text=chunk_text,
                        conds=conds,
                        exaggeration=float(req.exaggeration),
                        cfg_weight=float(req.cfg_weight),
                        temperature=chunk_temp,
                        repetition_penalty=float(req.repetition_penalty),
                        min_p=float(req.min_p),
                        top_p=float(req.top_p),
                        verbose=False,
                    )
                ]
                if gen_parts:
                    seg = np.concatenate(gen_parts)
                    segments.append(_fade_edges(_trim_silence(seg, sr), sr, ms=6.0))
                if s_idx < len(sentences) - 1 and req.sentence_pause > 0:
                    p_scale = 1.0 + float(rng.uniform(-0.34, 0.40)) * rh
                    sp_n = int(round(max(0.06, req.sentence_pause * p_scale) * sr))
                    segments.append(np.zeros(sp_n, dtype=np.float32))
            if p_idx < len(para_chunks) - 1 and req.paragraph_pause > 0:
                pp_scale = 1.0 + float(rng.uniform(-0.22, 0.28)) * rh
                pp_n = int(round(max(0.12, req.paragraph_pause * pp_scale) * sr))
                segments.append(np.zeros(pp_n, dtype=np.float32))

    dry_audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    infer_s = time.perf_counter() - t0

    t_dsp = time.perf_counter()
    mastered = apply_mastering(dry_audio, sr, _to_mastering_params(req.mastering))
    dsp_ms = (time.perf_counter() - t_dsp) * 1000.0

    take_id = f"cb_{uuid.uuid4().hex[:8]}"
    tuned_file = f"{take_id}_tuned.wav"
    raw_file = f"{take_id}_raw.wav"
    sf.write(str(OUTPUTS_DIR / tuned_file), mastered, sr, subtype="PCM_16")
    sf.write(str(OUTPUTS_DIR / raw_file), dry_audio, sr, subtype="PCM_16")

    tuned_duration = round(len(mastered) / sr, 3)
    raw_duration = round(len(dry_audio) / sr, 3)
    rtf = round(infer_s / max(tuned_duration, 0.01), 3)

    take_cache[take_id] = {
        "dry_audio": dry_audio,
        "sr": sr,
        "primary_voice": f"Chatterbox · {ref_label}",
        "raw_file": raw_file,
    }
    _evict_old_takes()

    return {
        "take_id": take_id,
        "engine": "chatterbox",
        "tuned_url": f"/api/audio/{tuned_file}",
        "raw_url": f"/api/audio/{raw_file}",
        "tuned_peaks": compute_waveform_peaks(mastered),
        "raw_peaks": compute_waveform_peaks(dry_audio),
        "tuned_duration": tuned_duration,
        "raw_duration": raw_duration,
        "sample_rate": sr,
        "infer_ms": round(infer_s * 1000.0, 1),
        "dsp_ms": round(dsp_ms, 1),
        "rtf": rtf,
        "primary_voice": f"Chatterbox ({ref_label})",
        "processed_text": processed_text,
        "phonemes": "",
    }


# ---------- Voice Export & OpenAI-Compatible App Narration API (/v1/audio/speech) ----------

class ExportVoiceRequest(BaseModel):
    name: str
    slug: str | None = None
    description: str = ""
    engine: str = "kokoro"  # 'kokoro' | 'chatterbox'
    config: dict[str, Any]
    pronunciation_rules: list[PronunciationRule] = Field(default_factory=list)


class OpenAISpeechRequest(BaseModel):
    model: str = "timbra"
    input: str
    voice: str = "eleven-warm-host"
    response_format: str = "wav"  # wav | mp3 | flac | ogg | pcm
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


def _encode_audio_bytes(audio: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    f = (fmt or "wav").lower().strip()
    x = np.asarray(audio, dtype=np.float32)
    if f == "pcm":
        pcm16 = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
        return pcm16.tobytes(), "audio/pcm"

    buf = io.BytesIO()
    if f == "mp3":
        sf.write(buf, x, sr, format="MP3")
        return buf.getvalue(), "audio/mpeg"
    if f == "flac":
        sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
        return buf.getvalue(), "audio/flac"
    if f in ("ogg", "opus"):
        sf.write(buf, x, sr, format="OGG", subtype="VORBIS")
        return buf.getvalue(), "audio/ogg"
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


def _build_style_from_config_dict(cfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    blend_items = [
        VoiceWeight(voice=b["voice"], weight=float(b["weight"]))
        for b in cfg.get("blend", [])
        if isinstance(b, dict) and "voice" in b
    ]
    dummy = SynthesizeRequest(
        text="x",
        lang=cfg.get("lang", "en-us"),
        blend=blend_items or [VoiceWeight(voice="af_heart", weight=1.0)],
        split_mode=bool(cfg.get("split_mode", False)),
        prosody_voice=str(cfg.get("prosody_voice", "af_heart")),
        prosody_intensity=float(cfg.get("prosody_intensity", 1.0)),
        style_offset=int(cfg.get("style_offset", 0)),
    )
    return _build_style_tensor(dummy)


def _mastering_from_dict(m: dict[str, Any] | None) -> MasteringParams:
    if not m:
        return MasteringParams()
    req = MasteringRequest(**{k: v for k, v in m.items() if k in MasteringRequest.model_fields})
    return _to_mastering_params(req)


@app.get("/api/exported-voices")
def get_exported_voices() -> dict[str, Any]:
    return {"voices": list_exported_voices()}


@app.post("/api/exported-voices")
def create_exported_voice(req: ExportVoiceRequest) -> dict[str, Any]:
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Please give your exported voice a name.")

    engine = "chatterbox" if req.engine == "chatterbox" else "kokoro"
    style_tensor: np.ndarray | None = None
    ref_wav_path: Path | None = None

    if engine == "kokoro":
        if kokoro is None:
            raise HTTPException(status_code=503, detail="Kokoro engine not ready.")
        style_tensor, _ = _build_style_from_config_dict(req.config)
    else:
        ref_id = req.config.get("ref_id")
        if ref_id:
            candidate = REFERENCES_DIR / f"{Path(str(ref_id)).name}.wav"
            if candidate.exists():
                ref_wav_path = candidate

    rules_dicts = [{"find": r.find, "replace": r.replace} for r in req.pronunciation_rules if r.find.strip()]
    saved = save_exported_voice(
        name=req.name,
        description=req.description,
        engine=engine,
        config=req.config,
        pronunciation_rules=rules_dicts,
        style_tensor=style_tensor,
        reference_wav_path=ref_wav_path,
        slug_override=req.slug,
    )
    return {"voice": saved}


@app.post("/api/exported-voices/import")
async def import_exported_voice(request: Request) -> dict[str, Any]:
    """Import a portable .timbra.json voice file back into Timbra."""
    raw = await request.body()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid .timbra.json file.")
    if not isinstance(data, dict) or "config" not in data:
        raise HTTPException(status_code=400, detail="Not a valid Timbra voice profile.")

    engine = "chatterbox" if data.get("engine") == "chatterbox" else "kokoro"
    style_tensor: np.ndarray | None = None
    if engine == "kokoro" and kokoro is not None:
        style_tensor, _ = _build_style_from_config_dict(data["config"])

    saved = save_exported_voice(
        name=str(data.get("name") or data.get("slug") or "Imported Voice"),
        description=str(data.get("description") or ""),
        engine=engine,
        config=data["config"],
        pronunciation_rules=data.get("pronunciation_rules") or [],
        style_tensor=style_tensor,
        slug_override=data.get("slug"),
    )
    return {"voice": saved}


@app.delete("/api/exported-voices/{slug}")
def remove_exported_voice(slug: str) -> dict[str, Any]:
    if not delete_exported_voice(slug):
        raise HTTPException(status_code=404, detail="Exported voice not found.")
    return {"deleted": slug}


@app.get("/api/exported-voices/{slug}/json")
def download_exported_json(slug: str) -> Response:
    loaded = load_exported_voice(slug)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Exported voice not found.")
    data, _ = loaded
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{data["slug"]}.timbra.json"'},
    )


@app.get("/api/exported-voices/{slug}/style.npy")
def download_exported_npy(slug: str) -> FileResponse:
    loaded = load_exported_voice(slug)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Exported voice not found.")
    data, voice_dir = loaded
    npy_path = voice_dir / "style.npy"
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail="Baked Kokoro style.npy is only available for Kokoro voices.")
    return FileResponse(
        str(npy_path),
        media_type="application/octet-stream",
        filename=f"{data['slug']}_style.npy",
    )


@app.get("/api/exported-voices/{slug}/bundle.zip")
def download_exported_bundle(slug: str) -> Response:
    zip_bytes = build_zip_bundle(slug)
    if zip_bytes is None:
        raise HTTPException(status_code=404, detail="Exported voice not found.")
    safe = Path(slug).name
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe}-timbra-pack.zip"'},
    )


@app.get("/v1/audio/voices")
def openai_list_voices() -> dict[str, Any]:
    """List all custom exported voices, curated presets, and built-in Kokoro voices."""
    exported = [
        {"id": v["slug"], "name": v["name"], "engine": v["engine"], "source": "exported"}
        for v in list_exported_voices()
    ]
    presets = [
        {"id": p["id"], "name": p["name"], "engine": "kokoro", "source": "preset"}
        for p in DEFAULT_PRESETS + _load_custom_presets()
    ]
    raw_kokoro = [
        {"id": v["id"], "name": v["name"], "engine": "kokoro", "source": "kokoro"}
        for v in voice_catalog
    ]
    return {"voices": exported + presets + raw_kokoro}


@app.post("/v1/audio/speech")
def openai_audio_speech(req: OpenAISpeechRequest) -> Response:
    """OpenAI-compatible TTS endpoint so any external app can narrate with Timbra voices."""
    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    voice_id = req.voice.strip()
    exported = load_exported_voice(voice_id)

    # 1. Custom exported Timbra voice (Kokoro or Chatterbox!)
    if exported is not None:
        vdata, voice_dir = exported
        cfg = vdata.get("config", {})
        rules = [
            PronunciationRule(find=r["find"], replace=r["replace"])
            for r in vdata.get("pronunciation_rules", [])
            if isinstance(r, dict) and r.get("find")
        ]
        processed = _preprocess_text(text, bool(cfg.get("smart_punctuation", True)), rules)
        m_params = _mastering_from_dict(cfg.get("mastering"))

        if vdata.get("engine") == "chatterbox":
            model = _get_chatterbox()
            sr = int(getattr(model, "sample_rate", SAMPLE_RATE))
            ref_file = voice_dir / "reference.wav"
            with chatterbox_lock:
                if ref_file.exists():
                    import mlx.core as mx
                    wav, rsr = sf.read(str(ref_file), dtype="float32")
                    if wav.ndim > 1:
                        wav = np.mean(wav, axis=1)
                    conds = model.prepare_conditionals(
                        mx.array(wav.astype(np.float32)),
                        int(rsr),
                        exaggeration=float(cfg.get("exaggeration", 0.55)),
                    )
                else:
                    conds, _ = _get_chatterbox_conds(model, cfg.get("ref_id"), float(cfg.get("exaggeration", 0.55)))

                para_chunks = _chunk_text_for_chatterbox(processed)
                segs: list[np.ndarray] = []
                sent_p = float(cfg.get("sentence_pause", 0.30))
                para_p = float(cfg.get("paragraph_pause", 0.60))
                for pi, sents in enumerate(para_chunks):
                    for si, chunk_txt in enumerate(sents):
                        parts = [
                            np.asarray(res.audio, dtype=np.float32).ravel()
                            for res in model.generate(
                                text=chunk_txt,
                                conds=conds,
                                exaggeration=float(cfg.get("exaggeration", 0.55)),
                                cfg_weight=float(cfg.get("cfg_weight", 0.50)),
                                temperature=float(cfg.get("temperature", 0.80)),
                                repetition_penalty=float(cfg.get("repetition_penalty", 1.20)),
                                verbose=False,
                            )
                        ]
                        if parts:
                            segs.append(_fade_edges(_trim_silence(np.concatenate(parts), sr), sr, ms=6.0))
                        if si < len(sents) - 1 and sent_p > 0:
                            segs.append(np.zeros(int(sent_p * sr), dtype=np.float32))
                    if pi < len(para_chunks) - 1 and para_p > 0:
                        segs.append(np.zeros(int(para_p * sr), dtype=np.float32))
                dry = np.concatenate(segs) if segs else np.zeros(0, dtype=np.float32)
            mastered = apply_mastering(dry, sr, m_params)
            payload_bytes, mime = _encode_audio_bytes(mastered, sr, req.response_format)
            return Response(content=payload_bytes, media_type=mime)

        # Exported Kokoro voice (load baked style.npy if present, else rebuild)
        if kokoro is None:
            raise HTTPException(status_code=503, detail="Kokoro engine not ready.")
        npy_path = voice_dir / "style.npy"
        if npy_path.exists():
            style = np.load(str(npy_path)).astype(np.float32)
        else:
            style, _ = _build_style_from_config_dict(cfg)

        eff_speed = float(np.clip(float(cfg.get("speed", 0.92)) * float(req.speed), 0.55, 1.65))
        rh = float(cfg.get("rhythm_humanize", 0.30)) if m_params.humanize_enabled else 0.0
        dry, sr, _ = _synthesize_paragraphs(
            processed,
            style=style,
            speed=eff_speed,
            lang=str(cfg.get("lang", "en-us")),
            sentence_pause=float(cfg.get("sentence_pause", 0.35)),
            clause_pause=float(cfg.get("clause_pause", 0.14)),
            paragraph_pause=float(cfg.get("paragraph_pause", 0.65)),
            continuous=bool(cfg.get("continuous", True)),
            is_phonemes=False,
            rhythm_humanize=rh,
        )
        mastered = apply_mastering(dry, sr, m_params)
        payload_bytes, mime = _encode_audio_bytes(mastered, sr, req.response_format)
        return Response(content=payload_bytes, media_type=mime)

    # 2. Built-in or custom preset ID
    all_presets = DEFAULT_PRESETS + _load_custom_presets()
    preset = next((p for p in all_presets if p.get("id") == voice_id), None)
    if preset is not None and kokoro is not None:
        cfg = preset["config"]
        style, _ = _build_style_from_config_dict(cfg)
        m_params = _mastering_from_dict(cfg.get("mastering"))
        processed = _preprocess_text(text, bool(cfg.get("smart_punctuation", True)), [])
        eff_speed = float(np.clip(float(cfg.get("speed", 0.92)) * float(req.speed), 0.55, 1.65))
        dry, sr, _ = _synthesize_paragraphs(
            processed,
            style=style,
            speed=eff_speed,
            lang=str(cfg.get("lang", "en-us")),
            sentence_pause=float(cfg.get("sentence_pause", 0.35)),
            clause_pause=float(cfg.get("clause_pause", 0.14)),
            paragraph_pause=float(cfg.get("paragraph_pause", 0.65)),
            continuous=bool(cfg.get("continuous", True)),
            is_phonemes=False,
            rhythm_humanize=0.25 if m_params.humanize_enabled else 0.0,
        )
        mastered = apply_mastering(dry, sr, m_params)
        payload_bytes, mime = _encode_audio_bytes(mastered, sr, req.response_format)
        return Response(content=payload_bytes, media_type=mime)

    # 3. Direct raw Kokoro voice ID (or default af_heart fallback for OpenAI names like alloy/nova)
    if kokoro is None:
        raise HTTPException(status_code=503, detail="Kokoro engine not ready.")
    openai_alias = {
        "alloy": "af_alloy",
        "echo": "am_echo",
        "fable": "bm_fable",
        "onyx": "am_onyx",
        "nova": "af_nova",
        "shimmer": "af_heart",
    }
    resolved_voice = openai_alias.get(voice_id.lower(), voice_id)
    if resolved_voice not in kokoro.voices.files:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_id}' not found. Export a voice in Timbra or inspect GET /v1/audio/voices.",
        )
    lang = "en-gb" if resolved_voice.startswith(("bf_", "bm_")) else "en-us"
    dry, sr, _ = _synthesize_paragraphs(
        text,
        style=resolved_voice,
        speed=float(np.clip(req.speed, 0.55, 1.65)),
        lang=lang,
        sentence_pause=0.30,
        clause_pause=0.12,
        paragraph_pause=0.55,
        continuous=True,
        is_phonemes=False,
    )
    payload_bytes, mime = _encode_audio_bytes(dry, sr, req.response_format)
    return Response(content=payload_bytes, media_type=mime)


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    # Restrict to safe filename inside OUTPUTS_DIR
    safe_name = Path(filename).name
    path = OUTPUTS_DIR / safe_name
    if not path.exists() or path.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(
        str(path),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


# Serve static frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8765, reload=False)
