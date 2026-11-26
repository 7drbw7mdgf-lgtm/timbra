"""Portable Voice Export, App Bundle (.zip) Builder, and Baked Tensor Helpers for Timbra."""

from __future__ import annotations

import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
EXPORTED_DIR = BASE_DIR / "exported_voices"
EXPORTED_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:48] or f"voice-{int(time.time())}"


def list_exported_voices() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not EXPORTED_DIR.exists():
        return items
    for d in sorted(EXPORTED_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_file = d / "voice.timbra.json"
        if not d.is_dir() or not meta_file.exists():
            continue
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            items.append(
                {
                    "slug": data.get("slug", d.name),
                    "name": data.get("name", d.name),
                    "description": data.get("description", ""),
                    "engine": data.get("engine", "kokoro"),
                    "created_at": data.get("created_at", 0),
                    "has_style_npy": (d / "style.npy").exists(),
                    "has_reference_wav": (d / "reference.wav").exists(),
                    "config": data.get("config", {}),
                    "pronunciation_rules": data.get("pronunciation_rules", []),
                }
            )
        except Exception:
            continue
    return items


def load_exported_voice(slug: str) -> tuple[dict[str, Any], Path] | None:
    safe_slug = Path(slug).name
    voice_dir = EXPORTED_DIR / safe_slug
    meta_file = voice_dir / "voice.timbra.json"
    if not meta_file.exists():
        return None
    data = json.loads(meta_file.read_text(encoding="utf-8"))
    return data, voice_dir


def save_exported_voice(
    name: str,
    description: str,
    engine: str,
    config: dict[str, Any],
    pronunciation_rules: list[dict[str, str]],
    style_tensor: np.ndarray | None = None,
    reference_wav_path: Path | None = None,
    slug_override: str | None = None,
) -> dict[str, Any]:
    slug = slugify(slug_override or name)
    voice_dir = EXPORTED_DIR / slug
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale artifacts when overwriting same slug across engines
    for old_f in ("style.npy", "reference.wav"):
        p = voice_dir / old_f
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    has_style = False
    if engine == "kokoro" and style_tensor is not None:
        np.save(str(voice_dir / "style.npy"), np.asarray(style_tensor, dtype=np.float32))
        has_style = True

    has_ref = False
    if engine == "chatterbox" and reference_wav_path is not None and reference_wav_path.exists():
        shutil.copy2(str(reference_wav_path), str(voice_dir / "reference.wav"))
        has_ref = True

    payload = {
        "schema": "timbra.voice.v1",
        "slug": slug,
        "name": name.strip() or slug,
        "description": description.strip(),
        "engine": engine,
        "created_at": int(time.time()),
        "has_style_npy": has_style,
        "has_reference_wav": has_ref,
        "config": config,
        "pronunciation_rules": pronunciation_rules,
    }
    (voice_dir / "voice.timbra.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def delete_exported_voice(slug: str) -> bool:
    safe_slug = Path(slug).name
    voice_dir = EXPORTED_DIR / safe_slug
    if not voice_dir.exists() or not voice_dir.is_dir():
        return False
    shutil.rmtree(voice_dir)
    return True


def _build_narrate_py(voice_data: dict[str, Any]) -> str:
    slug = voice_data["slug"]
    engine = voice_data["engine"]

    if engine == "chatterbox":
        return f'''#!/usr/bin/env python3
"""Standalone Timbra Voice Narrator — {voice_data["name"]} (Chatterbox MLX)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

from dsp import MasteringParams, apply_mastering

HERE = Path(__file__).resolve().parent
VOICE = json.loads((HERE / "voice.timbra.json").read_text(encoding="utf-8"))
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model("mlx-community/chatterbox-8bit")
    return _MODEL


def narrate(text: str, output_path: str | Path | None = None) -> tuple[np.ndarray, int]:
    """Synthesize text using the exported '{slug}' Chatterbox MLX voice profile."""
    model = _get_model()
    sr = int(getattr(model, "sample_rate", 24000))
    cfg = VOICE.get("config", {{}})
    m = cfg.get("mastering", {{}})

    ref_file = HERE / "reference.wav"
    ref_arg = str(ref_file) if ref_file.exists() else None

    paragraphs = [p.strip() for p in re.split(r"\\n\\s*\\n+", text.strip()) if p.strip()]
    segments = []
    sent_pause = float(cfg.get("sentence_pause", 0.30))
    para_pause = float(cfg.get("paragraph_pause", 0.60))

    for pi, para in enumerate(paragraphs):
        chunks = [
            np.asarray(r.audio, dtype=np.float32).ravel()
            for r in model.generate(
                text=para,
                ref_audio=ref_arg,
                exaggeration=float(cfg.get("exaggeration", 0.55)),
                cfg_weight=float(cfg.get("cfg_weight", 0.50)),
                temperature=float(cfg.get("temperature", 0.80)),
                repetition_penalty=float(cfg.get("repetition_penalty", 1.20)),
                verbose=False,
            )
        ]
        if chunks:
            segments.append(np.concatenate(chunks))
        if pi < len(paragraphs) - 1 and para_pause > 0:
            segments.append(np.zeros(int(para_pause * sr), dtype=np.float32))

    dry = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    mastered = apply_mastering(
        dry,
        sr,
        MasteringParams(**{{k: v for k, v in m.items() if hasattr(MasteringParams, k)}}),
    )

    if output_path is not None:
        sf.write(str(output_path), mastered, sr, subtype="PCM_16")
    return mastered, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Narrate text using Timbra voice '{slug}'")
    parser.add_argument("text", help="Text to narrate")
    parser.add_argument("-o", "--output", default="{slug}.wav", help="Output WAV/MP3/FLAC path")
    args = parser.parse_args()
    narrate(args.text, args.output)
    print(f"Saved {{args.output}}")
'''

    return f'''#!/usr/bin/env python3
"""Standalone Timbra Voice Narrator — {voice_data["name"]} (Kokoro 82M baked style)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from dsp import MasteringParams, apply_mastering

HERE = Path(__file__).resolve().parent
VOICE = json.loads((HERE / "voice.timbra.json").read_text(encoding="utf-8"))
STYLE = np.load(str(HERE / "style.npy")).astype(np.float32)


def narrate(
    text: str,
    output_path: str | Path | None = None,
    model_path: str = "models/kokoro-v1.0.onnx",
    voices_path: str = "models/voices-v1.0.bin",
    speed_scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Synthesize text using the exported '{slug}' baked Kokoro style tensor + DSP chain."""
    kokoro = Kokoro(model_path, voices_path)
    cfg = VOICE.get("config", {{}})
    m = cfg.get("mastering", {{}})
    lang = cfg.get("lang", "en-us")
    speed = float(cfg.get("speed", 0.92)) * float(speed_scale)
    sent_pause = float(cfg.get("sentence_pause", 0.35))
    para_pause = float(cfg.get("paragraph_pause", 0.65))

    paragraphs = [p.strip() for p in re.split(r"\\n\\s*\\n+", text.strip()) if p.strip()]
    segments = []
    sr = 24000

    for pi, para in enumerate(paragraphs):
        sents = [s.strip() for s in re.split(r"(?<=[.!?…])\\s+", para) if s.strip()]
        for si, sent in enumerate(sents):
            seg, sr = kokoro.create(sent, voice=STYLE, speed=speed, lang=lang)
            segments.append(np.asarray(seg, dtype=np.float32))
            if si < len(sents) - 1 and sent_pause > 0:
                segments.append(np.zeros(int(sent_pause * sr), dtype=np.float32))
        if pi < len(paragraphs) - 1 and para_pause > 0:
            segments.append(np.zeros(int(para_pause * sr), dtype=np.float32))

    dry = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    mastered = apply_mastering(
        dry,
        sr,
        MasteringParams(**{{k: v for k, v in m.items() if hasattr(MasteringParams, k)}}),
    )

    if output_path is not None:
        sf.write(str(output_path), mastered, sr, subtype="PCM_16")
    return mastered, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Narrate text using Timbra voice '{slug}'")
    parser.add_argument("text", help="Text to narrate")
    parser.add_argument("-o", "--output", default="{slug}.wav", help="Output WAV/MP3/FLAC path")
    parser.add_argument("--model", default="models/kokoro-v1.0.onnx")
    parser.add_argument("--voices", default="models/voices-v1.0.bin")
    args = parser.parse_args()
    narrate(args.text, args.output, model_path=args.model, voices_path=args.voices)
    print(f"Saved {{args.output}}")
'''


def _build_bundle_readme(voice_data: dict[str, Any]) -> str:
    slug = voice_data["slug"]
    name = voice_data["name"]
    engine = voice_data["engine"]
    return f"""# Timbra Voice Pack — {name} (`{slug}`)

Engine: **{engine}**

## Option 1: Use from ANY App via Timbra's Local OpenAI-Compatible API

While Timbra is running (`./run.sh` at `http://127.0.0.1:8765`), any app supporting OpenAI TTS (`/v1/audio/speech`) can narrate with this voice immediately:

```bash
curl http://127.0.0.1:8765/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"timbra","voice":"{slug}","input":"Hello from my app!","response_format":"mp3"}}' \\
  --output narration.mp3
```

### JavaScript / Web App
```js
const res = await fetch("http://127.0.0.1:8765/v1/audio/speech", {{
  method: "POST",
  headers: {{ "Content-Type": "application/json" }},
  body: JSON.stringify({{ model: "timbra", voice: "{slug}", input: "Hello from JS!", response_format: "wav" }})
}});
const audio = new Audio(URL.createObjectURL(await res.blob()));
audio.play();
```

### Swift (macOS / iOS App)
```swift
var req = URLRequest(url: URL(string: "http://127.0.0.1:8765/v1/audio/speech")!)
req.httpMethod = "POST"
req.setValue("application/json", forHTTPHeaderField: "Content-Type")
req.httpBody = try JSONSerialization.data(withJSONObject: [
    "model": "timbra", "voice": "{slug}", "input": "Hello from Swift!", "response_format": "wav"
])
let (data, _) = try await URLSession.shared.data(for: req)
let player = try AVAudioPlayer(data: data)
player.play()
```

## Option 2: Run Standalone Offline in Python

This bundle includes `voice.timbra.json`, `dsp.py`, `narrate.py`, and {"`style.npy` (baked 510x1x256 Kokoro tensor)" if engine == "kokoro" else "`reference.wav`"}.

```bash
python narrate.py "Hello from standalone Python!" -o out.wav
```
"""


def build_zip_bundle(slug: str) -> bytes | None:
    loaded = load_exported_voice(slug)
    if loaded is None:
        return None
    voice_data, voice_dir = loaded

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("voice.timbra.json", json.dumps(voice_data, indent=2))
        zf.writestr("narrate.py", _build_narrate_py(voice_data))
        zf.writestr("README.md", _build_bundle_readme(voice_data))
        dsp_path = BASE_DIR / "dsp.py"
        if dsp_path.exists():
            zf.write(str(dsp_path), arcname="dsp.py")
        style_path = voice_dir / "style.npy"
        if style_path.exists():
            zf.write(str(style_path), arcname="style.npy")
        ref_path = voice_dir / "reference.wav"
        if ref_path.exists():
            zf.write(str(ref_path), arcname="reference.wav")

    return buf.getvalue()
