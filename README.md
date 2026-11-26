# Voice Studio — Kokoro 82M (MLX) & Chatterbox MLX

A 100% local, zero-API-key **Voice Design, Cloning & Studio Mastering Console** built for Apple Silicon Macs.

Runs **Kokoro-82M** natively on Apple Silicon Metal GPU via MLX (**~10× faster than real time**) alongside **Chatterbox MLX** for expressive acting and zero-shot voice & accent cloning — all connected to a real-time **Studio Mastering** and **Vocal Humanizer** DSP chain.

---

## Features

- **Dual Local Engines (instant top-bar toggle)**
  - **Kokoro 82M (`MLX Metal GPU`)** — 54 voices across 9 languages, multi-voice style vector blending (up to 4 voices), **Split Timbre & Prosody** (borrow cadence from one voice while keeping another's vocal tone), prosody expressiveness, and phrasing cadence bias.
  - **Chatterbox MLX (`mlx-community/chatterbox-8bit`)** — expressive zero-shot voice & accent cloning from any 5–15 s `WAV` / `MP3` / `M4A` / `FLAC` / `OGG` clip, with live **Emotion Exaggeration**, **Clone Adherence / CFG**, **Spontaneity**, and **Repetition Penalty** controls plus a local Reference Voice Library.
- **Real-Time Live Sync & Instant A/B Waveform Deck**
  - Every right-panel slider and preset pill live-updates the left waveform deck.
  - Non-blocking background Raw A/B generation lets you toggle **A (Raw)** vs **B (Tuned + Mastered)** mid-playback at the same playhead position.
- **`03 · Studio Mastering` (Live ~150 ms DSP re-mastering)**
  - 3-Band Vocal EQ (**Warmth 210 Hz**, **Presence 3.1 kHz**, **Air/Silk 8.5 kHz**), **Tube Warmth** (broadcast preamp harmonics), **Vocal Depth** (cepstral formant shift), **De-Esser**, **Vocal Compression**, **Studio Booth Ambience**, **Pitch Shift**, **High-Pass**, and **LUFS Loudness** normalization.
- **`04 · Vocal Humanizer` (Live vocal-physics tuners)**
  - **Natural Breaths** (vocal-tract-filtered phrase inhales), **Close-Mic Airiness**, **Vocal Smile** (chest ↔ smile resonance), **Organic Shimmer & Drift**, **Phrase-End Vocal Fry**, **Live-Mic Room Tone**, and **Uneven "Thinking" Rhythm**.
- **Reference Acoustic Style Matcher**
  - Upload any speech clip to analyze pitch (`F0`), expressiveness, speaking pace, dynamic range, and 16-band spectral fingerprint and auto-tune Kokoro's closest voice blend + EQ/formant profile.
- **Custom Presets & Export Python**
  - Save custom voice presets locally or copy a self-contained Python snippet reproducing your exact blend, pacing, humanizer, and mastering settings.

---

## Quick Start (Apple Silicon macOS)

Requires **Python 3.10 – 3.13** on Apple Silicon (`arm64`).

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

Then open **http://127.0.0.1:8765** in your browser.

---

## Architecture

| File | Role |
| --- | --- |
| `server.py` | FastAPI backend — MLX Metal GPU Kokoro-82M, lazy Chatterbox MLX loader, LRU phoneme cache, Reference Voice Library, acoustic voice matcher, and live `/api/remaster` endpoint |
| `dsp.py` | Pure `scipy`/`numpy` studio mastering chain, vocal humanizer DSP processors, and acoustic voice profiler |
| `voices_meta.py` | Metadata for all 54 Kokoro v1.0 voices and curated presets |
| `static/` | Single-page studio console UI (`index.html`, `styles.css`, `app.js`) |
