"""Studio-grade audio mastering DSP & acoustic reference style analyzer for Kokoro TTS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass
class MasteringParams:
    enabled: bool = True
    highpass_hz: float = 75.0       # Clean sub-bass rumble (20 - 180 Hz)
    warmth_db: float = 2.0          # Low-mid body around 210 Hz (-6 to +6 dB)
    presence_db: float = 1.5        # Vocal clarity around 3.0 kHz (-6 to +6 dB)
    air_db: float = 1.8             # Silky high shelf around 8.5 kHz (-6 to +6 dB)
    tube_warmth: float = 0.0        # Subtle analog/broadcast harmonic warmth (0.0 - 1.0)
    formant_shift: float = 0.0      # Vocal tract resonance shift (-12% to +12%)
    deesser: float = 0.35           # Sibilance reduction 0.0 - 1.0 (5.2-9.0 kHz)
    compression: float = 0.35       # Vocal dynamics smoothing 0.0 - 1.0
    room_ambience: float = 0.12     # Subtle studio room warmth 0.0 - 0.6
    pitch_semitones: float = 0.0    # Subtle pitch tweak (-3.0 to +3.0 st)
    target_lufs: float = -16.0      # Output loudness target (-24 to -10)
    # Humanizer tuners
    humanize_enabled: bool = True
    breath_amount: float = 0.25     # Organic inhales before phrase onsets (0.0 - 1.0)
    airiness: float = 0.15          # Close-mic aspiration / breathy texture (0.0 - 1.0)
    smile: float = 0.15             # Serious/chest (-1.0) <-> Warm Smile (+1.0)
    organic_jitter: float = 0.20    # Vocal-fold micro-shimmer & pitch drift (0.0 - 1.0)
    vocal_fry: float = 0.12         # Subtle relaxed phrase-end vocal fry (0.0 - 1.0)
    room_floor: float = 0.20        # Live-mic warm room tone in silences (0.0 - 1.0)


def _biquad_peaking(fs: int, f0: float, gain_db: float, q: float = 1.0):
    if abs(gain_db) < 0.05 or f0 <= 0 or f0 >= fs * 0.48:
        return None
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    return np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64), np.array(
        [1.0, a1 / a0, a2 / a0], dtype=np.float64
    )


def _biquad_high_shelf(fs: int, f0: float, gain_db: float, slope: float = 0.7):
    if abs(gain_db) < 0.05 or f0 <= 0 or f0 >= fs * 0.48:
        return None
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64), np.array(
        [1.0, a1 / a0, a2 / a0], dtype=np.float64
    )


def _envelope_follower(x: np.ndarray, fs: int, attack_ms: float, release_ms: float) -> np.ndarray:
    rect = np.abs(x)
    tau_s = max(release_ms, 5.0) / 1000.0
    alpha = math.exp(-1.0 / (fs * tau_s))
    env = signal.lfilter([1.0 - alpha], [1.0, -alpha], rect)
    tau_a = max(attack_ms, 1.0) / 1000.0
    alpha_a = math.exp(-1.0 / (fs * tau_a))
    env_fast = signal.lfilter([1.0 - alpha_a], [1.0, -alpha_a], rect)
    return np.maximum(env, env_fast * 0.7)


def _apply_tube_warmth(x: np.ndarray, fs: int, amount: float) -> np.ndarray:
    """Gentle broadcast preamp harmonic saturation (adds warm, slightly husky chestiness)."""
    if amount <= 0.01:
        return x
    # Split mid-band (140Hz - 4.5kHz) for gentle asymmetric even+odd harmonics
    nyq = fs * 0.5
    sos = signal.butter(2, [140.0 / nyq, min(4800.0 / nyq, 0.94)], btype="bandpass", output="sos")
    mid = signal.sosfilt(sos, x)
    drive = 1.0 + 3.2 * amount
    # Asymmetric soft-clipping introduces warm 2nd-order harmonics
    sat = np.tanh(mid * drive + 0.12 * amount) - math.tanh(0.12 * amount)
    sat /= drive * 0.85
    mix = 0.38 * amount
    return x + mix * (sat - mid)


def _apply_formant_shift(x: np.ndarray, fs: int, shift_pct: float) -> np.ndarray:
    """Shift vocal tract spectral envelope independently of pitch via cepstral liftering."""
    if abs(shift_pct) < 0.4 or len(x) < 2048:
        return x
    # Negative shift_pct -> deeper/larger vocal tract; positive -> brighter/smaller
    ratio = 1.0 + (shift_pct / 100.0)
    nperseg = 1024
    hop = 256
    _, _, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)
    n_bins, n_frames = Zxx.shape
    if n_frames < 2:
        return x

    mag = np.abs(Zxx) + 1e-8
    phase = np.angle(Zxx)
    log_mag = np.log(mag)

    # Extract smooth spectral envelope (vocal tract formants) using real cepstrum liftering
    # Mirror to full FFT length for real cepstrum
    full_log = np.concatenate([log_mag, log_mag[-2:0:-1, :]], axis=0)
    ceps = np.fft.irfft(full_log[:n_bins, :], n=nperseg, axis=0)
    # Keep low-quefrency components (< ~1.4ms -> smooth formant envelope)
    cutoff = max(14, int(fs * 0.0014))
    lifter = np.zeros(nperseg, dtype=np.float64)
    lifter[0] = 1.0
    lifter[1:cutoff] = 1.0
    lifter[-cutoff + 1 :] = 1.0
    # Cosine taper on lifter edge
    taper = 4
    if cutoff > taper + 2:
        w = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, taper)))
        lifter[cutoff - taper : cutoff] = w
        lifter[-cutoff + 1 : -cutoff + 1 + taper] = w[::-1]

    env_log = np.real(np.fft.rfft(ceps * lifter[:, None], axis=0))
    fine_log = log_mag - env_log

    # Warp spectral envelope along frequency axis
    bins = np.arange(n_bins, dtype=np.float64)
    src_bins = np.clip(bins / ratio, 0.0, n_bins - 1.0)
    b0 = np.floor(src_bins).astype(int)
    b1 = np.minimum(b0 + 1, n_bins - 1)
    frac = (src_bins - b0)[:, None]
    warped_env = (1.0 - frac) * env_log[b0, :] + frac * env_log[b1, :]

    new_mag = np.exp(np.clip(fine_log + warped_env, -18.0, 12.0))
    Zxx_new = new_mag * np.exp(1j * phase)
    _, y = signal.istft(Zxx_new, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float64)


def _apply_deesser(x: np.ndarray, fs: int, amount: float) -> np.ndarray:
    """Split-band de-esser: turn down the 4.5 kHz+ band only on frames dominated by sibilance.

    Works in the STFT domain so the cut is exact (no phase-shifted band subtraction). The
    maximum cut scales with `amount` (about 14 dB at 100%); vowels and voiced consonants,
    whose energy sits below 4 kHz, are untouched.
    """
    if amount <= 0.01 or len(x) < 1024:
        return x
    nperseg, hop = 512, 128
    f, _, Z = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)
    power = np.abs(Z) ** 2
    hi = f >= 4500.0
    if not np.any(hi):
        return x
    share_db = 10.0 * np.log10((power[hi].sum(axis=0) + 1e-12) / (power.sum(axis=0) + 1e-12))
    level_db = 10.0 * np.log10(power.sum(axis=0) + 1e-12)
    audible = level_db > level_db.max() - 45.0

    max_cut_db = 14.0 * amount
    cut_db = np.clip((share_db + 8.0) * 1.3, 0.0, max_cut_db) * audible
    # Smooth over ~15 ms so the cut doesn't flutter between frames
    cut_db = np.convolve(cut_db, np.ones(3) / 3, mode="same")

    weight = np.clip((f - 3500.0) / 1500.0, 0.0, 1.0)[:, None]  # soft 3.5-5 kHz transition
    gain = 10.0 ** (-(weight * cut_db[None, :]) / 20.0)
    _, y = signal.istft(Z * gain, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def _apply_compressor(x: np.ndarray, fs: int, amount: float) -> np.ndarray:
    """Soft-knee compressor whose threshold follows the speech level, so it acts at any input level."""
    if amount <= 0.01:
        return x
    env = _envelope_follower(x, fs, attack_ms=10.0, release_ms=95.0) + 1e-7
    env_db = 20.0 * np.log10(env)

    speech = env_db[env_db > env_db.max() - 40.0]
    speech_ref_db = float(np.percentile(speech, 90)) if len(speech) else float(env_db.max())
    threshold_db = speech_ref_db - (3.0 + 12.0 * amount)
    ratio = 1.5 + 3.0 * amount
    knee_db = 6.0

    over = env_db - threshold_db
    gain_reduction_db = np.zeros_like(env_db)
    above = over > knee_db / 2.0
    in_knee = (over >= -knee_db / 2.0) & (~above)

    gain_reduction_db[above] = over[above] * (1.0 - 1.0 / ratio)
    gain_reduction_db[in_knee] = (
        (1.0 - 1.0 / ratio) * ((over[in_knee] + knee_db / 2.0) ** 2) / (2.0 * knee_db)
    )

    y = x * (10.0 ** (-gain_reduction_db / 20.0))
    rms_in = float(np.sqrt(np.mean(x ** 2)) + 1e-7)
    rms_out = float(np.sqrt(np.mean(y ** 2)) + 1e-7)
    return y * min(rms_in / rms_out, 4.0)


def _apply_room_ambience(x: np.ndarray, fs: int, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return x
    ir_len = int(fs * 0.16)
    rng = np.random.default_rng(42)
    t = np.arange(ir_len, dtype=np.float64) / fs
    ir = rng.standard_normal(ir_len) * np.exp(-t / 0.038) * 0.12
    for ms, amp in [(7.2, 0.32), (13.5, 0.24), (21.0, 0.18), (31.4, 0.12)]:
        idx = int(fs * ms / 1000.0)
        if idx < ir_len:
            ir[idx] += amp

    sos_lp = signal.butter(2, 4200.0 / (fs * 0.5), btype="lowpass", output="sos")
    ir = signal.sosfilt(sos_lp, ir)
    ir /= np.sqrt(np.sum(ir ** 2) + 1e-9)

    wet = signal.fftconvolve(x, ir, mode="full")[: len(x)]
    mix = min(amount, 0.55) * 0.42
    return (1.0 - mix * 0.25) * x + mix * wet


def _apply_pitch_shift(x: np.ndarray, fs: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 0.05 or len(x) < 2048:
        return x
    factor = 2.0 ** (semitones / 12.0)
    nperseg = 1024
    hop = 256
    _, _, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)
    n_bins, n_frames = Zxx.shape
    if n_frames < 4:
        return x

    time_steps = np.arange(0, n_frames - 1, 1.0 / factor)
    phase_advance = 2.0 * np.pi * hop * np.arange(n_bins) / nperseg
    phase_acc = np.angle(Zxx[:, 0])
    out = np.zeros((n_bins, len(time_steps)), dtype=np.complex128)

    for idx, step in enumerate(time_steps):
        i0 = int(math.floor(step))
        i1 = min(i0 + 1, n_frames - 1)
        frac = step - i0
        mag = (1.0 - frac) * np.abs(Zxx[:, i0]) + frac * np.abs(Zxx[:, i1])
        dp = np.angle(Zxx[:, i1]) - np.angle(Zxx[:, i0]) - phase_advance
        dp = dp - 2.0 * np.pi * np.round(dp / (2.0 * np.pi))
        phase_acc += phase_advance + dp
        out[:, idx] = mag * np.exp(1j * phase_acc)

    _, stretched = signal.istft(out, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)
    shifted = signal.resample(stretched, len(x))
    return shifted.astype(np.float64)


def integrated_lufs(x: np.ndarray, fs: int) -> float:
    """ITU-R BS.1770-4 integrated loudness: K-weighting, 400 ms blocks, absolute + relative gating."""
    if len(x) < int(fs * 0.4):
        return -70.0
    # K-weighting stage 1: high shelf (+4 dB above ~1.5 kHz)
    a_ = 10.0 ** (4.0 / 40.0)
    w0 = 2.0 * math.pi * 1500.0 / fs
    alpha = math.sin(w0) / (2.0 * (1.0 / math.sqrt(2.0)))
    c = math.cos(w0)
    b = [a_ * ((a_ + 1) + (a_ - 1) * c + 2 * math.sqrt(a_) * alpha),
         -2 * a_ * ((a_ - 1) + (a_ + 1) * c),
         a_ * ((a_ + 1) + (a_ - 1) * c - 2 * math.sqrt(a_) * alpha)]
    a = [(a_ + 1) - (a_ - 1) * c + 2 * math.sqrt(a_) * alpha,
         2 * ((a_ - 1) - (a_ + 1) * c),
         (a_ + 1) - (a_ - 1) * c - 2 * math.sqrt(a_) * alpha]
    y = signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x)
    # K-weighting stage 2: high-pass at ~38 Hz
    w0 = 2.0 * math.pi * 38.0 / fs
    alpha = math.sin(w0) / (2.0 * 0.5)
    c = math.cos(w0)
    b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    a = [1 + alpha, -2 * c, 1 - alpha]
    y = signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], y)

    block, hop = int(0.4 * fs), int(0.1 * fs)
    z = np.array([np.mean(y[i : i + block] ** 2) for i in range(0, len(y) - block + 1, hop)])
    loud = -0.691 + 10.0 * np.log10(z + 1e-12)
    z = z[loud > -70.0]
    if len(z) == 0:
        return -70.0
    relative_gate = -0.691 + 10.0 * math.log10(float(np.mean(z))) - 10.0
    z = z[(-0.691 + 10.0 * np.log10(z + 1e-12)) > relative_gate]
    return -0.691 + 10.0 * math.log10(float(np.mean(z)) + 1e-12) if len(z) else -70.0


def _soft_limit(y: np.ndarray, knee: float = 0.80, ceiling: float = 0.94) -> np.ndarray:
    mag = np.abs(y)
    over = mag > knee
    if np.any(over):
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + (ceiling - knee) * np.tanh((mag[over] - knee) / (ceiling - knee)))
    return y


def _normalize_and_limit(x: np.ndarray, fs: int, target_lufs: float) -> np.ndarray:
    """Normalize to a true BS.1770 integrated-loudness target, then soft-limit peaks.

    The limiter shaves a little loudness off hot peaks, so the gain is corrected once
    after limiting to land within a few tenths of a dB of the target.
    """
    if len(x) == 0:
        return x
    current = integrated_lufs(x, fs)
    if current <= -69.0:
        return x
    gain_db = float(np.clip(target_lufs - current, -24.0, 30.0))
    y = _soft_limit(x * (10.0 ** (gain_db / 20.0)))
    error_db = target_lufs - integrated_lufs(y, fs)
    if abs(error_db) > 0.2:
        y = _soft_limit(x * (10.0 ** ((gain_db + error_db) / 20.0)))
    return y


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Fast 1/f-ish warm pink noise via simple Paul Kellet filter approximation."""
    white = rng.standard_normal(n)
    # 1-pole low-pass cascade gives smooth natural pink tilt
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float64)
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float64)
    pink = signal.lfilter(b, a, white)
    pink /= float(np.std(pink) + 1e-8)
    return pink


def _detect_speech_regions(x: np.ndarray, fs: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return 100Hz envelope mask and list of (start_sample, end_sample) speech segments."""
    env = _envelope_follower(x, fs, attack_ms=12.0, release_ms=55.0)
    hop = max(1, fs // 100)  # 10ms frames
    env_ds = env[::hop]
    if len(env_ds) < 5:
        return env, [(0, len(x))]
    peak_env = float(np.percentile(env_ds, 96) + 1e-7)
    thresh = max(0.008, peak_env * 0.065)
    active = env_ds > thresh

    # Bridge tiny consonant gaps (< 90ms)
    min_gap_frames = 9
    i = 0
    while i < len(active):
        if not active[i]:
            j = i
            while j < len(active) and not active[j]:
                j += 1
            if i > 0 and j < len(active) and (j - i) < min_gap_frames:
                active[i:j] = True
            i = j
        else:
            i += 1

    regions: list[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) >= 12:  # >= 120ms real speech
                regions.append((i * hop, min(len(x), j * hop)))
            i = j
        else:
            i += 1
    return env, regions


def _synthesize_breath(fs: int, dur_s: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a soft, natural vocal-tract-filtered inhale (never harsh white noise)."""
    n = max(64, int(fs * dur_s))
    raw = _pink_noise(n, rng)
    nyq = fs * 0.5
    # Vocal tract inhale formants: soft breath body (420Hz), pharyngeal air (1350Hz), soft lip air (3100Hz)
    sos1 = signal.butter(2, [260.0 / nyq, 720.0 / nyq], btype="bandpass", output="sos")
    sos2 = signal.butter(2, [1050.0 / nyq, 2100.0 / nyq], btype="bandpass", output="sos")
    sos3 = signal.butter(2, [2500.0 / nyq, min(5200.0 / nyq, 0.94)], btype="bandpass", output="sos")
    breath = (
        0.50 * signal.sosfilt(sos1, raw)
        + 0.35 * signal.sosfilt(sos2, raw)
        + 0.22 * signal.sosfilt(sos3, raw)
    )
    # Asymmetric natural inhale envelope: gentle rise, soft peak near 65%, smooth taper before speech
    t = np.linspace(0.0, 1.0, n)
    env = np.sin(np.pi * (t ** 0.78)) ** 1.6
    breath *= env
    breath /= float(np.sqrt(np.mean(breath ** 2)) + 1e-8)
    return breath


def _apply_natural_breaths(
    x: np.ndarray,
    fs: int,
    amount: float,
    regions: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Place subtle, randomized vocal-tract inhales in pauses before new phrases."""
    if amount <= 0.01 or len(x) < int(fs * 0.5):
        return x
    if regions is None:
        _, regions = _detect_speech_regions(x, fs)
    if len(regions) < 1:
        return x

    out = x.copy()
    speech_rms = float(np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2)) + 1e-7) if np.any(np.abs(x) > 0.01) else 0.05
    rng = np.random.default_rng(107)

    for idx, (start, _end) in enumerate(regions):
        # Need a genuine pause gap before this phrase (>= 240ms after previous phrase, or leading silence)
        prev_end = regions[idx - 1][1] if idx > 0 else 0
        gap_samples = start - prev_end
        if idx == 0 and gap_samples < int(fs * 0.22):
            continue
        if idx > 0 and gap_samples < int(fs * 0.25):
            continue

        # Organic probability/variation: longer pauses almost always get a soft breath
        gap_s = gap_samples / fs
        dur_s = float(np.clip(rng.uniform(0.14, 0.23), 0.12, min(0.26, gap_s * 0.65)))
        breath = _synthesize_breath(fs, dur_s, rng)

        # Level-match subtly below speech (-30 dB at low amount up to -19 dB at high amount)
        level = speech_rms * (0.025 + 0.085 * amount) * float(rng.uniform(0.82, 1.18))
        lead_gap = int(fs * rng.uniform(0.025, 0.055))
        b_end = max(0, start - lead_gap)
        b_start = max(prev_end + int(fs * 0.02), b_end - len(breath))
        actual_len = b_end - b_start
        if actual_len > int(fs * 0.08):
            seg = breath[-actual_len:] * level
            out[b_start:b_end] += seg
    return out


def _apply_airiness(x: np.ndarray, fs: int, amount: float) -> np.ndarray:
    """Add envelope-tracked vocal aspiration/breathiness during active speech."""
    if amount <= 0.01 or len(x) < 512:
        return x
    rng = np.random.default_rng(211)
    noise = _pink_noise(len(x), rng)
    nyq = fs * 0.5
    sos = signal.butter(3, [2100.0 / nyq, min(7800.0 / nyq, 0.95)], btype="bandpass", output="sos")
    asp = signal.sosfilt(sos, noise)
    asp /= float(np.std(asp) + 1e-8)

    env = _envelope_follower(x, fs, attack_ms=8.0, release_ms=40.0)
    # Soft gate so aspiration only rides on active voice, never silence
    env_peak = float(np.percentile(env, 95) + 1e-7)
    gate = np.clip((env / env_peak - 0.06) / 0.30, 0.0, 1.0)
    mod = (env ** 0.85) * gate
    return x + asp * mod * (0.15 * amount)


def _apply_vocal_smile_eq(x: np.ndarray, fs: int, smile: float) -> np.ndarray:
    """Cheek/smile resonance EQ (-1.0 serious/chest <-> +1.0 warm broadcast smile)."""
    if abs(smile) < 0.03:
        return x
    y = x
    eq1 = _biquad_peaking(fs, 1750.0, float(smile * 2.2), q=1.1)
    if eq1 is not None:
        y = signal.lfilter(eq1[0], eq1[1], y)
    eq2 = _biquad_peaking(fs, 4200.0, float(smile * 1.8), q=1.2)
    if eq2 is not None:
        y = signal.lfilter(eq2[0], eq2[1], y)
    eq_chest = _biquad_peaking(fs, 165.0, float(-smile * 1.6), q=0.9)
    if eq_chest is not None:
        y = signal.lfilter(eq_chest[0], eq_chest[1], y)
    return y


def _apply_organic_jitter(x: np.ndarray, fs: int, amount: float) -> np.ndarray:
    """Subtle vocal-fold micro-shimmer and micro-pitch drift (breaks flat synthetic stillness)."""
    if amount <= 0.01 or len(x) < 1024:
        return x
    n = len(x)
    rng = np.random.default_rng(307)

    # 1. Slow organic amplitude shimmer (2 - 6 Hz random walk, max ±3.5% at 100%)
    sos_slow = signal.butter(2, 5.5 / (fs * 0.5), btype="lowpass", output="sos")
    shimmer = signal.sosfilt(sos_slow, rng.standard_normal(n))
    shimmer /= float(np.std(shimmer) + 1e-8)
    amp_mod = 1.0 + np.clip(shimmer * 0.022 * amount, -0.07, 0.07)

    # 2. Ultra-subtle micro-pitch drift via smooth fractional sample warp (max ±0.45ms jitter)
    sos_drift = signal.butter(2, 3.2 / (fs * 0.5), btype="lowpass", output="sos")
    drift = signal.sosfilt(sos_drift, rng.standard_normal(n))
    drift /= float(np.std(drift) + 1e-8)
    max_shift_samples = (fs * 0.00042) * amount
    idx = np.arange(n, dtype=np.float64) + np.clip(drift * max_shift_samples, -max_shift_samples * 2, max_shift_samples * 2)
    idx = np.clip(idx, 0.0, n - 1.0001)
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    warped = (1.0 - frac) * x[i0] + frac * x[i0 + 1]
    return warped * amp_mod


def _apply_vocal_fry(
    x: np.ndarray,
    fs: int,
    amount: float,
    regions: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Add subtle relaxed low-frequency glottal creak at the trailing tails of phrases."""
    if amount <= 0.01 or len(x) < int(fs * 0.4):
        return x
    if regions is None:
        _, regions = _detect_speech_regions(x, fs)
    if not regions:
        return x

    out = x.copy()
    rng = np.random.default_rng(409)
    for start, end in regions:
        seg_len = end - start
        if seg_len < int(fs * 0.45):
            continue
        # Target the last 130 - 210ms of the phrase tail
        tail_n = min(int(fs * rng.uniform(0.13, 0.21)), seg_len // 3)
        t_start = end - tail_n
        t = np.arange(tail_n, dtype=np.float64) / fs
        # Irregular glottal pulse train around 48 - 66 Hz
        freq = float(rng.uniform(48.0, 65.0))
        phase = 2.0 * np.pi * freq * t + 0.6 * np.sin(2.0 * np.pi * 7.5 * t)
        # Soft pulsing modulation (dips amplitude gently like relaxed glottal fry)
        pulses = 0.5 + 0.5 * np.cos(phase)
        fry_depth = 0.42 * amount
        fade_in = np.linspace(0.0, 1.0, tail_n) ** 1.5
        mod = 1.0 - fade_in * fry_depth * (1.0 - pulses ** 2.2)
        out[t_start:end] = out[t_start:end] * mod
    return out


def _apply_room_floor(x: np.ndarray, fs: int, amount: float) -> np.ndarray:
    """Replace sterile digital zero silence with subtle warm studio room-tone floor."""
    if amount <= 0.01 or len(x) < 256:
        return x
    rng = np.random.default_rng(521)
    pink = _pink_noise(len(x), rng)
    nyq = fs * 0.5
    sos = signal.butter(2, [55.0 / nyq, min(4500.0 / nyq, 0.9)], btype="bandpass", output="sos")
    room = signal.sosfilt(sos, pink)
    room /= float(np.std(room) + 1e-8)
    db = -58.0 + 15.0 * amount
    amp = 10.0 ** (db / 20.0)
    return x + room * amp


def apply_mastering(audio: np.ndarray, fs: int, params: MasteringParams) -> np.ndarray:
    if len(audio) == 0:
        return audio.astype(np.float32)

    x = np.asarray(audio, dtype=np.float64)
    x = x - np.mean(x)

    # Combine formant shift (from Mastering) + smile formant lift (from Humanize) into ONE STFT pass
    smile_val = float(np.clip(params.smile, -1.0, 1.0)) if params.humanize_enabled else 0.0
    base_formant = float(np.clip(params.formant_shift, -12.0, 12.0)) if params.enabled else 0.0
    total_formant = float(np.clip(base_formant + smile_val * 3.6, -12.0, 12.0))

    # Humanizer Stage A: Organic vocal micro-physics (compute speech regions ONCE)
    if params.humanize_enabled:
        need_regions = params.vocal_fry > 0.01 or params.breath_amount > 0.01
        regions = _detect_speech_regions(x, fs)[1] if need_regions else None
        x = _apply_organic_jitter(x, fs, float(np.clip(params.organic_jitter, 0.0, 1.0)))
        if params.vocal_fry > 0.01:
            x = _apply_vocal_fry(x, fs, float(np.clip(params.vocal_fry, 0.0, 1.0)), regions=regions)
        x = _apply_vocal_smile_eq(x, fs, smile_val)
        x = _apply_airiness(x, fs, float(np.clip(params.airiness, 0.0, 1.0)))
        if params.breath_amount > 0.01:
            x = _apply_natural_breaths(x, fs, float(np.clip(params.breath_amount, 0.0, 1.0)), regions=regions)

    # Single merged pitch + formant pass
    if params.enabled and abs(params.pitch_semitones) >= 0.05:
        x = _apply_pitch_shift(x, fs, params.pitch_semitones)
    if abs(total_formant) >= 0.4:
        x = _apply_formant_shift(x, fs, total_formant)

    if not params.enabled:
        if params.humanize_enabled:
            x = _apply_room_floor(x, fs, float(np.clip(params.room_floor, 0.0, 1.0)))
        peak = float(np.max(np.abs(x)) + 1e-8)
        if peak > 0.98:
            x = x * (0.98 / peak)
        return x.astype(np.float32)

    # 2. High-pass rumble filter
    hp_hz = float(np.clip(params.highpass_hz, 20.0, 220.0))
    if hp_hz > 22.0:
        sos_hp = signal.butter(2, hp_hz / (fs * 0.5), btype="highpass", output="sos")
        x = signal.sosfilt(sos_hp, x)

    # 3. Warmth EQ (210 Hz)
    eq_warm = _biquad_peaking(fs, 210.0, float(np.clip(params.warmth_db, -8.0, 8.0)), q=0.8)
    if eq_warm is not None:
        x = signal.lfilter(eq_warm[0], eq_warm[1], x)

    # 4. Broadcast Tube Warmth / Husky saturation
    x = _apply_tube_warmth(x, fs, float(np.clip(params.tube_warmth, 0.0, 1.0)))

    # 5. Presence EQ (3.1 kHz)
    eq_pres = _biquad_peaking(fs, 3100.0, float(np.clip(params.presence_db, -8.0, 8.0)), q=1.1)
    if eq_pres is not None:
        x = signal.lfilter(eq_pres[0], eq_pres[1], x)

    # 6. De-Esser
    x = _apply_deesser(x, fs, float(np.clip(params.deesser, 0.0, 1.0)))

    # 7. Air / Silk High Shelf (8.5 kHz)
    eq_air = _biquad_high_shelf(fs, 8500.0, float(np.clip(params.air_db, -8.0, 8.0)), slope=0.7)
    if eq_air is not None:
        x = signal.lfilter(eq_air[0], eq_air[1], x)

    # 8. Vocal Compressor
    x = _apply_compressor(x, fs, float(np.clip(params.compression, 0.0, 1.0)))

    # 9. Studio Room Ambience
    x = _apply_room_ambience(x, fs, float(np.clip(params.room_ambience, 0.0, 0.6)))

    # 10. Loudness Normalization & Soft Limiter
    x = _normalize_and_limit(x, fs, float(np.clip(params.target_lufs, -24.0, -10.0)))

    # 11. Live-Mic Studio Room Tone Floor
    if params.humanize_enabled:
        x = _apply_room_floor(x, fs, float(np.clip(params.room_floor, 0.0, 1.0)))

    fade_n = min(int(fs * 0.004), len(x) // 4)
    if fade_n > 1:
        fade = np.linspace(0.0, 1.0, fade_n)
        x[:fade_n] *= fade
        x[-fade_n:] *= fade[::-1]

    return x.astype(np.float32)


def compute_waveform_peaks(audio: np.ndarray, num_bars: int = 180) -> list[float]:
    if len(audio) == 0:
        return [0.0] * num_bars
    x = np.abs(audio)
    chunks = np.array_split(x, num_bars)
    peaks = [float(np.percentile(c, 92)) if len(c) > 0 else 0.0 for c in chunks]
    m = max(peaks) if peaks else 1.0
    if m < 1e-6:
        m = 1.0
    return [round(p / m, 4) for p in peaks]


# ---------- Reference Voice Acoustic Analyzer ----------

def _estimate_f0_hz(x: np.ndarray, fs: int) -> tuple[float, float]:
    """Estimate median fundamental frequency (Hz) and pitch variation (semitone std) via autocorrelation."""
    frame_len = int(fs * 0.04)  # 40ms frames
    hop = int(fs * 0.015)
    min_lag = max(2, int(fs / 380.0))  # 380 Hz max
    max_lag = min(frame_len - 2, int(fs / 65.0))  # 65 Hz min
    if len(x) < frame_len or min_lag >= max_lag:
        return 175.0, 2.0

    # Bandpass 65 - 600 Hz to isolate fundamental cleanly
    sos = signal.butter(3, [65.0 / (fs * 0.5), min(600.0 / (fs * 0.5), 0.9)], btype="bandpass", output="sos")
    xf = signal.sosfilt(sos, x)

    pitches: list[float] = []
    rms_all = float(np.sqrt(np.mean(xf ** 2)) + 1e-8)
    win = np.hanning(frame_len)

    for start in range(0, len(xf) - frame_len, hop):
        frame = xf[start : start + frame_len]
        if float(np.sqrt(np.mean(frame ** 2))) < rms_all * 0.35:
            continue
        frame_w = frame * win
        ac = np.correlate(frame_w, frame_w, mode="full")[frame_len - 1 :]
        if ac[0] <= 1e-9:
            continue
        ac /= ac[0]
        region = ac[min_lag:max_lag]
        peak_idx = int(np.argmax(region))
        if region[peak_idx] > 0.42:
            lag = min_lag + peak_idx
            pitches.append(fs / lag)

    if len(pitches) < 3:
        return 175.0, 2.0
    arr = np.array(pitches, dtype=np.float64)
    med = float(np.median(arr))
    semitones = 12.0 * np.log2(np.clip(arr / max(med, 50.0), 0.25, 4.0))
    std_st = float(np.std(semitones))
    return round(med, 1), round(std_st, 2)


def analyze_voice_profile(audio: np.ndarray, fs: int) -> dict[str, Any]:
    """Extract pitch, spectral band energies, vocal brightness, dynamics, and speaking cadence."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x - np.mean(x)

    # Resample to 24 kHz for consistent spectral comparisons
    if fs != 24000 and len(x) > 0:
        target_n = int(round(len(x) * 24000 / fs))
        x = signal.resample(x, max(1, target_n))
        fs = 24000

    # Limit analysis to first 35 seconds for speed
    if len(x) > fs * 35:
        x = x[: fs * 35]

    peak = float(np.max(np.abs(x)) + 1e-8)
    x = x / peak

    # 1. Pitch & expressiveness
    f0_hz, f0_std_st = _estimate_f0_hz(x, fs)

    # 2. Spectral band energies on active speech frames
    f, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    Pxx = np.maximum(Pxx, 1e-12)

    def band_db(lo: float, hi: float) -> float:
        mask = (f >= lo) & (f <= hi)
        if not np.any(mask):
            return -60.0
        return float(10.0 * np.log10(np.mean(Pxx[mask]) + 1e-12))

    full_db = band_db(100.0, 9500.0)
    warmth_rel = band_db(140.0, 360.0) - full_db
    mid_rel = band_db(600.0, 1800.0) - full_db
    presence_rel = band_db(2200.0, 4400.0) - full_db
    sib_rel = band_db(5200.0, 8800.0) - full_db
    air_rel = band_db(7500.0, 11000.0) - full_db

    # 16-band log-mel-like spectral fingerprint for cosine similarity matching
    edges = np.geomspace(120.0, 9500.0, 17)
    fp = np.array([band_db(edges[i], edges[i + 1]) - full_db for i in range(16)], dtype=np.float64)

    # 3. Speaking cadence & pause estimation via syllable envelope peaks
    env = _envelope_follower(x, fs, attack_ms=15.0, release_ms=65.0)
    env_ds = env[::240]  # 100 Hz envelope
    thresh = float(np.percentile(env_ds, 25) + 0.08 * (np.percentile(env_ds, 95) - np.percentile(env_ds, 25)))
    active_mask = env_ds > max(thresh, 0.025)
    speech_ratio = float(np.mean(active_mask))

    # Count syllable-like energy peaks per active speech second
    peaks_idx, _ = signal.find_peaks(env_ds, height=max(thresh * 1.4, 0.04), distance=12)
    active_sec = max(0.5, float(np.sum(active_mask)) / 100.0)
    syllables_per_sec = float(len(peaks_idx) / active_sec)

    # Dynamic range (crest/compression cue)
    active_env = env_ds[active_mask]
    if len(active_env) > 10:
        dyn_db = float(20.0 * np.log10((np.percentile(active_env, 92) + 1e-6) / (np.percentile(active_env, 25) + 1e-6)))
    else:
        dyn_db = 12.0

    return {
        "duration_s": round(len(x) / fs, 2),
        "f0_hz": f0_hz,
        "f0_std_st": f0_std_st,
        "warmth_rel": round(warmth_rel, 2),
        "mid_rel": round(mid_rel, 2),
        "presence_rel": round(presence_rel, 2),
        "sib_rel": round(sib_rel, 2),
        "air_rel": round(air_rel, 2),
        "speech_ratio": round(speech_ratio, 3),
        "syllables_per_sec": round(syllables_per_sec, 2),
        "dynamic_range_db": round(dyn_db, 2),
        "fingerprint": [round(float(v), 3) for v in fp],
    }
