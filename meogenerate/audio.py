from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def read_wav(path: str | Path) -> AudioData:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            raw = wav.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError(f"Only uncompressed PCM WAV is supported: {path}") from exc

    samples = _decode_pcm(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return AudioData(np.asarray(samples, dtype=np.float32), sample_rate)


def wav_duration(path: str | Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except (wave.Error, OSError, ZeroDivisionError):
        return None


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = np.round(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate or samples.size == 0:
        return samples.copy()
    target_size = max(1, int(round(samples.size * target_rate / source_rate)))
    old_positions = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    new_positions = np.linspace(0.0, 1.0, target_size, endpoint=False)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def crop_loudest_window(samples: np.ndarray, maximum_samples: int) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float32)
    maximum_samples = max(1, int(maximum_samples))
    if result.size <= maximum_samples:
        return result.copy()
    power = np.square(result, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    energies = cumulative[maximum_samples:] - cumulative[:-maximum_samples]
    start = int(np.argmax(energies))
    return result[start : start + maximum_samples].copy()


def apply_fade(samples: np.ndarray, sample_rate: int, fade_ms: float = 12.0) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float32).copy()
    fade_count = min(result.size // 2, max(0, int(sample_rate * fade_ms / 1000.0)))
    if fade_count:
        fade = np.linspace(0.0, 1.0, fade_count, dtype=np.float32)
        result[:fade_count] *= fade
        result[-fade_count:] *= fade[::-1]
    return result


def normalize_rms(
    samples: np.ndarray,
    target_db: float = -19.0,
    min_gain: float = 0.35,
    max_gain: float = 3.0,
) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float32)
    if result.size == 0:
        return result.copy()
    rms = float(np.sqrt(np.mean(np.square(result, dtype=np.float64))))
    if rms < 1e-5:
        return result.copy()
    target = 10.0 ** (target_db / 20.0)
    gain = float(np.clip(target / rms, min_gain, max_gain))
    return result * gain


def limit_peak(samples: np.ndarray, peak: float = 0.94) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float32)
    if result.size == 0:
        return result.copy()
    current = float(np.max(np.abs(result)))
    if current > peak and current > 0.0:
        return result * (peak / current)
    return result.copy()
