from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import (
    apply_fade,
    crop_loudest_window,
    limit_peak,
    normalize_rms,
    read_wav,
    resample_linear,
)
from .planner import BehaviorPlan, PlanEvent


@dataclass(frozen=True)
class RenderResult:
    samples: np.ndarray
    sample_rate: int
    events: tuple[dict[str, Any], ...]

    @property
    def duration(self) -> float:
        return self.samples.size / float(self.sample_rate)


@dataclass(frozen=True)
class VoiceProfile:
    pitch_semitones: float
    syllables_per_second: float
    overlap_ms: tuple[float, float]
    pitch_jitter: float
    flutter_depth: float
    roughness: float


# Higher pitch, restrained distortion and overlapping syllables create the cute,
# connected creature-voice character. Aggressive states remain denser but are not
# allowed to collapse into growls or noise.
VOICE_PROFILES: dict[str, VoiceProfile] = {
    "affectionate": VoiceProfile(7.0, 4.0, (80.0, 125.0), 0.60, 0.055, 0.004),
    "angry": VoiceProfile(4.0, 4.6, (70.0, 110.0), 0.90, 0.085, 0.032),
    "demanding": VoiceProfile(6.5, 4.9, (75.0, 118.0), 0.70, 0.065, 0.008),
    "distressed": VoiceProfile(5.5, 3.4, (90.0, 140.0), 0.65, 0.075, 0.014),
    "fearful": VoiceProfile(7.5, 4.3, (80.0, 125.0), 0.95, 0.090, 0.018),
    "fight": VoiceProfile(3.5, 5.3, (60.0, 100.0), 1.10, 0.105, 0.050),
    "neutral": VoiceProfile(6.0, 3.7, (90.0, 140.0), 0.50, 0.050, 0.004),
    "startled": VoiceProfile(8.5, 4.8, (70.0, 110.0), 0.80, 0.080, 0.012),
}

# Broad output-space formants. Picking a different pair for each grain turns the
# source voice into changing nonsense syllables rather than replaying words.
_VOWEL_FORMANTS = (
    (470.0, 2050.0),
    (530.0, 2200.0),
    (590.0, 1900.0),
    (650.0, 1650.0),
    (710.0, 1480.0),
)
_MEOW_NASAL_FORMANTS = (310.0, 1080.0)
_MEOW_BRIGHT_FORMANTS = (500.0, 2050.0)
_MEOW_ROUNDED_FORMANTS = (740.0, 1260.0)


def _active_voice_centers(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Find energetic, voice-like source positions without retaining source pauses."""
    frame_size = max(64, int(round(sample_rate * 0.030)))
    hop = max(32, int(round(sample_rate * 0.010)))
    if samples.size < frame_size:
        if float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) < 1e-4:
            raise ValueError("Input speech contains no usable voiced material")
        return np.array([samples.size // 2], dtype=np.int64), np.array([1.0], dtype=np.float64)

    starts = np.arange(0, samples.size - frame_size + 1, hop, dtype=np.int64)
    rms = np.empty(starts.size, dtype=np.float64)
    zcr = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        frame = samples[start : start + frame_size]
        rms[index] = math.sqrt(float(np.mean(np.square(frame, dtype=np.float64))))
        signs = frame >= 0.0
        zcr[index] = float(np.mean(signs[1:] != signs[:-1]))

    peak_rms = float(np.max(rms))
    if peak_rms < 1e-4:
        raise ValueError("Input speech contains no usable voiced material")
    threshold = max(1e-4, peak_rms * 0.14, float(np.percentile(rms, 58)) * 0.72)
    active = (rms >= threshold) & (zcr <= 0.34)
    indexes = np.flatnonzero(active)
    if indexes.size < 4:
        indexes = np.argsort(rms)[-min(12, rms.size) :]

    centers = starts[indexes] + frame_size // 2
    weights = np.maximum(rms[indexes], 1e-6)
    weights = np.square(weights / float(np.max(weights))) + 0.04
    return centers.astype(np.int64), weights.astype(np.float64)


def _extract_centered(samples: np.ndarray, center: int, size: int) -> np.ndarray:
    size = max(8, int(size))
    start = int(center) - size // 2
    end = start + size
    left = max(0, -start)
    right = max(0, end - samples.size)
    start = max(0, start)
    end = min(samples.size, end)
    chunk = np.asarray(samples[start:end], dtype=np.float32)
    if left or right:
        mode = "reflect" if chunk.size > 1 else "edge"
        chunk = np.pad(chunk, (left, right), mode=mode)
    if chunk.size != size:
        chunk = np.resize(chunk, size)
    return chunk.astype(np.float32)


def _resample_with_glide(
    samples: np.ndarray,
    size: int,
    glide: tuple[float, float, float],
) -> np.ndarray:
    """Resample with a smooth intra-syllable speed curve for a meow-like pitch glide."""
    size = max(1, int(size))
    if samples.size <= 1:
        return np.resize(np.asarray(samples, dtype=np.float32), size)
    output_position = np.linspace(0.0, 1.0, size, endpoint=True)
    speed = np.interp(output_position, (0.0, 0.52, 1.0), glide)
    speed = np.maximum(0.35, speed)
    source_position = np.cumsum(speed, dtype=np.float64)
    source_position -= source_position[0]
    if source_position[-1] > 0.0:
        source_position *= (samples.size - 1) / source_position[-1]
    return np.interp(source_position, np.arange(samples.size), samples).astype(np.float32)


def _spectral_colour(
    samples: np.ndarray,
    sample_rate: int,
    start_formants: tuple[float, float],
    end_formants: tuple[float, float] | None = None,
) -> np.ndarray:
    if samples.size < 16:
        return samples.copy()
    spectrum = np.fft.rfft(np.asarray(samples, dtype=np.float64))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    high_pass = 1.0 - np.exp(-np.square(frequencies / 115.0))
    nasal_band = np.exp(-0.5 * np.square((frequencies - 950.0) / 520.0))
    high_tilt = 1.0 / (1.0 + np.square(frequencies / 5200.0))

    def coloured_for(formants: tuple[float, float]) -> np.ndarray:
        first, second = formants
        first_band = np.exp(-0.5 * np.square((frequencies - first) / 260.0))
        second_band = np.exp(-0.5 * np.square((frequencies - second) / 470.0))
        shape = high_pass * high_tilt * (
            0.48 + 1.05 * first_band + 0.72 * second_band + 0.38 * nasal_band
        )
        return np.fft.irfft(spectrum * shape, n=samples.size)

    start_colour = coloured_for(start_formants)
    if end_formants is None or end_formants == start_formants:
        coloured = start_colour
    else:
        end_colour = coloured_for(end_formants)
        position = np.linspace(0.0, 1.0, samples.size)
        blend = position * position * (3.0 - 2.0 * position)
        coloured = start_colour * (1.0 - blend) + end_colour * blend
    before = math.sqrt(float(np.mean(np.square(samples, dtype=np.float64))))
    after = math.sqrt(float(np.mean(np.square(coloured, dtype=np.float64))))
    if before > 1e-6 and after > 1e-6:
        coloured *= before / after
    return coloured.astype(np.float32)


def _apply_meow_character(
    samples: np.ndarray,
    sample_rate: int,
    strength: float,
) -> np.ndarray:
    """Add a restrained closed-mouth onset and rounded 'ow' release."""
    strength = min(1.0, max(0.0, float(strength)))
    if strength <= 0.0 or samples.size < 16:
        return samples.copy()
    nasal = _spectral_colour(samples, sample_rate, _MEOW_NASAL_FORMANTS)
    bright = _spectral_colour(samples, sample_rate, _MEOW_BRIGHT_FORMANTS)
    rounded = _spectral_colour(samples, sample_rate, _MEOW_ROUNDED_FORMANTS)
    position = np.linspace(0.0, 1.0, samples.size)
    onset_progress = np.clip(position / 0.30, 0.0, 1.0)
    onset_progress = onset_progress * onset_progress * (3.0 - 2.0 * onset_progress)
    bright_position = np.clip((position - 0.12) / 0.58, 0.0, 1.0)
    bright_weight = strength * 0.42 * np.sin(math.pi * bright_position) ** 2
    tail_progress = np.clip((position - 0.52) / 0.48, 0.0, 1.0)
    tail_progress = tail_progress * tail_progress * (3.0 - 2.0 * tail_progress)
    nasal_weight = strength * (1.0 - onset_progress)
    rounded_weight = strength * 0.84 * tail_progress
    dry_weight = np.maximum(0.0, 1.0 - nasal_weight - bright_weight - rounded_weight)
    result = (
        samples * dry_weight
        + nasal * nasal_weight
        + bright * bright_weight
        + rounded * rounded_weight
    )
    return result.astype(np.float32)


def _pitch_contour(state: str, syllable_index: int, syllable_count: int) -> float:
    position = syllable_index / max(1, syllable_count - 1)
    if state in {"demanding", "startled"}:
        return -0.8 + position * 2.5
    if state == "distressed":
        return 0.8 - position * 2.0
    if state in {"angry", "fight"}:
        return math.sin(position * math.pi * 2.5) * 0.8 - position * 0.5
    if state == "affectionate":
        return math.sin(position * math.pi) * 1.5
    return math.sin(position * math.pi * 2.0) * 0.55


def _syllable_glide(state: str, category: str | None) -> tuple[float, float, float]:
    if category in {"meow_up", "chirp", "trill"} or state in {"demanding", "startled"}:
        return (0.82, 0.98, 1.18)
    if category in {"meow_plaintive", "whimper"} or state == "distressed":
        return (1.10, 1.01, 0.84)
    if category in {"yowl_long", "scream"}:
        return (0.88, 1.16, 0.92)
    if state in {"angry", "fight"}:
        return (0.94, 1.08, 0.93)
    return (0.90, 1.07, 0.98)


def _micro_grain_count(size: int, sample_rate: int) -> int:
    return max(2, min(12, int(round(size / max(1, sample_rate * 0.055)))))


def _formants_at(
    start: tuple[float, float],
    end: tuple[float, float],
    position: float,
) -> tuple[float, float]:
    position = min(1.0, max(0.0, position))
    smooth = position * position * (3.0 - 2.0 * position)
    return (
        start[0] + (end[0] - start[0]) * smooth,
        start[1] + (end[1] - start[1]) * smooth,
    )


def _make_syllable(
    source: np.ndarray,
    micro_centers: list[int],
    size: int,
    sample_rate: int,
    profile: VoiceProfile,
    event: PlanEvent,
    state: str,
    intensity: float,
    syllable_index: int,
    syllable_count: int,
    pitch_variation: float,
    start_formants: tuple[float, float],
    end_formants: tuple[float, float],
    meow_strength: float,
    rng: random.Random,
) -> np.ndarray:
    contour = _pitch_contour(state, syllable_index, syllable_count)
    pitch = profile.pitch_semitones + event.pitch_semitones + contour + pitch_variation
    ratio = 2.0 ** (pitch / 12.0)
    micro_count = len(micro_centers)
    overlap = min(int(sample_rate * 0.030), max(1, size // max(2, micro_count * 2)))
    available = size + overlap * max(0, micro_count - 1)
    micro_sizes = [available // micro_count] * micro_count
    micro_sizes[-1] += available - sum(micro_sizes)

    base_glide = _syllable_glide(state, event.category)
    meow_glide = (0.82, 1.20, 0.74)
    full_glide = tuple(
        base + (meow - base) * meow_strength
        for base, meow in zip(base_glide, meow_glide)
    )

    grains: list[np.ndarray] = []
    for index, (center, micro_size) in enumerate(zip(micro_centers, micro_sizes)):
        source_size = max(16, int(round(micro_size * ratio)))
        search_size = min(source.size, max(source_size, int(round(source_size * 1.45))))
        grain = _extract_centered(source, center, search_size)
        grain = crop_loudest_window(grain, source_size)
        grain -= float(np.mean(grain))
        emphasized = np.concatenate(([grain[0]], grain[1:] - 0.92 * grain[:-1]))
        grain = grain * 0.88 + emphasized * 0.12

        start_position = index / micro_count
        end_position = (index + 1) / micro_count
        middle_position = (start_position + end_position) * 0.5
        local_glide = (
            float(np.interp(start_position, (0.0, 0.52, 1.0), full_glide)),
            float(np.interp(middle_position, (0.0, 0.52, 1.0), full_glide)),
            float(np.interp(end_position, (0.0, 0.52, 1.0), full_glide)),
        )
        grain = _resample_with_glide(grain, micro_size, local_glide)
        grain = _spectral_colour(
            grain,
            sample_rate,
            _formants_at(start_formants, end_formants, start_position),
            _formants_at(start_formants, end_formants, end_position),
        )
        grains.append(apply_fade(grain, sample_rate, fade_ms=2.0))

    syllable = grains[0]
    for grain in grains[1:]:
        syllable = _crossfade_append(syllable, grain, overlap)
    if syllable.size < size:
        syllable = np.pad(syllable, (0, size - syllable.size))
    elif syllable.size > size:
        syllable = syllable[:size]
    syllable = _apply_meow_character(syllable, sample_rate, meow_strength)

    t = np.arange(size, dtype=np.float64) / sample_rate
    flutter_rate = 18.0 + intensity * 6.0
    flutter = 1.0 - profile.flutter_depth + profile.flutter_depth * np.sin(
        2.0 * math.pi * flutter_rate * t
    )
    syllable = syllable * flutter.astype(np.float32)

    if profile.roughness > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        noise = noise_rng.normal(0.0, 1.0, size).astype(np.float32)
        noise = np.concatenate(([0.0], np.diff(noise))).astype(np.float32)
        envelope = np.minimum(1.0, np.abs(syllable) * 8.0)
        syllable += noise * envelope * profile.roughness * 0.07

    syllable = np.tanh(syllable * (1.24 + profile.roughness * 1.6)).astype(np.float32)
    return normalize_rms(syllable, target_db=-18.0, min_gain=0.25, max_gain=4.0)


def _syllable_layout(
    duration: float,
    sample_rate: int,
    profile: VoiceProfile,
    intensity: float,
    density_scale: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    target_size = max(1, int(round(duration * sample_rate)))
    rate = profile.syllables_per_second * (0.88 + intensity * 0.20) * density_scale
    count = max(1, min(20, int(round(duration * rate))))
    count = min(count, max(1, int(duration / 0.075)))
    if duration < 0.20:
        count = 1

    overlaps = [
        int(round(rng.uniform(*profile.overlap_ms) * sample_rate / 1000.0))
        for _ in range(max(0, count - 1))
    ]
    # Overlap-add keeps adjacent syllables voiced. The raw grains are longer by the
    # total overlap so the final phrase still has exactly the planned duration.
    available = max(count * 16, target_size + sum(overlaps))
    proportions = np.array([rng.uniform(0.88, 1.14) for _ in range(count)], dtype=np.float64)
    sizes = np.maximum(16, np.floor(proportions / float(np.sum(proportions)) * available).astype(np.int64))
    sizes[-1] += available - int(np.sum(sizes))
    for index, overlap in enumerate(overlaps):
        safe_overlap = max(1, min(overlap, int(sizes[index]) - 8, int(sizes[index + 1]) - 8))
        overlaps[index] = safe_overlap
    return [int(value) for value in sizes], overlaps


def _diverse_source_centers(
    centers: np.ndarray,
    weights: np.ndarray,
    count: int,
    sample_rate: int,
    rng: random.Random,
) -> list[int]:
    """Choose non-adjacent source units without repeating one until the deck is exhausted."""
    spacing = max(1, int(round(sample_rate * 0.14)))
    buckets: dict[int, list[int]] = {}
    for index, center in enumerate(centers):
        buckets.setdefault(int(center) // spacing, []).append(index)
    representatives: list[int] = []
    for indexes in buckets.values():
        selected = rng.choices(indexes, weights=[float(weights[index]) for index in indexes], k=1)[0]
        representatives.append(int(centers[selected]))
    if not representatives:
        representatives = [int(centers[0])]

    result: list[int] = []
    while len(result) < count:
        deck = representatives.copy()
        rng.shuffle(deck)
        if result and len(deck) > 1 and deck[0] == result[-1]:
            deck.append(deck.pop(0))
        result.extend(deck)
    return result[:count]


def _crossfade_append(base: np.ndarray, following: np.ndarray, overlap: int) -> np.ndarray:
    overlap = max(0, min(int(overlap), base.size - 1, following.size - 1))
    if overlap <= 0:
        return np.concatenate((base, following))
    # Keep the voiced carrier in the same polarity to avoid a hollow dip at joins.
    if float(np.dot(base[-overlap:], following[:overlap])) < 0.0:
        following = -following
    angle = np.linspace(0.0, math.pi / 2.0, overlap, endpoint=True)
    mixed = base[-overlap:] * np.cos(angle) + following[:overlap] * np.sin(angle)
    return np.concatenate((base[:-overlap], mixed.astype(np.float32), following[overlap:]))


def _stabilize_phrase_envelope(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reduce syllable-boundary dips without flattening phrase-level expression."""
    frame = max(16, int(round(sample_rate * 0.012)))
    hop = max(8, frame // 2)
    if samples.size < frame * 2:
        return samples.copy()
    starts = np.arange(0, samples.size - frame + 1, hop)
    rms = np.array(
        [
            math.sqrt(float(np.mean(np.square(samples[start : start + frame], dtype=np.float64))))
            for start in starts
        ],
        dtype=np.float64,
    )
    reference = float(np.percentile(rms, 58))
    if reference < 1e-5:
        return samples.copy()
    floor = reference * 0.34
    gain = np.clip(floor / np.maximum(rms, 1e-5), 1.0, 2.2)
    kernel = np.ones(9, dtype=np.float64) / 9.0
    gain = np.convolve(np.pad(gain, (4, 4), mode="edge"), kernel, mode="valid")
    positions = starts + frame // 2
    sample_gain = np.interp(np.arange(samples.size), positions, gain, left=gain[0], right=gain[-1])
    return (samples * sample_gain).astype(np.float32)


def _render_phrase(
    source: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    event: PlanEvent,
    plan: BehaviorPlan,
    sample_rate: int,
    profile: VoiceProfile,
    meowiness: float,
    meow_tail_mix: float,
    add_reference_meow: bool,
    density_scale: float,
    rng: random.Random,
) -> tuple[np.ndarray, int, int, int, int, int, float]:
    sizes, overlaps = _syllable_layout(
        event.duration,
        sample_rate,
        profile,
        plan.intensity,
        density_scale,
        rng,
    )
    micro_counts = [_micro_grain_count(size, sample_rate) for size in sizes]
    source_centers = _diverse_source_centers(
        centers,
        weights,
        sum(micro_counts),
        sample_rate,
        rng,
    )
    meow_strengths = [
        meowiness * 0.78 * rng.uniform(0.90, 1.10)
        for _ in sizes
    ]
    if sizes and meowiness > 0.0:
        # Every unit gets a restrained m-ee-ow contour; the phrase ending carries
        # the clearest meow so the result remains language-like rather than repetitive.
        meow_strengths[-1] = meowiness
        if len(sizes) >= 4:
            accent_index = rng.randrange(1, len(sizes) - 1)
            meow_strengths[accent_index] = max(
                meow_strengths[accent_index],
                meowiness * 0.96,
            )
    current_formants = rng.choice(_VOWEL_FORMANTS)
    pitch_drift = rng.uniform(-profile.pitch_jitter, profile.pitch_jitter)

    syllables: list[np.ndarray] = []
    center_offset = 0
    for index, size in enumerate(sizes):
        if index:
            pitch_drift = 0.90 * pitch_drift + 0.10 * rng.uniform(
                -profile.pitch_jitter, profile.pitch_jitter
            )
        if rng.random() < 0.42:
            next_choices = [value for value in _VOWEL_FORMANTS if value != current_formants]
            next_formants = rng.choice(next_choices)
        else:
            next_formants = current_formants
        micro_centers = source_centers[center_offset : center_offset + micro_counts[index]]
        center_offset += micro_counts[index]
        syllables.append(
            _make_syllable(
                source,
                micro_centers,
                size,
                sample_rate,
                profile,
                event,
                plan.state,
                plan.intensity,
                index,
                len(sizes),
                pitch_drift,
                current_formants,
                next_formants,
                meow_strengths[index],
                rng,
            )
        )
        current_formants = next_formants

    phrase = syllables[0] if syllables else np.zeros(1, dtype=np.float32)
    for index, syllable in enumerate(syllables[1:]):
        phrase = _crossfade_append(phrase, syllable, overlaps[index])
    target_size = max(1, int(round(event.duration * sample_rate)))
    if phrase.size < target_size:
        phrase = np.pad(phrase, (0, target_size - phrase.size))
    elif phrase.size > target_size:
        phrase = phrase[:target_size]

    reference_meow_units = 0
    reference_meow_duration = 0.0
    if add_reference_meow and meow_tail_mix > 0.0 and phrase.size >= int(sample_rate * 0.20):
        tail_size = min(
            phrase.size,
            int(round(sample_rate * rng.uniform(0.48, 0.72))),
        )
        tail_micro_count = _micro_grain_count(tail_size, sample_rate)
        tail_centers = source_centers[-tail_micro_count:]
        if len(tail_centers) < tail_micro_count:
            tail_centers = (source_centers * tail_micro_count)[:tail_micro_count]
        reference_meow = _make_syllable(
            source,
            tail_centers,
            tail_size,
            sample_rate,
            profile,
            event,
            plan.state,
            plan.intensity,
            max(0, len(sizes) - 1),
            max(1, len(sizes)),
            pitch_drift,
            _MEOW_NASAL_FORMANTS,
            _MEOW_ROUNDED_FORMANTS,
            1.0,
            rng,
        )
        reference_meow = apply_fade(reference_meow, sample_rate, fade_ms=16.0)
        position = np.linspace(0.0, 1.0, tail_size, dtype=np.float64)
        attack = np.clip(position / 0.34, 0.0, 1.0)
        attack = attack * attack * (3.0 - 2.0 * attack)
        release = 1.0 - 0.30 * np.clip((position - 0.84) / 0.16, 0.0, 1.0)
        blend = (attack * release * meow_tail_mix).astype(np.float32)
        tail_start = phrase.size - tail_size
        original_tail = phrase[tail_start:].copy()
        phrase[tail_start:] = (
            original_tail * (1.0 - blend * 0.55)
            + reference_meow * blend
        )
        reference_meow_units = len(set(tail_centers))
        reference_meow_duration = tail_size / float(sample_rate)

    phrase = _stabilize_phrase_envelope(phrase, sample_rate)
    phrase = normalize_rms(phrase, target_db=-18.0, min_gain=0.4, max_gain=2.5)
    phrase = apply_fade(phrase, sample_rate, fade_ms=14.0)
    phrase *= 10.0 ** (event.gain_db / 20.0)
    accent_count = sum(strength > 0.0 for strength in meow_strengths)
    return (
        phrase.astype(np.float32),
        len(sizes),
        len(source_centers),
        len(set(source_centers)),
        accent_count,
        reference_meow_units,
        reference_meow_duration,
    )


def render_granular_voice(
    plan: BehaviorPlan,
    source_voice: str | Path,
    sample_rate: int = 24000,
    master_peak: float = 0.94,
    meowiness: float = 0.86,
    meow_tail_mix: float = 0.42,
    reverse_source: bool = True,
    voice_rate: float = 0.52,
    pause_scale: float = 0.50,
) -> RenderResult:
    """Recompose voiced grains into emotion-paced, non-lexical cartoon-cat speech."""
    source_path = Path(source_voice)
    audio = read_wav(source_path)
    source = resample_linear(audio.samples, audio.sample_rate, sample_rate)
    source = np.asarray(source, dtype=np.float32)
    if reverse_source:
        source = source[::-1].copy()
    source -= float(np.mean(source)) if source.size else 0.0
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    if peak < 1e-5:
        raise ValueError("Input speech contains no usable voiced material")
    source /= max(peak, 1e-5)
    centers, weights = _active_voice_centers(source, sample_rate)

    profile = VOICE_PROFILES.get(plan.state, VOICE_PROFILES["neutral"])
    meowiness = min(1.0, max(0.0, float(meowiness)))
    meow_tail_mix = min(0.60, max(0.0, float(meow_tail_mix)))
    voice_rate = min(1.25, max(0.45, float(voice_rate)))
    pause_scale = min(1.0, max(0.20, float(pause_scale)))
    # Slower delivery must also reduce articulation changes; otherwise stretching the
    # timeline while keeping the same syllable count still sounds hurried.
    density_scale = min(1.8, max(0.20, voice_rate * voice_rate * 1.35))
    raw_render_duration = 0.0
    for planned_event in plan.events:
        if planned_event.kind == "sound":
            raw_render_duration += planned_event.duration / voice_rate
        else:
            planned_pause_scale = (
                pause_scale
                if planned_event.duration < 0.75
                else min(1.0, pause_scale + 0.25)
            )
            raw_render_duration += max(0.025, planned_event.duration * planned_pause_scale)
    timeline_scale = (
        plan.target_duration / raw_render_duration if raw_render_duration > 0.0 else 1.0
    )

    rng = random.Random(plan.seed + 32452843)
    chunks: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    cursor = 0
    source_duration = audio.samples.size / float(audio.sample_rate)
    sound_event_total = sum(event.kind == "sound" for event in plan.events)
    reference_meow_indexes = set(range(1, sound_event_total, 2))
    if sound_event_total:
        reference_meow_indexes.add(sound_event_total - 1)
    sound_event_index = 0

    for event in plan.events:
        start = cursor / float(sample_rate)
        rendered_event = event
        effective_pause_scale: float | None = None
        if event.kind == "silence":
            effective_pause_scale = (
                pause_scale if event.duration < 0.75 else min(1.0, pause_scale + 0.25)
            )
            rendered_event = PlanEvent(
                kind=event.kind,
                duration=max(0.025, event.duration * effective_pause_scale) * timeline_scale,
            )
            chunk = np.zeros(
                max(1, int(round(rendered_event.duration * sample_rate))),
                dtype=np.float32,
            )
            source_kind = "silence"
            event_source_path = None
            syllable_count = 0
            micro_grain_count = 0
            source_unit_count = 0
            meow_accent_count = 0
            reference_meow_units = 0
            reference_meow_duration = 0.0
        elif event.kind == "sound":
            rendered_event = PlanEvent(
                kind=event.kind,
                duration=event.duration / voice_rate * timeline_scale,
                category=event.category,
                gain_db=event.gain_db,
                pitch_semitones=event.pitch_semitones,
            )
            (
                chunk,
                syllable_count,
                micro_grain_count,
                source_unit_count,
                meow_accent_count,
                reference_meow_units,
                reference_meow_duration,
            ) = _render_phrase(
                source,
                centers,
                weights,
                rendered_event,
                plan,
                sample_rate,
                profile,
                meowiness,
                meow_tail_mix,
                sound_event_index in reference_meow_indexes,
                density_scale,
                rng,
            )
            sound_event_index += 1
            source_kind = "granular_voice"
            event_source_path = str(source_path.resolve())
        else:
            raise ValueError(f"Invalid plan event: {event}")

        chunks.append(chunk)
        cursor += chunk.size
        events.append(
            {
                "kind": event.kind,
                "category": "stylized_phrase" if event.kind == "sound" else None,
                "planned_category": event.category,
                "start": round(start, 6),
                "end": round(cursor / float(sample_rate), 6),
                "planned_duration": round(event.duration, 6),
                "rendered_duration": round(rendered_event.duration, 6),
                "voice_rate": voice_rate if event.kind == "sound" else None,
                "pause_scale": effective_pause_scale,
                "timeline_scale": timeline_scale,
                "gain_db": round(event.gain_db, 4),
                "pitch_semitones": round(event.pitch_semitones, 4),
                "source": source_kind,
                "source_path": event_source_path,
                "source_reversed": bool(reverse_source) if event.kind == "sound" else None,
                "source_original_duration": round(source_duration, 6) if event.kind == "sound" else None,
                "selected_duration_before_pitch": round(rendered_event.duration, 6) if event.kind == "sound" else None,
                "syllable_count": syllable_count,
                "micro_grain_count": micro_grain_count,
                "source_unit_count": source_unit_count,
                "meow_accent_count": meow_accent_count,
                "meowiness": meowiness if event.kind == "sound" else None,
                "reference_meow": reference_meow_units > 0,
                "reference_meow_mix": (
                    meow_tail_mix if reference_meow_units > 0 else 0.0
                ),
                "reference_meow_source_units": reference_meow_units,
                "reference_meow_duration": round(reference_meow_duration, 6),
                "reference_meow_source": (
                    "source_voice_grains" if reference_meow_units > 0 else None
                ),
                "reference_meow_blend": (
                    "overlaid_tail_crossfade" if reference_meow_units > 0 else None
                ),
                "coarticulated": event.kind == "sound",
                "voice_character": "cute_cat" if event.kind == "sound" else None,
                "style_pitch_semitones": profile.pitch_semitones if event.kind == "sound" else None,
            }
        )

    samples = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    samples = limit_peak(samples, peak=min(0.999, max(0.1, master_peak)))
    if not np.all(np.isfinite(samples)):
        raise ValueError("Granular renderer produced non-finite audio values")
    return RenderResult(
        samples=samples,
        sample_rate=sample_rate,
        events=tuple(events),
    )
