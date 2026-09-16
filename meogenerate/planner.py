from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlanEvent:
    kind: str
    duration: float
    category: str | None = None
    gain_db: float = 0.0
    pitch_semitones: float = 0.0


@dataclass(frozen=True)
class BehaviorPlan:
    state: str
    emotion: str
    intensity: float
    target_duration: float
    estimated_duration: float
    seed: int
    events: tuple[PlanEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["events"] = [asdict(event) for event in self.events]
        return result


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("version") != 1:
        raise ValueError(f"Unsupported config version in {config_path}")
    if not isinstance(config.get("states"), dict) or not config["states"]:
        raise ValueError(f"Config has no states: {config_path}")
    return config


def available_states(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(config["states"]))


def _range_value(value: Any, rng: random.Random, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        if low > high:
            raise ValueError(f"Invalid {field} range: {value}")
        return rng.uniform(low, high)
    raise ValueError(f"Expected number or [min, max] for {field}, got {value!r}")


def _weighted_choice(
    choices: list[dict[str, Any]],
    rng: random.Random,
    previous_index: int | None,
) -> tuple[int, dict[str, Any]]:
    indexes = list(range(len(choices)))
    weights = [max(0.0, float(choice.get("weight", 1.0))) for choice in choices]
    if previous_index is not None and len(choices) > 1:
        weights[previous_index] *= 0.18
    if not any(weights):
        raise ValueError("At least one sequence must have a positive weight")
    selected = rng.choices(indexes, weights=weights, k=1)[0]
    return selected, choices[selected]


def _eligible_sequences(state_config: dict[str, Any], intensity: float) -> list[dict[str, Any]]:
    result = []
    for sequence in state_config.get("sequences", []):
        minimum = float(sequence.get("min_intensity", 0.0))
        maximum = float(sequence.get("max_intensity", 1.0))
        if minimum <= intensity <= maximum:
            result.append(sequence)
    if not result:
        raise ValueError("No behavior sequence is eligible for the requested intensity")
    return result


def _midpoint(value: Any, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 2:
        return (float(value[0]) + float(value[1])) * 0.5
    raise ValueError(f"Expected number or [min, max] for {field}, got {value!r}")


def _expected_sequence_duration(sequence: dict[str, Any], intensity: float) -> float:
    sound_scale = 0.90 + intensity * 0.18
    duration = 0.0
    for event in sequence.get("events", []):
        if "pause" in event:
            duration += _midpoint(event["pause"], "pause")
        elif "sound" in event:
            duration += _midpoint(event.get("duration", [0.35, 0.90]), "duration") * sound_scale
    return duration


def _expected_next_duration(
    choices: list[dict[str, Any]],
    state_config: dict[str, Any],
    intensity: float,
) -> float:
    weights = [max(0.0, float(choice.get("weight", 1.0))) for choice in choices]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.0
    sequence_duration = sum(
        weight * _expected_sequence_duration(choice, intensity)
        for choice, weight in zip(choices, weights)
    ) / total_weight
    gap = _midpoint(state_config.get("inter_sequence_pause", [0.5, 1.4]), "inter_sequence_pause")
    return gap + sequence_duration


def _build_sequence(
    sequence: dict[str, Any],
    intensity: float,
    rng: random.Random,
) -> list[PlanEvent]:
    events: list[PlanEvent] = []
    intensity_gain = (intensity - 0.5) * 3.0
    intensity_length = 0.90 + intensity * 0.18

    for raw_event in sequence.get("events", []):
        if "pause" in raw_event:
            duration = max(0.02, _range_value(raw_event["pause"], rng, "pause"))
            events.append(PlanEvent(kind="silence", duration=duration))
            continue
        category = raw_event.get("sound")
        if not category:
            raise ValueError(f"Sequence event needs 'sound' or 'pause': {raw_event!r}")
        duration = _range_value(raw_event.get("duration", [0.35, 0.90]), rng, "duration")
        duration *= intensity_length
        gain_db = _range_value(raw_event.get("gain_db", [-2.0, 1.0]), rng, "gain_db")
        pitch = _range_value(raw_event.get("pitch", [-0.5, 0.5]), rng, "pitch")
        events.append(
            PlanEvent(
                kind="sound",
                category=str(category),
                duration=max(0.08, duration),
                gain_db=gain_db + intensity_gain,
                pitch_semitones=pitch,
            )
        )
    return events


def plan_behavior(
    config: dict[str, Any],
    state: str,
    emotion: str,
    intensity: float,
    target_duration: float,
    seed: int,
) -> BehaviorPlan:
    if state not in config["states"]:
        choices = ", ".join(available_states(config))
        raise ValueError(f"Unknown state '{state}'. Available states: {choices}")
    intensity = min(1.0, max(0.0, float(intensity)))
    target_duration = min(60.0, max(0.5, float(target_duration)))
    rng = random.Random(seed)
    state_config = config["states"][state]
    choices = _eligible_sequences(state_config, intensity)

    minimum_sequences = max(1, int(state_config.get("min_sequences", 1)))
    maximum_sequences = max(minimum_sequences, int(state_config.get("max_sequences", 4)))
    expected_next_duration = _expected_next_duration(choices, state_config, intensity)
    # Configured maxima protect short generations from excessive repetition.
    # Longer speech-derived performances need enough independently randomized phrases to
    # approach their requested timeline instead of stopping after only a few seconds.
    if expected_next_duration > 0.0:
        duration_sequence_limit = min(
            64,
            int(math.ceil(target_duration / max(0.25, expected_next_duration))) + 2,
        )
        maximum_sequences = max(maximum_sequences, duration_sequence_limit)

    events: list[PlanEvent] = []
    estimated = 0.0
    previous_index: int | None = None
    sequence_count = 0

    while sequence_count < maximum_sequences:
        selected_index, sequence = _weighted_choice(choices, rng, previous_index)
        previous_index = selected_index
        sequence_events = _build_sequence(sequence, intensity, rng)
        if events:
            gap = _range_value(state_config.get("inter_sequence_pause", [0.5, 1.4]), rng, "inter_sequence_pause")
            events.append(PlanEvent(kind="silence", duration=max(0.04, gap)))
            estimated += gap
        events.extend(sequence_events)
        estimated += sum(event.duration for event in sequence_events)
        sequence_count += 1

        reached_budget = estimated >= target_duration * 0.82
        exceeded_hard_budget = estimated >= target_duration
        next_sequence_would_overshoot = (
            estimated + expected_next_duration > target_duration * 1.15
        )
        if sequence_count >= minimum_sequences and (
            exceeded_hard_budget or next_sequence_would_overshoot
        ):
            break
        if sequence_count >= minimum_sequences and reached_budget:
            stop_probability = float(state_config.get("stop_probability", 0.72))
            if rng.random() <= stop_probability:
                break

    tail = _range_value(state_config.get("tail_silence", [0.35, 1.0]), rng, "tail_silence")
    # Keep the requested duration approximate without cutting a behavior sequence in half.
    tail_budget = max(0.15, target_duration * 1.35 - estimated)
    tail = min(tail, tail_budget)
    events.append(PlanEvent(kind="silence", duration=max(0.04, tail)))
    estimated += tail

    return BehaviorPlan(
        state=state,
        emotion=emotion,
        intensity=intensity,
        target_duration=target_duration,
        estimated_duration=estimated,
        seed=seed,
        events=tuple(events),
    )
