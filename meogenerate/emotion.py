from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EMOTIONS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)


@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float
    intensity: float
    scores: dict[str, float]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "scores": self.scores,
            "source": self.source,
        }


_AROUSAL_PRIOR = {
    "angry": 0.90,
    "disgusted": 0.68,
    "fearful": 0.82,
    "happy": 0.68,
    "neutral": 0.28,
    "other": 0.45,
    "sad": 0.42,
    "surprised": 0.86,
    "unknown": 0.40,
}


_STATE_BY_EMOTION = {
    "angry": "angry",
    "disgusted": "angry",
    "fearful": "fearful",
    "happy": "affectionate",
    "neutral": "neutral",
    "other": "neutral",
    "sad": "distressed",
    "surprised": "startled",
    "unknown": "neutral",
}


def _canonical_label(label: Any) -> str:
    text = str(label).strip().lower()
    for candidate in EMOTIONS:
        if text == candidate or candidate in text:
            return candidate
    return "unknown"


def _derive_intensity(label: str, confidence: float) -> float:
    prior = _AROUSAL_PRIOR.get(label, 0.45)
    return min(1.0, max(0.05, prior * 0.62 + confidence * 0.38))


def manual_emotion(label: str, intensity: float) -> EmotionResult:
    canonical = _canonical_label(label)
    if canonical == "unknown" and label.strip().lower() != "unknown":
        raise ValueError(f"Unknown emotion '{label}'. Available: {', '.join(EMOTIONS)}")
    intensity = min(1.0, max(0.0, float(intensity)))
    return EmotionResult(
        label=canonical,
        confidence=1.0,
        intensity=intensity,
        scores={canonical: 1.0},
        source="manual",
    )


def emotion_to_state(label: str) -> str:
    return _STATE_BY_EMOTION.get(_canonical_label(label), "neutral")


def parse_emotion2vec_result(result: Any) -> EmotionResult:
    if isinstance(result, list):
        if not result:
            raise ValueError("emotion2vec returned an empty result")
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected emotion2vec result type: {type(result).__name__}")

    labels = result.get("labels")
    scores = result.get("scores")
    if labels is None or scores is None:
        raise ValueError("emotion2vec result does not contain labels and scores")
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    while (
        isinstance(labels, (list, tuple))
        and len(labels) == 1
        and isinstance(labels[0], (list, tuple))
    ):
        labels = labels[0]
    while (
        isinstance(scores, (list, tuple))
        and len(scores) == 1
        and isinstance(scores[0], (list, tuple))
    ):
        scores = scores[0]
    if not isinstance(labels, (list, tuple)) or not isinstance(scores, (list, tuple)):
        raise ValueError("emotion2vec labels and scores must be arrays")
    if len(labels) != len(scores) or not labels:
        raise ValueError("emotion2vec returned mismatched labels and scores")

    normalized: dict[str, float] = {}
    for raw_label, raw_score in zip(labels, scores):
        label = _canonical_label(raw_label)
        score = float(raw_score)
        normalized[label] = max(score, normalized.get(label, 0.0))

    label, confidence = max(normalized.items(), key=lambda item: item[1])
    return EmotionResult(
        label=label,
        confidence=confidence,
        intensity=_derive_intensity(label, confidence),
        scores=normalized,
        source="emotion2vec",
    )


def analyze_with_emotion2vec(
    audio_path: str | Path,
    model_id: str = "iic/emotion2vec_plus_seed",
    hub: str = "ms",
    device: str | None = None,
) -> EmotionResult:
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "emotion2vec dependencies are not installed. Run: "
            "python -m pip install -r tools/MeoGenerate/requirements-emotion.txt"
        ) from exc

    model_arguments: dict[str, Any] = {
        "model": model_id,
        "hub": hub,
        "disable_update": True,
    }
    if device and device != "auto":
        model_arguments["device"] = device
    model = AutoModel(**model_arguments)
    with tempfile.TemporaryDirectory(prefix="meogenerate-emotion-") as output_dir:
        result = model.generate(
            str(Path(audio_path).resolve()),
            output_dir=output_dir,
            granularity="utterance",
            extract_embedding=False,
        )
    return parse_emotion2vec_result(result)
