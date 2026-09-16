"""Emotion-driven cartoon-cat language generator."""

from .emotion import EmotionResult, emotion_to_state
from .granular import RenderResult, render_granular_voice
from .planner import BehaviorPlan, PlanEvent, plan_behavior

__all__ = [
    "BehaviorPlan",
    "EmotionResult",
    "PlanEvent",
    "RenderResult",
    "emotion_to_state",
    "plan_behavior",
    "render_granular_voice",
]

__version__ = "0.2.0"
