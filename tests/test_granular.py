from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from meogenerate.audio import write_wav  # noqa: E402
from meogenerate.granular import render_granular_voice  # noqa: E402
from meogenerate.planner import load_config, plan_behavior  # noqa: E402


def _speech_like_audio(sample_rate: int = 12000) -> np.ndarray:
    samples = np.zeros(sample_rate * 3, dtype=np.float32)
    for start_seconds, end_seconds, base in ((0.45, 1.05, 145.0), (1.55, 2.65, 190.0)):
        start = int(start_seconds * sample_rate)
        end = int(end_seconds * sample_rate)
        t = np.arange(end - start, dtype=np.float64) / sample_rate
        phase = 2.0 * math.pi * np.cumsum(base + 18.0 * np.sin(2.0 * math.pi * 3.2 * t)) / sample_rate
        voice = 0.32 * np.sin(phase) + 0.15 * np.sin(2.0 * phase) + 0.07 * np.sin(3.0 * phase)
        envelope = np.sin(np.linspace(0.0, math.pi, end - start)) ** 0.4
        samples[start:end] = (voice * envelope).astype(np.float32)
    return samples


class GranularVoiceTests(unittest.TestCase):
    def test_recomposes_voice_into_stylized_syllables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "speech.wav"
            write_wav(source, _speech_like_audio(), 12000)
            plan = plan_behavior(load_config(), "demanding", "angry", 0.8, 3.0, 77)
            first = render_granular_voice(plan, source, sample_rate=12000)
            second = render_granular_voice(plan, source, sample_rate=12000)
            without_meow = render_granular_voice(plan, source, sample_rate=12000, meowiness=0.0)
            without_tail = render_granular_voice(plan, source, sample_rate=12000, meow_tail_mix=0.0)
            forward_source = render_granular_voice(plan, source, sample_rate=12000, reverse_source=False)

            self.assertTrue(np.array_equal(first.samples, second.samples))
            self.assertFalse(np.array_equal(first.samples, without_meow.samples))
            self.assertFalse(np.array_equal(first.samples, without_tail.samples))
            self.assertFalse(np.array_equal(first.samples, forward_source.samples))
            self.assertAlmostEqual(plan.target_duration, first.duration, places=3)
            self.assertGreater(float(np.max(np.abs(first.samples))), 0.05)
            self.assertLessEqual(float(np.max(np.abs(first.samples))), 0.941)
            sound_events = [event for event in first.events if event["kind"] == "sound"]
            silence_events = [event for event in first.events if event["kind"] == "silence"]
            self.assertTrue(sound_events)
            self.assertTrue(silence_events)
            self.assertTrue(
                all(event["rendered_duration"] < event["planned_duration"] for event in silence_events)
            )
            self.assertTrue(all(event["source"] == "granular_voice" for event in sound_events))
            self.assertTrue(all(event["source_reversed"] is True for event in sound_events))
            self.assertTrue(
                all(
                    event["source_reversed"] is False
                    for event in forward_source.events
                    if event["kind"] == "sound"
                )
            )
            self.assertTrue(all(event["category"] == "stylized_phrase" for event in sound_events))
            self.assertTrue(all(event["syllable_count"] >= 1 for event in sound_events))
            self.assertTrue(
                all(event["micro_grain_count"] >= event["syllable_count"] * 2 for event in sound_events)
            )
            self.assertTrue(
                all(
                    event["source_unit_count"] >= min(2, event["syllable_count"])
                    for event in sound_events
                )
            )
            self.assertTrue(
                all(event["meow_accent_count"] == event["syllable_count"] for event in sound_events)
            )
            self.assertTrue(all(event["meowiness"] == 0.86 for event in sound_events))
            self.assertTrue(all(event["voice_rate"] == 0.52 for event in sound_events))
            reference_meows = [event for event in sound_events if event["reference_meow"]]
            self.assertGreaterEqual(len(reference_meows), len(sound_events) // 2)
            self.assertTrue(all(event["reference_meow_mix"] == 0.42 for event in reference_meows))
            self.assertTrue(all(event["reference_meow_source_units"] >= 1 for event in reference_meows))
            self.assertTrue(all(event["reference_meow_duration"] >= 0.20 for event in reference_meows))
            self.assertTrue(
                all(event["reference_meow_source"] == "source_voice_grains" for event in reference_meows)
            )
            self.assertTrue(
                all(event["reference_meow_blend"] == "overlaid_tail_crossfade" for event in reference_meows)
            )
            self.assertTrue(
                all(event["rendered_duration"] > event["planned_duration"] for event in sound_events)
            )
            self.assertTrue(all(event["coarticulated"] for event in sound_events))
            self.assertTrue(all(event["voice_character"] == "cute_cat" for event in sound_events))
            for event in sound_events:
                start = int(event["start"] * first.sample_rate)
                end = int(event["end"] * first.sample_rate)
                phrase = first.samples[start:end]
                frame_size = max(1, int(first.sample_rate * 0.010))
                frame_count = phrase.size // frame_size
                frame_rms = np.sqrt(
                    np.mean(
                        phrase[: frame_count * frame_size].reshape(frame_count, frame_size) ** 2,
                        axis=1,
                    )
                )
                self.assertGreater(float(np.mean(frame_rms > 0.001)), 0.94)

    def test_silent_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "silent.wav"
            write_wav(source, np.zeros(12000, dtype=np.float32), 12000)
            plan = plan_behavior(load_config(), "neutral", "neutral", 0.5, 2.0, 2)
            with self.assertRaisesRegex(ValueError, "no usable voiced material"):
                render_granular_voice(plan, source, sample_rate=12000)


if __name__ == "__main__":
    unittest.main()
