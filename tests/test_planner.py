from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from meogenerate.planner import load_config, plan_behavior  # noqa: E402


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_plan_is_reproducible(self) -> None:
        first = plan_behavior(self.config, "angry", "angry", 0.9, 7.0, 123)
        second = plan_behavior(self.config, "angry", "angry", 0.9, 7.0, 123)
        self.assertEqual(first, second)
        self.assertTrue(any(event.kind == "sound" for event in first.events))
        self.assertTrue(any(event.kind == "silence" for event in first.events))

    def test_fight_uses_fight_vocabulary(self) -> None:
        plan = plan_behavior(self.config, "fight", "angry", 0.95, 7.0, 9)
        categories = {event.category for event in plan.events if event.category}
        self.assertTrue(categories & {"fight_burst", "scream", "yowl_long"})
        self.assertGreaterEqual(sum(event.kind == "sound" for event in plan.events), 4)

    def test_distressed_has_longer_silence(self) -> None:
        plan = plan_behavior(self.config, "distressed", "sad", 0.6, 5.0, 44)
        pauses = [event.duration for event in plan.events if event.kind == "silence"]
        self.assertTrue(pauses)
        self.assertGreater(max(pauses), 0.7)

    def test_duration_budget_does_not_repeat_whole_sequences_excessively(self) -> None:
        fight = plan_behavior(self.config, "fight", "angry", 0.95, 6.0, 42)
        distressed = plan_behavior(self.config, "distressed", "sad", 0.62, 6.0, 43)
        self.assertLessEqual(fight.estimated_duration, 8.11)
        self.assertLessEqual(distressed.estimated_duration, 8.11)

    def test_long_speech_budget_is_not_capped_by_short_sequence_limit(self) -> None:
        plan = plan_behavior(self.config, "affectionate", "happy", 0.8, 30.0, 20260316)
        self.assertGreaterEqual(plan.estimated_duration, 24.0)
        self.assertGreater(sum(event.kind == "sound" for event in plan.events), 10)

    def test_removed_categories_are_not_planned(self) -> None:
        categories = {
            event["sound"]
            for state in self.config["states"].values()
            for sequence in state.get("sequences", [])
            for event in sequence.get("events", [])
            if "sound" in event
        }
        self.assertTrue(categories.isdisjoint({"purr", "growl", "hiss"}))

    def test_unknown_state_fails(self) -> None:
        with self.assertRaises(ValueError):
            plan_behavior(self.config, "missing", "neutral", 0.5, 5.0, 1)


if __name__ == "__main__":
    unittest.main()
