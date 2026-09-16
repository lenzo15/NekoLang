from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from meogenerate.emotion import (  # noqa: E402
    emotion_to_state,
    manual_emotion,
    parse_emotion2vec_result,
)


class EmotionTests(unittest.TestCase):
    def test_parse_emotion2vec_nested_result(self) -> None:
        result = parse_emotion2vec_result(
            [
                {
                    "labels": [["angry", "happy", "neutral"]],
                    "scores": [[0.81, 0.12, 0.07]],
                }
            ]
        )
        self.assertEqual("angry", result.label)
        self.assertAlmostEqual(0.81, result.confidence)
        self.assertGreater(result.intensity, 0.7)

    def test_default_state_mapping(self) -> None:
        self.assertEqual("angry", emotion_to_state("disgusted"))
        self.assertEqual("distressed", emotion_to_state("sad"))
        self.assertEqual("affectionate", emotion_to_state("happy"))
        self.assertEqual("startled", emotion_to_state("surprised"))

    def test_manual_intensity_is_clamped(self) -> None:
        self.assertEqual(1.0, manual_emotion("angry", 2.0).intensity)
        self.assertEqual(0.0, manual_emotion("sad", -1.0).intensity)


if __name__ == "__main__":
    unittest.main()
