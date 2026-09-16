from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from meogenerate.audio import (  # noqa: E402
    crop_loudest_window,
    read_wav,
    resample_linear,
    write_wav,
)


class AudioTests(unittest.TestCase):
    def test_pcm_wav_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            source = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
            write_wav(path, source, 16000)
            loaded = read_wav(path)
            self.assertEqual(16000, loaded.sample_rate)
            self.assertEqual(source.size, loaded.samples.size)
            self.assertLess(float(np.max(np.abs(source - loaded.samples))), 0.0001)

    def test_resample_and_crop_keep_expected_lengths(self) -> None:
        source = np.zeros(1000, dtype=np.float32)
        source[400:600] = 0.8
        resampled = resample_linear(source, 1000, 2000)
        self.assertEqual(2000, resampled.size)
        cropped = crop_loudest_window(resampled, 200)
        self.assertEqual(200, cropped.size)
        self.assertGreater(float(np.mean(np.abs(cropped))), 0.7)


if __name__ == "__main__":
    unittest.main()
