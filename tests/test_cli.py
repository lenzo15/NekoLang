from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from meogenerate.audio import write_wav  # noqa: E402
from meogenerate.cli import main  # noqa: E402


class CliTests(unittest.TestCase):
    def test_plan_flag_writes_granular_wav_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "speech.wav"
            t = np.arange(24000, dtype=np.float32) / 12000.0
            voice = (np.sin(2.0 * np.pi * 180.0 * t) * 0.3).astype(np.float32)
            write_wav(source, voice, 12000)
            output = root / "stylized.wav"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exit_code = main(
                    [
                        "generate",
                        "--input",
                        str(source),
                        "--emotion",
                        "happy",
                        "--duration",
                        "2",
                        "--seed",
                        "9",
                        "--plan",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(0, exit_code)
            self.assertTrue(output.is_file())
            manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual("granular_voice", manifest["render"]["renderer"])
            self.assertNotIn("style", manifest["render"])
            self.assertEqual(0.86, manifest["render"]["meowiness"])
            self.assertEqual(0.42, manifest["render"]["meow_tail_mix"])
            self.assertEqual(0.52, manifest["render"]["voice_rate"])
            self.assertEqual(0.50, manifest["render"]["pause_scale"])
            self.assertEqual(str(source.resolve()), manifest["render"]["source_voice"])
            self.assertTrue(manifest["render"]["source_reversed"])
            sound_events = [event for event in manifest["render"]["events"] if event["kind"] == "sound"]
            self.assertTrue(sound_events)
            self.assertTrue(all(event["source"] == "granular_voice" for event in sound_events))

    def test_generate_does_not_write_manifest_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "speech.wav"
            t = np.arange(12000, dtype=np.float32) / 12000.0
            voice = (np.sin(2.0 * np.pi * 180.0 * t) * 0.3).astype(np.float32)
            write_wav(source, voice, 12000)
            output = root / "stylized.wav"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exit_code = main(
                    [
                        "generate",
                        "--input",
                        str(source),
                        "--emotion",
                        "happy",
                        "--duration",
                        "2",
                        "--seed",
                        "9",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(0, exit_code)
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".json").exists())

    def test_generate_help_has_no_legacy_rendering_options(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["generate", "--help"])
        self.assertEqual(0, raised.exception.code)
        help_text = stdout.getvalue()
        self.assertNotIn("--style", help_text)
        self.assertNotIn("--bank", help_text)
        self.assertNotIn("--strict-bank", help_text)
        self.assertIn("--no-reverse-source", help_text)
        self.assertIn("--plan", help_text)
        self.assertIn("--no-plan", help_text)


if __name__ == "__main__":
    unittest.main()
