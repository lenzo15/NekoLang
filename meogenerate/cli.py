from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from typing import Sequence

from .audio import wav_duration, write_wav
from .emotion import (
    EMOTIONS,
    EmotionResult,
    analyze_with_emotion2vec,
    emotion_to_state,
    manual_emotion,
)
from .granular import render_granular_voice
from .planner import available_states, load_config, plan_behavior


def _add_emotion_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="iic/emotion2vec_plus_seed",
        help="emotion2vec model id",
    )
    parser.add_argument(
        "--hub",
        choices=("ms", "modelscope", "hf", "huggingface"),
        default="ms",
        help="model download source; use ms in mainland China",
    )
    parser.add_argument("--device", default="auto", help="FunASR device, e.g. cpu or cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meo_generate",
        description="Generate emotion-paced cartoon-cat gibberish from speech.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="render a stylized cat-language WAV")
    generate.add_argument(
        "--input",
        type=Path,
        required=True,
        help="PCM WAV source voice; also analyzed by emotion2vec unless --emotion is set",
    )
    generate.add_argument("--output", type=Path, required=True, help="output mono PCM WAV")
    generate.add_argument(
        "--emotion",
        choices=EMOTIONS,
        help="manual emotion; skips the emotion2vec model but still uses --input as the voice source",
    )
    generate.add_argument(
        "--intensity",
        type=float,
        help="emotion intensity from 0 to 1; overrides model-derived intensity",
    )
    generate.add_argument(
        "--state",
        help="cat behavior override, e.g. fight, demanding, distressed or angry",
    )
    generate.add_argument(
        "--duration",
        type=float,
        help="target output duration in seconds; defaults to the input duration",
    )
    generate.add_argument("--seed", type=int, help="random seed; omitted means a new seed")
    generate.add_argument("--config", type=Path, help="behavior JSON; defaults to config/default.json")
    generate.add_argument(
        "--meowiness",
        type=float,
        default=0.86,
        help="per-syllable synthetic meow character from 0 to 1 (default: 0.86)",
    )
    generate.add_argument(
        "--meow-tail-mix",
        type=float,
        default=0.42,
        help="source-conditioned meow tail blend from 0 to 0.60 (default: 0.42)",
    )
    generate.add_argument(
        "--reverse-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reverse the source waveform before granular recomposition (default: enabled)",
    )
    generate.add_argument(
        "--voice-rate",
        type=float,
        default=0.52,
        help="articulation speed and density from 0.45 to 1.25 (default: 0.52)",
    )
    generate.add_argument(
        "--pause-scale",
        type=float,
        default=0.50,
        help="shorten planned pauses from 0.20 to 1.0 (default: 0.50)",
    )
    generate.add_argument(
        "--plan-output",
        type=Path,
        help="render manifest path; also enables manifest output",
    )
    generate.add_argument(
        "--plan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write a JSON manifest next to the output WAV (default: disabled)",
    )
    _add_emotion_model_arguments(generate)

    analyze = subparsers.add_parser("analyze", help="print emotion2vec scores for a speech file")
    analyze.add_argument("input", type=Path)
    _add_emotion_model_arguments(analyze)

    states = subparsers.add_parser("list-states", help="list states and vocal categories")
    states.add_argument("--config", type=Path)
    return parser


def _analyze(args: argparse.Namespace) -> EmotionResult:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {args.input}")
    return analyze_with_emotion2vec(
        args.input,
        model_id=args.model,
        hub=args.hub,
        device=args.device,
    )


def _command_generate(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {args.input}")

    if args.emotion:
        emotion = manual_emotion(args.emotion, args.intensity if args.intensity is not None else 0.65)
    else:
        emotion = _analyze(args)
        if args.intensity is not None:
            emotion = EmotionResult(
                label=emotion.label,
                confidence=emotion.confidence,
                intensity=min(1.0, max(0.0, args.intensity)),
                scores=emotion.scores,
                source=emotion.source,
            )

    config = load_config(args.config)
    state = args.state or emotion_to_state(emotion.label)
    if state not in config["states"]:
        raise ValueError(
            f"Unknown state '{state}'. Available: {', '.join(available_states(config))}"
        )

    source_duration = wav_duration(args.input)
    if args.duration is not None:
        target_duration = args.duration
    elif source_duration is not None:
        target_duration = max(2.5, min(60.0, source_duration))
    else:
        raise ValueError(f"Unable to read PCM WAV duration: {args.input}")

    if not 0.0 <= args.meowiness <= 1.0:
        raise ValueError("--meowiness must be between 0 and 1")
    if not 0.0 <= args.meow_tail_mix <= 0.60:
        raise ValueError("--meow-tail-mix must be between 0 and 0.60")
    if not 0.45 <= args.voice_rate <= 1.25:
        raise ValueError("--voice-rate must be between 0.45 and 1.25")
    if not 0.20 <= args.pause_scale <= 1.0:
        raise ValueError("--pause-scale must be between 0.20 and 1.0")

    seed = args.seed if args.seed is not None else secrets.randbits(31)
    plan = plan_behavior(
        config=config,
        state=state,
        emotion=emotion.label,
        intensity=emotion.intensity,
        target_duration=target_duration,
        seed=seed,
    )
    audio_config = config.get("audio", {})
    sample_rate = int(audio_config.get("sample_rate", 24000))
    master_peak = float(audio_config.get("master_peak", 0.94))
    rendered = render_granular_voice(
        plan,
        source_voice=args.input,
        sample_rate=sample_rate,
        master_peak=master_peak,
        meowiness=args.meowiness,
        meow_tail_mix=args.meow_tail_mix,
        reverse_source=args.reverse_source,
        voice_rate=args.voice_rate,
        pause_scale=args.pause_scale,
    )
    write_wav(args.output, rendered.samples, rendered.sample_rate)

    manifest = {
        "version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "emotion": emotion.to_dict(),
        "plan": plan.to_dict(),
        "render": {
            "renderer": "granular_voice",
            "duration": round(rendered.duration, 6),
            "sample_rate": rendered.sample_rate,
            "source_voice": str(args.input.resolve()),
            "source_reversed": args.reverse_source,
            "meowiness": args.meowiness,
            "meow_tail_mix": args.meow_tail_mix,
            "voice_rate": args.voice_rate,
            "pause_scale": args.pause_scale,
            "events": list(rendered.events),
        },
    }
    manifest_path = args.plan_output or args.output.with_suffix(".json")
    write_plan = args.plan is True or (args.plan is None and args.plan_output is not None)
    if write_plan:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote: {args.output} ({rendered.duration:.2f}s, state={state}, seed={seed})")
    if write_plan:
        print(f"Plan:  {manifest_path}")
    return 0


def _command_analyze(args: argparse.Namespace) -> int:
    emotion = _analyze(args)
    print(json.dumps(emotion.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _command_list_states(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    for state in available_states(config):
        categories = sorted(
            {
                event["sound"]
                for sequence in config["states"][state].get("sequences", [])
                for event in sequence.get("events", [])
                if "sound" in event
            }
        )
        print(f"{state}: {', '.join(categories) if categories else '(silence only)'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            return _command_generate(args)
        if args.command == "analyze":
            return _command_analyze(args)
        if args.command == "list-states":
            return _command_list_states(args)
        parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 2
