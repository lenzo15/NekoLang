# NekoLang

[English](README.md) | [简体中文](README_CN.md)

Turn human speech into original, Animal Crossing-style cartoon-cat gibberish with emotion-driven rhythm.

## How it works

1. Use `emotion2vec` to identify the emotion and intensity of the source speech.
2. Reverse the input waveform by default to reduce the chance of recognizable speech remaining.
3. Extract voiced regions and split them into micro-grains of approximately 50–90 ms.
4. Rearrange grains across regions and plan new phrases, accents, and pauses.
5. Apply `m-ee-ow` formant changes and cat-like pitch slides to each syllable.
6. Periodically overlay a reference meow generated from the same source voice at phrase endings.
7. Smooth transitions with crossfades and loudness envelopes to avoid abrupt cuts.

## Installation

Run the commands below from the `NekoLang` directory. From the repository root:

```bash
cd NekoLang
```

Basic audio processing requires Python 3.10+ and NumPy:

```bash
python -m pip install -r requirements.txt
```

Automatic emotion recognition also requires:

```bash
python -m pip install -r requirements-emotion.txt
```

If you need a specific CUDA version, install the matching `torch` build first by following the official PyTorch instructions.

## Quick start

The input must be an uncompressed PCM WAV file.

### Automatic emotion recognition

```bash
python meo_generate.py generate \
  --input path/to/speech.wav \
  --hub ms \
  --output tmp/cat-language.wav
```

The first run downloads the default model, `iic/emotion2vec_plus_seed`; later runs use the local cache. `--hub ms` is usually preferred in mainland China. You can also use `--hub hf`.

### Specify the emotion manually

Setting `--emotion` skips loading the emotion model, but `--input` is still required as the source of the cat voice's timbre:

```bash
python meo_generate.py generate \
  --input path/to/speech.wav \
  --emotion angry \
  --intensity 0.9 \
  --state fight \
  --duration 8 \
  --seed 42 \
  --output tmp/cat-fight.wav
```

### Analyze emotion only

```bash
python meo_generate.py analyze path/to/speech.wav --hub ms
```

> The multiline examples use Bash line continuations (`\`). In PowerShell or Command Prompt, enter each command on a single line without the continuation characters.

## Main options

| Option            | Range / default             | Purpose                                                                 |
| ----------------- | --------------------------- | ----------------------------------------------------------------------- |
| `--input`         | Required                    | PCM WAV input for emotion analysis and voice timbre extraction          |
| `--output`        | Required                    | Output mono 16-bit PCM WAV                                              |
| `--meowiness`     | `0–1`, default `0.86`       | Nasal character, formant shaping, and pitch-slide strength per syllable |
| `--meow-tail-mix` | `0–0.60`, default `0.42`    | Source-derived reference meow blend; set to `0` to disable              |
| `--voice-rate`    | `0.45–1.25`, default `0.52` | Articulation speed and syllable density                                 |
| `--pause-scale`   | `0.20–1.0`, default `0.50`  | Pause duration multiplier                                               |
| `--duration`      | Up to 60 seconds            | Target output duration; follows the input duration when omitted         |
| `--state`         | State name                  | Override the automatic emotion-to-behavior mapping                      |
| `--seed`          | Integer                     | Fix the random seed for reproducible results                            |
| `--config`        | JSON path                   | Use a custom behavior configuration                                     |

For the complete option list:

```bash
python meo_generate.py generate --help
```

## Source reversal

The input waveform is reversed by default before grain extraction. Emotion recognition still uses the original, unreversed speech. When `--plan` is enabled, the output JSON records:

```json
{
  "render": {
    "source_reversed": true
  }
}
```

To use the forward waveform instead:

```bash
python meo_generate.py generate \
  --input path/to/speech.wav \
  --no-reverse-source \
  --output tmp/cat-language-forward.wav
```

## Source-derived reference meows

The renderer adds a reference meow approximately every two phrases, lasting about 0.48–0.72 seconds. It is generated from micro-grains of the current input voice and inherits the speaker's timbre, the behavior state's pitch, and the phrase-ending pitch drift. The reference meow is faded in over the original phrase ending while retaining the underlying audio to avoid a sudden timbre change.

Related JSON fields when `--plan` is enabled:

- `reference_meow`: whether the phrase includes a reference meow.
- `reference_meow_mix`: blend strength.
- `reference_meow_duration`: overlay duration.
- `reference_meow_source_units`: number of source grain units used.
- `reference_meow_source`: always `source_voice_grains`.
- `reference_meow_blend`: always `overlaid_tail_crossfade`.

## Emotions and behavior states

| Speech emotion                  | Default state  |
| ------------------------------- | -------------- |
| `angry` / `disgusted`           | `angry`        |
| `fearful`                       | `fearful`      |
| `happy`                         | `affectionate` |
| `sad`                           | `distressed`   |
| `surprised`                     | `startled`     |
| `neutral` / `other` / `unknown` | `neutral`      |

Emotion recognition does not infer semantic situations such as fighting or demanding attention. Specify a state explicitly when needed:

```text
--state fight
--state demanding
```

List all available states:

```bash
python meo_generate.py list-states
```

## Duration and reproducibility

When `--duration` is omitted, the output follows the input duration, clamped to 2.5–60 seconds. The renderer calibrates phrases and pauses so the actual output stays close to the target duration.

Use the same input, configuration, and `--seed` to reproduce a result. By default, only the WAV file is written. With `--plan`, a JSON file with the same basename is written alongside the output WAV. Use `--plan-output` to choose a different path. The manifest includes:

- Emotion and confidence.
- Behavior state and random seed.
- Planned and actual durations.
- Start and end times for each phrase.
- Syllable, micro-grain, and reference meow statistics.
- Current rendering parameters.

## Behavior configuration

[`config/default.json`](config/default.json) defines states, candidate onomatopoeic sequences, intensity conditions, phrase durations, pitch, loudness, and pauses. Copy the configuration and pass it with `--config` to customize behavior:

```bash
python meo_generate.py generate \
  --input path/to/speech.wav \
  --config path/to/custom-cat.json \
  --output tmp/custom-cat.wav
```

Character voice parameters are defined in `VOICE_PROFILES` in [`meogenerate/granular.py`](meogenerate/granular.py).

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

NekoLang is licensed under the [MIT License](LICENSE).
