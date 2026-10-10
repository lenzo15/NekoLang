# NekoLang

[English](README.md) | [简体中文](README_CN.md)

将人类语音转换为带情绪节奏的原创（动森like）卡通猫语。

## 工作方式

1. 使用 `emotion2vec` 识别原始语音的情绪和强度；
2. 默认反转输入波形，降低残留可辨识语言的概率；
3. 提取有声区域并切成约 50–90 ms 的微颗粒；
4. 跨区域重排颗粒，重新规划短句、重音和停顿；
5. 对每个音节施加 `m-ee-ow` 共振峰变化和猫叫式滑音；
6. 定期在句尾叠加由同一输入人声生成的参考喵声；
7. 使用交叉淡化和响度包络平滑连接，避免明显断口。

## 安装

以下命令均在 `NekoLang` 目录下执行。从仓库根目录进入：

```bash
cd NekoLang
```

基础音频处理需要 Python 3.10+ 和 NumPy：

```bash
python -m pip install -r requirements.txt
```

自动情绪识别还需要：

```bash
python -m pip install -r requirements-emotion.txt
```

如需特定 CUDA 版本，请先按 PyTorch 官方说明安装对应的 `torch`。

## 快速开始

输入必须是未压缩 PCM WAV。

### 自动识别情绪

```bash
python meo_generate.py generate \
  --input path/to/speech.wav \
  --hub ms \
  --output tmp/cat-language.wav
```

首次运行会下载默认模型 `iic/emotion2vec_plus_seed`，后续使用本地缓存。中国大陆通常使用 `--hub ms`，也可使用 `--hub hf`。

### 手工指定情绪

指定 `--emotion` 后不会加载情绪模型，但仍必须提供 `--input` 作为猫语音色来源：

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

### 只分析情绪

```bash
python meo_generate.py analyze path/to/speech.wav --hub ms
```

> 多行示例使用 Bash 续行符（`\`）。在 PowerShell 或命令提示符中，请去掉续行符，将每条命令合并为一行执行。

## 主要参数

| 参数              | 范围/默认值              | 作用                                  |
| ----------------- | ------------------------ | ------------------------------------- |
| `--input`         | 必填                     | PCM WAV 输入；用于情绪分析和音色提取  |
| `--output`        | 必填                     | 输出单声道 16-bit PCM WAV             |
| `--meowiness`     | `0–1`，默认 `0.86`       | 每个音节的鼻音、共振峰和滑音强度      |
| `--meow-tail-mix` | `0–0.60`，默认 `0.42`    | 参考原声喵声的混合量；设为 `0` 可关闭 |
| `--voice-rate`    | `0.45–1.25`，默认 `0.52` | 发音速度和音节密度                    |
| `--pause-scale`   | `0.20–1.0`，默认 `0.50`  | 停顿长度比例                          |
| `--duration`      | 最长 60 秒               | 目标输出时长；省略时跟随输入时长      |
| `--state`         | 状态名                   | 覆盖情绪到行为状态的自动映射          |
| `--seed`          | 整数                     | 固定随机结果，便于复现                |
| `--config`        | JSON 路径                | 使用自定义行为配置                    |

完整参数：

```bash
python meo_generate.py generate --help
```

## 默认反转

颗粒提取前默认反转输入波形；情绪识别仍使用未反转的原始语音。启用 `--plan` 后，输出 JSON 会记录：

```json
{
  "render": {
    "source_reversed": true
  }
}
```

需要使用正向波形时：

```bash
python meo_generate.py generate \
  --input path/to/speech.wav \
  --no-reverse-source \
  --output tmp/cat-language-forward.wav
```

## 参考原声喵声

渲染器约每两个短句加入一次参考喵声，持续约 0.48–0.72 秒。它由当前输入人声的微颗粒生成，并继承说话者音色、行为状态音高和句尾音高漂移。参考喵声覆盖在原短句尾部，通过渐入叠加并保留原声底层，避免突然切换音色。

启用 `--plan` 后的相关 JSON 字段：

- `reference_meow`：该短句是否包含参考喵声；
- `reference_meow_mix`：混合强度；
- `reference_meow_duration`：覆盖时长；
- `reference_meow_source_units`：使用的原声颗粒单元数；
- `reference_meow_source`：固定为 `source_voice_grains`；
- `reference_meow_blend`：固定为 `overlaid_tail_crossfade`。

## 情绪与行为状态

| 语音情绪                        | 默认状态       |
| ------------------------------- | -------------- |
| `angry` / `disgusted`           | `angry`        |
| `fearful`                       | `fearful`      |
| `happy`                         | `affectionate` |
| `sad`                           | `distressed`   |
| `surprised`                     | `startled`     |
| `neutral` / `other` / `unknown` | `neutral`      |

情绪识别不能判断“打架”或“催促”等语义场景，需要时请显式指定：

```text
--state fight
--state demanding
```

查看全部状态：

```bash
python meo_generate.py list-states
```

## 时长与复现

未指定 `--duration` 时，输出时长跟随输入，并限制在 2.5–60 秒。渲染器会校准短句和停顿，使实际输出接近目标时长。

使用相同输入、配置和 `--seed` 可复现相同结果。默认只写入 WAV；指定 `--plan` 后会在输出 WAV 旁写入同名 JSON（指定 `--plan-output` 可自定义路径），包含：

- 情绪及置信度；
- 行为状态和随机种子；
- 计划与实际时长；
- 每个短句的起止时间；
- 音节、微颗粒和参考喵声统计；
- 当前渲染参数。

## 行为配置

[`config/default.json`](config/default.json) 定义状态、候选拟声序列、强度条件、短句时长、音高、响度和停顿。可复制配置并通过 `--config` 使用：

```bash
python meo_generate.py generate \
  --input path/to/speech.wav \
  --config path/to/custom-cat.json \
  --output tmp/custom-cat.wav
```

角色音色参数位于 [`meogenerate/granular.py`](meogenerate/granular.py) 的 `VOICE_PROFILES`。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 开源协议

NekoLang 使用 [MIT 协议](LICENSE) 开源。
