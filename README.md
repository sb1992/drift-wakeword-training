# drift-wakeword-training

Training configs and trained models for the wake words shipped in the
[Drift](https://github.com/sb1992/drift) macOS app. Uses the upstream
[livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword)
training pipeline (VITS TTS for synthetic data + ACAV100M negatives +
conv_attention classifier).

## What's here

- `configs/` - the three YAML training configs (one per wake word) with
  hand-tuned adversarial negatives
- `trained-models/` - the exported `.onnx` classifiers as a backup so you
  don't have to retrain from scratch if anything goes sideways

The actual training pipeline (~30 GB of downloaded ML data + the Python
package) is **not** in this repo - it's all regenerable. See "Setup" below.

## Wake words

| Phrase | Action in app | Classifier file | Recall @ thr=0.5 | FPPH @ thr=0.5 |
|---|---|---|---|---|
| "Hey Drift" | activate session | `hey_drift.onnx` | 85.7% | 0.05 |
| "Drift Stop" | exit session | `drift_stop.onnx` | 87.4% | 0.00 |
| "Drift Code" | switch to code mode | `drift_code.onnx` | 85.1% | 0.00 |

All three were trained on 10K synthetic positives + adversarial negatives,
conv_attention small architecture, 50K steps + 5K Phase 2 + 5K Phase 3
checkpoint averaging. Total training time ~50 min per keyword on Apple
Silicon (M-series) with MPS.

## Setup (to retrain or train new wake words)

```bash
# 1. System deps
brew install espeak-ng ffmpeg portaudio

# 2. Clone the upstream training repo somewhere (NOT inside Drift's repo!)
git clone https://github.com/livekit/livekit-wakeword.git
cd livekit-wakeword

# 3. Python env
uv sync --all-extras

# 4. Download training data (~17.5 GB: VITS, ACAV100M, MUSAN, RIRs, validation)
uv run livekit-wakeword setup

# 5. Drop a config from this repo into the configs/ folder
cp /path/to/drift-wakeword-training/configs/hey_drift.yaml configs/

# 6. Run the full pipeline (~50 min). The env var is REQUIRED — see gotchas.
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 uv run livekit-wakeword run configs/hey_drift.yaml

# 7. The exported model lands at output/hey_drift/hey_drift.onnx
#    Copy it into Drift's Resources/livekit-wakeword/ folder
```

## Critical gotchas (don't lose hours rediscovering these)

### 1. `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` is mandatory on Apple Silicon

Without this env var, the feature extraction phase ballooned MPS memory to
**84 GB** during testing — system swap exploded, machine became unusable.
With it set, memory stayed under ~700 MB throughout. Set it before every
training run:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0
```

The classifier training phase still loads ACAV100M features (~13 GB) into
RAM, so don't run training while Xcode + Chrome + 14 other apps are
chewing through memory.

### 2. Mel spectrogram tensor shape is `[1, 1, time, 32]`, not `[1, time, 32]`

This is a Drift integration detail (not a training one), but worth
documenting because the failure mode is silent. When parsing the output of
`melspectrogram.onnx`, the time dimension is at **index 2**, not index 0.
Reading `shape[0]` gives you `1` (batch), so you get one frame, zero
embeddings, and all-zero classifier scores — model "works" but never
detects anything. See `LiveKitWakeWordEngine.swift` in Drift for the
correct parsing.

### 3. Disk usage per training run

A full pipeline run dumps roughly:
- ~3-4 GB of generated WAV training data per keyword (positive + negative + background)
- ~270 MB of `.npy` feature files per keyword
- ~340 KB of trained models (.pt + .onnx)

**Delete the WAV folders after training** — only the `.npy` features are
needed if you want to re-train without regenerating audio:

```bash
rm -rf output/hey_drift/{positive_train,positive_test,negative_train,negative_test,background_train,background_test}
```

### 4. Custom negatives matter a lot

Each `configs/*.yaml` has a `custom_negative_phrases` list of phonetically
similar phrases that should NOT trigger the wake word. These are the most
hand-tuned part of each config — don't skip them. Examples from
`drift_code.yaml`:

```yaml
custom_negative_phrases:
  - "code"           # bare action word
  - "drift"          # bare wake word
  - "drift coat"     # phonetic neighbor
  - "drift cold"
  - "drift mode"
  - "drift load"
  - "dress code"     # confused alternate
  - "trip code"
  ...
```

Without these, the model happily fires on any "drift X" or "X code" phrase.

## Adding a new wake word

1. Copy an existing config as a starting point: `cp configs/hey_drift.yaml configs/your_word.yaml`
2. Edit `model_name`, `target_phrases`, and `custom_negative_phrases`
3. Train with the env var set (see Setup)
4. Copy the resulting `.onnx` into Drift's `Resources/livekit-wakeword/`
5. Register the filename → tag mapping in
   `Features/Voice/WakeWord/WakeWordDetector.swift` (look for
   `knownClassifiers`)
6. Rebuild Drift

The engine handles N classifiers; no architectural change needed.

## Branch where this lands in Drift

- **App integration PR**: https://github.com/sb1992/drift/pull/57
  (`livekit-wakeword` → `glasses`)
- The Sherpa-ONNX KWS that this replaces is removed in the same PR — about
  62 MB of dylibs and models gone, replaced with 3 MB of LiveKit ONNX files.

## License

The trained models are derived from the upstream livekit-wakeword pipeline,
which is Apache 2.0. The configs in this repo are MIT.
