# Project Instructions

## Dataset Preparation

Create raw data folders:

```powershell
python scripts/prepare_data.py --init
```

Put source files here:

- `data/raw/videos`
- `data/raw/audio`

Audio and video files must have the same filename stem, for example `sample_001.mp4` and `sample_001.wav`.

Generate train/validation data:

```powershell
python scripts/prepare_data.py --overwrite
```

Prepared files and manifests will be written to `data/processed`.

## Training

To be documented.

## Inference / CLI

To be documented.