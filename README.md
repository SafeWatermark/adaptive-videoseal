# Adaptive VideoSeal

Adaptive VideoSeal extends Meta's VideoSeal video watermarking system with adaptive frame selection, propagation, and evaluation tooling for high-frame-rate content. The repository ships with the original VideoSeal inference stack, additional research notebooks, and an adaptive embedding pipeline designed for reproducible experiments.

## 1. Overview

- **Baseline VideoSeal**: official inference and evaluation code (video and optional audio watermarking).
- **Adaptive pipeline**: `src/main.py` builds on VideoSeal with dynamic key-frame selection and propagation.
- **Research assets**: notebooks for ablations, dataset configs, and attenuation presets.
- **Checkpoints**: automatically downloaded model weights (cache under `ckpts/`).

Use this README as a lab manual for installation, model setup, running experiments, and extending the system.

## 2. Environment Setup

### 2.1 Prerequisites
- Python 3.11
- [FFmpeg](https://ffmpeg.org/) available on the `PATH`
- PyTorch + torchvision compiled for your CUDA runtime

### 2.2 Create an Environment
Using [uv](https://github.com/astral-sh/uv):
```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

Using pip:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> **PyTorch** needs to be installed manually for your hardware, e.g.
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.

### 2.3 Optional Tooling
```bash
pip install pre-commit
pre-commit install
```
Enables linting (ruff) and formatting checks before each commit.

### 2.4 Linting
```bash
bash scripts/format.sh
```
Run Ruff’s linting checks on `src/` folders,
automatically fix simple style issues,
and ignore everything inside `src/videoseal/`.


## 3. Checkpoint Management

The default card `videoseal_1.0.yaml` pulls the official checkpoint (`y_256b_img.pth`). First run of any script that calls `setup_model_from_model_card("videoseal")` will:
1. download the checkpoint to `ckpts/`,
2. copy it into `src/videoseal/ckpts/` for library-relative access.

To pre-download:
```python
from pathlib import Path
from videoseal.videoseal.utils.cfg import setup_model_from_model_card
Path("ckpts").mkdir(exist_ok=True)
setup_model_from_model_card("videoseal")
```

To use custom weights, edit the YAML card (`checkpoint_path`) and rerun the loader.


## 4. Quick Start

### 4.1 Adaptive Embedding Demo
```bash
uv run src/main.py
```

Flow:
1. Load baseline VideoSeal weights.
2. Instantiate `AdaptiveVideoseal` to enable adaptive propagation.
3. Embed watermarks into `data/sample_video.mp4`.
4. Save watermarked outputs to `outputs/` (both baseline and adaptive variants).
5. Extract and report bit accuracy for baseline vs adaptive runs.

Artifacts:
- `outputs/1.mp4`: baseline watermarked video.
- `outputs/adaptive_1.mp4`: adaptive watermarked video.
- `outputs/*.txt`: payload bits for reproducibility.



## 5. Evaluation Utilities

- `videoseal.evals.metrics.bit_accuracy`: bit-wise accuracy metric.
- `detect_video()` in `src/main.py`: chunked video decoding + detection pipeline.
- `inference_av.py --detect`: CLI detection.
- Additional metrics (LPIPS, SSIM, PSNR) available via installed dependencies (`lpips`, `scikit-image`, `pytorch-msssim`).

## 6. Development Tips

- Enable `pre-commit` to keep formatting consistent.
