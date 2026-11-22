# **Adaptive VideoSeal**

<p align="left">
    <a href="README.ko.md">한국어</a>&nbsp ｜ &nbspEnglish&nbsp
</p>


**Adaptive VideoSeal** is an extended version of Meta’s VideoSeal video watermarking system, enhanced with **adaptive frame selection** and **watermark propagation** mechanisms.
This repository includes the original VideoSeal implementation, research notebooks, and a **reproducible adaptive embedding pipeline** for experimentation.

## 1. Overview

* **Baseline VideoSeal**: Official video/audio watermarking inference and evaluation code (`src/videoseal`).
* **Adaptive pipeline**: `src/adaptive_models/adaptive_videoseal.py` extends VideoSeal with **adaptive keyframe selection and watermark propagation logic**.
* **Checkpoints**: Model weights are downloaded automatically and cached in the `ckpts/` directory.

## 2. Environment Setup

### 2.1 Prerequisites

* Python 3.11
* [FFmpeg](https://ffmpeg.org/) must be available in your system `PATH`
* PyTorch + torchvision compiled for your CUDA environment

### 2.2 Creating a Virtual Environment

```bash
uv venv .venv --python 3.11
source .venv/bin/activate  # for Linux / macOS
uv sync
```

### 2.3 Optional Development Tools

```bash
uv pip install pre-commit
pre-commit install
```

* Runs **linting (ruff)** and **format checks** automatically before each commit.

### 2.4 Code Formatting

```bash
bash scripts/format.sh
```

* Checks and auto-formats code within the `src/` directory
* The `src/videoseal/` directory is excluded from linting.

## 3. Quick Start

### 3.1 Video Scene Detector Overview

AdaptiveVideoSeal detects scene changes to optimize keyframe selection and watermark propagation.
During experiments, you can choose a detection method using the `--scene_detector` option.

#### **Detector Types**

| Detector      | Description                                                                                                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **content**   | Detects scene changes based on variations in color, saturation, and brightness (H, S, V). Default option. Inspired by the PySceneDetect project, it computes frame-to-frame RMSE (mean pixel difference). |
| **histogram** | Detects scene changes by comparing differences in Y-channel histogram distributions between frames.                                                                                                       |
| **hash**      | Uses perceptual image hashes to measure frame similarity—fast and effective for detecting abrupt scene changes.                                                                                           |
| **lpips** | Detects scene changes based on the perceptual difference between consecutive frames using the deep-learning–based LPIPS metric. |
---

#### **Model Types**

The **Adaptive VideoSeal** project supports not only the **VideoSeal** model but also the **StegaStamp** model.

| Model          | Details                                                                                                                      |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **VideoSeal**  | The default model. It applies the Adaptive Pipeline based on Meta's official VideoSeal 1.0 weights.                          |
| **StegaStamp** | Requires a separate ONNX model file (`stega_stamp.onnx`). <br> (For detailed setup, see **3.3 Using the StegaStamp Model**.) |
| **REVMark**    | Requires ME_Spynet_Full.pth. <br> (For detailed setup, see **3.4 Using the REVMark Model**.)     |

---

#### **ECC Mode (Error Correction)**

Use the `--ecc_mode` flag to enable error correction for watermark messages. This improves robustness against bit errors during extraction.

| ECC Mode | Description |
| -------- | ----------- |
| **rs** | Reed-Solomon code. Classic algebraic ECC with configurable redundancy. |
| **polar** | 5G NR Polar code via Sionna. Supports SC, SCL, BP decoders with soft-input decoding for better noise resilience. |
| **conv** | Convolutional code with Viterbi decoding via Sionna. Supports rate 1/2, 1/3 and constraint lengths 3-8. Effective for burst error correction. |
| **hamming** | Hamming code (pure PyTorch). Supports Hamming(7,4), (15,11), (31,26), (63,57). Single-bit error correction per block. Fast and lightweight. |
| **bch** | BCH code (pure PyTorch). Supports BCH(15,7,2), (31,21,2), (63,51,2), etc. Multi-bit error correction with Berlekamp-Massey decoding. |

```bash
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=polar ...    # Polar code
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=rs ...       # Reed-Solomon code
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=conv ...     # Convolutional code
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=hamming ...  # Hamming code
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=bch ...      # BCH code
```


#### **MAC Mode (Message Authentication Code)**
Use the `--mac_mode` flag to **authenticate watermark messages**. This verifies whether a **decoded watermark** was produced by the watermark model or forged by an attacker.

* Improves resistance to spoofing/forgery
* May increase **false negatives** (a valid watermark may be rejected)
* Recommended to use **ECC mode** together to reduce detection failures


| MAC Mode | Description |
| -------- | ----------- |
| **hmac** | uses a secret key + a cryptographic hash function (e.g., SHA-256) to generate an authentication tag for the watermark message. |

```bash
uv run python src/poc.py --model videoseal --mac_mode --mac_algorithm=hmac ... # hmac
```


### 3.2 Adaptive Watermarking Experiment

```bash
bash scripts/poc/videoseal/run_content_detector.sh     # Run experiment using the content-based detector
bash scripts/poc/videoseal/run_histogram_detector.sh   # Run experiment using the histogram-based detector
bash scripts/poc/videoseal/run_hash_detector.sh        # Run experiment using the hash-based detector
bash scripts/poc/videoseal/run_lpips_detector.sh        # Run experiment using the lpips detector
```


### Execution Flow

1. **Load baseline VideoSeal weights**
2. Instantiate `AdaptiveVideoSeal` with adaptive propagation enabled
3. Embed watermark into `data/sample_video.mp4`
4. Save results to the `outputs/` folder (both baseline and adaptive versions)
5. Compare bit accuracy between baseline and adaptive models

### Generated Outputs

* `outputs/1.mp4` → Baseline watermarking result
* `outputs/adaptive_1.mp4` → Adaptive watermarking result
* `outputs/*.txt` → Reproducible payload bit data

### 3.3 Using the StegaStamp Model

To run inference with the StegaStamp model, the **`stega_stamp.onnx`** file is required.

* **Where to download:**

  * The file can be downloaded from this repository’s **GitHub Releases** section.

    * Original source: [WAVES repository](https://github.com/umd-huang-lab/WAVES/tree/main/decoders)

* **File placement:**

  * After downloading, place the `stega_stamp.onnx` file at the following path relative to the project root:
    `onnx/stega_stamp.onnx`

Once the file is prepared, you can run watermarking experiments using the `StegaStamp` model within the Adaptive VideoSeal environment.

#### Execution Scripts

```bash
bash scripts/poc/stega_stamp/run_content_detector.sh     # Run experiment using the content detector
bash scripts/poc/stega_stamp/run_histogram_detector.sh   # Run experiment using the histogram detector
...
```

### 3.4 Using the REVMark Model

To run inference with the REVMark model:

1. **Initialize the git submodule**:
   ```bash
   git submodule update --init --recursive
   ```
   `Encoder.pth` and `Decoder.pth` will be placed automatically in `src/videomarkbench/REVMark/checkpoints/` directory

2. **Download `ME_Spynet_Full.pth`:**
   * Available from this repository's **GitHub Releases** section (v0.0.1).
   * Place the file at the project root directory.

Once prepared, you can use the REVMark model within the Adaptive VideoSeal environment.

### 3.5 SA-V Validation Image-to-Video Conversion

The SA-V validation dataset provides each sample as a **directory of JPEG frames**.
To evaluate Adaptive VideoSeal on this dataset, you must first convert each frame
sequence into an MP4 video.

A utility script is provided:

```bash
uv run scripts/sav_val_image2video.py --data_path <path_to_JPEGImages directory>
```

This will generate an `.mp4` file for each directory of frames.

### 3.6 Scene-Detector Threshold Experiments

* Use `src/threshold_experiment.py` to sweep scene-detector thresholds and watermark step sizes.

* Run the sweeps with the helper scripts:
  * `bash scripts/run_content_threshold_exp.sh`
  * `bash scripts/run_hist_threshold_exp.sh`
  * `bash scripts/run_hash_threshold_exp.sh`
  * `bash scripts/run_lpips_threshold_exp.sh`

* Notes:

  * **videoseal 1.0.** automatically loads the `attenuation` module (JND masking) using `configs/attenuation.yaml` and the implementation in modules/jnd.py.

  * For additional context, see [videoseal issue #57](https://github.com/facebookresearch/videoseal/issues/57).

* Outputs are stored under:
  ```
  output/step_size_eval/<Detector>/threshold_<τ>/step_<n>/
  ```

Analyze results in `src/notebooks/default_threshold_selection.ipynb`, which visualizes accuracy vs. embed ratio and helps which threshold to select per detector.

## 4. Testing

All major project components are validated through **pytest-based unit tests**, located in the `tests/` directory.

Run all tests:

```bash
uv run pytest
# or
bash scripts/test.sh
```

The `conftest.py` file fixes a global random seed to ensure experiment reproducibility.

## 5. Development Tips

* Enable `pre-commit` to maintain consistent code style
* Run `scripts/format.sh` regularly to auto-format and clean your code

## 6. Evaluation

### Scene-Detector Threshold × Step-Size Sweep Results

Results of the sweeps in [3.6](#36-scene-detector-threshold-experiments), using each detector's default threshold.

* **Setup:** SA-V validation, 30 videos (360p–1080p), seed 42, VideoSeal 1.0 with JND attenuation, `chunk_size=32`, `mode=repeat`
* **Condition:** identity only (no attacks), so these numbers say nothing about robustness. See VideoMarkBench below for attacks.
* **Cell format:** `bit accuracy (embed ratio)`. Embed ratio is the fraction of frames passed through the embedder; the rest receive a propagated watermark.

| Detector (τ) | step 8 | step 16 | step 32 | step 64 | step 128 |
|---|---|---|---|---|---|
| Content (42.0) | 99.8% (13.6%) | 99.2% (7.4%) | 97.0% (4.3%) | 92.8% (2.7%) | 84.5% (2.0%) |
| Histogram (0.15) | 99.8% (13.9%) | 99.2% (7.7%) | 97.1% (4.5%) | 92.9% (3.0%) | 84.9% (2.2%) |
| Hash (0.45) | 99.8% (13.3%) | 99.2% (7.1%) | 97.0% (3.9%) | 93.0% (2.4%) | 84.6% (1.7%) |
| LPIPS (0.125) | 99.9% (20.9%) | 99.5% (15.2%) | 98.1% (12.4%) | 95.9% (10.9%) | 92.3% (10.3%) |
| Uniform keyframes (1/step) | – (12.5%) | – (6.2%) | – (3.1%) | – (1.6%) | – (0.8%) |

**Observations**

* At step 8, embedding about 13–14% of frames keeps bit accuracy at about 99.8% for every detector (lowest single video: 97.3%; 98.0% for LPIPS).
* Embed ratio exceeds the uniform 1/step rate because scene cuts add extra keyframes. LPIPS triggers the most cuts, so it degrades least at large steps but uses more embeddings.
* At a similar embedding budget, periodic keyframes do better than scene-cut keyframes: Content at step 16 gives 99.2% with 7.4% embedded, while LPIPS at step 128 gives 92.3% with 10.3% embedded.
* Wall-clock embedding time stayed about the same across step sizes (≈53 s per video for Content), so the lower embed ratio has **not** yet translated into a measured speed-up.

**Limitations**

* No fixed-step vanilla VideoSeal baseline was run on the same 30 videos, so this sweep alone cannot show that adaptive selection beats uniform keyframes.
* Visual-quality metrics (PSNR/SSIM/VMAF) were skipped (`--skip_image_metrics True`), and ECC/MAC modes were not part of this sweep.
* Full per-threshold results (per-step means, per-video metrics, plots): [`results/step_size_eval/<Detector>/threshold_<τ>/`](results/step_size_eval/)
* Runs for Hash τ=0.5, Histogram τ=0.2 and LPIPS τ=0.15 did not finish and are not included.


### VideoMarkBench

This project includes the VideoMarkBench repository as a submodule, enabling systematic evaluation of the robustness of VideoSeal and AdaptiveVideoSeal.

📄 For detailed information → [VideoMarkBench Integration Guide](Videomarkbench_README.md)

#### Dataset Download

**1. VideoMarkBench Dataset (for Removal Attacks)**

Download the VideoMarkBench dataset:

1. Download the dataset from Kaggle: [VideoMarkBench Dataset](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)
2. Place the dataset in the `data/videomarkbench/` directory with the following structure:

```
data/videomarkbench/
  {model}/          # Sora, Hunyuan, SVD
    {style}/        # realistic, cartoon, sci-fi
      {id}.mp4      # Video files
```

**2. Kinetics-400 Dataset (for Forgery Attacks)**

For forgery attack evaluation, you need to download a sample of the Kinetics-400 dataset. We provide a script to download a subset:

**Recommended: Use pre-validated CSV file (avoids inaccessible videos)**

```bash
# Download using pre-validated CSV file (recommended)
# The script will automatically use data/kinetics/kinetics400_available_1000.csv if it exists
PYTHONPATH=src uv run python scripts/download_kinetics_sample.py \
    --output_dir data/kinetics

# Or specify a custom CSV file
PYTHONPATH=src uv run python scripts/download_kinetics_sample.py \
    --output_dir data/kinetics \
    --csv_file data/kinetics/kinetics400_available_1000.csv
```

**Alternative: Random sampling mode (may include inaccessible videos)**

```bash
# Download a sample subset (default: 10 classes, 5 videos per class)
PYTHONPATH=src uv run python scripts/download_kinetics_sample.py \
    --output_dir data/kinetics \
    --split val \
    --num_classes 10 \
    --videos_per_class 5 \
    --seed 42
```

**Options:**
- `--output_dir`: Output directory (default: `data/kinetics`)
- `--csv_file`: Path to CSV file with accessible video IDs (youtube_id,time_start,label). If not specified, uses `data/kinetics/kinetics400_available_1000.csv` if it exists. Otherwise, uses random sampling mode.
- `--split`: Dataset split to download - `train`, `val`, or `test` (only used in random sampling mode, default: `val`)
- `--num_classes`: Number of action classes to sample (only used in random sampling mode, default: 10)
- `--videos_per_class`: Number of videos per class (only used in random sampling mode, default: 5)
- `--seed`: Random seed for sampling (only used in random sampling mode, default: 42)

**Requirements:**
- `yt-dlp` must be installed (automatically installed via `uv sync`)

The downloaded videos will be organized as:

```
data/kinetics/
  kinetics400_available_1000.csv  # Pre-validated video list (optional)
  videos/
    {label}_{youtube_id}_{time_start}.mp4
```

**3. Complete Dataset Structure**

After downloading both datasets, your `data/` directory should have the following structure:

```
data/
  videomarkbench/          # Removal 공격용 (생성형 AI 비디오)
    {model}/               # Sora, Hunyuan, SVD
      {style}/             # realistic, cartoon, sci-fi
        {id}.mp4
  kinetics/                # Forgery 공격용 (Kinetics-400 샘플)
    kinetics400_available_1000.csv  # Pre-validated video list (optional)
    videos/
      {label}_{youtube_id}_{time_start}.mp4
```

#### Running Evaluation

**Common Perturbations, Blackbox Attacks, and Whitebox Attacks:**

```bash
# Common perturbations and whitebox attacks
PYTHONPATH=src uv run python src/evaluate/main.py \
    --model videoseal \
    --dataset data/videomarkbench \
    --kinetics_dataset data/kinetics/videos \
    --output_dir outputs/evaluation \
    --max_videos 10 \
    --evaluate_common \
    --evaluate_whitebox \
    --wevade_version standard \
    --wevade_rb 0.1

# Blackbox attacks (Square Attack)
PYTHONPATH=src uv run python src/evaluate/main.py \
    --model videoseal \
    --dataset data/videomarkbench \
    --kinetics_dataset data/kinetics/videos \
    --output_dir outputs/blackbox_eval \
    --max_videos 40 \
    --max_frames_per_video 14 \
    --evaluate_blackbox_removal \
    --evaluate_blackbox_forgery \
    --blackbox_attack_method square

# Blackbox attacks (Triangle Attack)
PYTHONPATH=src uv run python src/evaluate/main.py \
    --model videoseal \
    --dataset data/videomarkbench \
    --kinetics_dataset data/kinetics/videos \
    --output_dir outputs/blackbox_eval \
    --max_videos 40 \
    --max_frames_per_video 14 \
    --evaluate_blackbox_removal \
    --evaluate_blackbox_forgery \
    --blackbox_attack_method triangle
```

**Key Arguments:**
- `--dataset`: Path to VideoMarkBench dataset (used for removal attacks)
- `--kinetics_dataset`: Path to Kinetics dataset (used for forgery attacks)
- `--evaluate_common`: Evaluate common perturbations (JPEG, Gaussian, Crop, etc.)
- `--evaluate_blackbox_removal`: Evaluate blackbox removal attacks (Square/Triangle Attack)
- `--evaluate_blackbox_forgery`: Evaluate blackbox forgery attacks (Square/Triangle Attack)
- `--blackbox_attack_method`: `square` or `triangle` (default: `square`)
- `--evaluate_whitebox`: Evaluate whitebox attacks (WEvade)
- `--wevade_version`: `standard` or `adaptive` (default: `standard`)
- `--wevade_rb`: Perturbation bound (L-infinity norm) (default: 0.1)

**Note:**
- **Removal attacks** use the VideoMarkBench dataset (generated AI videos)
- **Forgery attacks** use the Kinetics dataset (real-world videos)
- If `--kinetics_dataset` is not specified, forgery attacks will use the same dataset as removal attacks (for backward compatibility)
- **Triangle Attack** requires `torch_dct` package (`uv add torch-dct`)
