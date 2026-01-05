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

Use the `--ecc_mode` flag to enable Reed-Solomon error correction for watermark messages. This improves robustness against bit errors during extraction.

```bash
uv run python src/poc.py --model videoseal --ecc_mode ...
```

---

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

**Common Perturbations and Whitebox Attacks:**

```bash
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
```

**Key Arguments:**
- `--dataset`: Path to VideoMarkBench dataset (used for removal attacks)
- `--kinetics_dataset`: Path to Kinetics dataset (used for forgery attacks)
- `--evaluate_common`: Evaluate common perturbations (JPEG, Gaussian, Crop, etc.)
- `--evaluate_whitebox`: Evaluate whitebox attacks (WEvade)
- `--wevade_version`: `standard` or `adaptive` (default: `standard`)
- `--wevade_rb`: Perturbation bound (L-infinity norm) (default: 0.1)

**Note:**
- **Removal attacks** use the VideoMarkBench dataset (generated AI videos)
- **Forgery attacks** use the Kinetics dataset (real-world videos)
- If `--kinetics_dataset` is not specified, forgery attacks will use the same dataset as removal attacks (for backward compatibility)
