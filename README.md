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

To download the VideoMarkBench dataset:

1. Download the dataset from Kaggle: [VideoMarkBench Dataset](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)
2. Place the dataset in the `data/videomarkbench/` directory with the following structure:

```
data/videomarkbench/
  {model}/          # Sora, Hunyuan, SVD
    {style}/        # realistic, cartoon, sci-fi
      {id}.mp4      # Video files
```
