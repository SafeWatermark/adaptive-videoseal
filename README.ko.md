# **Adaptive VideoSeal**

<p align="left">
    한국어 ｜ <a href="README.md">English</a>
</p>

**Adaptive VideoSeal**은 Meta의 VideoSeal 비디오 워터마킹 시스템을 기반으로,
**adaptive 프레임 선택**, **워터마크 전파** 등를 추가한 확장 버전입니다.
이 저장소에는 원본 VideoSeal, 연구용 노트북, 그리고 **재현 가능한 실험을 위한 Adaptive embedding 파이프라인**이 포함되어 있습니다.


## 1. 개요 (Overview)

* **Baseline VideoSeal**: 공식 비디오/오디오 워터마킹 추론 및 평가 코드.(`src/videoseal`)
* **Adaptive pipeline**: `src/adaptive_models/adaptive_videoseal.py`는 VideoSeal 기반으로 **adaptive 키프레임 선택 및 전파 로직**을 추가.
* **체크포인트 (Checkpoints)**: 모델 가중치는 자동으로 다운로드되며, `ckpts/` 경로에 캐시됨.

## 2. 환경 설정 (Environment Setup)

### 2.1 사전 요구사항

* Python 3.11
* [FFmpeg](https://ffmpeg.org/) 이 `PATH` 환경변수에 등록되어 있어야 함
* CUDA 환경에 맞게 컴파일된 PyTorch + torchvision

---

### 2.2 가상환경 생성


```bash
uv venv .venv --python 3.11
source .venv/bin/activate # linux, mac
uv sync
```

### 2.3 선택적 개발 도구

```bash
uv pip install pre-commit
pre-commit install
```

* 커밋 전에 자동으로 **linting(ruff)** 및 **포맷팅 검사**를 수행합니다.

### 2.4 코드 포맷팅

```bash
bash scripts/format.sh
```

* `src/` 내부의 코드 스타일을 검사 및 자동 수정
* `src/videoseal/` 디렉터리는 검사에서 제외됩니다.

## 3. 빠른 시작 (Quick Start)

### 3.1 Video Scene Detector 소개

AdaptiveVideoseal은 장면 변화(scene change)를 감지해 키프레임 선택 및 워터마크 전파를 최적화합니다.
실험 시 --scene_detector 옵션을 통해 다양한 감지 방식을 선택할 수 있습니다.

#### **Detector Types**

| Detector      | Description                                                                                                    |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| **content**   | 프레임 간의 색상·채도·밝기(H, S, V) 변화를 기반으로 장면 전환을 감지합니다. 기본값이며, <br>PySceneDetect 프로젝트를 참고해 프레임별 RMSE(평균 픽셀 변화량)를 계산합니다. |
| **histogram** | 프레임의 Y 히스토그램 분포 차이를 비교하여 장면 변화를 감지합니다.                                                                        |
| **hash**      | 이미지 perception hash 기반으로 프레임 유사도를 계산하며, 빠른 장면 변화 검출에 적합합니다.                                                                 |
| **lpips**    | 딥러닝 기반 지각적 유사도 지표(LPIPS) 를 사용해 연속 프레임 간의 ‘지각적 차이’를 기반으로 장면 전환을 감지합니다. |

---

#### **Model Types**

Adaptive VideoSeal 프로젝트는 **VideoSeal** 모델 외에도 **StegaStamp** 모델을 지원합니다.

| Model      | Details                                                                                                    |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| **Videoseal**   | 기본 모델입니다. Meta의 공식 VideoSeal 1.0 가중치를 기반으로 Adaptive Pipeline을 적용합니다. |
| **StegaStamp** | 별도의 ONNX 모델 파일(`stega_stamp.onnx`)이 필요합니다. <br> (자세한 설정은 **3.3 StegaStamp 모델 사용** 안내 참고). |
| **REVMark**      | ME_Spynet_Full.pth 파일이 필요합니다. <br> (자세한 설정은 **3.4 REVMark 모델 사용** 안내 참고). |

---

#### **ECC Mode (오류 정정)**

`--ecc_mode` 플래그를 사용하면 오류 정정 코드를 워터마크 메시지에 적용합니다. 추출 시 비트 오류에 대한 강건성이 향상됩니다.

| ECC Mode | Description |
| -------- | ----------- |
| **rs** | Reed-Solomon 코드. 설정 가능한 중복도를 가진 전통적인 대수적 ECC. |
| **polar** | Sionna 기반 5G NR Polar 코드. SC, SCL, BP 디코더를 지원하며 soft-input 디코딩으로 노이즈에 강건함. |
| **conv** | Sionna 기반 Convolutional 코드 (Viterbi 디코딩). Rate 1/2, 1/3 및 constraint length 3-8 지원. 연속 에러(burst error) 정정에 효과적. |

```bash
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=rs ...    # Reed-Solomon
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=polar ... # Polar code
uv run python src/poc.py --model videoseal --ecc_mode --ecc_codec=conv ...  # Convolutional code
```

#### **MAC 모드 (메시지 인증 코드, Message Authentication Code)**

`--mac_mode` 플래그를 사용해 **워터마크 메시지를 인증(Authentication)** 합니다. 이는 **디코딩된 워터마크**가 워터마크 모델에 의해 생성된 것인지, 아니면 공격자에 의해 위조된 것인지를 검증합니다.

* 스푸핑/위조(Forge) 공격에 대한 저항성을 향상
* **거짓 음성(False Negative)** 이 증가할 수 있음 (유효한 워터마크가 거부될 수 있음)
* 탐지 실패를 줄이기 위해 **ECC 모드**와 함께 사용하는 것을 권장

| MAC 모드   | 설명                                                                                 |
| -------- | ---------------------------------------------------------------------------------- |
| **hmac** | 비밀 키 + 암호학적 해시 함수(예: SHA-256)를 사용하여 워터마크 메시지에 대한 인증 태그(Authentication Tag)를 생성합니다. |

```bash
uv run python src/poc.py --model videoseal --mac_mode --mac_algorithm=hmac ... # hmac
```



### 3.2 Adaptive watermarking 실험

```bash
bash scripts/poc/videoseal/run_content_detector.sh     # content_detector를 활용한 experiment 실행
bash scripts/poc/videoseal/run_histogram_detector.sh   # histogram detector를 활용한 experiment 실행
bash scripts/poc/videoseal/run_hash_detector.sh        # hash detector를 활용한 experiment 실행
bash scripts/poc/videoseal/run_lpips_detector.sh        # lpips detector를 활용한 experiment 실행
```

### 실행 흐름

1. **VideoSeal 1.0. 가중치 로드**
2. `AdaptiveVideoseal` 인스턴스화 → adaptive propagation 활성화
3. `data/sample_video.mp4`에 워터마크 삽입
4. `outputs/` 폴더에 결과 저장 (기본 및 적응형 두 버전)
5. 기본 모델 vs adaptive 모델 비트 정확도(bit accuracy) 비교 및 보고

### 생성 결과

* `outputs/1.mp4` → 기본 모델 워터마킹 결과
* `outputs/adaptive_1.mp4` → 적응형 모델 워터마킹 결과
* `outputs/*.txt` → 재현 가능한 페이로드 비트 저장

### 3.3 Stega Stamp 모델 사용 안내
StegaStamp 모델의 추론(inference)을 위해서는 **`stega_stamp.onnx`** 파일이 필요합니다.

  * **파일 다운로드 출처:**

      * 해당 파일은 본 저장소의 **GitHub Releases** 섹션에서 다운로드할 수 있습니다.
        * 원본 출처: [Waves repository](https://github.com/umd-huang-lab/WAVES/tree/main/decoders)

  * **파일 저장 경로:**

      * 다운로드한 `stega_stamp.onnx` 파일을 프로젝트 루트 디렉터리 기준으로 `onnx/stega_stamp.onnx` 경로에 저장해야 합니다.


해당 파일을 준비하면 Adaptive VideoSeal 환경에서 `StegaStamp` 모델의 워터마킹 실험도 수행할 수 있습니다.

#### 실행 스크립트

```bash
bash scripts/poc/stega_stamp/run_content_detector.sh     # content_detector를 활용한 experiment 실행
bash scripts/poc/stega_stamp/run_histogram_detector.sh   # histogram detector를 활용한 experiment 실행
...
```

### 3.4 REVMark 모델 사용 안내

REVMark 모델의 추론(inference)을 위해서는:

1. **git submodule 초기화**:
   ```bash
   git submodule update --init --recursive
   ```
   `src/videomarkbench/REVMark/checkpoints/` 디렉토리에 `Encoder.pth`, `Decoder.pth`가 자동으로 다운로드됩니다.

2. **`ME_Spynet_Full.pth` 다운로드:**
   * 본 저장소의 **GitHub Releases** 섹션(v0.0.1)에서 다운로드할 수 있습니다.
   * 프로젝트 루트 디렉터리에 파일을 저장하세요.

파일을 준비하면 Adaptive VideoSeal 환경에서 REVMark 모델을 사용할 수 있습니다.

### 3.5 SA-V validation 이미지 → 비디오 변환

SA-V validation 데이터셋은 각 샘플이 **JPEG 이미지 시퀀스(프레임 폴더)** 형태로 제공됩니다.
Adaptive VideoSeal로 평가하려면 먼저 이 이미지 프레임들을 MP4 비디오로 변환해야 합니다.

이를 위한 간단한 유틸리티 스크립트를 제공합니다:

```bash
uv run scripts/sav_val_image2video.py --data_path <JPEGImages directory>
```

각 폴더의 프레임을 하나의 `.mp4` 비디오로 자동 변환합니다.

### 3.6 Scene-Detector Threshold Experiments

* `src/threshold_experiment.py`를 사용해 Scene-Detector Threshold(τ)와 step_size를 실험합니다.

* 아래 스크립트로 실험을 실행할 수 있습니다:

  * `bash scripts/run_content_threshold_exp.sh`
  * `bash scripts/run_hist_threshold_exp.sh`
  * `bash scripts/run_hash_threshold_exp.sh`
  * `bash scripts/run_lpips_threshold_exp.sh`

* 참고 사항:

  * **videoseal 1.0**은 `configs/attenuation.yaml` 설정과 `modules/jnd.py` 구현을 기반으로 `attenuation` 모듈(JND 마스킹)을 자동으로 로드합니다.

  * 자세한 내용은 [videoseal issue #57](https://github.com/facebookresearch/videoseal/issues/57)을 참고하세요.

* 출력 결과는 다음 경로에 저장됩니다:

  ```
  output/step_size_eval/<Detector>/threshold_<τ>/step_<n>/
  ```

결과 분석은 `src/notebooks/default_threshold_selection.ipynb`에서 수행하며, accuracy-embed ratio를 시각화하고 각 detector별 기본 임계값을 선택할 수 있는 분석 결과를 제공합니다.

## 4. test
프로젝트의 주요 구성 요소는 모두 pytest 기반 단위 테스트로 검증됩니다.
테스트 코드는 tests/ 디렉터리에 포함되어 있습니다.

테스트 실행:
```bash
uv run pytest
# 또는
bash scripts/test.sh
```

conftest.py에서 모든 테스트의 랜덤 시드(seed) 를 고정하여
재현 가능한 실험 환경을 유지합니다.


## 5. 개발 팁 (Development Tips)

* `pre-commit`을 활성화해 코드 스타일 일관성 유지
* `scripts/format.sh` 실행으로 코드 자동 정리

## 6. Evaluation

### VideoMarkBench

본 프로젝트는 VideoMarkBench 레포지토리를 submodule로 포함하며, VideoSeal 및 AdaptiveVideoSeal의 강건성을 체계적으로 평가할 수 있습니다.

📄 자세한 설명 보기 → [VideoMarkBench 통합 가이드](Videomarkbench_README.md)

#### Dataset Download

**1. VideoMarkBench 데이터셋 (제거 공격용)**

VideoMarkBench 데이터셋을 다운로드하려면:

1. Kaggle에서 데이터셋을 다운로드하세요: [VideoMarkBench Dataset](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)
2. 데이터셋을 `data/videomarkbench/` 디렉토리에 다음 구조로 배치하세요:

```
data/videomarkbench/
  {model}/          # Sora, Hunyuan, SVD
    {style}/        # realistic, cartoon, sci-fi
      {id}.mp4      # Video files
```
