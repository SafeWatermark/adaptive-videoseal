# VideoMarkBench 통합 가이드

## 개요

이 프로젝트에 VideoMarkBench 벤치마크가 통합되어 있습니다. VideoSeal과 Adaptive VideoSeal의 robustness를 다양한 perturbation과 aggregation 전략으로 평가할 수 있습니다.

## 데이터셋 구조

데이터셋은 다음과 같은 구조로 되어 있어야 합니다:

```
data/videomarkbench/
  {model}/          # Sora, Hunyuan, SVD
    {style}/        # realistic, cartoon, sci-fi
      {id}.mp4      # 비디오 파일
```

현재 데이터셋이 올바르게 인식되는지 확인하려면:

```bash
PYTHONPATH=src python src/videomarkbench/test_dataset.py
```

## 빠른 시작

### 1. 기본 평가 (Common perturbations만)

```bash
PYTHONPATH=src python src/videomarkbench/main.py \
    --model videoseal \
    --dataset data/videomarkbench \
    --output_dir outputs/videomarkbench \
    --perturbations common \
    --max_videos 5
```

또는 스크립트 사용:

```bash
bash scripts/run_videomarkbench.sh videoseal 5
```

### 2. 모든 perturbations로 평가

```bash
PYTHONPATH=src python src/videomarkbench/main.py \
    --model adaptive-videoseal \
    --dataset data/videomarkbench \
    --output_dir outputs/videomarkbench \
    --perturbations all \
    --max_videos 10
```

### 3. 특정 aggregation 전략만 사용

```bash
PYTHONPATH=src python src/videomarkbench/main.py \
    --model videoseal \
    --dataset data/videomarkbench \
    --output_dir outputs/videomarkbench \
    --perturbations common \
    --aggregations logit_mean bitacc_mean \
    --max_videos 5
```

## 주요 기능

### Perturbations

**Common (17가지):**
- `JPEG`: JPEG 압축 (quality: 90, 75, 50)
- `Gaussian`: 가우시안 노이즈 (std: 0.01, 0.05)
- `GaussianBlur`: 가우시안 블러 (std: 1.0, 2.0)
- `Crop`: 크롭 (crop_ratio: 0.9, 0.8)
- `MPEG4`: MPEG4 압축 (quality: 28, 40)
- `FrameAverage`: 프레임 평균 (num_frames: 2, 3)
- `FrameRemove`: 프레임 제거 (p: 0.1, 0.2)
- `FrameSwitch`: 프레임 교체 (p: 0.1, 0.2)

**Black-box (구현 예정):**
- `square_attack`: Square Attack
- `triangle_attack`: Triangle Attack

**White-box (구현 예정):**
- `wevade`: WEvade 공격

### Aggregation 전략 (8가지)

- `logit_mean`: 로짓 평균
- `logit_g_median`: 로짓 기하학적 중앙값
- `bit_majority`: 비트 다수결
- `bit_g_median`: 비트 기하학적 중앙값
- `bitacc_mean`: 비트 정확도 평균
- `bitacc_median`: 비트 정확도 중앙값
- `detection_threshold`: 검출 임계값
- `detection_majority`: 검출 다수결

## 출력 결과

평가가 완료되면 `output_dir`에 다음 파일들이 생성됩니다:

- `results.csv`: 상세 평가 결과 (각 비디오, perturbation, aggregation별 bit accuracy)
- `summary.txt`: 요약 통계 (텍스트 형식)
- `summary.json`: 요약 통계 (JSON 형식)
- `logs/`: 로그 파일

## 문제 해결

### Import 에러

만약 import 에러가 발생하면:

```bash
# 가상환경 활성화 확인
source .venv/bin/activate  # 또는 해당하는 가상환경

# 필요한 패키지 설치 확인
uv sync
```

### 데이터셋 인식 문제

데이터셋이 인식되지 않으면:

```bash
# 데이터셋 구조 확인
PYTHONPATH=src python src/videomarkbench/test_dataset.py
```

### 모델 로딩 문제

모델이 로드되지 않으면 체크포인트 파일이 올바른 위치에 있는지 확인:

```bash
ls -lh ckpts/y_256b_img.pth
```

## 고급 사용법

### 커스텀 Perturbation 추가

`src/videomarkbench/common_perturabtions/` 디렉토리에 새로운 perturbation 클래스를 추가하고, `src/videomarkbench/main.py`의 `get_perturbations()` 함수에 등록하세요.

### 커스텀 Aggregation 전략 추가

`src/videomarkbench/aggregations.py`에 새로운 aggregation 함수를 추가하고 `AGGREGATION_STRATEGIES` 딕셔너리에 등록하세요.

## 참고

- 평가는 시간이 오래 걸릴 수 있습니다. `--max_videos`로 테스트할 비디오 수를 제한하세요.
- GPU를 사용하면 훨씬 빠릅니다. `--device cuda`를 사용하세요.
- 결과는 CSV 형식으로 저장되므로 pandas로 분석하기 쉽습니다.
