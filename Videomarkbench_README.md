# VideoMarkBench 통합 가이드

## 개요

이 프로젝트는 원본 VideoMarkBench 레포지토리(https://github.com/zhengyuan-jiang/VideoMarkBench)를 git submodule로 사용하여, VideoSeal과 Adaptive VideoSeal의 robustness를 평가합니다.

**중요**: 이 프로젝트는 원본 VideoMarkBench 레포지토리를 git submodule로 사용합니다.

## Submodule 초기화

프로젝트를 처음 클론하거나 submodule을 업데이트하려면:

```bash
git submodule update --init --recursive
```

## 원본 레포 구조

원본 VideoMarkBench 레포지토리는 다음과 같은 구조를 가지고 있습니다:

- **common_perturabtions/**: 일반적인 perturbation들 (JPEG, Gaussian, Crop, FrameAverage, FrameRemove, FrameSwitch 등)
- **blackbox_perturbations/**: 
  - `square_attack/`: forgery_attack.py, removal_attack.py
  - `triangle_attack/`: forgery_TA.py, removal_TA.py
- **whitebox_perturbations/**: WEvade 공격 (각 워터마크 방법별로 구현)
- **aggregation.py**: 8가지 aggregation 전략
- 각 워터마크 방법별 디렉토리 (REVMark, StegaStamp, VideoSeal, VideoShield)

## 데이터셋 구조

데이터셋은 다음과 같은 구조로 되어 있어야 합니다:

```
data/videomarkbench/
  {model}/          # Sora, Hunyuan, SVD
    {style}/        # realistic, cartoon, sci-fi
      {id}.mp4      # 비디오 파일
```

## 빠른 시작

### 1. 기본 평가 (Common perturbations만)

```bash
bash scripts/run_videomarkbench.sh videoseal 5
```

또는 직접 실행:

```bash
PYTHONPATH=src uv run python src/evaluate/main.py \
    --model videoseal \
    --dataset data/videomarkbench \
    --output_dir outputs/videomarkbench \
    --max_videos 5 \
    --evaluate_common
```

### 2. Adaptive VideoSeal 평가

```bash
bash scripts/run_videomarkbench.sh adaptive-videoseal 5
```

### 3. 모든 평가 타입 실행 (향후 구현)

```bash
PYTHONPATH=src uv run python src/evaluate/main.py \
    --model videoseal \
    --dataset data/videomarkbench \
    --output_dir outputs/videomarkbench \
    --evaluate_common \
    --evaluate_blackbox_forgery \
    --evaluate_blackbox_removal \
    --evaluate_whitebox
```

## 평가 타입

### Common Perturbations (현재 구현됨)

원본 VideoMarkBench 레포지토리의 `common_perturabtions/`를 사용합니다.

**18가지 perturbations:**
- `JPEG`: JPEG 압축 (quality: 90, 75, 50) - **albumentations 최적화 적용**
- `Gaussian`: 가우시안 노이즈 (std: 0.01, 0.05) - 원본 구현 사용 (정확도 보장)
- `GaussianBlur`: 가우시안 블러 (std: 1.0, 2.0) - **albumentations 최적화 적용**
- `Crop`: 크롭 (crop_ratio: 0.9, 0.8) - **albumentations 최적화 적용**
- `MPEG4`: MPEG-4 비디오 압축 (CRF: 23, 28, 33) - decord 최적화 적용
- `FrameAverage`: 프레임 평균 (num_frames: 2, 3) - 벡터화 최적화 적용
- `FrameRemove`: 프레임 제거 (p: 0.1, 0.2)
- `FrameSwitch`: 프레임 교체 (p: 0.1, 0.2)

**성능 최적화:**
- albumentations 라이브러리를 사용하여 JPEG, GaussianBlur, Crop 변형 처리 속도 **40% 향상** (약 1.7배 빠름)
- 비디오 augmentation API를 사용하여 모든 프레임에 일관된 변형 적용
- 벡터화된 연산으로 FrameAverage 처리 속도 개선
- decord를 사용한 비디오 로딩 최적화

각 perturbation에 대해 **removal**과 **forgery** 공격 모두 평가되며, 8가지 aggregation 전략으로 평가됩니다.

**공격 타입:**
- **Removal 공격**: 워터마크가 삽입된 비디오에 perturbation을 적용한 후 워터마크가 여전히 검출되는지 확인 (FNR 측정)
- **Forgery 공격**: 워터마크가 없는 원본 비디오에 perturbation을 적용한 후 워터마크가 잘못 검출되는지 확인 (FPR 측정)

### Blackbox Attacks (향후 구현 예정)

- **Forgery Attacks**: 원본 비디오에 워터마크를 위조하는 공격
  - Square Attack forgery
  - Triangle Attack forgery
- **Removal Attacks**: 워터마크가 있는 비디오에서 워터마크를 제거하는 공격
  - Square Attack removal
  - Triangle Attack removal

### Whitebox Attacks (향후 구현 예정)

- **WEvade**: White-box 공격
  - Forgery: 원본 비디오에 워터마크 위조
  - Removal: 워터마크 제거

## Aggregation 전략 (8가지)

원본 VideoMarkBench 레포지토리의 `aggregation.py`를 사용합니다.

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

- `common_perturbations_results.csv`: Common perturbations 평가 결과
  - 각 비디오, perturbation, aggregation, attack_type(removal/forgery)별 detection 결과
  - 비디오 이름은 상대 경로 형식 (예: `Hunyuan/cartoon/1.mp4`)
- `common_perturbations_stats.json`: 통계 정보
  - 전체 FNR/FPR 통계
  - Perturbation별 통계
  - Aggregation별 통계
- `summary.json`: 평가 요약 정보
- `logs/`: 로그 파일

## 통계 계산

평가 결과에서 다음 통계가 자동으로 계산됩니다:

- **FNR (False Negative Rate)**: Removal 공격에서 워터마크 검출 실패 비율
- **FPR (False Positive Rate)**: Forgery 공격에서 False Positive 비율

통계는 `src/evaluate/utils.py`의 `calculate_statistics()` 함수로 계산되며, 다른 evaluator에서도 재사용 가능합니다.

## 랜덤 샘플링

`--max_videos` 옵션을 사용하면 전체 데이터셋에서 랜덤하게 비디오를 샘플링합니다:

- 시드는 `--seed` 파라미터로 고정 (기본값: 42)
- 같은 seed로 실행하면 항상 동일한 비디오들이 선택되어 재현 가능
- 선택된 비디오들은 정렬되어 일관된 순서로 평가됨

## 문제 해결

### Submodule 초기화 문제

만약 submodule이 초기화되지 않았다면:

```bash
git submodule update --init --recursive
```

### Import 에러

만약 import 에러가 발생하면:

```bash
# 가상환경 활성화 확인
source .venv/bin/activate

# 필요한 패키지 설치 확인
uv sync

# PYTHONPATH 설정 확인
export PYTHONPATH=src:$PYTHONPATH
```

**주의**: 이 프로젝트는 `albumentations` 라이브러리가 필수입니다. JPEG, GaussianBlur, Crop 변형에 사용됩니다.

### 데이터셋 인식 문제

데이터셋이 인식되지 않으면:

```bash
# 데이터셋 구조 확인
ls -la data/videomarkbench/
```

## 아키텍처

이 통합은 다음과 같이 구성됩니다:

- `src/videomarkbench/`: 원본 VideoMarkBench 레포지토리 (git submodule)
  - `aggregation.py`: Aggregation 전략
  - `common_perturabtions/`: Common perturbations
  - `blackbox_perturbations/`: Black-box attacks
  - `whitebox_perturbations/`: White-box attacks

- `src/evaluate/`: 통합 평가 시스템
  - `main.py`: 메인 entry point
  - `model_wrapper.py`: VideoSeal 및 AdaptiveVideoSeal 래퍼
  - `dataset_loader.py`: 데이터셋 로더
  - `common_evaluator.py`: Common perturbations 평가 엔진
  - `utils.py`: 통계 계산 등 유틸리티 함수

## 참고

- 평가는 시간이 오래 걸릴 수 있습니다. `--max_videos`로 테스트할 비디오 수를 제한하세요.
- GPU를 사용하면 훨씬 빠릅니다. `--device cuda`를 사용하세요.
- 결과는 CSV 형식으로 저장되므로 pandas로 분석하기 쉽습니다.
- 원본 VideoMarkBench 레포지토리: https://github.com/zhengyuan-jiang/VideoMarkBench
