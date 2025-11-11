# 🚀 Quick Start Guide

> Mini-R1 프로젝트를 5분 안에 시작하기

## ⚡ 초고속 시작 (RunPod)

```bash
# 1. Pod 접속 후 한 번에 실행
cd /workspace && \
git clone https://github.com/YOUR_USERNAME/MiniR1.git && \
cd MiniR1 && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
source $HOME/.cargo/env && \
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121 && \
uv sync

# 2. HuggingFace 로그인
export HF_TOKEN="your_token_here"

# 3. 데이터 준비 (2분)
uv run python scripts/dataset_prep.py --num_samples 5000

# 4. 학습 시작 (3-4시간)
nohup uv run python scripts/train_grpo.py --config configs/training_config.yaml > training.log 2>&1 &

# 5. 로그 확인
tail -f training.log
```

## 📋 체크리스트

### 시작 전 확인사항
- [ ] RunPod GPU 선택 (RTX 4090/A5000 권장)
- [ ] 볼륨 50GB 이상
- [ ] HuggingFace 토큰 준비
- [ ] CUDA 12.1+ 확인

### 필수 파일
- [ ] `configs/training_config.yaml` 존재 확인
- [ ] `.cache/datasets/` 디렉토리 생성됨
- [ ] `training.log` 파일 생성 확인

## 🎯 핵심 명령어

### 데이터 준비
```bash
# 기본 (5,000 샘플)
uv run python scripts/dataset_prep.py --num_samples 5000

# 더 많이 (10,000 샘플)
uv run python scripts/dataset_prep.py --num_samples 10000
```

### 학습 실행
```bash
# 포그라운드 (터미널 종료 시 중단)
uv run python scripts/train_grpo.py --config configs/training_config.yaml

# 백그라운드 (권장)
nohup uv run python scripts/train_grpo.py --config configs/training_config.yaml > training.log 2>&1 &

# 프로세스 확인
ps aux | grep train_grpo
```

### 학습 재개
```bash
# 마지막 체크포인트부터
uv run python scripts/train_grpo.py \
  --config configs/training_config.yaml \
  --resume_from_checkpoint checkpoints/qwen-r1-countdown/checkpoint-100
```

### 평가
```bash
# 최종 체크포인트 평가
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/qwen-r1-countdown/checkpoint-200 \
  --num_samples 100
```

## 📊 모니터링

### 실시간 로그
```bash
tail -f training.log
```

### TensorBoard (선택)
```bash
# 별도 터미널
tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006
```

### 생성 샘플 확인
```bash
# 최신 성공 샘플
ls -t completion_samples/*success.txt | head -1 | xargs cat

# 특정 step 샘플
cat completion_samples/step_0100_success.txt
```

### GPU 상태
```bash
watch -n 1 nvidia-smi
```

## ⚠️ 문제 발생 시

### OOM 에러
```yaml
# configs/training_config.yaml 수정
grpo:
  num_generations: 1  # 2 → 1
  max_completion_length: 256  # 512 → 256
```

### 느린 학습
```bash
# Flash Attention 설치
uv add flash-attn --no-build-isolation
```

### UV 명령어 안 됨
```bash
source $HOME/.cargo/env
```

## 📈 예상 타임라인

| 단계 | 소요 시간 | 누적 시간 |
|------|----------|----------|
| 환경 설정 | 10분 | 10분 |
| 데이터 준비 | 3분 | 13분 |
| 학습 (200 steps) | 3-4시간 | ~4시간 |
| 평가 | 5분 | ~4시간 5분 |

**총 소요**: 약 4시간 | **비용**: ~$2.50 (RTX 4090 기준)

## 🎓 핵심 개념

### 데이터 형태
```
입력: {nums: [19,36,55,7], target: 65}
  ↓
프롬프트: "Using [19,36,55,7], make 65. <think>"
  ↓
모델 출력: "추론... </think>\n<answer>55+36-7-19</answer>"
  ↓
보상: Format(1.0) + Equation(1.0) = 2.0
```

### 보상 함수
- **Format**: `<think></think><answer></answer>` 형식 검사
- **Equation**: 수식 정확성 검사 (숫자 사용, 계산 결과)

### 학습 진행
- **0-50 steps**: 형식 학습
- **50-100 steps**: 초기 추론
- **100-150 steps**: 패턴 인식
- **150-200 steps**: 성능 수렴

## 🔗 더 알아보기

- **전체 가이드**: [README.md](README.md)
- **RunPod 설정**: [RUNPOD_SETUP.md](RUNPOD_SETUP.md)
- **TensorBoard**: [TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md)
- **진행 상황**: [PROGRESS.md](PROGRESS.md)

---

**문제가 있나요?** → [README.md - 문제 해결](README.md#-문제-해결-troubleshooting) 섹션 참고
