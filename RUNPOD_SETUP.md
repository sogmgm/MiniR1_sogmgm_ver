# RunPod 실행 가이드

> Mini-R1 프로젝트를 RunPod에서 **순차적으로 실행**하기 위한 가이드

---

## 🎮 Step 1: GPU 선택

### 💰 추천 GPU (가성비 + 성능)

| GPU | VRAM | 시간당 비용 | Qwen 1.5B | Qwen 3B | 200 Steps 예상 시간 | 총 예상 비용 |
|-----|------|------------|-----------|---------|-------------------|-------------|
| **RTX 4090** ⭐ | 24GB | ~$0.69 | ✅ 최적 | ✅ 최적 | 3-4시간 | ~$2.50 |
| **RTX A5000** | 24GB | ~$0.50 | ✅ 최적 | ✅ 최적 | 4-6시간 | ~$2.50 |
| **RTX 3090** | 24GB | ~$0.44 | ✅ 좋음 | ✅ 좋음 | 5-7시간 | ~$2.50 |
| **L4** | 24GB | ~$0.45 | ✅ 가능 | ✅ 가능 | 6-8시간 | ~$3.00 |

**최종 추천**: 
- **가성비 최고**: RTX 3090
- **속도 최우선**: RTX 4090
- **안정성**: RTX A5000

### � 모델 선택

- **Qwen2.5-1.5B** (추천 ⭐)
  - VRAM: ~12-14GB
  - 빠른 학습 속도
  - 저렴한 비용
  - 첫 실험에 최적
  
- **Qwen2.5-3B**
  - VRAM: ~14-18GB
  - 더 나은 추론 성능
  - 약간 느린 속도
  - VRAM 16GB+ 권장

---

## 🚀 Step 2: RunPod Pod 생성

1. [RunPod](https://www.runpod.io/) 로그인
2. **Community Cloud** 또는 **Secure Cloud** 선택
3. 위에서 선택한 GPU 찾기
4. **템플릿**: `RunPod PyTorch 2.4` 또는 `CUDA 12.1` 포함된 것
5. **볼륨**: 최소 30GB (50GB 권장)
6. **Deploy** 클릭!
7. SSH 또는 **Web Terminal** 접속

---

## 📦 Step 3: 프로젝트 업로드

### 방법 1: GitHub (추천)
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/MiniR1.git
cd MiniR1
```

### 방법 2: 직접 업로드
```bash
# 로컬에서
cd /Users/kb.yang/Desktop/kb/repo
tar -czf minir1.tar.gz MiniR1/

# RunPod 파일 브라우저로 업로드 후
cd /workspace
tar -xzf minir1.tar.gz
cd MiniR1
```

---

## 🛠️ Step 4: UV 및 환경 설정

### 4-1. UV 설치
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version
```

### 4-2. PyTorch 설치 (CUDA 12.1)
```bash
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121
```

### 4-3. 프로젝트 의존성 동기화
```bash
uv sync
```

### 4-4. Flash Attention 설치 (선택, 속도 20% 향상)
```bash
uv add flash-attn --no-build-isolation
```
> ⚠️ 실패해도 괜찮음. 없으면 조금 느릴 뿐.

### 4-5. HuggingFace 로그인
```bash
# 토큰 없으면: https://huggingface.co/settings/tokens
uv run huggingface-cli login
```

---

## ✅ Step 5: 환경 검증

```bash
uv run python scripts/check_environment.py
```

이 스크립트가 확인:
- ✅ GPU 및 VRAM
- ✅ CUDA 버전
- ✅ 디스크 공간
- ✅ RAM
- ✅ 패키지 설치 여부
- 💡 최적 config 추천

---

## 📊 Step 6: 데이터셋 준비

```bash
uv run python scripts/dataset_prep.py --num_samples 5000
```

**예상 시간**: 2-5분  
**생성 파일**:
- `.cache/datasets/train_countdown_r1.json` (4500개)
- `.cache/datasets/test_countdown_r1.json` (500개)

---

## 🧪 Step 7: 보상 함수 테스트

```bash
uv run python scripts/rewards.py
```

**예상 결과**: 모든 테스트 통과 ✅

---

## 🎯 Step 8: 모델 설정 (필요시 수정)

### 8-1. 모델 선택
```bash
nano configs/training_config.yaml
```

**Qwen 1.5B 사용** (추천):
```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
```

**Qwen 3B 사용** (VRAM 18GB+ 필요):
```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
```

### 8-2. GPU 메모리 부족하면
```yaml
grpo:
  max_completion_length: 384  # 512 → 384
  num_generations: 1          # 2 → 1

training:
  gradient_accumulation_steps: 16  # 8 → 16
```

---

## 🚀 Step 9: 학습 시작!

```bash
uv run python scripts/train_grpo.py --config configs/training_config.yaml
```

**예상 시간**: 
- RTX 4090: 3-4시간
- RTX 3090: 5-7시간
- L4: 6-8시간

**체크포인트 저장**: 50, 100, 150, 200 steps

---

## � Step 10: 모니터링 (학습 중)

### 터미널 1: 학습 로그
## 📈 Step 10: 모니터링 (학습 중)

### 터미널 1: TensorBoard 실행 (선택)
```bash
# TensorBoard 시작
tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006
```
**접속**: RunPod의 포트 포워딩 또는 `http://localhost:6006`

### 터미널 2: 학습 진행상황 확인
```bash
# 실시간 로그 보기
tail -f logs/training.log
```

### 터미널 3: GPU 사용량 모니터링
```bash
# 1초마다 GPU 상태 확인
watch -n 1 nvidia-smi
```

### 생성 샘플 확인
```bash
# Step 50 샘플
cat completion_samples/step_0050_success.txt

# Step 100 샘플
cat completion_samples/step_0100_success.txt
```

### TensorBoard에서 확인 가능한 메트릭
- **Loss**: 학습 손실
- **Learning Rate**: 학습률 변화
- **Rewards**: 보상 점수 변화
- **GPU Utilization**: GPU 사용률

> 💡 TensorBoard는 25 step마다 업데이트됩니다 (메모리 절약)

---

## 🎓 Step 11: 학습 완료 후

### 결과 확인
```bash
# 진행 상황 보기
cat PROGRESS.md

# 체크포인트 확인
ls -lh checkpoints/qwen-r1-countdown/
```

### 최종 모델 평가
```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/qwen-r1-countdown/checkpoint-200
```

---

## 🔧 문제 해결

### ❌ CUDA Out of Memory
**증상**: RuntimeError: CUDA out of memory

**해결책**:
```bash
# configs/training_config.yaml 수정
nano configs/training_config.yaml
```

```yaml
# 1.5B로 변경
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"

# 시퀀스 길이 축소
grpo:
  max_completion_length: 384
  num_generations: 1

# Gradient accumulation 증가
training:
  gradient_accumulation_steps: 16
```

### ❌ 학습 중단되었을 때
```bash
# 마지막 체크포인트에서 재개
uv run python scripts/train_grpo.py \
  --config configs/training_config.yaml \
  --resume_from_checkpoint checkpoints/qwen-r1-countdown/checkpoint-100
```

### ❌ Flash Attention 설치 실패
```bash
# configs/training_config.yaml 수정
nano configs/training_config.yaml
```

```yaml
model:
  attn_implementation: "eager"  # flash_attention_2 → eager
```

### ❌ UV 설치 실패
```bash
# 대체: pip 사용
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### ❌ Disk Space 부족
```bash
# 데이터셋 샘플 줄이기
uv run python scripts/dataset_prep.py --num_samples 2000

# 체크포인트 개수 줄이기
# configs/training_config.yaml에서
training:
  save_total_limit: 2  # 4 → 2
```

---

## 💡 팁과 트릭

### 빠른 테스트 (10 steps만)
```bash
uv run python scripts/train_grpo.py \
  --config configs/training_config.yaml \
  --max_steps 10
```

### GPU 활용률 최대화
```yaml
# 메모리 여유 있으면
training:
  per_device_train_batch_size: 2  # 1 → 2
  gradient_accumulation_steps: 4   # 8 → 4
```

### 더 작은 데이터셋으로 실험
```bash
uv run python scripts/dataset_prep.py --num_samples 1000
```

### 학습 중 다른 터미널에서 샘플 생성 (TODO)
```bash
uv run python scripts/generate_samples.py --checkpoint checkpoints/qwen-r1-countdown/checkpoint-100
```

---

## � 예상 결과

### 학습 진행 (200 Steps)
| Step | Format 정확도 | 정답률 | 특징 |
|------|--------------|--------|------|
| 50   | ~90% | ~5% | `<think></think><answer></answer>` 학습 완료 |
| 100  | ~95% | ~15-20% | 간단한 계산 시작 |
| 150  | ~95% | ~25-30% | 연산 조합 시도 |
| 200  | ~95% | ~35-40% | 복잡한 추론 패턴 |

### 리소스 사용량
- **GPU 메모리**: 
  - Qwen 1.5B: 12-14GB
  - Qwen 3B: 14-18GB
- **디스크**: ~10GB
- **RAM**: ~16GB

---

## ✅ 체크리스트

실행 전 확인:

- [ ] GPU 선택 완료 (24GB VRAM 권장)
- [ ] RunPod Pod 생성 및 접속
- [ ] 프로젝트 업로드 (GitHub 또는 직접)
- [ ] UV 설치 및 가상 환경 생성
- [ ] PyTorch + 의존성 설치
- [ ] HuggingFace 로그인
- [ ] 환경 검증 통과 (`check_environment.py`)
- [ ] 디스크 여유 공간 20GB+
- [ ] 모델 선택 (1.5B 또는 3B)
- [ ] Pod 자동 종료 방지 설정

---

## 📞 추가 도움말

**로그 확인**:
```bash
# 학습 로그
cat logs/training.log

# GPU 상태
nvidia-smi

# 진행 상황
cat PROGRESS.md
```

**파일 구조**:
```
MiniR1/
├── checkpoints/           # 모델 체크포인트
│   └── qwen-r1-countdown/
│       ├── checkpoint-50/
│       ├── checkpoint-100/
│       ├── checkpoint-150/
│       └── checkpoint-200/
├── completion_samples/    # 생성 샘플
├── logs/                  # 학습 로그
├── .cache/datasets/       # 전처리된 데이터셋
└── configs/               # 설정 파일
```

---

**준비 완료! Step 4부터 순차적으로 실행하세요!** 🚀
