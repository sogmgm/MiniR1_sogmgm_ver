# RunPod 완전 가이드

> Mini-R1 프로젝트를 RunPod에서 실행하기 위한 **완벽 가이드**

---

## ⚡ 빠른 시작 (5분 설치)

```bash
# 1. 프로젝트 클론
cd /workspace
git clone https://github.com/sogmgm/MiniR1_sogmgm_ver.git
cd MiniR1_sogmgm_ver

# 2. UV 설치 및 PATH 설정
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 3. 의존성 설치 (프로젝트 빌드 제외)
uv sync --no-install-project

# 4. PyTorch 설치 (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. HuggingFace 로그인
export HF_TOKEN="your_token_here"

# 6. 환경 확인
uv run python scripts/check_environment.py

# 7. 데이터 준비 (2-3분)
uv run python scripts/dataset_prep.py --num_samples 5000

# 8. 학습 시작 (백그라운드)
nohup uv run python scripts/train_grpo.py --config configs/training_config.yaml > training.log 2>&1 &

# 9. 로그 확인
tail -f training.log
```

**예상 총 소요 시간**: 
- 설치: ~10분
- 데이터 준비: ~3분
- 학습 (200 steps): 3-4시간 (RTX 4090 기준)

---

## 📋 빠른 체크리스트

### 시작 전 확인
- [ ] RunPod GPU 선택 (RTX 4090/A5000 권장)
- [ ] 템플릿: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` 또는 CUDA 12.1+
- [ ] 볼륨: 50GB 이상
- [ ] HuggingFace 토큰 준비

### 설치 후 확인
- [ ] `pwd` → `/workspace/MiniR1_sogmgm_ver`
- [ ] `uv --version` → `uv 0.9.8` 이상
- [ ] `ls pyproject.toml` → 파일 존재 확인
- [ ] 환경 검증 통과 (`check_environment.py`)

---

## 🎮 Step 1: GPU 선택

### 💰 추천 GPU (가성비 + 성능)

| GPU | VRAM | 시간당 비용 | Qwen 1.5B | Qwen 3B | 학습 시간 | 총 비용 |
|-----|------|------------|-----------|---------|----------|---------|
| **RTX 4090** ⭐ | 24GB | ~$0.69 | ✅ 최적 | ✅ 최적 | 3-4시간 | ~$2.50 |
| **RTX A5000** | 24GB | ~$0.50 | ✅ 최적 | ✅ 최적 | 4-6시간 | ~$2.50 |
| **RTX 3090** | 24GB | ~$0.44 | ✅ 좋음 | ✅ 좋음 | 5-7시간 | ~$2.60 |
| **L4** | 24GB | ~$0.45 | ✅ 가능 | ✅ 가능 | 6-8시간 | ~$3.00 |

**최종 추천**: 
- **가성비 최고**: RTX 3090
- **속도 최우선**: RTX 4090
- **안정성**: RTX A5000

### 💡 모델 선택

- **Qwen2.5-1.5B** (추천 ⭐)
  - VRAM: ~12-14GB
  - 빠른 학습 속도
  - 첫 실험에 최적
  
- **Qwen2.5-3B**
  - VRAM: ~14-18GB
  - 더 나은 추론 성능
  - VRAM 18GB+ 권장

---

## 🚀 Step 2: RunPod Pod 생성

1. [RunPod](https://www.runpod.io/) 로그인
2. **Community Cloud** 또는 **Secure Cloud** 선택
3. GPU 선택
4. **템플릿**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (권장)
5. **볼륨**: 50GB 이상
6. **Deploy** 클릭
7. SSH 또는 **Web Terminal** 접속

---

## 📦 Step 3: 프로젝트 클론

```bash
cd /workspace
git clone https://github.com/sogmgm/MiniR1_sogmgm_ver.git
cd MiniR1_sogmgm_ver

# 현재 위치 확인 (중요!)
pwd
# 출력: /workspace/MiniR1_sogmgm_ver
```

---

## 🛠️ Step 4: UV 설치 및 환경 설정

### 4-1. UV 설치 및 PATH 설정
```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH 설정 (중요!)
export PATH="$HOME/.local/bin:$PATH"

# 또는 (설치 위치에 따라)
source $HOME/.local/bin/env

# 버전 확인
uv --version
```

### 4-2. 프로젝트 디렉토리 확인
```bash
# 반드시 프로젝트 디렉토리에 있어야 함
pwd
# /workspace/MiniR1_sogmgm_ver

# pyproject.toml 확인
ls -la pyproject.toml
```

### 4-3. 의존성 설치 (프로젝트 빌드 제외)
```bash
# 의존성만 설치 (권장, 빠름)
uv sync --no-install-project
```

### 4-4. PyTorch 설치 (CUDA 12.1)
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4-5. Flash Attention 설치 (선택)
```bash
# 속도 20% 향상, 하지만 5-10분 소요
uv pip install flash-attn --no-build-isolation
```
> ⚠️ 실패해도 괜찮음. 없으면 조금 느릴 뿐입니다.

### 4-6. HuggingFace 로그인
```bash
# 방법 1: 토큰으로 (권장)
export HF_TOKEN="your_token_here"

# 방법 2: 대화형
uv run huggingface-cli login
```

**토큰 발급**: https://huggingface.co/settings/tokens

---

## ✅ Step 5: 환경 검증

```bash
uv run python scripts/check_environment.py
```

**예상 출력**:
```
✅ GPU: NVIDIA GeForce RTX 4090
✅ VRAM: 24.0 GB
✅ CUDA: 12.1
✅ PyTorch: 2.5.0+cu121
✅ All dependencies installed
```

---

## 📊 Step 6: 데이터셋 준비

```bash
# 기본 5,000 샘플 (추천)
uv run python scripts/dataset_prep.py --num_samples 5000

# 빠른 테스트용
uv run python scripts/dataset_prep.py --num_samples 1000

# 긴 학습용
uv run python scripts/dataset_prep.py --num_samples 10000
```

**예상 시간**: 2-5분  
**생성 파일**:
- `.cache/datasets/train_countdown_r1.json`
- `.cache/datasets/test_countdown_r1.json`

---

## 🎯 Step 7: 모델 설정 (선택)

```bash
nano configs/training_config.yaml
```

### 모델 선택
```yaml
# Qwen 1.5B (추천)
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"

# Qwen 3B (VRAM 18GB+ 필요)
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
```

### 메모리 부족 시
```yaml
grpo:
  max_completion_length: 256  # 512 → 256
  num_generations: 1          # 2 → 1

training:
  gradient_accumulation_steps: 8  # 4 → 8
```

---

## 🚀 Step 8: 학습 시작!

### 방법 1: 백그라운드 실행 (권장)
```bash
nohup uv run python scripts/train_grpo.py \
  --config configs/training_config.yaml \
  > training.log 2>&1 &

# 로그 확인
tail -f training.log
```

### 방법 2: tmux 사용
```bash
# 세션 생성
tmux new -s training

# 학습 실행
uv run python scripts/train_grpo.py --config configs/training_config.yaml

# 나가기: Ctrl+B, D
# 재접속: tmux attach -t training
```

### 방법 3: 포그라운드
```bash
uv run python scripts/train_grpo.py --config configs/training_config.yaml
```

---

## 📈 Step 9: 모니터링

### 실시간 로그
```bash
tail -f training.log
```

### TensorBoard (선택)
```bash
# 별도 터미널에서
tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006
```

**RunPod 포트 연결**:
1. RunPod UI → Pod 클릭 → "Connect"
2. "TCP Port Mappings" → Port 6006 추가
3. 생성된 URL 접속

### GPU 모니터링
```bash
watch -n 1 nvidia-smi
```

### 생성 샘플 확인
```bash
ls -lh completion_samples/
cat completion_samples/step_0050_success.txt
cat completion_samples/step_0100_success.txt
```

---

## 🎓 Step 10: 평가

```bash
# 최종 평가
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/qwen-r1-countdown/checkpoint-200 \
  --num_samples 100

# 특정 체크포인트
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/qwen-r1-countdown/checkpoint-100 \
  --num_samples 50
```

---

## 🔧 문제 해결

### ❌ UV 명령어를 찾을 수 없음
```bash
export PATH="$HOME/.local/bin:$PATH"
# 또는
source $HOME/.local/bin/env

# 영구 설정
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### ❌ pyproject.toml을 찾을 수 없음
```bash
# 프로젝트 디렉토리로 이동
cd /workspace/MiniR1_sogmgm_ver
pwd
ls -la pyproject.toml
```

### ❌ CUDA Out of Memory
```yaml
# configs/training_config.yaml 수정
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"  # 3B → 1.5B

grpo:
  max_completion_length: 256
  num_generations: 1

training:
  gradient_accumulation_steps: 8
```

### ❌ 학습 중단 후 재개
```bash
uv run python scripts/train_grpo.py \
  --config configs/training_config.yaml \
  --resume_from_checkpoint checkpoints/qwen-r1-countdown/checkpoint-100
```

### ❌ Flash Attention 설치 실패
```bash
# 무시하고 진행 (선택사항이므로 OK)

# 또는 설정에서 비활성화
nano configs/training_config.yaml
```

```yaml
model:
  attn_implementation: "eager"  # flash_attention_2 → eager
```

---

## 💡 유용한 팁

### 빠른 테스트 (10 steps)
```bash
uv run python scripts/train_grpo.py \
  --config configs/training_config.yaml \
  --max_steps 10
```

### 프로세스 관리
```bash
# 확인
ps aux | grep train_grpo

# 종료
pkill -f train_grpo
```

### 디스크 공간 확인
```bash
df -h /workspace
```

---

## � 예상 결과

### 학습 진행 (200 Steps)

| Step | Format 정확도 | 정답률 | 특징 |
|------|--------------|--------|------|
| 50   | ~90% | ~5% | 형식 학습 완료 |
| 100  | ~95% | ~15-20% | 초기 추론 시작 |
| 150  | ~95% | ~25-30% | 패턴 인식 |
| 200  | ~95% | ~35-40% | 안정적 추론 |

### 리소스 사용량

- **GPU 메모리**: 
  - Qwen 1.5B: 12-14GB
  - Qwen 3B: 14-18GB
- **디스크**: ~10GB
- **RAM**: ~16GB

---

## 📊 핵심 개념

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

### 학습 진행
- **0-50 steps**: 형식 학습 (`<think></think>` 구조)
- **50-100 steps**: 초기 추론 (간단한 계산)
- **100-150 steps**: 패턴 인식 (숫자 조합)
- **150-200 steps**: 성능 수렴 (안정적 추론)

---

## 🔗 더 알아보기

- **전체 가이드**: [README.md](README.md)
- **TensorBoard**: [TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md)
- **진행 상황**: [PROGRESS.md](PROGRESS.md)

---

## ✅ 최종 체크리스트

- [ ] GPU 선택 및 Pod 생성
- [ ] 프로젝트 클론
- [ ] UV 설치 및 PATH 설정
- [ ] 의존성 설치 (`uv sync --no-install-project`)
- [ ] PyTorch 설치
- [ ] HuggingFace 로그인
- [ ] 환경 검증
- [ ] 데이터 준비
- [ ] 학습 시작
- [ ] 모니터링 설정

---

**🚀 모든 준비 완료! 학습을 시작하세요!**

**예상 총 비용**: ~$2.50 (RTX 4090, 200 steps 기준)
