# Mini-R1: DeepSeek R1 "Aha Moment" Reproduction

**단일 GPU** RunPod 환경에서 경량화된 GRPO 학습으로 DeepSeek R1의 추론 능력 재현

## 🔧 기술 스택 및 시스템 사양

### 핵심 기술
- **패키지 관리**: UV (빠른 Python 패키지 관리자)
- **모델**: Qwen2.5-1.5B/3B-Instruct (메모리 효율화)
- **학습 방법**: GRPO (Group Relative Policy Optimization)
- **환경**: **단일 GPU** (Multi-GPU 지원 없음)


**실습 환경 참고**: NVIDIA A100 80GB 단일 GPU 기준으로 약 10~12시간 소요되었습니다.

### 📦 필수 소프트웨어
- Python 3.10+
- CUDA 12.1+
- UV 패키지 관리자
- 50GB 이상 디스크 공간

## 📁 프로젝트 구조

```
MiniR1/
├── README.md                    # 📖 프로젝트 전체 가이드 (이 문서)
├── RUNPOD_SETUP.md              # 🚀 RunPod 완전 가이드 (설치+실행+문제해결)
├── TENSORBOARD_GUIDE.md         # 📊 TensorBoard 사용법
├── PROGRESS.md                  # 📝 개발 진행 상황 기록
├── pyproject.toml               # 📦 UV 패키지 설정 및 의존성
│
├── configs/                     # ⚙️ 설정 파일
│   └── training_config.yaml     #    학습 하이퍼파라미터 (모델, 배치, GRPO 등)
│
├── scripts/                     # 🛠️ 실행 스크립트
│   ├── dataset_prep.py          #    데이터셋 준비 (HF → R1 형식 변환)
│   ├── rewards.py               #    형식/수식/길이 보상 + 안전한 수식 평가
│   ├── train_grpo.py            #    가중치 보상 적용 GRPO 학습 스크립트
│   ├── evaluate.py              #    모델 평가 및 테스트
│   └── check_environment.py     #    환경 설정 확인
│
├── notebooks/                   # 📓 Jupyter 노트북
│   └── data_exploration.ipynb   #    데이터 탐색 및 실험
│
├── .cache/                      # 💾 캐시 디렉토리 (생성됨)
│   └── datasets/                #    준비된 데이터셋 저장
│       ├── train_countdown_r1.json      # 학습 데이터 (3,600 샘플)
│       ├── test_countdown_r1.json       # 테스트 데이터 (400 샘플)
│       └── dataset_metadata.json        # 데이터셋 메타정보
│
├── logs/                        # 📊 학습 로그 (생성됨)
│   ├── generation_samples/      #    step_00005.txt 등 생성 샘플 기록
│   └── tensorboard_qwen2.5-3B_* #    TensorBoard 로그 파일
│
├── checkpoints/                 # 💾 모델 체크포인트 (생성됨)
│   └── qwen2.5-3B/              #    학습된 모델 저장
│       ├── checkpoint-1000/     #    기본 저장 주기(1000 step)
│       └── ...                  #    save_steps마다 자동 저장
```

### 주요 파일 설명

**설정 파일**:
- `training_config.yaml`: 모든 학습 파라미터 (모델, 배치, GRPO, 보상 등)

**핵심 스크립트**:
- `dataset_prep.py`: Countdown 데이터를 R1 프롬프트 형식으로 변환
- `rewards.py`: 형식/수식/길이 보상 및 안전한 수식 평가 구현
- `train_grpo.py`: 가중치 보상 기반 GRPO 학습 루프 + 생성 샘플 로깅
- `evaluate.py`: 테스트 데이터로 모델 성능 평가

**생성되는 디렉토리**:
- `.cache/datasets/`: 전처리된 데이터셋 (~50MB)
- `logs/generation_samples/`: 학습 중 생성된 텍스트 & 보상 로그
- `logs/tensorboard_*`: TensorBoard 시각화 데이터
- `checkpoints/`: 모델/LoRA 체크포인트 (각 ~3GB)

## 🧮 Reward 시스템 한눈에 보기

- **Format Reward (최대 0.5)**: `<think>...</think><answer>...</answer>` 태그 구조를 검증하고 순서를 확인합니다.
- **Equation Reward (최대 2.0)**: `<answer>` 내부 수식을 `safe_eval`로 계산해 정답 일치 여부와 숫자 사용 정확도를 평가합니다.
- **Length Penalty (최소 -0.5)**: 생성 길이가 `max_completion_length` 근처를 초과하면 페널티를 부여합니다.
- **가중치 설정**: `configs/training_config.yaml`의 `reward_weights`에서 각 보상 항목의 영향력을 조정할 수 있습니다.
- **샘플 로깅**: `logs/generation_samples/step_*.txt`에 원시/가중치 보상 값이 모두 기록됩니다.

## 🚀 빠른 시작 가이드

### 🎮 RunPod 구동 전체 과정

#### 1️⃣ Pod 생성 및 접속
```bash
# RunPod에서 RTX 4090/A5000 선택
# 템플릿: PyTorch 2.4 + CUDA 12.1
# 볼륨: 최소 50GB
# Web Terminal 또는 SSH로 접속
```

#### 2️⃣ 프로젝트 클론
```bash
cd /workspace
git clone https://github.com/sogmgm/MiniR1_sogmgm_ver.git
cd MiniR1_sogmgm_ver
```

#### 3️⃣ 환경 설정 (5-10분 소요)
```bash
# UV 설치 및 환경 로드
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 의존성 설치 (프로젝트 빌드 제외)
uv sync --no-install-project

# PyTorch CUDA 확인 (RunPod pytorch 템플릿 사용 시 이미 설치됨)
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# ❌ "False" 또는 CPU 버전이면 → CUDA 버전 설치
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Flash Attention 설치 (선택, 20% 속도 향상)
uv pip install flash-attn --no-build-isolation
```

> 💡 **Tip**: RunPod pytorch 템플릿을 사용했다면 PyTorch CUDA가 이미 설치되어 있을 가능성이 높습니다.

#### 4️⃣ Hugging Face 인증
```bash
uv run huggingface-cli login

# 또는 환경 변수로 설정
export HF_TOKEN="your_hf_token_here"
```

#### 5️⃣ GPU 확인
```bash
# GPU 상태 확인
nvidia-smi

# PyTorch CUDA 확인
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

#### 6️⃣ 데이터셋 준비 (~2-3분)
```bash
uv run python scripts/dataset_prep.py --config configs/training_config.yaml
# --num_samples 5000 처럼 인자를 주면 설정값을 덮어쓸 수 있습니다.
# 기본 설정 결과: train 3,600 / test 400 샘플 생성
```

#### 7️⃣ 학습 시작 (10-12시간 소요, A100 80GB 기준)
```bash
# 단일 GPU 학습
uv run python scripts/train_grpo.py --config configs/training_config.yaml

# 백그라운드 실행 (권장)
nohup uv run python scripts/train_grpo.py --config configs/training_config.yaml > training.log 2>&1 &

# 로그 실시간 확인
tail -f training.log

# 생성 샘플은 logs/generation_samples/step_*.txt 로 저장됩니다.
```

#### 8️⃣ TensorBoard 모니터링 (선택)
```bash
# 별도 터미널에서
tensorboard --logdir=logs/tensorboard_qwen2.5-3B_25112 --host=0.0.0.0 --port=6006
# 또는 로그 디렉토리에 맞춰 패턴 사용 (예: logs/tensorboard_qwen2.5-3B_*)

# RunPod 포트 포워딩 설정 후
# 브라우저에서: http://localhost:6006
```


#### 주요 메트릭
```
📈 train/loss              - 학습 손실 (감소 추세 확인)
📈 train/rewards/format    - 포맷 보상 
📈 train/rewards/equation  - 수식 보상 
📈 train/rewards/length    - 길이 페널티 감시 
📈 train/learning_rate     - 학습률 (Cosine 감소)
```


### 4️⃣ 체크포인트 관리
```bash
# 저장 위치 (save_steps: 1000)
checkpoints/qwen2.5-3B/
├── checkpoint-1000/
├── checkpoint-2000/
└── ...

# save_total_limit 설정에 따라 오래된 체크포인트 자동 정리
```


### 외부 참고자료
- **[원본 튜토리얼](https://www.philschmid.de/mini-deepseek-r1)**: Philipp Schmid의 Mini-DeepSeek R1 가이드
- **[DeepSeek R1 논문](https://arxiv.org/abs/2501.12948)**: 원본 연구 논문
- **[GRPO 논문](https://arxiv.org/abs/2402.03300)**: Group Relative Policy Optimization
- **[TRL Documentation](https://huggingface.co/docs/trl)**: Hugging Face TRL 라이브러리
- **[Qwen 2.5 모델](https://huggingface.co/Qwen)**: Qwen 모델 카드
- **[UV 패키지 관리자](https://github.com/astral-sh/uv)**: UV 공식 문서

### 관련 프로젝트
- **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)**: 원본 DeepSeek R1
- **[OpenAI O1](https://openai.com/o1/)**: 추론 모델 비교
- **[Countdown Dataset](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)**: 사용한 데이터셋

## 🤝 기여 및 라이선스

### 기여 방법
```bash
# 1. Fork 후 Clone
git clone https://github.com/YOUR_USERNAME/MiniR1.git

# 2. 브랜치 생성
git checkout -b feature/your-feature

```

