# TensorBoard 사용 가이드

## 🚀 빠른 시작

### 1. TensorBoard 실행
```bash
# 학습 시작 전 또는 학습 중 별도 터미널에서
tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006
```

### 2. 접속
- **로컬**: `http://localhost:6006`
- **RunPod**: 포트 포워딩 설정 후 접속

---

## 📊 주요 메트릭

### Scalars (스칼라 메트릭)
1. **train/loss** - 학습 손실
   - 감소하면 학습이 잘 되는 중
   - 200 steps 기준: 초기 > 1.0 → 최종 < 0.5

2. **train/learning_rate** - 학습률
   - Cosine 스케줄러: 점진적 감소
   - 5e-7에서 시작

3. **train/global_step** - 전체 스텝 수

4. **train/rewards/format** - 포맷 보상
   - 목표: ~50 steps에 0.9 이상

5. **train/rewards/equation** - 수식 보상
   - 목표: ~100 steps부터 증가 시작

6. **train/rewards/combined** - 총 보상
   - 목표: 200 steps에 ~1.5-1.8

### Distributions (분포)
- 가중치 분포
- Gradient 분포

---

## ⚙️ 설정

### 현재 설정 (가벼운 모드)
```yaml
# configs/training_config.yaml
training:
  logging_steps: 25          # 25 step마다 로깅
  logging_dir: "logs/tensorboard"
  report_to: ["tensorboard"]
```

### 더 자주 로깅하고 싶다면
```yaml
training:
  logging_steps: 10  # 25 → 10
```
⚠️ GPU 메모리 사용량 증가 가능

---

## 🔍 TensorBoard 팁

### 1. 스무스 적용
- Scalars 탭에서 `Smoothing` 슬라이더 조정
- 추천: 0.6-0.8 (노이즈 제거)

### 2. 여러 Run 비교
```bash
# 체크포인트별 비교
logs/tensorboard/
├── run_1/  # 첫 번째 실험
├── run_2/  # 두 번째 실험
└── run_3/  # 세 번째 실험
```

### 3. 특정 메트릭만 보기
- 좌측 메뉴에서 원하는 메트릭 선택
- Regex 필터: `train/rewards.*` (보상만)

---

## 🐛 문제 해결

### TensorBoard 안 보일 때
```bash
# 1. 포트 확인
lsof -i :6006

# 2. 로그 디렉토리 확인
ls -la logs/tensorboard/

# 3. TensorBoard 재시작
pkill -f tensorboard
tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006
```

### RunPod 포트 포워딩
```bash
# SSH 터널링
ssh -L 6006:localhost:6006 user@runpod-instance

# 로컬 브라우저에서
http://localhost:6006
```

### 메모리 부족 시
```yaml
# TensorBoard 비활성화
training:
  report_to: []  # ["tensorboard"] → []
```

---

## 📈 학습 진행 예시

### 정상적인 학습 곡선
```
Step 0   : Loss ~2.0, Format Reward ~0.0, Equation Reward ~0.0
Step 25  : Loss ~1.5, Format Reward ~0.3, Equation Reward ~0.0
Step 50  : Loss ~1.0, Format Reward ~0.9, Equation Reward ~0.05
Step 100 : Loss ~0.7, Format Reward ~0.95, Equation Reward ~0.15
Step 150 : Loss ~0.5, Format Reward ~0.95, Equation Reward ~0.30
Step 200 : Loss ~0.4, Format Reward ~0.95, Equation Reward ~0.40
```

### 문제 신호
- ❌ Loss가 증가: 학습률 너무 높음
- ❌ Format Reward가 50 step 후에도 < 0.5: 포맷 학습 실패
- ❌ Loss가 변하지 않음: 학습률 너무 낮음

---

## 💾 데이터 저장

TensorBoard 로그 위치:
```
logs/tensorboard/
└── events.out.tfevents.*
```

백업:
```bash
# 로컬로 다운로드
scp -r user@runpod:/workspace/MiniR1/logs/tensorboard ./
```

---

## 🎨 추천 뷰 설정

1. **학습 진행** 탭
   - `train/loss`
   - `train/learning_rate`

2. **보상 분석** 탭
   - `train/rewards/format`
   - `train/rewards/equation`
   - `train/rewards/combined`

3. **모델 상태** 탭
   - Distributions → Weights
   - Distributions → Gradients

---

**TensorBoard + 파일 로깅 하이브리드 = 최고!** 🎉
