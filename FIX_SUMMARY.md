# 🔧 학습 문제 해결 요약

## 📊 문제 진단

TensorBoard 로그를 확인한 결과, **모든 스텝에서 보상이 0**이었습니다:
- `reward: 0.0`
- `reward_std: 0.0`  
- `rewards/equation_reward_func/mean: 0.0`
- `rewards/format_reward_func/mean: 0.0`
- `loss: 0.0`
- `grad_norm: 0.0`

이는 **학습이 전혀 되지 않는다**는 의미입니다.

## 🐛 발견한 버그들

### 1. **치명적 버그: 보상 함수의 로직 오류**

**위치:** `scripts/rewards.py`

**문제:**
```python
# 기존 코드 (잘못됨)
completion = "<think>" + completion  # ❌ 중복 추가!
```

**원인:**
- 프롬프트가 이미 `<|im_start|>assistant\n<think>`로 끝남
- 모델은 `<think>` **이후부터** 텍스트를 생성
- 하지만 보상 함수가 `<think>`를 다시 앞에 추가
- 결과: `<think><think>내용...</think>...` 형태가 되어 정규식 매칭 실패
- 모든 보상이 0.0으로 계산됨

**해결:**
```python
# 수정된 코드 (올바름)
# completion에 <think>를 추가하지 않음! ✓
# 대신 </think>와 <answer> 태그만 확인
```

### 2. **하이퍼파라미터 문제**

**위치:** `configs/training_config.yaml`

#### 문제 1: Learning Rate가 너무 낮음
```yaml
# 기존
learning_rate: 5.0e-7  # ❌ 너무 낮음!

# 수정
learning_rate: 1.0e-5  # ✓ 20배 증가 (GRPO 표준 범위)
```

#### 문제 2: Beta (KL divergence coefficient)가 너무 낮음
```yaml
# 기존
beta: 0.001  # ❌ 너무 낮아서 학습 신호가 약함

# 수정
beta: 0.05  # ✓ 50배 증가 (일반적인 GRPO 범위: 0.01-0.1)
```

#### 문제 3: Warmup이 너무 짧음
```yaml
# 기존
warmup_ratio: 0.05  # ❌ 5%만 warmup

# 수정
warmup_ratio: 0.1  # ✓ 10%로 증가, 더 안정적인 시작
```

#### 문제 4: Reward Scale이 너무 작음
```yaml
# 기존
reward_scale: 1.0  # ❌ 보상 신호가 약함

# 수정
reward_scale: 10.0  # ✓ 10배 증가, 더 강한 학습 신호
```

## ✅ 수정된 파일들

### 1. `scripts/rewards.py`
- `format_reward_func()`: `<think>` 중복 추가 제거
- `equation_reward_func()`: `<think>` 중복 추가 제거
- 더 간단하고 정확한 형식 검증 로직

### 2. `configs/training_config.yaml`
- `learning_rate`: 5e-7 → 1e-5
- `beta`: 0.001 → 0.05
- `warmup_ratio`: 0.05 → 0.1
- `reward_scale`: 1.0 → 10.0

### 3. `scripts/test_rewards.py` (신규)
- 보상 함수를 독립적으로 테스트할 수 있는 스크립트
- 6가지 테스트 케이스 포함

## 🎯 예상 효과

### Before (버그 상태):
- ❌ 모든 보상 = 0.0
- ❌ Gradient norm = 0.0
- ❌ Loss = 0.0
- ❌ **학습이 전혀 안 됨**

### After (수정 후):
- ✅ 보상이 제대로 계산됨 (0.0 ~ 2.0 범위)
- ✅ Gradient가 업데이트됨
- ✅ Loss가 감소함
- ✅ **실제 학습이 진행됨**

## 🚀 다음 단계

1. **기존 체크포인트 삭제** (잘못된 학습 상태)
   ```bash
   rm -rf checkpoints/qwen-r1-countdown/checkpoint-*
   ```

2. **새로 학습 시작**
   ```bash
   python scripts/train_grpo.py --config configs/training_config.yaml
   ```

3. **TensorBoard로 모니터링**
   ```bash
   tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6007
   ```

4. **확인할 지표들:**
   - `rewards/format_reward_func/mean`: 0.5 이상으로 증가해야 함
   - `rewards/equation_reward_func/mean`: 서서히 증가 (0.1 → 0.3 → 0.5)
   - `reward`: 전체 보상 증가
   - `loss`: 감소 추세
   - `grad_norm`: 0이 아닌 값 (0.1 ~ 10 범위)

## 📝 참고사항

### GRPO 학습의 정상적인 패턴:
1. **초기 (0-20 steps)**: 
   - Format reward가 먼저 증가 (0.0 → 0.8)
   - Equation reward는 여전히 낮음 (0.0 ~ 0.1)

2. **중기 (20-60 steps)**:
   - Format reward가 높게 유지 (0.8 ~ 1.0)
   - Equation reward가 서서히 증가 (0.1 → 0.4)

3. **후기 (60-100 steps)**:
   - 둘 다 높은 값 유지
   - Equation reward 계속 증가 (0.4 → 0.6+)

### 주의사항:
- 첫 5-10 스텝은 불안정할 수 있음 (warmup 기간)
- Reward가 증가하지 않으면 completion samples를 확인할 것
- Gradient norm이 너무 크면 (>10) learning rate를 줄일 것
