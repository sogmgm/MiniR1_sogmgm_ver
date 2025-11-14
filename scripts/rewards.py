"""
Reward Functions for Mini-R1 GRPO Training
Implements format and equation correctness rewards

보상 함수 설명:
--------------
GRPO 학습에서는 생성된 텍스트에 "보상(reward)"을 부여하여 모델이 올바른 방향으로 학습하도록 합니다.
이 파일은 두 가지 주요 보상 함수를 구현합니다:

1. format_reward_func: 출력 형식이 올바른지 검사 (<think></think><answer></answer>)
2. equation_reward_func: 수학적으로 정답인지 검사 (수식이 맞는지, 목표값을 만족하는지)

보상 점수:
- 1.0 = 완벽하게 올바름
- 0.0 = 잘못됨 또는 실패

데이터 형태:
-----------
입력 예시:
  - nums: [19, 36, 55, 7]  (사용 가능한 숫자들)
  - target: 65             (만들어야 하는 목표 숫자)

모델 출력 예시:
  "추론 과정을 작성합니다... </think>
  <answer> 55 + 36 - 7 - 19 </answer>"

학습 과정:
---------
1. 모델이 프롬프트를 받고 "<think>" 이후부터 텍스트 생성
2. 생성된 텍스트를 보상 함수로 평가 (형식 + 정확성)
3. 높은 보상을 받을수록 모델이 그런 방식으로 생성하도록 학습됨
"""

import re
from typing import List


def format_reward_func(completions: List[str], target: List[str], **kwargs) -> List[float]:
    """
    포맷 보상 함수: 생성된 텍스트가 올바른 형식을 따르는지 검사
    
    올바른 형식:
    <think>
    여기에 추론 과정...
    </think>
    <answer> 최종 수식 </answer>
    
    예시:
    ✅ 올바른 형식:
    "Let me think... </think>\n<answer> 55 + 36 - 7 - 19 </answer>"
    
    ❌ 잘못된 형식:
    "The answer is 55 + 36 - 7 - 19"  (태그 없음)
    "<think> ... <think> ... </think></think><answer>...</answer>"  (중첩된 태그)
    
    Args:
        completions: 모델이 생성한 텍스트 리스트
        target: 목표 숫자 리스트 (이 함수에서는 사용 안 함)
        **kwargs: 추가 인자
    
    Returns:
        각 completion에 대한 보상 점수 리스트 (1.0 = 올바름, 0.0 = 잘못됨)
    """
    rewards = []
    
    for completion in completions:
        try:
            # 프롬프트가 이미 '<think>'로 끝나므로 모델은 그 이후부터 생성
            # completion에 <think>를 추가하지 않음!
            # 대신 </think>와 <answer> 태그가 올바르게 있는지만 확인
            
            # Step 1: </think> 태그가 있는지 확인
            if "</think>" not in completion:
                rewards.append(0.0)
                continue
            
            # Step 2: <answer> ... </answer> 태그가 있는지 확인
            answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if answer_match is None:
                rewards.append(0.0)
                continue
            
            # Step 3: </think>가 <answer>보다 먼저 나오는지 확인
            think_end_pos = completion.find("</think>")
            answer_start_pos = completion.find("<answer>")
            
            if think_end_pos == -1 or answer_start_pos == -1 or think_end_pos >= answer_start_pos:
                rewards.append(0.0)
                continue
            
            # 모든 조건을 만족하면 보상 1.0
            rewards.append(1.0)
                
        except Exception:
            # 에러 발생 시 보상 0.0
            rewards.append(0.0)
    
    return rewards


def equation_reward_func(
    completions: List[str],
    target: List[str],
    nums: List[List[int]],
    **kwargs
) -> List[float]:
    """
    수식 정확성 보상 함수: 생성된 수식이 수학적으로 올바르고 정답인지 검사
    
    검사 항목:
    1. <answer> 태그 안에 수식이 있는가?
    2. 주어진 숫자를 정확히 한 번씩만 사용했는가?
    3. 허용된 연산자만 사용했는가? (+, -, *, /, 괄호, 소수점)
    4. 수식을 계산했을 때 목표 숫자와 일치하는가?
    
    예시:
    ✅ 정답:
    - 숫자: [19, 36, 55, 7], 목표: 65
    - 수식: "55 + 36 - 7 - 19"
    - 계산: 55 + 36 - 7 - 19 = 65 ✓
    - 모든 숫자 사용 ✓
    - 보상: 1.0
    
    ❌ 오답 예시 1 (계산 틀림):
    - 숫자: [19, 36, 55, 7], 목표: 65
    - 수식: "55 + 36 - 7 - 18"  (18은 주어지지 않은 숫자)
    - 보상: 0.0
    
    ❌ 오답 예시 2 (숫자 누락):
    - 숫자: [19, 36, 55, 7], 목표: 65
    - 수식: "55 + 36 - 7"  (19를 사용 안 함)
    - 보상: 0.0
    
    ❌ 오답 예시 3 (결과 틀림):
    - 숫자: [19, 36, 55, 7], 목표: 65
    - 수식: "55 + 36 + 7 - 19"
    - 계산: 79 ≠ 65 ✗
    - 보상: 0.0
    
    Args:
        completions: 모델이 생성한 텍스트 리스트
        target: 목표 숫자 리스트
        nums: 각 문제에 사용 가능한 숫자 리스트
        **kwargs: 추가 인자
    
    Returns:
        각 completion에 대한 보상 점수 리스트 (1.0 = 정답, 0.0 = 오답)
    """
    rewards = []
    
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # 프롬프트가 이미 '<think>'로 끝나므로 completion에 추가하지 않음
            
            # Step 1: <answer> 태그에서 수식 추출
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                # <answer> 태그가 없으면 보상 0.0
                rewards.append(0.0)
                continue
            
            # Step 2: 수식 가져오기
            equation = match.group(1).strip()
            
            # Step 3: 수식에서 모든 숫자 추출
            # 예: "55 + 36 - 7 - 19" → [55, 36, 7, 19]
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
            
            # Step 4: 주어진 숫자를 정확히 한 번씩만 사용했는지 확인
            # sorted([55, 36, 7, 19]) == sorted([19, 36, 55, 7]) ✓
            if sorted(used_numbers) != sorted(numbers):
                # 숫자를 잘못 사용했으면 보상 0.0
                rewards.append(0.0)
                continue
            
            # Step 5: 허용된 문자만 사용했는지 확인
            # 허용: 숫자, +, -, *, /, (, ), 소수점, 공백
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                # 허용되지 않은 문자 사용 시 보상 0.0
                rewards.append(0.0)
                continue
            
            # Step 6: 수식을 안전하게 계산
            # eval()을 사용하지만 __builtins__를 제한하여 안전하게 실행
            # 예: "55 + 36 - 7 - 19" → 65
            result = eval(equation, {"__builtins__": None}, {})
            
            # Step 7: 계산 결과가 목표값과 일치하는지 확인 (부동소수점 오차 허용)
            # abs(65.0 - 65) < 0.00001 ✓
            if abs(float(result) - float(gt)) < 1e-5:
                # 정답! 보상 1.0
                rewards.append(1.0)
            else:
                # 계산 결과가 틀렸으면 보상 0.0
                rewards.append(0.0)
                
        except Exception:
            # 어떤 에러든 발생하면 보상 0.0
            # (예: eval 에러, 0으로 나누기, 형변환 에러 등)
            rewards.append(0.0)
    
    return rewards


def combined_reward_func(
    completions: List[str],
    target: List[str],
    nums: List[List[int]],
    format_weight: float = 1.0,
    equation_weight: float = 1.0,
    **kwargs
) -> List[float]:
    """
    결합 보상 함수: 포맷 보상과 수식 보상을 가중합
    
    최종 보상 = (format_weight × 포맷 보상) + (equation_weight × 수식 보상)
    
    예시:
    1. 완벽한 정답:
       - 포맷 보상: 1.0
       - 수식 보상: 1.0
       - 최종: 1.0 × 1.0 + 1.0 × 1.0 = 2.0
    
    2. 포맷만 맞음:
       - 포맷 보상: 1.0
       - 수식 보상: 0.0 (계산 틀림)
       - 최종: 1.0 × 1.0 + 1.0 × 0.0 = 1.0
    
    3. 전부 틀림:
       - 포맷 보상: 0.0
       - 수식 보상: 0.0
       - 최종: 1.0 × 0.0 + 1.0 × 0.0 = 0.0
    
    학습 초기에는 포맷 보상이 높아지고,
    학습이 진행되면서 수식 보상도 함께 높아지는 것이 정상입니다.
    
    Args:
        completions: 모델이 생성한 텍스트 리스트
        target: 목표 숫자 리스트
        nums: 각 문제에 사용 가능한 숫자 리스트
        format_weight: 포맷 보상 가중치 (기본값: 1.0)
        equation_weight: 수식 보상 가중치 (기본값: 1.0)
        **kwargs: 추가 인자
    
    Returns:
        각 completion에 대한 결합 보상 점수 리스트
    """
    format_rewards = format_reward_func(completions, target, **kwargs)
    equation_rewards = equation_reward_func(completions, target, nums, **kwargs)
    
    combined = [
        format_weight * f + equation_weight * e
        for f, e in zip(format_rewards, equation_rewards)
    ]
    
    return combined


# Test cases
def test_reward_functions():
    """
    보상 함수 테스트
    
    다양한 시나리오로 보상 함수가 제대로 작동하는지 확인합니다:
    - 올바른 형식 + 올바른 수식 (정답)
    - 올바른 형식 + 틀린 수식
    - 틀린 형식
    - 중첩된 태그
    - 잘못된 숫자 사용
    """
    
    print("Testing Reward Functions")
    print("=" * 80)
    
    
    # Test samples (without <think> prefix as it's added automatically)
    # 테스트 샘플들 (모델이 생성하는 부분만, <think>는 자동으로 추가됨)
    
    # 샘플 1: 완벽한 정답 (긴 추론 포함)
    correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. One possible combination is 55 + 36 - 19 + 7... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""
    
    # 샘플 2: 완벽한 정답 (짧은 추론)
    correct_sample_2 = """ ... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""
    
    # 샘플 3: 포맷 틀림 (태그 없음)
    wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""
    
    # 샘플 4: 포맷 틀림 (중첩된 태그)
    wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:
95 + 88 = 183
Now, let's subtract 104 from 183 to get 79:
183 - 104 = 79
<think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""
    
    # 샘플 5: 포맷 맞지만 수식 틀림 (18은 주어진 숫자가 아님)
    wrong_result = """ ... </think>
<answer> 55 + 36 - 7 - 18 </answer>"""    # Test data
    test_completions = [
        correct_sample_1,
        correct_sample_2,
        wrong_format,
        wrong_format_2,
        wrong_result
    ]
    test_targets = ["65", "65", "65", "65", "65"]
    test_nums = [[19, 36, 55, 7]] * 5
    
    # Test format rewards
    print("\n1. Format Reward Tests:")
    print("   (형식이 올바른지 검사: <think></think><answer></answer>)")
    print("-" * 80)
    format_rewards = format_reward_func(
        completions=test_completions,
        target=test_targets,
        nums=test_nums
    )
    
    expected_format = [1.0, 1.0, 0.0, 0.0, 1.0]
    for i, (completion, reward, expected) in enumerate(zip(test_completions, format_rewards, expected_format)):
        status = "✓" if reward == expected else "✗"
        print(f"{status} Sample {i+1}: {reward:.1f} (expected {expected:.1f})")
        if reward != expected:
            print(f"  Preview: {completion[:100]}...")
    
    assert format_rewards == expected_format, f"Format rewards mismatch! Got {format_rewards}"
    print("\n✓ All format tests passed!")
    
    # Test equation rewards
    print("\n2. Equation Reward Tests:")
    print("   (수식이 수학적으로 올바르고 정답인지 검사)")
    print("-" * 80)
    equation_rewards = equation_reward_func(
        completions=test_completions,
        target=test_targets,
        nums=test_nums
    )
    
    expected_equation = [1.0, 1.0, 0.0, 0.0, 0.0]
    for i, (completion, reward, expected) in enumerate(zip(test_completions, equation_rewards, expected_equation)):
        status = "✓" if reward == expected else "✗"
        print(f"{status} Sample {i+1}: {reward:.1f} (expected {expected:.1f})")
        if reward != expected:
            print(f"  Preview: {completion[:100]}...")
    
    assert equation_rewards == expected_equation, f"Equation rewards mismatch! Got {equation_rewards}"
    print("\n✓ All equation tests passed!")
    
    # Test combined rewards
    print("\n3. Combined Reward Tests:")
    print("   (포맷 보상 + 수식 보상 = 최종 보상)")
    print("-" * 80)
    combined_rewards = combined_reward_func(
        completions=test_completions,
        target=test_targets,
        nums=test_nums,
        format_weight=1.0,
        equation_weight=1.0
    )
    
    expected_combined = [2.0, 2.0, 0.0, 0.0, 1.0]
    for i, (reward, expected) in enumerate(zip(combined_rewards, expected_combined)):
        status = "✓" if reward == expected else "✗"
        print(f"{status} Sample {i+1}: {reward:.1f} (expected {expected:.1f})")
    
    assert combined_rewards == expected_combined, f"Combined rewards mismatch! Got {combined_rewards}"
    print("\n✓ All combined tests passed!")
    
    print("\n" + "=" * 80)
    print("✓ All reward function tests passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_reward_functions()
