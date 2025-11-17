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
    Format reward function - microR1 style (strict binary check)
    
    Check if the generated text follows the correct format:
    - Starts with <think>
    - Contains </think>
    - Contains <answer> ... </answer>
    - Proper order: <think> → </think> → <answer> → </answer>
    - Pattern appears exactly 2 times (system + response)
    
    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    rewards = []
    
    for completion in completions:
        try:
            # Step 1: Check if starts with <think>
            if not completion.strip().startswith("<think>"):
                rewards.append(0.0)
                continue
            
            # Step 2: Check for </think>
            if "</think>" not in completion:
                rewards.append(0.0)
                continue
            
            # Step 3: Check for <answer> ... </answer>
            answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if answer_match is None:
                rewards.append(0.0)
                continue
            
            # Step 4: Check order (<think> → </think> → <answer>)
            think_start = completion.find("<think>")
            think_end = completion.find("</think>")
            answer_start = completion.find("<answer>")
            answer_end = completion.find("</answer>")
            
            if not (think_start < think_end < answer_start < answer_end):
                rewards.append(0.0)
                continue
            
            # All conditions met - binary reward
            rewards.append(1.0)
                
        except Exception:
            rewards.append(0.0)
    
    return rewards


def equation_reward_func(
    completions: List[str],
    target: List[str],
    nums: List[List[int]],
    **kwargs
) -> List[float]:
    """
    Equation correctness reward function - microR1 style (strict binary check)
    
    Check if the generated equation is mathematically correct:
    - All and only given numbers are used
    - No equals sign in answer
    - Result equals target
    
    Returns:
        1.0 if equation is correct, 0.0 otherwise
    
    Example:
    ✅ Correct (1.0):
    - Numbers: [19, 36, 55, 7], Target: 65
    - Equation: "55 + 36 - 7 - 19"
    - Result: 55 + 36 - 7 - 19 = 65 ✓
    
    ❌ Wrong (0.0):
    - Missing number: "55 + 36 - 7" (19 not used)
    - Extra number: "55 + 36 - 7 - 18" (18 not given)
    - Wrong result: "55 + 36 + 7 - 19" = 79 ≠ 65
    - Has equals sign: "55 + 36 - 7 - 19 = 65"
    
    Args:
        completions: List of generated text
        target: List of target numbers
        nums: List of available numbers for each problem
        **kwargs: Additional arguments
    
    Returns:
        List of reward scores (1.0 or 0.0)
    """
    rewards = []
    
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # Step 1: Extract equation from <answer> tag
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue
            
            equation = match.group(1).strip()
            
            # Step 2: Handle equation with or without '='
            # If '=' exists, extract left (equation) and right (answer) parts
            # Example: "75 - (47 - 43) = 71" -> left="75 - (47 - 43)", right="71"
            if '=' in equation:
                parts = equation.split('=')
                equation_part = parts[0].strip()
                answer_part = parts[1].strip() if len(parts) > 1 else None
                
                # Verify the answer part matches target (optional check)
                if answer_part:
                    try:
                        answer_value = float(answer_part)
                        target_float = float(gt)
                        # If answer part exists but doesn't match target, fail
                        if abs(answer_value - target_float) >= 1e-5:
                            rewards.append(0.0)
                            continue
                    except:
                        # If answer part is not a valid number, ignore it
                        pass
                
                # Use only the equation part for further validation
                equation = equation_part
            
            # Step 3: Extract all numbers from equation
            used_numbers = []
            for n in re.findall(r'-?\d+', equation):
                num = int(n)
                used_numbers.append(abs(num))
            
            # Step 4: Check if all and only given numbers are used
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            
            # Step 5: Validate allowed characters
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue
            
            # Step 6: Evaluate equation
            try:
                result = eval(equation, {"__builtins__": None}, {})
                result_float = float(result)
                target_float = float(gt)
                
                # Step 7: Check if result matches target
                if abs(result_float - target_float) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                
            except Exception:
                # Evaluation error (division by zero, etc.)
                rewards.append(0.0)
                
        except Exception:
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
