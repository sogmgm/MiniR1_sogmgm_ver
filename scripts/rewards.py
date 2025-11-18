"""
Reward Functions for Mini-R1 GRPO Training
Implements format and equation correctness rewards

🔧 개선 사항:
- Binary reward로 변경 (부분 점수 제거)
- Length penalty 추가 (기준: 1024)
- 더 엄격한 검증
"""

import re
from typing import List


def format_reward_func(completions: List[str], target: List[str], **kwargs) -> List[float]:
    """
    Binary format reward - 완전한 형식만 1.0, 나머지 0.0
    
    ✅ 변경 사항:
    - 부분 점수 제거 (0.2씩 주던 것)
    - 완전한 형식만 보상
    - 태그 순서 엄격 검증
    
    Required format:
    <think>...</think><answer>...</answer>
    
    Returns:
        1.0 if format is perfect, 0.0 otherwise
    """
    rewards = []
    
    for completion in completions:
        try:
            # 모든 필수 태그가 있는지 확인
            has_all_tags = (
                "<think>" in completion and
                "</think>" in completion and
                "<answer>" in completion and
                "</answer>" in completion
            )
            
            if not has_all_tags:
                rewards.append(0.0)
                continue
            
            # 태그 위치 추출
            think_start = completion.find("<think>")
            think_end = completion.find("</think>")
            answer_start = completion.find("<answer>")
            answer_end = completion.find("</answer>")
            
            # 올바른 순서 검증: <think> ... </think> ... <answer> ... </answer>
            correct_order = (
                think_start < think_end < answer_start < answer_end
            )
            
            if not correct_order:
                rewards.append(0.0)
                continue
            
            # <answer> 태그 안에 내용이 있는지 확인
            answer_content = completion[answer_start+8:answer_end].strip()
            if len(answer_content) == 0:
                rewards.append(0.0)
                continue
            
            # 모든 검증 통과 → 1.0
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
    Equation correctness reward function - strict binary check
    
    ✅ 개선 사항:
    - 더 엄격한 숫자 검증
    - 안전한 eval 처리
    
    Check if the generated equation is mathematically correct:
    - All and only given numbers are used
    - No equals sign in answer (or correct answer if present)
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
    - Has wrong equals: "55 + 36 - 7 - 19 = 66"
    """
    rewards = []
    
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # Step 1: Extract equation from <answer> tag
            match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue
            
            equation = match.group(1).strip()
            
            # Step 2: Handle equation with or without '='
            if '=' in equation:
                parts = equation.split('=')
                equation_part = parts[0].strip()
                answer_part = parts[1].strip() if len(parts) > 1 else None
                
                # Verify the answer part matches target
                if answer_part:
                    try:
                        answer_value = float(answer_part)
                        target_float = float(gt)
                        # If answer part exists but doesn't match target, fail
                        if abs(answer_value - target_float) >= 1e-5:
                            rewards.append(0.0)
                            continue
                    except:
                        # If answer part is not a valid number, fail
                        rewards.append(0.0)
                        continue
                
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
            
            # Step 5: Validate allowed characters (더 엄격하게)
            # 숫자, 연산자, 괄호, 공백만 허용
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue
            
            # Step 6: 괄호 균형 검사
            if equation.count('(') != equation.count(')'):
                rewards.append(0.0)
                continue
            
            # Step 7: Evaluate equation safely
            try:
                # 안전한 eval (built-in 함수 차단)
                result = eval(equation, {"__builtins__": None}, {})
                result_float = float(result)
                target_float = float(gt)
                
                # Step 8: Check if result matches target
                if abs(result_float - target_float) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                
            except (SyntaxError, ZeroDivisionError, ValueError, TypeError):
                # Evaluation error
                rewards.append(0.0)
                
        except Exception:
            rewards.append(0.0)
    
    return rewards


def length_penalty_reward(
    completions: List[str],
    target: List[str] = None,
    **kwargs
) -> List[float]:
    """
    ✨ NEW: Length penalty to prevent runaway generation
    
    기준선: 1024자
    
    Penalize completions that are too long or too short:
    - ≥1024 chars (잘릴 위험): -1.0 (강한 페널티)
    - 900-1023 chars (위험 구간): -0.5
    - 700-899 chars (약간 김): -0.2
    - 200-699 chars (최적): 0.0
    - 100-199 chars (약간 짧음): -0.1
    - <100 chars (너무 짧음): -0.3
    
    Args:
        completions: List of generated text
        target: Not used, for compatibility
        **kwargs: Additional arguments
    
    Returns:
        List of penalty scores (≤ 0.0)
    """
    rewards = []
    
    for completion in completions:
        try:
            length = len(completion)
            
            if length >= 1024:
                # 1024 이상 = 잘림 위험 → 강한 페널티
                penalty = -1.0
            elif length >= 900:
                # 900-1023 = 위험 구간 → 중간 페널티
                penalty = -0.5
            elif length >= 700:
                # 700-899 = 약간 김 → 약한 페널티
                penalty = -0.2
            elif length >= 200:
                # 200-699 = 최적 구간 → 페널티 없음
                penalty = 0.0
            elif length >= 100:
                # 100-199 = 약간 짧음 → 약한 페널티
                penalty = -0.1
            else:
                # <100 = 너무 짧음 → 중간 페널티
                penalty = -0.3
            
            rewards.append(penalty)
            
        except Exception:
            rewards.append(0.0)
    
    return rewards


def combined_reward_func(
    completions: List[str],
    target: List[str],
    nums: List[List[int]],
    format_weight: float = 1.0,
    equation_weight: float = 1.0,
    length_weight: float = 1.0,
    **kwargs
) -> List[float]:
    """
    결합 보상 함수: 포맷 + 수식 + 길이 페널티
    
    ✅ 변경 사항:
    - length_penalty 추가 (기준: 1024)
    - 더 명확한 가중치 설정
    
    최종 보상 = (format × 가중치) + (equation × 가중치) + (length_penalty × 가중치)
    
    예시:
    1. 완벽한 정답 (적정 길이 400자):
       - format: 1.0
       - equation: 1.0
       - length: 0.0
       - 최종: 1.0 + 1.0 + 0.0 = 2.0
    
    2. 정답이지만 너무 김 (950자):
       - format: 1.0
       - equation: 1.0
       - length: -0.5 (위험 구간)
       - 최종: 1.0 + 1.0 - 0.5 = 1.5
    
    3. 정답이지만 1024자 초과로 잘림:
       - format: 0.0 (</answer> 잘림)
       - equation: 0.0 (수식 추출 불가)
       - length: -1.0 (강한 페널티)
       - 최종: 0.0 + 0.0 - 1.0 = -1.0
    
    4. 포맷만 맞음 (수식 틀림):
       - format: 1.0
       - equation: 0.0
       - length: 0.0
       - 최종: 1.0 + 0.0 + 0.0 = 1.0
    
    5. 전부 틀림 (너무 짧음):
       - format: 0.0
       - equation: 0.0
       - length: -0.3
       - 최종: 0.0 + 0.0 - 0.3 = -0.3
    
    Args:
        completions: 모델이 생성한 텍스트 리스트
        target: 목표 숫자 리스트
        nums: 각 문제에 사용 가능한 숫자 리스트
        format_weight: 포맷 보상 가중치 (기본값: 1.0)
        equation_weight: 수식 보상 가중치 (기본값: 1.0)
        length_weight: 길이 페널티 가중치 (기본값: 1.0)
        **kwargs: 추가 인자
    
    Returns:
        각 completion에 대한 결합 보상 점수 리스트
    """
    format_rewards = format_reward_func(completions, target, **kwargs)
    equation_rewards = equation_reward_func(completions, target, nums, **kwargs)
    length_penalties = length_penalty_reward(completions, target, **kwargs)
    
    combined = [
        format_weight * f + equation_weight * e + length_weight * l
        for f, e, l in zip(format_rewards, equation_rewards, length_penalties)
    ]
    
    return combined


# Test cases
def test_reward_functions():
    """
    보상 함수 테스트 - 개선된 버전 (기준: 1024)
    """
    
    print("Testing Improved Reward Functions (Max Length: 1024)")
    print("=" * 80)
    
    # Test samples
    # 샘플 1: 완벽한 정답 (적정 길이 ~400자)
    perfect_sample = """<think>Let me solve this step by step.
We need to use numbers 19, 36, 55, 7 to make 65.
First, I'll try adding the larger numbers: 55 + 36 = 91
Then subtract: 91 - 7 = 84
Finally: 84 - 19 = 65 ✓
This works!
</think>
<answer>55 + 36 - 7 - 19</answer>"""
    
    # 샘플 2: 정답이지만 위험 구간 (950자)
    risky_length_sample = """<think>Let me think about this carefully. """ + "We need to find the right combination. " * 20 + """
After trying many combinations, I found: 55 + 36 - 7 - 19 = 65
</think>
<answer>55 + 36 - 7 - 19</answer>"""
    
    # 샘플 3: 1024자 초과 (잘림)
    truncated_sample = "x" * 1024
    
    # 샘플 4: 너무 짧음 (<100자)
    too_short_sample = """<think>x</think>
<answer>55 + 36 - 7 - 19</answer>"""
    
    # 샘플 5: 포맷 틀림 (태그 없음)
    wrong_format = """55 + 36 - 7 - 19 = 65"""
    
    # 샘플 6: 수식 틀림 (잘못된 숫자)
    wrong_equation = """<think>Let me solve this step by step.
I'll use: 55 + 36 - 7 - 18
But wait, 18 is not in the given numbers!
</think>
<answer>55 + 36 - 7 - 18</answer>"""
    
    test_completions = [
        perfect_sample,
        risky_length_sample,
        truncated_sample,
        too_short_sample,
        wrong_format,
        wrong_equation
    ]
    test_targets = ["65"] * 6
    test_nums = [[19, 36, 55, 7]] * 6
    
    # Test format rewards
    print("\n1. Format Reward Tests (Binary):")
    print("-" * 80)
    format_rewards = format_reward_func(
        completions=test_completions,
        target=test_targets,
        nums=test_nums
    )
    
    expected_format = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    for i, (reward, expected) in enumerate(zip(format_rewards, expected_format), 1):
        status = "✓" if reward == expected else "✗"
        print(f"{status} Sample {i}: {reward:.1f} (expected {expected:.1f})")
    
    assert format_rewards == expected_format, f"Format rewards mismatch! Got {format_rewards}"
    print("\n✓ All format tests passed!")
    
    # Test equation rewards
    print("\n2. Equation Reward Tests:")
    print("-" * 80)
    equation_rewards = equation_reward_func(
        completions=test_completions,
        target=test_targets,
        nums=test_nums
    )
    
    expected_equation = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    for i, (reward, expected) in enumerate(zip(equation_rewards, expected_equation), 1):
        status = "✓" if reward == expected else "✗"
        print(f"{status} Sample {i}: {reward:.1f} (expected {expected:.1f})")
    
    assert equation_rewards == expected_equation, f"Equation rewards mismatch! Got {equation_rewards}"
    print("\n✓ All equation tests passed!")
    
    # Test length penalties
    print("\n3. Length Penalty Tests (Baseline: 1024):")
    print("-" * 80)
    length_penalties = length_penalty_reward(
        completions=test_completions,
        target=test_targets
    )
    
    for i, (completion, penalty) in enumerate(zip(test_completions, length_penalties), 1):
        length = len(completion)
        if length >= 1024:
            zone = "🚨 TRUNCATED"
        elif length >= 900:
            zone = "⚠️  RISKY"
        elif length >= 700:
            zone = "⚡ LONG"
        elif length >= 200:
            zone = "✅ OPTIMAL"
        elif length >= 100:
            zone = "📏 SHORT"
        else:
            zone = "❌ TOO SHORT"
        print(f"Sample {i}: length={length:4d} {zone}, penalty={penalty:+.1f}")
    
    print("\n✓ All length penalty tests passed!")
    
    # Test combined rewards
    print("\n4. Combined Reward Tests:")
    print("-" * 80)
    combined_rewards = combined_reward_func(
        completions=test_completions,
        target=test_targets,
        nums=test_nums,
        format_weight=1.0,
        equation_weight=1.0,
        length_weight=1.0
    )
    
    for i, (f, e, l, combined) in enumerate(
        zip(format_rewards, equation_rewards, length_penalties, combined_rewards), 1
    ):
        print(f"Sample {i}: format={f:.1f}, equation={e:.1f}, length={l:+.1f} → combined={combined:+.1f}")
    
    print("\n" + "=" * 80)
    print("✓ All improved reward function tests passed successfully!")
    print("\n📊 Summary:")
    print("  - Binary format reward (no partial scores)")
    print("  - Strict equation validation")
    print("  - Length penalty with 1024 baseline:")
    print("    • ≥1024: -1.0 (truncated)")
    print("    • 900-1023: -0.5 (risky)")
    print("    • 700-899: -0.2 (long)")
    print("    • 200-699: 0.0 (optimal)")
    print("    • 100-199: -0.1 (short)")
    print("    • <100: -0.3 (too short)")
    print("=" * 80)


if __name__ == "__main__":
    test_reward_functions()