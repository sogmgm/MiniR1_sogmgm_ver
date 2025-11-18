"""
Reward Functions for Mini-R1 GRPO Training
Implements format and equation correctness rewards

🔧 개선 사항:
- 부분 점수 + 조기 완성 보너스
- Efficiency bonus 추가 (별도 함수)
- 더 명확한 보상 체계
"""

import re
from typing import List


def format_reward_func(completions: List[str], target: List[str], **kwargs) -> List[float]:
    """
    부분 점수 + 조기 완성 보너스
    
    기본 점수: 0.0 ~ 1.0 (태그별 0.25점)
    보너스: 0.0 ~ 0.5 (효율성)
    최대: 1.5점
    최소: 0.0점
    """
    rewards = []
    
    for completion in completions:
        try:
            score = 0.0
            length = len(completion)
            
            # === 1단계: 기본 태그 점수 (부분 점수) ===
            has_think_start = "<think>" in completion
            has_think_end = "</think>" in completion
            has_answer_start = "<answer>" in completion
            has_answer_end = "</answer>" in completion
            
            if has_think_start:
                score += 0.25
            if has_think_end:
                score += 0.25
            if has_answer_start:
                score += 0.25
            if has_answer_end:
                score += 0.25
            
            # === 2단계: 순서 검증 (필수) ===
            if has_think_start and has_think_end and has_answer_start and has_answer_end:
                think_start = completion.find("<think>")
                think_end = completion.find("</think>")
                answer_start = completion.find("<answer>")
                answer_end = completion.find("</answer>")
                
                correct_order = think_start < think_end < answer_start < answer_end
                
                if not correct_order:
                    score = 0.0  # 순서 틀리면 전체 무효
                else:
                    # === 3단계: 효율성 보너스 ===
                    
                    # 보너스 1: Think section 길이 (<500자)
                    think_length = think_end - think_start
                    if think_length < 300:
                        score += 0.15
                    elif think_length < 500:
                        score += 0.10
                    elif think_length < 700:
                        score += 0.05
                    
                    # 보너스 2: Answer 위치 (Think 직후)
                    gap = answer_start - think_end
                    if gap < 50:
                        score += 0.15
                    elif gap < 100:
                        score += 0.10
                    elif gap < 150:
                        score += 0.05
                    
                    # 보너스 3: 전체 길이 (<600자)
                    if answer_end < 500:
                        score += 0.20
                    elif answer_end < 600:
                        score += 0.15
                    elif answer_end < 700:
                        score += 0.10
                    elif answer_end < 800:
                        score += 0.05
                    
                    # 보너스 4: Answer 내용 검증
                    answer_content = completion[answer_start+8:answer_end].strip()
                    if len(answer_content) == 0:
                        score -= 0.5  # 빈 답변 큰 페널티
            
            rewards.append(max(0.0, score))
                
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
    수식 정확도 리워드 - Binary (0 or 1)
    
    Check if the generated equation is mathematically correct:
    - All and only given numbers are used
    - No equals sign in answer (or correct answer if present)
    - Result equals target
    
    Returns:
        1.0 if equation is correct, 0.0 otherwise
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
                        if abs(answer_value - target_float) >= 1e-5:
                            rewards.append(0.0)
                            continue
                    except:
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
            
            # Step 5: Validate allowed characters
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
                result = eval(equation, {"__builtins__": None}, {})
                result_float = float(result)
                target_float = float(gt)
                
                # Step 8: Check if result matches target
                if abs(result_float - target_float) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                
            except (SyntaxError, ZeroDivisionError, ValueError, TypeError):
                rewards.append(0.0)
                
        except Exception:
            rewards.append(0.0)
    
    return rewards


def efficiency_bonus_func(
    completions: List[str],
    target: List[str] = None,
    **kwargs
) -> List[float]:
    """
    ✨ 효율성 보너스 - 길이 기반 보상/페널티
    
    max_completion_length: 1024 기준
    
    보상 체계:
    - 400자 이하: +0.3 (매우 효율적)
    - 400-600자: +0.2 (효율적)
    - 600-800자: +0.1 (적정)
    - 800-900자: 0.0 (보통)
    - 900-1000자: -0.1 (긴 편)
    - 1000자 이상: -0.2 (너무 김, 잘릴 위험)
    
    Args:
        completions: 생성된 텍스트 리스트
        target: 사용 안 함 (호환성)
        **kwargs: 추가 인자
    
    Returns:
        효율성 보너스/페널티 점수 리스트 (-0.2 ~ +0.3)
    """
    rewards = []
    
    for completion in completions:
        try:
            length = len(completion)
            
            if length <= 400:
                bonus = 0.3
            elif length <= 600:
                bonus = 0.2
            elif length <= 800:
                bonus = 0.1
            elif length <= 900:
                bonus = 0.0
            elif length <= 1000:
                bonus = -0.1
            else:
                bonus = -0.2
            
            rewards.append(bonus)
            
        except Exception:
            rewards.append(0.0)
    
    return rewards


# Test cases
def test_reward_functions():
    """
    보상 함수 테스트 (3개 함수)
    """
    
    print("Testing Reward Functions (Format + Equation + Efficiency)")
    print("=" * 80)
    
    # Test samples
    # 샘플 1: 완벽한 정답 (짧고 정확 ~250자)
    perfect_short = """<think>
55 + 36 = 91
91 - 7 = 84
84 - 19 = 65 ✓
</think>
<answer>55 + 36 - 7 - 19</answer>"""
    
    # 샘플 2: 정답이지만 중간 길이 (~600자)
    correct_medium = """<think>Let me solve this step by step.
We need to use numbers 19, 36, 55, 7 to make 65.
First, I'll try adding the larger numbers: 55 + 36 = 91
Then subtract: 91 - 7 = 84
Finally: 84 - 19 = 65 ✓
This works!
</think>
<answer>55 + 36 - 7 - 19</answer>""" + " " * 400
    
    # 샘플 3: 정답이지만 위험 구간 (950자)
    risky_length_sample = """<think>Let me think about this carefully. """ + "We need to find the right combination. " * 20 + """
After trying many combinations, I found: 55 + 36 - 7 - 19 = 65
</think>
<answer>55 + 36 - 7 - 19</answer>"""
    
    # 샘플 4: 1024자 초과 (잘림)
    truncated_sample = "x" * 1024
    
    # 샘플 5: 너무 짧음 (<100자)
    too_short_sample = """<think>x</think>
<answer>55 + 36 - 7 - 19</answer>"""
    
    # 샘플 6: 포맷 틀림 (태그 없음)
    wrong_format = """55 + 36 - 7 - 19 = 65"""
    
    # 샘플 7: 수식 틀림 (잘못된 숫자)
    wrong_equation = """<think>Let me solve this step by step.
I'll use: 55 + 36 - 7 - 18
But wait, 18 is not in the given numbers!
</think>
<answer>55 + 36 - 7 - 18</answer>"""
    
    test_completions = [
        perfect_short,
        correct_medium,
        risky_length_sample,
        truncated_sample,
        too_short_sample,
        wrong_format,
        wrong_equation
    ]
    test_targets = ["65"] * 7
    test_nums = [[19, 36, 55, 7]] * 7
    
    # Test format rewards
    print("\n1. Format Reward Tests (0.0 ~ 1.5):")
    print("-" * 80)
    format_rewards = format_reward_func(
        completions=test_completions,
        target=test_targets
    )
    
    for i, (completion, reward) in enumerate(zip(test_completions, format_rewards), 1):
        length = len(completion)
        print(f"Sample {i} (len={length:4d}): {reward:.2f}")
    
    # Test equation rewards
    print("\n2. Equation Reward Tests (0 or 1):")
    print("-" * 80)
    equation_rewards = equation_reward_func(
        completions=test_completions,
        target=test_targets,
        nums=test_nums
    )
    
    for i, reward in enumerate(equation_rewards, 1):
        print(f"Sample {i}: {reward:.1f}")
    
    # Test efficiency bonus
    print("\n3. Efficiency Bonus Tests (-0.2 ~ +0.3):")
    print("-" * 80)
    efficiency_rewards = efficiency_bonus_func(
        completions=test_completions,
        target=test_targets
    )
    
    for i, (completion, bonus) in enumerate(zip(test_completions, efficiency_rewards), 1):
        length = len(completion)
        if length <= 400:
            zone = "⭐⭐⭐ EXCELLENT"
        elif length <= 600:
            zone = "⭐⭐ GOOD"
        elif length <= 800:
            zone = "⭐ OK"
        elif length <= 900:
            zone = "✓ NORMAL"
        elif length <= 1000:
            zone = "⚠ LONG"
        else:
            zone = "🚨 TOO LONG"
        print(f"Sample {i} (len={length:4d}): {bonus:+.1f} {zone}")
    
    # Test combined (with weights from config)
    print("\n4. Combined Rewards (format×1.0 + equation×1.0 + efficiency×0.5):")
    print("-" * 80)
    
    format_weight = 1.0
    equation_weight = 1.0
    efficiency_weight = 0.5
    
    for i, (f, e, eff) in enumerate(zip(format_rewards, equation_rewards, efficiency_rewards), 1):
        combined = format_weight * f + equation_weight * e + efficiency_weight * eff
        print(f"Sample {i}: format={f:.2f} + equation={e:.1f} + efficiency={eff:+.1f}×0.5 = {combined:.2f}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("\n📊 Scoring Summary:")
    print("  Format Reward (0.0 ~ 1.5):")
    print("    - Basic tags: 0.25 each (total 1.0)")
    print("    - Bonuses: up to +0.5 (efficiency)")
    print("  Equation Reward (0 or 1):")
    print("    - Correct equation: 1.0")
    print("    - Wrong/missing: 0.0")
    print("  Efficiency Bonus (-0.2 ~ +0.3):")
    print("    - ≤400 chars: +0.3")
    print("    - 400-600: +0.2")
    print("    - 600-800: +0.1")
    print("    - 800-900: 0.0")
    print("    - 900-1000: -0.1")
    print("    - >1000: -0.2")
    print("\n  Total Range (with config weights):")
    print("    - Best: 1.5 + 1.0 + 0.15 = 2.65")
    print("    - Worst: 0.0 + 0.0 - 0.1 = -0.1")
    print("=" * 80)


if __name__ == "__main__":
    test_reward_functions()