"""
Test script to verify reward functions work correctly
"""

from rewards import format_reward_func, equation_reward_func, combined_reward_func

# Test cases
test_cases = [
    {
        "name": "Perfect answer",
        "completion": "Let me solve this step by step.\n1. I need to make 88 using [95, 21, 3]\n2. 95 - 21 / 3 = 95 - 7 = 88\n</think>\n<answer> 95 - 21 / 3 </answer>",
        "target": "88",
        "nums": [95, 21, 3],
        "expected_format": 1.0,
        "expected_equation": 1.0
    },
    {
        "name": "Correct format, wrong equation",
        "completion": "Let me try this.\n</think>\n<answer> 95 + 21 - 3 </answer>",
        "target": "88",
        "nums": [95, 21, 3],
        "expected_format": 1.0,
        "expected_equation": 0.0  # 95+21-3 = 113, not 88
    },
    {
        "name": "No think tag closure",
        "completion": "Let me think about this...\n<answer> 95 - 21 / 3 </answer>",
        "target": "88",
        "nums": [95, 21, 3],
        "expected_format": 0.0,
        "expected_equation": 0.0
    },
    {
        "name": "No answer tags",
        "completion": "Let me solve this.\n</think>\nThe answer is 95 - 21 / 3",
        "target": "88",
        "nums": [95, 21, 3],
        "expected_format": 0.0,
        "expected_equation": 0.0
    },
    {
        "name": "Wrong numbers used",
        "completion": "Here's my solution.\n</think>\n<answer> 90 - 2 </answer>",
        "target": "88",
        "nums": [95, 21, 3],
        "expected_format": 1.0,
        "expected_equation": 0.0  # Wrong numbers
    },
    {
        "name": "Correct with parentheses",
        "completion": "Step by step calculation.\n</think>\n<answer> (95 - 21) / 3 + 63 </answer>",
        "target": "88",
        "nums": [95, 21, 3, 63],
        "expected_format": 1.0,
        "expected_equation": 0.0  # Would need to check if this equals 88
    }
]

def run_tests():
    print("=" * 80)
    print("Testing Reward Functions")
    print("=" * 80)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        
        # Prepare inputs
        completions = [test_case['completion']]
        targets = [test_case['target']]
        nums = [test_case['nums']]
        
        # Test format reward
        format_rewards = format_reward_func(completions, targets)
        format_score = format_rewards[0]
        
        # Test equation reward
        equation_rewards = equation_reward_func(completions, targets, nums)
        equation_score = equation_rewards[0]
        
        # Test combined reward
        combined_rewards = combined_reward_func(completions, targets, nums)
        combined_score = combined_rewards[0]
        
        # Check results
        format_pass = abs(format_score - test_case['expected_format']) < 0.01
        equation_pass = abs(equation_score - test_case['expected_equation']) < 0.01
        
        print(f"Completion (first 100 chars): {test_case['completion'][:100]}...")
        print(f"Format Reward: {format_score:.2f} (expected {test_case['expected_format']:.2f}) {'✓' if format_pass else '✗'}")
        print(f"Equation Reward: {equation_score:.2f} (expected {test_case['expected_equation']:.2f}) {'✓' if equation_pass else '✗'}")
        print(f"Combined Reward: {combined_score:.2f}")
        
        if not (format_pass and equation_pass):
            all_passed = False
            print("❌ FAILED")
        else:
            print("✅ PASSED")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    run_tests()
