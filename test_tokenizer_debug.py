#!/usr/bin/env python3
"""
🔍 Tokenizer Debug Script
Qwen2.5-3B-Instruct 토크나이저 숫자 인코딩 테스트
"""

import torch
from transformers import AutoTokenizer
import json

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(f"✅ Tokenizer loaded")
print(f"📊 Vocab size: {len(tokenizer)}")
print(f"🔢 PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"🔢 EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# ============================================================================
# 1️⃣ 실제 프롬프트 토크나이징 테스트
# ============================================================================
print("\n" + "="*80)
print("1️⃣  ACTUAL PROMPT TOKENIZATION TEST")
print("="*80)

test_prompt = """<|im_start|>system
You are a helpful assistant. You first think about the reasoning process in <think></think> tags and then provide the answer in <answer></answer> tags.<|im_end|>
<|im_start|>user
Using the numbers [75, 25, 3, 1, 7, 10], create an equation that equals 111. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Think step by step in <think> tags, then provide your final equation in <answer> tags.<|im_end|>
<|im_start|>assistant"""

print("\n📝 Original Prompt:")
print(test_prompt)

# 토크나이징
tokens = tokenizer.encode(test_prompt)
print(f"\n🔢 Token IDs ({len(tokens)} tokens):")
print(tokens[:50], "...")

# 디코딩
decoded = tokenizer.decode(tokens)
print("\n🔄 Decoded back:")
print(decoded[:300], "...")

# 일치 여부
if test_prompt == decoded:
    print("\n✅ PERFECT MATCH!")
else:
    print("\n❌ MISMATCH DETECTED!")

# ============================================================================
# 2️⃣ 숫자 토크나이징 상세 테스트
# ============================================================================
print("\n" + "="*80)
print("2️⃣  NUMBER TOKENIZATION DETAILED TEST")
print("="*80)

numbers_test = [
    "[75, 25, 3, 1, 7, 10]",
    "75", "25", "3", "1", "7", "10", "111",
    "75 + 25 + 3 + 1 + 7 = 111",
    "(75 + 25) * 3",
    "66", "99", "80",
]

for text in numbers_test:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    match = "✅" if text == decoded else "❌ MISMATCH!"
    
    print(f"\nInput:   '{text}'")
    print(f"Tokens:  {tokens}")
    print(f"Decoded: '{decoded}'")
    print(f"Status:  {match}")
    
    if len(tokens) > 1 and len(tokens) <= 10:
        print(f"  Token breakdown:")
        for i, tid in enumerate(tokens):
            decoded_token = tokenizer.decode([tid])
            print(f"    [{i}] {tid} -> '{decoded_token}'")

# ============================================================================
# 3️⃣ 전각 문자 및 특수 유니코드 테스트
# ============================================================================
print("\n" + "="*80)
print("3️⃣  FULL-WIDTH AND UNICODE CHARACTER TEST")
print("="*80)

weird_outputs = [
    ("7", "regular"),
    ("７", "full-width"),
    ("1", "regular"),
    ("１", "full-width"),
    ("8", "regular"),
    ("８", "full-width"),
]

for text, desc in weird_outputs:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    
    print(f"\n{desc}: '{text}'")
    print(f"  Unicode: U+{ord(text):04X}")
    print(f"  Bytes:   {text.encode('utf-8').hex()}")
    print(f"  Tokens:  {tokens}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Match:   {'✅' if text == decoded else '❌'}")

# ============================================================================
# 4️⃣ Vocabulary 내 숫자 존재 여부 확인
# ============================================================================
print("\n" + "="*80)
print("4️⃣  VOCABULARY CHECK FOR NUMBERS (0-120)")
print("="*80)

single_token_nums = []
multi_token_nums = []

for num in range(0, 121):
    num_str = str(num)
    token_ids = tokenizer.encode(num_str, add_special_tokens=False)
    
    if len(token_ids) == 1:
        single_token_nums.append((num, token_ids[0]))
    else:
        multi_token_nums.append((num, token_ids))

print(f"\n✅ Single-token numbers ({len(single_token_nums)}):")
for num, tid in single_token_nums[:20]:
    print(f"  {num:3d} -> Token ID: {tid}")
if len(single_token_nums) > 20:
    print(f"  ... and {len(single_token_nums) - 20} more")

print(f"\n⚠️  Multi-token numbers ({len(multi_token_nums)}):")
for num, tids in multi_token_nums[:10]:
    decoded_parts = [tokenizer.decode([tid]) for tid in tids]
    print(f"  {num:3d} -> {tids} = {decoded_parts}")
if len(multi_token_nums) > 10:
    print(f"  ... and {len(multi_token_nums) - 10} more")

# ============================================================================
# 5️⃣ 특수 문자 및 연산자 토크나이징
# ============================================================================
print("\n" + "="*80)
print("5️⃣  SPECIAL CHARACTERS AND OPERATORS")
print("="*80)

special_chars = ['[', ']', ',', ' ', '+', '-', '*', '/', '=', '(', ')', '<', '>']

for char in special_chars:
    tokens = tokenizer.encode(char, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    match = '✅' if char == decoded else '❌'
    print(f"'{char}' -> {tokens} -> '{decoded}' {match}")

# ============================================================================
# 6️⃣ 실제 학습 데이터 샘플 테스트
# ============================================================================
print("\n" + "="*80)
print("6️⃣  TRAINING DATA SAMPLE TEST")
print("="*80)

try:
    with open('.cache/datasets/train_countdown_r1.json', 'r') as f:
        train_data = json.load(f)
    
    print(f"Total samples: {len(train_data)}")
    
    for i, sample in enumerate(train_data[:2]):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}:")
        print(f"{'='*80}")
        
        prompt = sample['prompt']
        print(f"Prompt (first 200 chars):\n{prompt[:200]}...")
        
        tokens = tokenizer.encode(prompt)
        print(f"\nToken count: {len(tokens)}")
        
        decoded = tokenizer.decode(tokens)
        
        if prompt == decoded:
            print("✅ Encode/Decode: PERFECT")
        else:
            print("❌ Encode/Decode: MISMATCH")
            
except FileNotFoundError:
    print("⚠️  Training data file not found")

# ============================================================================
# 7️⃣ 모델 생성 시뮬레이션
# ============================================================================
print("\n" + "="*80)
print("7️⃣  SIMULATING MODEL GENERATION")
print("="*80)

model_outputs = [
    "Given numbers: 85, 40, -15",
    "Target: 99",
    "(7 * (3 *)) + (5 *) + (-1 *)",
]

for output in model_outputs:
    print(f"\nModel output: '{output}'")
    tokens = tokenizer.encode(output, add_special_tokens=False)
    print(f"Tokens: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")
    print(f"Match: {'✅' if output == decoded else '❌'}")

print("\n" + "="*80)
print("🎯 TEST COMPLETE")
print("="*80)
