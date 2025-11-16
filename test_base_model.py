import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

def test_base_model():
    """
    Loads the base Qwen/Qwen3 model and generates a response to a specific prompt.
    Always saves output to base_model_output.txt
    """
    model_name = "Qwen/Qwen3-4B"
    output_file = "base_model_output.txt"
    
    # 파일 열기 (콘솔과 파일 동시 출력)
    with open(output_file, 'w', encoding='utf-8') as f:
        def log(msg):
            """콘솔과 파일에 동시 출력"""
            print(msg)
            f.write(msg + "\n")
        
        log(f"{'='*80}")
        log(f"Base Model Test - {datetime.now().isoformat()}")
        log(f"{'='*80}\n")
        log(f"Loading model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            log("Model and tokenizer loaded successfully.\n")
        except Exception as e:
            log(f"Error loading model: {e}")
            return

        messages = [
            {"role": "system", "content": "Respond in the following format: <think> ... </think> <answer> ... </answer>"},
            {"role": "user", "content": "Create an equation using only the numbers [66, 34, 56, 3] that equals 27. Using the numbers [66, 34, 56, 3], create an equation that equals 27. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        log("--- PROMPT ---")
        log(text)
        log("")

        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        log("Generating response...\n")
        try:
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            log("--- MODEL RESPONSE ---")
            log(response)
            log(f"\n{'='*80}")
            log(f"Output saved to: {output_file}")
            log(f"{'='*80}")
        except Exception as e:
            log(f"Error during generation: {e}")

if __name__ == "__main__":
    test_base_model()
