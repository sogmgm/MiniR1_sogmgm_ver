"""
Mini-R1 GRPO Training Script
Single GPU training with memory optimizations for RunPod
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
import yaml

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config
from peft import LoraConfig, prepare_model_for_kbit_training

from rewards import format_reward_func, equation_reward_func


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def save_completion_samples(
    step: int,
    completions: list,
    prompts: list,
    targets: list,
    nums: list,
    rewards: list,
    output_dir: str = "completion_samples"
):
    """
    학습 중 생성된 샘플을 파일로 저장
    
    Args:
        step: 현재 학습 스텝
        completions: 생성된 텍스트들
        prompts: 입력 프롬프트들
        targets: 정답 목표값들
        nums: 사용 가능한 숫자들
        rewards: 보상 점수들
        output_dir: 저장 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 전체 샘플 저장
    all_samples_file = output_path / f"step_{step:04d}_all.txt"
    with open(all_samples_file, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"Training Step: {step}\n")
        f.write(f"Total Samples: {len(completions)}\n")
        f.write(f"=" * 80 + "\n\n")
        
        for i, (completion, prompt, target, nums_list, reward) in enumerate(
            zip(completions, prompts, targets, nums, rewards)
        ):
            f.write(f"--- Sample {i+1} ---\n")
            f.write(f"Target: {target}\n")
            f.write(f"Numbers: {nums_list}\n")
            f.write(f"Reward: {reward:.2f}\n\n")
            f.write(f"Completion:\n{completion}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # 성공한 샘플만 따로 저장
    successful_samples = [
        (completion, target, nums_list, reward)
        for completion, target, nums_list, reward in zip(completions, targets, nums, rewards)
        if reward > 1.5  # 포맷(1.0) + 수식(1.0) = 2.0에 가까우면 성공
    ]
    
    if successful_samples:
        success_file = output_path / f"step_{step:04d}_success.txt"
        with open(success_file, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"Training Step: {step} - Successful Samples\n")
            f.write(f"Success Count: {len(successful_samples)}/{len(completions)}\n")
            f.write(f"Success Rate: {len(successful_samples)/len(completions)*100:.1f}%\n")
            f.write(f"=" * 80 + "\n\n")
            
            for i, (completion, target, nums_list, reward) in enumerate(successful_samples, 1):
                f.write(f"--- Success {i} ---\n")
                f.write(f"Target: {target}\n")
                f.write(f"Numbers: {nums_list}\n")
                f.write(f"Reward: {reward:.2f}\n\n")
                f.write(f"Completion:\n{completion}\n\n")
                f.write("-" * 80 + "\n\n")
    
    logger.info(f"Saved samples to {output_dir}/ (Success: {len(successful_samples)}/{len(completions)})")


def update_progress_md(step: int, metrics: dict, output_file: str = "PROGRESS.md"):
    """
    PROGRESS.md 파일에 학습 진행 상황 업데이트
    
    Args:
        step: 현재 학습 스텝
        metrics: 학습 메트릭 딕셔너리
        output_file: PROGRESS.md 파일 경로
    """
    progress_path = Path(output_file)
    
    if not progress_path.exists():
        return
    
    # 기존 내용 읽기
    with open(progress_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 학습 진행 섹션 찾기 또는 생성
    training_section_marker = "## 🔥 학습 진행 상황 (실시간 업데이트)"
    
    if training_section_marker not in content:
        # 섹션이 없으면 추가
        content += f"\n\n{training_section_marker}\n\n"
        content += f"| Step | Loss | Format Acc | Equation Acc | Success Rate | 시간 |\n"
        content += f"|------|------|------------|--------------|--------------|------|\n"
    
    # 새로운 메트릭 라인 추가
    format_acc = metrics.get('format_accuracy', 0.0)
    equation_acc = metrics.get('equation_accuracy', 0.0)
    success_rate = metrics.get('success_rate', 0.0)
    loss = metrics.get('loss', 0.0)
    timestamp = datetime.now().strftime("%H:%M")
    
    new_line = f"| {step} | {loss:.4f} | {format_acc:.1%} | {equation_acc:.1%} | {success_rate:.1%} | {timestamp} |\n"
    
    # 테이블에 라인 추가
    lines = content.split('\n')
    table_end_idx = -1
    for i, line in enumerate(lines):
        if training_section_marker in line:
            # 테이블 끝 찾기
            for j in range(i+1, len(lines)):
                if lines[j].startswith('|') and 'Step' not in lines[j] and '---' not in lines[j]:
                    table_end_idx = j
                elif lines[j].strip() == '' or not lines[j].startswith('|'):
                    break
    
    if table_end_idx > 0:
        lines.insert(table_end_idx + 1, new_line.strip())
    else:
        # 테이블 헤더 다음에 추가
        for i, line in enumerate(lines):
            if '|------|' in line:
                lines.insert(i + 1, new_line.strip())
                break
    
    # 파일 다시 쓰기
    with open(progress_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Updated PROGRESS.md with step {step} metrics")


class SampleSavingCallback(TrainerCallback):
    """
    학습 중 생성 샘플을 주기적으로 저장하는 콜백 (5 step마다)
    """
    def __init__(self, save_steps: int = 5, tokenizer=None, config=None, eval_dataset=None):
        self.save_steps = save_steps
        self.tokenizer = tokenizer
        self.config = config
        self.eval_dataset = eval_dataset
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        각 스텝 종료 시 호출되어 샘플 생성 및 저장 (1, 5, 10, 15, ...)
        """
        step = state.global_step
        
        # 1, 5, 10, 15, 20... 스텝에서 샘플 생성
        if step == 1 or (step % self.save_steps == 0):
            logger.info(f"📊 Step {step}: Generating sample...")
            
            # Trainer의 model과 tokenizer 사용
            model = kwargs.get('model', None)
            if model is not None and self.tokenizer is not None:
                self._generate_and_save_samples(model, step)
        
        return control
    

    def _generate_and_save_samples(self, model, step):
        """모델에서 샘플 생성 및 저장 (데이터셋에서 랜덤 샘플, dataset_prep.py 프롬프트와 동일)"""
        import random
        try:
            gen_start = time.time()  # 타이머 시작

            # ✅ 중요: 모델을 eval mode로 전환
            was_training = model.training
            model.eval()

            # eval_dataset에서 랜덤 샘플 선택
            if self.eval_dataset is not None and len(self.eval_dataset) > 0:
                sample = self.eval_dataset[random.randint(0, len(self.eval_dataset)-1)]
                numbers = sample.get('nums', []) if 'nums' in sample else sample.get('numbers', [])
                target = sample.get('target', None)
            else:
                # fallback: 기존 방식
                numbers = random.sample(range(1, 101), 6)
                target = random.randint(10, 999)

            # dataset_prep.py와 동일한 프롬프트 생성
            messages = [
                {
                    "role": "system",
                    "content": "Respond in the following format: <think> ... </think> <answer> ... </answer>"
                },
                {
                    "role": "user",
                    "content": f"Create an equation using only the numbers {numbers} that equals {target}. "
                               f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
                }
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 샘플 저장
            samples_dir = Path("logs/generation_samples")
            samples_dir.mkdir(parents=True, exist_ok=True)

            model_suffix = self.config.get('model', {}).get('model_suffix', '')
            if model_suffix:
                sample_file = samples_dir / f"step_{step:05d}_{model_suffix}.txt"
            else:
                sample_file = samples_dir / f"step_{step:05d}.txt"

            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(f"{'='*80}\n")
                f.write(f"Generation Sample - Step {step}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"{'='*80}\n\n")

                f.write(f"Target: {target}\n")
                f.write(f"Numbers: {numbers}\n\n")

                # 프롬프트 전체 출력
                f.write(f"{'─'*80}\n")
                f.write(f"PROMPT:\n")
                f.write(f"{'─'*80}\n")
                f.write(f"{prompt}\n\n")

                # 생성
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

                gen_sample_start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                gen_sample_time = time.time() - gen_sample_start

                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                completion = full_output[len(prompt):]

                f.write(f"{'─'*80}\n")
                f.write(f"GENERATED (in {gen_sample_time:.2f}s):\n")
                f.write(f"{'─'*80}\n")
                f.write(f"{completion}\n\n")

                # 보상 계산 (리스트로 전달해야 함)
                format_rewards = format_reward_func([completion], [target])
                equation_rewards = equation_reward_func([completion], [target], [numbers])
                format_reward = format_rewards[0]
                equation_reward = equation_rewards[0]
                total_reward = format_reward + equation_reward

                f.write(f"{'─'*80}\n")
                f.write(f"REWARDS:\n")
                f.write(f"{'─'*80}\n")
                f.write(f"Format:   {format_reward:.2f}\n")
                f.write(f"Equation: {equation_reward:.2f}\n")
                f.write(f"Total:    {total_reward:.2f}\n")
                f.write(f"Status:   {'✅ SUCCESS' if total_reward >= 1.8 else '❌ FAIL'}\n")
                f.write(f"{'='*80}\n")

            # ✅ 모델을 원래 상태로 복원
            if was_training:
                model.train()

            gen_time = time.time() - gen_start
            logger.info(f"✅ Step {step}: Sample saved (Reward: {total_reward:.2f}, Time: {gen_time:.2f}s)")

        except Exception as e:
            logger.error(f"❌ Failed to generate sample at step {step}: {e}")
            import traceback
            logger.error(traceback.format_exc())



def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_from_json(file_path: str) -> Dataset:
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


def save_training_metrics(metrics: dict, output_file: str):
    """Append training metrics to JSON file"""
    output_path = Path(output_file)
    
    # Load existing metrics if file exists
    if output_path.exists():
        with open(output_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    # Append new metrics
    all_metrics.append({
        "timestamp": datetime.now().isoformat(),
        **metrics
    })
    
    # Save back to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Mini-R1 with GRPO")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    seed = config.get('dataset', {}).get('shuffle_seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Load datasets
    dataset_dir = Path(config['dataset']['cache_dir'])
    train_file = dataset_dir / "train_countdown_r1.json"
    test_file = dataset_dir / "test_countdown_r1.json"
    
    logger.info(f"Loading training data from {train_file}")
    train_dataset = load_dataset_from_json(str(train_file))
    
    logger.info(f"Loading test data from {test_file}")
    test_dataset = load_dataset_from_json(str(test_file))
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Load tokenizer
    model_name = config['model']['name']
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ✅ LoRA on/off 설정
    use_peft = config['model'].get('use_peft', True)
    logger.info(f"{'='*80}")
    logger.info(f"TRAINING MODE: {'LoRA (PEFT)' if use_peft else 'Full Fine-tuning'}")
    logger.info(f"{'='*80}")
    
    # Configure model with QLoRA
    logger.info(f"Loading model {model_name}")
    
    # A100 80GB에서는 full precision 사용
    logger.info("Loading model with full precision (optimized for A100 80GB)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config['model'].get('attn_implementation', 'eager'),
    )
    
    # ✅ Vocab size 불일치 수정 (중요!)
    if len(tokenizer) != model.config.vocab_size:
        logger.warning(f"Vocab size mismatch: Tokenizer={len(tokenizer)}, Model={model.config.vocab_size}")
        logger.info("Resizing model embeddings to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"✓ Model embeddings resized to {len(tokenizer)}")
    
    # ✅ LoRA 설정 (use_peft=True일 때만)
    if use_peft:
        logger.info("🔧 Configuring LoRA...")
        peft_config_dict = config.get('peft', {})
        peft_config = LoraConfig(
            r=peft_config_dict.get('r', 16),
            lora_alpha=peft_config_dict.get('lora_alpha', 32),
            lora_dropout=peft_config_dict.get('lora_dropout', 0.05),
            target_modules=peft_config_dict.get('target_modules', None),
            task_type=peft_config_dict.get('task_type', "CAUSAL_LM"),
        )
        logger.info(f"✅ LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}")
    else:
        logger.info("⚠️  LoRA disabled - Full Fine-tuning mode")
        logger.info(f"   Total parameters: {model.num_parameters():,}")
        peft_config = None
    
    # Configure GRPO training arguments
    training_config = config['training']
    grpo_config = config['grpo']
    
    training_args = GRPOConfig(
        # Output
        output_dir=training_config['output_dir'],
        
        # Learning rate
        learning_rate=training_config['learning_rate'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_ratio=training_config.get('warmup_ratio', 0.05),
        
        # Training steps
        max_steps=training_config['max_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config.get('eval_steps', 50),
        
        # Batch size
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        
        # Optimization
        gradient_checkpointing=training_config['gradient_checkpointing'],
        gradient_checkpointing_kwargs=training_config.get('gradient_checkpointing_kwargs', {}),
        bf16=training_config.get('bf16', True),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        optim=training_config.get('optim', 'adamw_torch'),
        weight_decay=training_config.get('weight_decay', 0.01),
        
        # Logging
        logging_dir=training_config['logging_dir'],
        report_to=training_config.get('report_to', []),
        
        # Checkpointing
        save_total_limit=training_config.get('save_total_limit', 4),
        
        # GRPO specific
        max_prompt_length=grpo_config['max_prompt_length'],
        max_completion_length=grpo_config['max_completion_length'],
        num_generations=grpo_config['num_generations'],
        temperature=grpo_config.get('temperature', 0.7),
        top_p=grpo_config.get('top_p', 0.9),
        top_k=grpo_config.get('top_k', 50),
        beta=grpo_config['beta'],
    )
    
    logger.info("Training configuration:")
    logger.info(f"  Max steps: {training_args.max_steps}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  GRPO beta: {training_args.beta}")
    logger.info(f"  Num generations: {training_args.num_generations}")
    
    # Create trainer
    logger.info("Creating GRPO Trainer")
    
    # 타이밍 측정 시작
    trainer_creation_start = time.time()
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config if use_peft else None,  # ✅ LoRA off시 None 전달
        reward_funcs=[format_reward_func, equation_reward_func],
    )
    
    trainer_creation_time = time.time() - trainer_creation_start
    logger.info(f"⏱️  Trainer created in {trainer_creation_time:.2f}s")
    
    # 샘플 저장 콜백 추가 (5 step마다)
    sample_callback = SampleSavingCallback(
        save_steps=5,
        tokenizer=tokenizer,
        config=config,
        eval_dataset=test_dataset
    )
    trainer.add_callback(sample_callback)
    
    # ✅ 학습 시작 전 초기 샘플 생성 (step 0)
    logger.info("📊 Generating initial sample before training (step 0)...")
    sample_callback._generate_and_save_samples(model, step=0)
    
    # Check GPU memory
    if torch.cuda.is_available():
        logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"📊 GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"📊 GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Train
    logger.info("🚀 Starting training...")
    training_start_time = time.time()
    
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        training_total_time = time.time() - training_start_time
        logger.info(f"⏱️  Total training time: {training_total_time / 60:.2f} minutes")
        
        # Save final model
        logger.info("💾 Saving final model...")
        save_start = time.time()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        save_time = time.time() - save_start
        logger.info(f"✅ Model saved in {save_time:.2f}s")
        
        logger.info("✨ Training completed successfully!")
        
        # 최종 GPU 메모리 상태
        if torch.cuda.is_available():
            logger.info(f"📊 Final GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"📊 Final GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # Save final metrics
    metrics_file = Path(training_args.logging_dir) / "training_metrics.json"
    save_training_metrics(
        {
            "status": "completed",
            "final_step": training_args.max_steps,
            "model": model_name,
        },
        str(metrics_file)
    )
    
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
