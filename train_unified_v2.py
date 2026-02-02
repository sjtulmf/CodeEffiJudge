#!/usr/bin/env python3
"""
统一训练脚本 v2
================

支持多语言和多种数据格式的 LoRA 微调:

语言: python, cpp, java
格式: ab, abdiff, adiff, all

使用方法:
CUDA_VISIBLE_DEVICES=0 python train_unified_v2.py --lang python --format ab --epochs 3
CUDA_VISIBLE_DEVICES=1 python train_unified_v2.py --lang cpp --format abdiff --model_size 7b

数据格式 (SFT格式):
- instruction: 系统提示词
- input: 代码输入 (Code A, Code B, Diff)
- output: FASTER/SLOWER X.XX
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


# ==================== 配置 ====================

# 默认数据集目录
DEFAULT_DATASET_DIR = Path("./dataset_unified")

# 支持的语言和格式
LANGUAGES = ["python", "cpp", "java"]
FORMATS = ["ab", "abdiff", "adiff", "all"]

# 模型配置
MODEL_CONFIGS = {
    "32b": {
        "model_name": "Qwen/Qwen3-32B",
        "lora_prefix": "lora_qwen32b",
    },
    "7b": {
        "model_name": "Qwen/Qwen2.5-Coder-7B",
        "lora_prefix": "lora_qwen7b",
    }
}

# 默认Qwen ChatML模板
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def get_dataset_path(dataset_dir: Path, lang: str, fmt: str, split: str) -> Path:
    """获取数据集文件路径"""
    return dataset_dir / f"{lang}_{fmt}_{split}.jsonl"


def load_jsonl(filepath: Path) -> list:
    """加载JSONL文件"""
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(dataset_dir: Path, lang: str, fmt: str, split: str) -> list:
    """加载指定配置的数据"""
    if fmt == "all":
        all_data = []
        for f in ["ab", "abdiff", "adiff"]:
            filepath = get_dataset_path(dataset_dir, lang, f, split)
            print(f"  加载 {filepath}...")
            data = load_jsonl(filepath)
            print(f"    {len(data)} 条样本")
            all_data.extend(data)
        return all_data
    else:
        filepath = get_dataset_path(dataset_dir, lang, fmt, split)
        print(f"  加载 {filepath}...")
        data = load_jsonl(filepath)
        print(f"    {len(data)} 条样本")
        return data


def print_data_stats(data: list, name: str):
    """打印数据集统计"""
    print(f"\n{name} 统计:")
    print(f"  样本总数: {len(data)}")
    
    forward = sum(1 for x in data if x.get('direction') == 'forward')
    reverse = sum(1 for x in data if x.get('direction') == 'reversed')
    print(f"  Forward: {forward}, Reversed: {reverse}")
    
    problem_ids = set(x.get('problem_id', '') for x in data)
    print(f"  问题数: {len(problem_ids)}")


def format_to_chat(sample: dict, tokenizer) -> dict:
    """
    将SFT格式样本转换为chat格式
    所有语言都使用统一的 instruction/input/output 格式
    """
    messages = [
        {"role": "system", "content": sample['instruction']},
        {"role": "user", "content": sample['input']},
        {"role": "assistant", "content": sample['output']}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="统一训练脚本 v2")
    
    # 必需参数
    parser.add_argument("--lang", type=str, required=True, choices=LANGUAGES,
                        help="语言: python, cpp, java")
    parser.add_argument("--format", type=str, required=True, choices=FORMATS,
                        help="格式: ab, abdiff, adiff, all")
    
    # 模型配置
    parser.add_argument("--model_size", type=str, default="32b", choices=["7b", "32b"],
                        help="模型大小 (默认: 32b)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="自定义模型路径 (覆盖 model_size)")
    
    # 数据配置
    parser.add_argument("--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR),
                        help="数据集目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认自动生成)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=4096)
    
    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # DDP
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    # 解析配置
    dataset_dir = Path(args.dataset_dir)
    model_config = MODEL_CONFIGS[args.model_size]
    model_name = args.model_name or model_config["model_name"]
    
    # 生成输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./{model_config['lora_prefix']}_{args.lang}_{args.format}_v1"
    
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = local_rank in [-1, 0]
    
    if is_main:
        print("\n" + "="*60)
        print("统一训练脚本 v2")
        print("="*60)
        print(f"  模型: {model_name}")
        print(f"  语言: {args.lang}")
        print(f"  格式: {args.format}")
        print(f"  数据目录: {dataset_dir}")
        print(f"  输出目录: {output_dir}")
        print("="*60)
    
    # ==================== 1. 加载数据 ====================
    if is_main:
        print("\n[1] 加载数据集...")
    
    try:
        train_data = load_data(dataset_dir, args.lang, args.format, "train")
        val_data = load_data(dataset_dir, args.lang, args.format, "val")
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print(f"\n请确保数据文件在 {dataset_dir}/ 目录下")
        sys.exit(1)
    
    if is_main:
        print_data_stats(train_data, "训练集")
        print_data_stats(val_data, "验证集")
    
    # ==================== 2. 加载Tokenizer ====================
    if is_main:
        print("\n[2] 加载Tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None:
        if is_main:
            print("  [Warning] 未找到chat_template，使用默认Qwen模板")
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    
    # ==================== 3. 格式化数据 ====================
    if is_main:
        print("\n[3] 格式化和Tokenize...")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(
        lambda x: format_to_chat(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: format_to_chat(x, tokenizer),
        remove_columns=val_dataset.column_names
    )
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    if is_main:
        print(f"  训练样本: {len(train_dataset)}")
        print(f"  验证样本: {len(val_dataset)}")
    
    # ==================== 4. 加载模型 ====================
    if is_main:
        print("\n[4] 加载模型 (4-bit量化)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # ==================== 5. LoRA配置 ====================
    if is_main:
        print(f"\n[5] 应用LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    
    # ==================== 6. 训练配置 ====================
    if is_main:
        print("\n[6] 配置训练参数...")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation}")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
        print(f"  Learning rate: {args.learning_rate}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        ddp_find_unused_parameters=False,
    )
    
    # ==================== 7. 训练 ====================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    if is_main:
        print("\n" + "="*60)
        print("开始训练...")
        print("="*60)
    
    trainer.train()
    
    # ==================== 8. 保存 ====================
    if is_main:
        print("\n" + "="*60)
        print(f"保存模型到 {output_dir}...")
        print("="*60)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if is_main:
        print(f"\n✓ 训练完成！模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
