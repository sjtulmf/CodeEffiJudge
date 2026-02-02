#!/usr/bin/env python3
"""
统一推理脚本 v4
================

支持4种推理模式：
1. base_format: Base模型 + 训练格式测试集 (基准)
2. finetune_format: 微调模型 + 对应训练格式测试集 (训练成果)
3. base_zeroshot: Base模型 + zeroshot测试集 (泛化基准)
4. finetune_zeroshot: 微调模型 + zeroshot测试集 (泛化能力)

使用方法:
# Base模型基准
CUDA_VISIBLE_DEVICES=6 python inference_v4.py --mode base_format --lang python --format ab

# 微调模型训练成果  
CUDA_VISIBLE_DEVICES=6 python inference_v4.py --mode finetune_format --lang python --format ab --lora python_ab

# Base模型泛化基准
CUDA_VISIBLE_DEVICES=6 python inference_v4.py --mode base_zeroshot --lang python

# 微调模型泛化测试
CUDA_VISIBLE_DEVICES=6 python inference_v4.py --mode finetune_zeroshot --lang python --lora python_ab
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ==================== 配置 ====================

DEFAULT_DATASET_DIR = Path('./dataset_unified')
DEFAULT_OUTPUT_DIR = Path('./inference_results_4096')

# 模型配置
MODEL_CONFIGS = {
    "32b": {
        "base_model": "Qwen/Qwen3-32B",
        "lora_prefix": "lora_qwen32b",
    },
    "7b": {
        "base_model": "Qwen/Qwen2.5-Coder-7B", 
        "lora_prefix": "lora_qwen7b",
    }
}

LANGUAGES = ["python", "cpp", "java"]
FORMATS = ["ab", "abdiff", "adiff"]

# 默认Qwen ChatML模板
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


# ==================== 工具函数 ====================

def load_jsonl(filepath: str) -> List[dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[dict], filepath: str):
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"保存 {len(data)} 条结果到 {filepath}")


def get_difficulty(speedup: float) -> str:
    """根据speedup计算难度"""
    if speedup >= 8:
        return 'easy'
    elif speedup >= 4:
        return 'medium'
    else:
        return 'hard'


def get_lora_path(lang: str, fmt: str, model_size: str) -> Path:
    """获取LoRA模型路径"""
    prefix = MODEL_CONFIGS[model_size]["lora_prefix"]
    return Path(f"./{prefix}_{lang}_{fmt}_v1")


def find_best_checkpoint(lora_path: Path, use_final: bool = False) -> Optional[Path]:
    """寻找最佳checkpoint或最终checkpoint
    
    Args:
        lora_path: LoRA模型路径
        use_final: True=使用最终checkpoint(最大epoch), False=使用最优checkpoint(最小metric)
    """
    if not lora_path.exists():
        return None
    
    checkpoints = []
    for d in lora_path.iterdir():
        if d.is_dir() and d.name.startswith('checkpoint-'):
            state_file = d / 'trainer_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                metric = state.get('best_metric', float('inf'))
                epoch = state.get('epoch', 0)
                checkpoints.append((d, metric, epoch))
    
    if not checkpoints:
        # 检查根目录是否有adapter文件
        if (lora_path / 'adapter_config.json').exists():
            return lora_path
        return None
    
    if use_final:
        # 选择最终checkpoint（最大epoch）
        checkpoints.sort(key=lambda x: -x[2])  # 按epoch降序
        final = checkpoints[0]
        print(f"  最终checkpoint: {final[0].name} (epoch={final[2]:.2f})")
        return final[0]
    else:
        # 选择最优checkpoint（最小metric）
        checkpoints.sort(key=lambda x: (x[1], x[2]))
        best = checkpoints[0]
        print(f"  最佳checkpoint: {best[0].name} (metric={best[1]:.4f})")
        return best[0]


# ==================== 模型加载 ====================

def load_model(model_type: str, model_size: str, lora_key: Optional[str] = None, 
               custom_model_path: str = None, custom_lora_path: str = None, use_final: bool = False):
    """加载模型
    
    Args:
        use_final: True=使用最终checkpoint(epoch最大), False=使用最优checkpoint(metric最小)
    """
    
    config = MODEL_CONFIGS[model_size]
    base_path = custom_model_path or config["base_model"]
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    print(f"加载Base模型: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    if model_type == 'lora' and lora_key:
        # 解析 lora_key: "python_ab" -> lang="python", fmt="ab"
        parts = lora_key.split('_')
        if len(parts) >= 2:
            lang, fmt = parts[0], parts[1]
            lora_path = custom_lora_path or get_lora_path(lang, fmt, model_size)
        else:
            raise ValueError(f"无效的LoRA key: {lora_key}")
        
        if not Path(lora_path).exists():
            raise ValueError(f"LoRA模型路径不存在: {lora_path}")
        
        best_ckpt = find_best_checkpoint(Path(lora_path), use_final=use_final)
        adapter_path = best_ckpt or lora_path
        
        print(f"加载LoRA: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        model = base_model
    
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None:
        print("  [Warning] 使用默认Qwen chat模板")
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    
    model.eval()
    return model, tokenizer


# ==================== 生成函数 ====================

def generate_response(model, tokenizer, messages: List[dict], max_new_tokens: int = 64) -> str:
    """生成模型回复"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


# ==================== 解析函数 ====================

def parse_format_output(text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    解析训练格式输出: FASTER/SLOWER X.XX
    
    策略:
    1. 取最后一行
    2. 找数字，取离数字最近的FASTER/SLOWER
    3. 如果没有数字，取最后一个FASTER/SLOWER
    """
    text = text.strip()
    if not text:
        return None, None
    
    # 取最后一行
    last_line = text.split('\n')[-1].strip().upper()
    
    # 找所有FASTER/SLOWER的位置
    directions = []
    for m in re.finditer(r'(FASTER|SLOWER)', last_line):
        directions.append((m.start(), m.group(1)))
    
    if not directions:
        # 最后一行没有，在全文找
        text_upper = text.upper()
        for m in re.finditer(r'(FASTER|SLOWER)', text_upper):
            directions.append((m.start(), m.group(1)))
        if not directions:
            return None, None
        return directions[-1][1], None
    
    # 找所有数字
    numbers = []
    for m in re.finditer(r'[-+]?\d+\.?\d*', last_line):
        try:
            val = float(m.group())
            numbers.append((m.start(), val))
        except:
            pass
    
    if not numbers:
        return directions[-1][1], None
    
    # 找离数字最近的方向标签
    best_direction = None
    best_value = None
    min_distance = float('inf')
    
    for num_pos, num_val in numbers:
        for dir_pos, dir_name in directions:
            if dir_pos < num_pos:
                distance = num_pos - dir_pos
                if distance < min_distance:
                    min_distance = distance
                    best_direction = dir_name
                    best_value = num_val
    
    if best_direction:
        return best_direction, best_value
    
    return directions[-1][1], numbers[-1][1] if numbers else None


def parse_zeroshot_output(text: str) -> Optional[str]:
    """
    解析zero-shot输出: A 或 B
    """
    text = text.strip()
    if not text:
        return None
    
    last_line = text.split('\n')[-1].strip().upper()
    
    if last_line == 'A':
        return 'A'
    if last_line == 'B':
        return 'B'
    
    last_a = last_line.rfind('A')
    last_b = last_line.rfind('B')
    
    if last_a == -1 and last_b == -1:
        text_upper = text.upper()
        last_a = text_upper.rfind('A')
        last_b = text_upper.rfind('B')
    
    if last_a == -1 and last_b == -1:
        return None
    elif last_a == -1:
        return 'B'
    elif last_b == -1:
        return 'A'
    else:
        return 'A' if last_a > last_b else 'B'


# ==================== Zero-Shot Prompt ====================

def get_zeroshot_prompt(code_a: str, code_b: str, lang: str) -> dict:
    """Zero-Shot prompt模板"""
    lang_display = {"python": "Python", "cpp": "C++", "java": "Java"}.get(lang.lower(), lang)
    lang_tag = {"python": "python", "cpp": "cpp", "java": "java"}.get(lang.lower(), lang)
    
    system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs faster based on algorithmic complexity, data structure efficiency, and implementation details."""
    
    user_prompt = f"""Compare the following two functionally equivalent {lang_display} implementations and determine which one is more efficient.

## Code A:
```{lang_tag}
{code_a}
```

## Code B:
```{lang_tag}
{code_b}
```

Based on algorithm complexity, data structures, loop efficiency, and other performance metrics, determine which implementation runs faster.

You must output ONLY the identifier of the faster implementation:
- Output "A" if Code A is faster
- Output "B" if Code B is faster

Your response must be exactly one character: A or B

No explanation or additional text."""
    
    return {"system": system_prompt, "user": user_prompt}


# ==================== 推理函数 ====================

def run_format_inference(model, tokenizer, test_data: List[dict], lang: str) -> List[dict]:
    """运行训练格式推理 (ab/abdiff/adiff)"""
    results = []
    
    for sample in tqdm(test_data, desc="推理中"):
        # 所有语言统一使用 SFT 格式 (instruction/input/output)
        messages = [
            {"role": "system", "content": sample['instruction']},
            {"role": "user", "content": sample['input']}
        ]
        
        response = generate_response(model, tokenizer, messages)
        
        pred_dir, pred_val = parse_format_output(response)
        exp_dir, exp_val = parse_format_output(sample['output'])
        correct = (pred_dir == exp_dir) if pred_dir and exp_dir else False
        
        speedup = sample.get('speedup_ratio', sample.get('speedup', 1.0))
        difficulty = get_difficulty(speedup)
        
        result = {
            'pair_id': sample.get('pair_id', ''),
            'problem_id': sample.get('problem_id', ''),
            'language': lang,
            'direction': sample.get('direction', ''),
            'speedup': speedup,
            'difficulty': difficulty,
            'expected_output': sample['output'],
            'expected_direction': exp_dir,
            'expected_value': exp_val,
            'response': response,
            'predicted_direction': pred_dir,
            'predicted_value': pred_val,
            'correct': correct,
        }
        
        results.append(result)
    
    return results


def run_zeroshot_inference(model, tokenizer, test_data: List[dict]) -> List[dict]:
    """运行zero-shot推理"""
    results = []
    
    for sample in tqdm(test_data, desc="推理中"):
        lang = sample.get('language', 'python')
        
        # 检查是否已有预构造的prompt
        if 'system_prompt' in sample and 'user_prompt' in sample:
            messages = [
                {"role": "system", "content": sample['system_prompt']},
                {"role": "user", "content": sample['user_prompt']}
            ]
            expected = sample['expected']
        else:
            # 从 code_a/code_b 构造
            direction = sample.get('direction', 'forward')
            code_a = sample.get('code_a', sample.get('slow_code', ''))
            code_b = sample.get('code_b', sample.get('fast_code', ''))
            
            # direction=forward时，A是slow，B是fast，期望输出B
            # direction=reversed时，A是fast，B是slow，期望输出A
            if direction == 'forward':
                expected = 'B'
            else:
                expected = 'A'
            
            prompt = get_zeroshot_prompt(code_a, code_b, lang)
            messages = [
                {"role": "system", "content": prompt['system']},
                {"role": "user", "content": prompt['user']}
            ]
        
        response = generate_response(model, tokenizer, messages, max_new_tokens=16)
        predicted = parse_zeroshot_output(response)
        correct = (predicted == expected) if predicted else False
        
        speedup = sample.get('speedup', sample.get('speedup_ratio', 1.0))
        difficulty = sample.get('difficulty', get_difficulty(speedup))
        
        result = {
            'pair_id': sample.get('pair_id', ''),
            'problem_id': sample.get('problem_id', ''),
            'language': lang,
            'direction': sample.get('direction', ''),
            'speedup': speedup,
            'difficulty': difficulty,
            'expected': expected,
            'response': response,
            'predicted': predicted,
            'correct': correct,
        }
        
        results.append(result)
    
    return results


def print_quick_stats(results: List[dict]):
    """打印快速统计"""
    correct_count = sum(1 for r in results if r.get('correct', False))
    total_count = len(results)
    
    print(f"\n{'='*60}")
    print(f"快速统计")
    print(f"{'='*60}")
    print(f"总样本数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {correct_count/total_count*100:.2f}%")
    
    # 按难度统计
    difficulty_stats = {}
    for r in results:
        diff = r.get('difficulty', 'unknown')
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'correct': 0, 'total': 0}
        difficulty_stats[diff]['total'] += 1
        if r.get('correct', False):
            difficulty_stats[diff]['correct'] += 1
    
    print(f"\n按难度统计:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in difficulty_stats:
            s = difficulty_stats[diff]
            acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
            print(f"  {diff}: {s['correct']}/{s['total']} = {acc:.2f}%")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="统一推理脚本 v4")
    
    # 推理模式
    parser.add_argument('--mode', required=True,
                        choices=['base_format', 'finetune_format', 'base_zeroshot', 'finetune_zeroshot'],
                        help="推理模式")
    
    # 数据配置
    parser.add_argument('--lang', required=True, choices=LANGUAGES,
                        help="语言: python, cpp, java")
    parser.add_argument('--format', type=str, default='ab', choices=FORMATS,
                        help="数据格式 (仅format模式需要)")
    
    # 模型配置
    parser.add_argument('--lora', type=str, default=None,
                        help="LoRA模型key，如 python_ab, cpp_abdiff")
    parser.add_argument('--model_size', type=str, default='32b', choices=['7b', '32b'],
                        help="模型大小")
    parser.add_argument('--model_path', type=str, default=None,
                        help="自定义base模型路径")
    parser.add_argument('--lora_path', type=str, default=None,
                        help="自定义LoRA路径")
    
    # 路径配置
    parser.add_argument('--dataset_dir', type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--test_file', type=str, default=None,
                        help="自定义测试文件")
    
    # 其他
    parser.add_argument('--limit', type=int, default=0,
                        help="限制测试样本数 (0=全部)")
    parser.add_argument('--output', type=str, default=None,
                        help="自定义输出文件路径")
    parser.add_argument('--use_final', action='store_true',
                        help="使用最终checkpoint(epoch最大)，默认使用最优checkpoint(metric最小)")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 确定测试数据 ====================
    
    if args.test_file:
        test_file = Path(args.test_file)
        if not test_file.exists():
            print(f"错误: 测试文件不存在: {test_file}")
            sys.exit(1)
        test_data = load_jsonl(str(test_file))
        
    elif 'zeroshot' in args.mode:
        # Zeroshot 使用专门的测试集
        zeroshot_file = dataset_dir / f'{args.lang}_zeroshot_test.jsonl'
        if zeroshot_file.exists():
            print(f"加载zeroshot测试集: {zeroshot_file}")
            test_data = load_jsonl(str(zeroshot_file))
        else:
            print(f"错误: Zeroshot测试文件不存在: {zeroshot_file}")
            print("请先运行 build_zeroshot_test.py 构建测试集")
            sys.exit(1)
    else:
        # Format模式使用对应格式的测试集
        test_file = dataset_dir / f'{args.lang}_{args.format}_test.jsonl'
        if not test_file.exists():
            print(f"错误: 测试文件不存在: {test_file}")
            sys.exit(1)
        test_data = load_jsonl(str(test_file))
    
    if args.limit > 0:
        test_data = test_data[:args.limit]
    
    print(f"\n测试样本数: {len(test_data)}")
    
    # ==================== 加载模型 ====================
    
    print(f"\n加载模型 (size={args.model_size})...")
    
    if 'finetune' in args.mode:
        if not args.lora:
            print("错误: finetune模式需要指定 --lora")
            sys.exit(1)
        model, tokenizer = load_model(
            'lora', 
            args.model_size, 
            lora_key=args.lora,
            custom_model_path=args.model_path,
            custom_lora_path=args.lora_path,
            use_final=args.use_final
        )
        model_tag = f"{args.lora}_{args.model_size}"
    else:
        model, tokenizer = load_model(
            'base', 
            args.model_size,
            custom_model_path=args.model_path,
            use_final=False
        )
        model_tag = f'base_{args.model_size}'
    
    # ==================== 运行推理 ====================
    
    print(f"\n开始推理 (mode={args.mode}, lang={args.lang})...")
    
    if 'zeroshot' in args.mode:
        results = run_zeroshot_inference(model, tokenizer, test_data)
    else:
        results = run_format_inference(model, tokenizer, test_data, args.lang)
    
    # ==================== 保存结果 ====================
    
    if args.output:
        output_file = args.output
    elif 'zeroshot' in args.mode:
        output_file = output_dir / f'{args.mode}_{args.lang}_{model_tag}.jsonl'
    else:
        output_file = output_dir / f'{args.mode}_{args.lang}_{args.format}_{model_tag}.jsonl'
    
    save_jsonl(results, str(output_file))
    
    # 打印统计
    print_quick_stats(results)


if __name__ == '__main__':
    main()
