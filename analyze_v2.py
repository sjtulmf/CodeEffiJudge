#!/usr/bin/env python3
"""
推理结果分析脚本 v2
====================

分析指标:
1. 准确率 (Accuracy) = 正确数 / 总样本数
2. Both Correct = 双向都对的pair数 / 总pair数  
3. Both Wrong = 双向都错的pair数 / 总pair数
4. Bias = 单向对单向错的pair数 / 总pair数

难度分层:
- easy: speedup >= 8
- medium: 4 <= speedup < 8
- hard: 2 <= speedup < 4

使用方法:
python analyze_v2.py --input inference_results/
python analyze_v2.py --input inference_results/ --output report.json
python analyze_v2.py --file inference_results/xxx.jsonl
"""

import json
import argparse
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ==================== 7B模型修复逻辑 ====================

def parse_format_output_7b(text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    解析7B模型的训练格式输出: FASTER/SLOWER X.XX
    
    7B模型特点：答案在开头
    策略: 取第一行，匹配 FASTER/SLOWER 后面跟着的数字
    """
    text = text.strip()
    if not text:
        return None, None
    
    # 取第一行
    first_line = text.split('\n')[0].strip()
    
    # 直接匹配开头的 FASTER/SLOWER + 数字
    match = re.match(r'(FASTER|SLOWER)\s+([-+]?\d+\.?\d*)', first_line, re.IGNORECASE)
    if match:
        direction = match.group(1).upper()
        try:
            value = float(match.group(2))
            return direction, value
        except:
            return direction, None
    
    # 只匹配 FASTER/SLOWER
    match = re.match(r'(FASTER|SLOWER)', first_line, re.IGNORECASE)
    if match:
        direction = match.group(1).upper()
        # 尝试在同一行找数字
        numbers = re.findall(r'[-+]?\d+\.?\d*', first_line[match.end():])
        if numbers:
            try:
                return direction, float(numbers[0])
            except:
                pass
        return direction, None
    
    # 如果第一行没有，在全文开头找
    text_upper = text.upper()
    match = re.search(r'(FASTER|SLOWER)', text_upper)
    if match:
        direction = match.group(1)
        # 找紧跟着的数字
        remaining = text[match.end():match.end()+20]
        numbers = re.findall(r'[-+]?\d+\.?\d*', remaining)
        if numbers:
            try:
                return direction, float(numbers[0])
            except:
                pass
        return direction, None
    
    return None, None


def parse_zeroshot_output_7b(text: str) -> Optional[str]:
    """
    解析7B模型的zero-shot输出: A 或 B
    
    7B模型特点：答案在开头
    策略: 取第一行，找第一个A或B
    """
    text = text.strip()
    if not text:
        return None
    
    # 取第一行
    first_line = text.split('\n')[0].strip().upper()
    
    # 检查第一行是否就是 A 或 B
    if first_line == 'A':
        return 'A'
    if first_line == 'B':
        return 'B'
    
    # 在第一行找第一个A或B
    first_a = first_line.find('A')
    first_b = first_line.find('B')
    
    if first_a == -1 and first_b == -1:
        # 第一行没有，在全文开头找
        text_upper = text.upper()
        first_a = text_upper.find('A')
        first_b = text_upper.find('B')
    
    if first_a == -1 and first_b == -1:
        return None
    elif first_a == -1:
        return 'B'
    elif first_b == -1:
        return 'A'
    else:
        return 'A' if first_a < first_b else 'B'


def is_7b_model(filename: str) -> bool:
    """判断是否是7B模型的结果文件"""
    return '_7b.jsonl' in filename.lower()


def fix_7b_item_format(item: dict) -> dict:
    """修复单个7B模型的format类型样本"""
    response = item.get('response', '')
    
    # 重新解析
    new_pred_dir, new_pred_val = parse_format_output_7b(response)
    exp_dir = item.get('expected_direction')
    
    # 更新预测
    item['predicted_direction'] = new_pred_dir
    item['predicted_value'] = new_pred_val
    item['correct'] = (new_pred_dir == exp_dir) if new_pred_dir and exp_dir else False
    
    return item


def fix_7b_item_zeroshot(item: dict) -> dict:
    """修复单个7B模型的zeroshot类型样本"""
    response = item.get('response', '')
    
    # 重新解析
    new_predicted = parse_zeroshot_output_7b(response)
    expected = item.get('expected')
    
    # 更新预测
    item['predicted'] = new_predicted
    item['correct'] = (new_predicted == expected) if new_predicted and expected else False
    
    return item


def auto_fix_7b_results(data: List[dict], filename: str) -> List[dict]:
    """自动检测并修复7B模型的结果"""
    if not is_7b_model(filename):
        return data
    
    # 判断是format还是zeroshot
    is_format = 'format' in filename.lower()
    is_zeroshot = 'zeroshot' in filename.lower()
    
    if is_format:
        return [fix_7b_item_format(item.copy()) for item in data]
    elif is_zeroshot:
        return [fix_7b_item_zeroshot(item.copy()) for item in data]
    
    return data


# ==================== 原有分析逻辑 ====================


@dataclass
class PairStats:
    """单个pair的统计"""
    pair_id: str
    speedup: float
    difficulty: str
    forward_correct: bool = False
    reversed_correct: bool = False
    has_forward: bool = False
    has_reversed: bool = False
    forward_expected: Optional[float] = None
    forward_predicted: Optional[float] = None
    reversed_expected: Optional[float] = None
    reversed_predicted: Optional[float] = None


@dataclass 
class AnalysisResult:
    """分析结果"""
    total_samples: int = 0
    total_correct: int = 0
    
    total_pairs: int = 0
    both_correct: int = 0
    both_wrong: int = 0
    bias: int = 0
    
    by_difficulty: Dict[str, dict] = field(default_factory=dict)
    
    speedup_errors: List[float] = field(default_factory=list)
    speedup_abs_errors: List[float] = field(default_factory=list)
    
    def accuracy(self) -> float:
        return self.total_correct / self.total_samples * 100 if self.total_samples > 0 else 0
    
    def both_correct_rate(self) -> float:
        return self.both_correct / self.total_pairs * 100 if self.total_pairs > 0 else 0
    
    def both_wrong_rate(self) -> float:
        return self.both_wrong / self.total_pairs * 100 if self.total_pairs > 0 else 0
    
    def bias_rate(self) -> float:
        return self.bias / self.total_pairs * 100 if self.total_pairs > 0 else 0
    
    def mae(self) -> float:
        return sum(self.speedup_abs_errors) / len(self.speedup_abs_errors) if self.speedup_abs_errors else 0
    
    def rmse(self) -> float:
        if not self.speedup_errors:
            return 0
        mse = sum(e ** 2 for e in self.speedup_errors) / len(self.speedup_errors)
        return math.sqrt(mse)


def load_jsonl(filepath: str) -> List[dict]:
    """加载JSONL文件，自动修复7B模型结果"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 自动检测并修复7B模型结果
    filename = Path(filepath).name
    data = auto_fix_7b_results(data, filename)
    
    return data


def get_difficulty(speedup: float) -> str:
    """计算难度等级"""
    if speedup >= 8:
        return 'easy'
    elif speedup >= 4:
        return 'medium'
    else:
        return 'hard'


def get_base_pair_id(pair_id: str) -> str:
    """获取基础pair_id，用于匹配forward和reversed"""
    pair_id = str(pair_id)
    if pair_id.endswith('_forward'):
        return pair_id[:-8]
    elif pair_id.endswith('_reversed'):
        return pair_id[:-9]
    return pair_id


def analyze_results(data: List[dict]) -> AnalysisResult:
    """分析推理结果"""
    result = AnalysisResult()
    
    pair_stats: Dict[str, PairStats] = {}
    
    for item in data:
        result.total_samples += 1
        
        correct = item.get('correct', False)
        if correct:
            result.total_correct += 1
        
        pair_id = item.get('pair_id', item.get('problem_id', ''))
        direction = item.get('direction', 'forward')
        speedup = item.get('speedup', item.get('speedup_ratio', 1.0))
        difficulty = item.get('difficulty', get_difficulty(speedup))
        
        base_id = get_base_pair_id(pair_id)
        
        if base_id not in pair_stats:
            pair_stats[base_id] = PairStats(
                pair_id=base_id,
                speedup=speedup,
                difficulty=difficulty
            )
        
        ps = pair_stats[base_id]
        
        if direction == 'forward':
            ps.has_forward = True
            ps.forward_correct = correct
            ps.forward_expected = item.get('expected_value')
            ps.forward_predicted = item.get('predicted_value')
        else:
            ps.has_reversed = True
            ps.reversed_correct = correct
            ps.reversed_expected = item.get('expected_value')
            ps.reversed_predicted = item.get('predicted_value')
        
        # Speedup预测误差
        exp_val = item.get('expected_value')
        pred_val = item.get('predicted_value')
        if exp_val is not None and pred_val is not None:
            try:
                error = float(pred_val) - float(exp_val)
                result.speedup_errors.append(error)
                result.speedup_abs_errors.append(abs(error))
            except (ValueError, TypeError):
                pass
        
        # 难度统计
        if difficulty not in result.by_difficulty:
            result.by_difficulty[difficulty] = {
                'total': 0, 'correct': 0,
                'pairs': 0, 'both_correct': 0, 'both_wrong': 0
            }
        result.by_difficulty[difficulty]['total'] += 1
        if correct:
            result.by_difficulty[difficulty]['correct'] += 1
    
    # 计算pair级别统计
    for base_id, ps in pair_stats.items():
        if ps.has_forward and ps.has_reversed:
            result.total_pairs += 1
            
            if ps.forward_correct and ps.reversed_correct:
                result.both_correct += 1
            elif not ps.forward_correct and not ps.reversed_correct:
                result.both_wrong += 1
            else:
                result.bias += 1
            
            # 难度统计
            if ps.difficulty in result.by_difficulty:
                result.by_difficulty[ps.difficulty]['pairs'] += 1
                if ps.forward_correct and ps.reversed_correct:
                    result.by_difficulty[ps.difficulty]['both_correct'] += 1
                elif not ps.forward_correct and not ps.reversed_correct:
                    result.by_difficulty[ps.difficulty]['both_wrong'] += 1
    
    return result


def print_result(name: str, result: AnalysisResult):
    """打印分析结果"""
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print('='*70)
    
    print(f"\n【样本级别统计】")
    print(f"  总样本数: {result.total_samples}")
    print(f"  正确数: {result.total_correct}")
    print(f"  准确率: {result.accuracy():.2f}%")
    
    if result.total_pairs > 0:
        print(f"\n【Pair级别统计】")
        print(f"  总Pair数: {result.total_pairs}")
        print(f"  Both Correct: {result.both_correct} ({result.both_correct_rate():.2f}%)")
        print(f"  Both Wrong: {result.both_wrong} ({result.both_wrong_rate():.2f}%)")
        print(f"  Bias: {result.bias} ({result.bias_rate():.2f}%)")
    
    if result.speedup_abs_errors:
        print(f"\n【Speedup预测误差】")
        print(f"  MAE: {result.mae():.4f}")
        print(f"  RMSE: {result.rmse():.4f}")
    
    print(f"\n【按难度分层】")
    for diff in ['easy', 'medium', 'hard']:
        if diff in result.by_difficulty:
            d = result.by_difficulty[diff]
            acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            bc_rate = d['both_correct'] / d['pairs'] * 100 if d['pairs'] > 0 else 0
            print(f"  {diff:8s}: {d['correct']:4d}/{d['total']:<4d} = {acc:5.2f}%  "
                  f"(pairs: {d['pairs']}, BC: {d['both_correct']} = {bc_rate:.1f}%)")


def analyze_all_results(result_dir: Path) -> Dict[str, AnalysisResult]:
    """分析目录下所有结果文件"""
    all_results = {}
    
    for jsonl_file in sorted(result_dir.glob('*.jsonl')):
        try:
            data = load_jsonl(str(jsonl_file))
            if data:
                result = analyze_results(data)
                all_results[jsonl_file.stem] = result
        except Exception as e:
            print(f"警告: 处理 {jsonl_file} 失败: {e}")
    
    return all_results


def generate_summary_table(all_results: Dict[str, AnalysisResult]):
    """生成汇总表格"""
    print("\n" + "="*100)
    print("📈 汇总表格")
    print("="*100)
    
    # 按类别分组
    base_results = {k: v for k, v in all_results.items() if 'base_' in k}
    ft_results = {k: v for k, v in all_results.items() if 'finetune_' in k}
    
    print(f"\n{'文件名':<55} | {'准确率':>8} | {'BC率':>8} | {'BW率':>8} | {'MAE':>8}")
    print("-"*100)
    
    # Base模型结果
    if base_results:
        print("【Base模型】")
        for name in sorted(base_results.keys()):
            r = base_results[name]
            mae = f"{r.mae():.4f}" if r.speedup_abs_errors else "N/A"
            print(f"  {name:<53} | {r.accuracy():>7.2f}% | {r.both_correct_rate():>7.2f}% | "
                  f"{r.both_wrong_rate():>7.2f}% | {mae:>8}")
    
    # 微调模型结果
    if ft_results:
        print("\n【微调模型】")
        for name in sorted(ft_results.keys()):
            r = ft_results[name]
            mae = f"{r.mae():.4f}" if r.speedup_abs_errors else "N/A"
            print(f"  {name:<53} | {r.accuracy():>7.2f}% | {r.both_correct_rate():>7.2f}% | "
                  f"{r.both_wrong_rate():>7.2f}% | {mae:>8}")


def save_report(all_results: Dict[str, AnalysisResult], output_path: str):
    """保存JSON报告"""
    report = {}
    
    for name, result in all_results.items():
        report[name] = {
            'total_samples': result.total_samples,
            'total_correct': result.total_correct,
            'accuracy': result.accuracy(),
            'total_pairs': result.total_pairs,
            'both_correct': result.both_correct,
            'both_wrong': result.both_wrong,
            'bias': result.bias,
            'both_correct_rate': result.both_correct_rate(),
            'both_wrong_rate': result.both_wrong_rate(),
            'bias_rate': result.bias_rate(),
            'mae': result.mae(),
            'rmse': result.rmse(),
            'by_difficulty': result.by_difficulty,
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="推理结果分析脚本 v2")
    
    parser.add_argument('--input', type=str, default='./inference_results',
                        help="结果目录")
    parser.add_argument('--file', type=str, default=None,
                        help="单个结果文件")
    parser.add_argument('--output', type=str, default=None,
                        help="JSON报告输出路径")
    
    args = parser.parse_args()
    
    if args.file:
        # 分析单个文件
        data = load_jsonl(args.file)
        result = analyze_results(data)
        print_result(Path(args.file).stem, result)
    else:
        # 分析整个目录
        result_dir = Path(args.input)
        if not result_dir.exists():
            print(f"错误: 目录不存在: {result_dir}")
            return
        
        all_results = analyze_all_results(result_dir)
        
        if not all_results:
            print("未找到任何结果文件")
            return
        
        # 打印每个文件的详细结果
        for name, result in sorted(all_results.items()):
            print_result(name, result)
        
        # 生成汇总表格
        generate_summary_table(all_results)
        
        # 保存报告
        if args.output:
            save_report(all_results, args.output)


if __name__ == '__main__':
    main()
