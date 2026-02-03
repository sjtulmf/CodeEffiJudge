#!/usr/bin/env python3
"""
PIE Dataset Builder (Unified) - 构建均匀分布的代码优化数据集（含去重补充）

流程：
1. 加载原始数据，计算speedup
2. 分档分层采样（保证各档均匀）
3. 去重（同一problem_id+同一bin内的相似代码）
4. 补充（从原始数据中补充，保持目标数量，遵循problem_id均衡）
5. 生成最终数据集和图表

修复内容：
- 补充阶段现在遵循 problem_id 均衡策略
- 补充时会随机打乱候选，避免同一 problem_id 被连续选中
- seed 确保完全可复现

Usage:
    python extract.py -i merged.jsonl -n 900 -o ./output --output-name java --balance-problem-id -l java -d uniform --bins 2,4,8,inf --dedup-threshold 0.9 
"""

import argparse
import json
import re
import hashlib
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from difflib import SequenceMatcher
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='构建均匀分布的代码优化数据集（含去重补充）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
流程说明：
  1. 加载数据 → 计算speedup → 分档
  2. 分层采样（每档均分）
  3. 去重（同一problem_id+bin内的相似代码对）
  4. 从原始数据补充（保持目标数量，遵循problem_id均衡）
  5. 生成最终数据集和图表

Examples:
  # 基础用法
  python pie_dataset_builder_unified.py -i train.jsonl -n 10000 -o ./output --output-name balanced
  
  # 启用问题ID均衡 + 自定义去重阈值
  python pie_dataset_builder_unified.py -i train.jsonl -n 900 -o ./output --output-name java --balance-problem-id --dedup-threshold 0.9
  
  # 不去重
  python pie_dataset_builder_unified.py -i train.jsonl -n 10000 -o ./output --output-name balanced --no-dedup
        """
    )
    
    # ==================== 输入输出 ====================
    parser.add_argument('--input', '-i', required=True, 
                        help='输入jsonl文件路径')
    parser.add_argument('--total', '-n', type=int, required=True, 
                        help='目标数据集总pair数量')
    parser.add_argument('--output-dir', '-o', required=True, 
                        help='输出目录路径')
    parser.add_argument('--output-name', required=True, 
                        help='输出数据集名称（不含扩展名）')
    parser.add_argument('--language', '-l', type=str, default='cpp',
                        help='数据集语言 (用于注入language字段, default: cpp)')
    
    # ==================== 采样参数 ====================
    parser.add_argument('--distribution', '-d', default='natural',
                        choices=['uniform', 'normal', 'natural'], 
                        help='档内speedup分布方式 (default: natural)')
    parser.add_argument('--bins', type=str, default='1.5,2,4,8,16,inf', 
                        help='分档边界，逗号分隔 (default: 1.5,2,4,8,16,inf)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子 (default: 42)')
    parser.add_argument('--sub-bins', type=int, default=10,
                        help='档内均匀采样时的子区间数量 (default: 10)')
    
    # ==================== 字段名 ====================
    parser.add_argument('--speedup-key', type=str, default='speedup', 
                        help='speedup字段名 (default: speedup)')
    parser.add_argument('--time-v0-key', type=str, default='slow_time', 
                        help='慢版本时间字段名 (default: slow_time)')
    parser.add_argument('--time-v1-key', type=str, default='fast_time', 
                        help='快版本时间字段名 (default: fast_time)')
    parser.add_argument('--problem-id-key', type=str, default='problem_id', 
                        help='问题ID字段名 (default: problem_id)')
    parser.add_argument('--src-code-key', default='slow_code',
                        help='源代码字段名 (default: slow_code)')
    parser.add_argument('--tgt-code-key', default='fast_code',
                        help='目标代码字段名 (default: fast_code)')
    
    # ==================== 问题ID均衡 ====================
    parser.add_argument('--balance-problem-id', action='store_true', 
                        help='启用问题ID全局均衡采样')
    parser.add_argument('--max-pairs-per-problem', type=int, default=10, 
                        help='每个问题ID最多采样的pair数量')
    
    # ==================== 补充参数 ====================
    parser.add_argument('--no-replenish', action='store_true',
                        help='禁用数据补充（仅去重，不补齐到目标数量）')

    # ==================== 去重参数 ====================
    parser.add_argument('--no-dedup', action='store_true',
                        help='禁用去重')
    parser.add_argument('--dedup-threshold', '-t', type=float, default=0.9,
                        help='去重相似度阈值 (default: 0.9)')
    parser.add_argument('--dedup-method', default='combined',
                        choices=['hash', 'jaccard', 'edit', 'combined'],
                        help='去重相似度计算方法 (default: combined)')
    parser.add_argument('--compare-mode', default='both',
                        choices=['src', 'tgt', 'both', 'either'],
                        help='去重比较模式 (default: both)')
    parser.add_argument('--keep-strategy', default='max-speedup',
                        choices=['max-speedup', 'min-speedup', 'first', 'random'],
                        help='去重时保留策略 (default: max-speedup)')
    
    # ==================== 图表参数 ====================
    parser.add_argument('--fig-format', type=str, default='png', 
                        choices=['png', 'pdf', 'svg'],
                        help='图片输出格式 (default: png)')
    parser.add_argument('--fig-dpi', type=int, default=150,
                        help='图片DPI (default: 150)')
    parser.add_argument('--no-plot', action='store_true', 
                        help='不生成分布图')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息')
    
    return parser.parse_args()


# ============================================================
# 工具函数
# ============================================================

def parse_bins(bins_str: str) -> List[float]:
    """解析分档边界字符串"""
    bins = []
    for b in bins_str.split(','):
        b = b.strip()
        bins.append(float('inf') if b.lower() == 'inf' else float(b))
    return sorted(bins)


def load_data(input_path: str) -> List[Dict]:
    """加载jsonl数据"""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data


def compute_speedup(data: List[Dict], speedup_key: Optional[str], 
                    time_v0_key: str, time_v1_key: str, problem_id_key: str) -> List[Dict]:
    """计算speedup并过滤无效数据（仅保留 speedup >= 1.5x 以对抗测量 RSD）"""
    valid_data = []
    for d in data:
        try:
            if speedup_key and speedup_key in d and d[speedup_key] is not None:
                speedup = float(d[speedup_key])
            else:
                time_v0 = float(d[time_v0_key])
                time_v1 = float(d[time_v1_key])
                if time_v1 <= 0:
                    continue
                speedup = time_v0 / time_v1
            
            # 最低阈值从 1.0x 提升到 1.5x，对抗测量误差
            if speedup >= 1.5:
                d['_speedup'] = speedup
                d['_pair_uid'] = (
                    d.get(problem_id_key, 'unknown'),
                    d.get(time_v0_key),
                    d.get(time_v1_key),
                )
                valid_data.append(d)
        except (KeyError, ValueError, TypeError):
            continue
    return valid_data


def assign_bins(data: List[Dict], bins: List[float]) -> Dict[int, List[Dict]]:
    """将数据分配到各个档位"""
    binned = defaultdict(list)
    for d in data:
        speedup = d['_speedup']
        for i in range(len(bins) - 1):
            if bins[i] <= speedup < bins[i + 1]:
                d['_bin_idx'] = i
                d['_bin_label'] = f"[{bins[i]}x, {bins[i+1]}x)" if bins[i+1] != float('inf') else f"[{bins[i]}x, inf)"
                binned[i].append(d)
                break
    return binned


# ============================================================
# 采样函数
# ============================================================

def sample_within_bin_uniform(items: List[Dict], n: int, rng: np.random.Generator, 
                               num_sub_bins: int = 4) -> List[Dict]:
    """档内均匀采样：分子区间均匀采样"""
    if len(items) <= n:
        return items.copy()
    
    speedups = np.array([d['_speedup'] for d in items])
    min_sp, max_sp = speedups.min(), speedups.max()
    
    if max_sp - min_sp < 1e-6:
        indices = rng.choice(len(items), n, replace=False)
        return [items[i] for i in indices]
    
    sub_bin_edges = np.linspace(min_sp, max_sp + 1e-6, num_sub_bins + 1)
    per_sub_bin = n // num_sub_bins
    remainder = n % num_sub_bins
    
    sampled = []
    for i in range(num_sub_bins):
        mask = (speedups >= sub_bin_edges[i]) & (speedups < sub_bin_edges[i+1])
        sub_items = [items[j] for j in range(len(items)) if mask[j]]
        target = per_sub_bin + (1 if i < remainder else 0)
        
        if len(sub_items) == 0:
            continue
        elif len(sub_items) <= target:
            sampled.extend(sub_items)
        else:
            indices = rng.choice(len(sub_items), target, replace=False)
            sampled.extend([sub_items[j] for j in indices])
    
    if len(sampled) < n:
        sampled_uids = set(d['_pair_uid'] for d in sampled)
        remaining = [d for d in items if d['_pair_uid'] not in sampled_uids]
        if remaining:
            need = n - len(sampled)
            if len(remaining) <= need:
                sampled.extend(remaining)
            else:
                indices = rng.choice(len(remaining), need, replace=False)
                sampled.extend([remaining[i] for i in indices])
    
    return sampled


def sample_within_bin_normal(items: List[Dict], n: int, rng: np.random.Generator) -> List[Dict]:
    """档内正态分布采样"""
    if len(items) <= n:
        return items.copy()
    
    speedups = np.array([d['_speedup'] for d in items])
    log_speedups = np.log2(speedups)
    mean_log = np.mean(log_speedups)
    std_log = np.std(log_speedups)
    
    if std_log < 1e-6:
        indices = rng.choice(len(items), n, replace=False)
    else:
        weights = np.exp(-0.5 * ((log_speedups - mean_log) / std_log) ** 2)
        weights /= weights.sum()
        indices = rng.choice(len(items), n, replace=False, p=weights)
    
    return [items[i] for i in indices]


def sample_within_bin_natural(items: List[Dict], n: int, rng: np.random.Generator) -> List[Dict]:
    """档内自然分布采样"""
    if len(items) <= n:
        return items.copy()
    indices = rng.choice(len(items), n, replace=False)
    return [items[i] for i in indices]


def sample_within_bin_with_problem_balance(items: List[Dict],
                                            n: int,
                                            distribution: str,
                                            rng: np.random.Generator,
                                            problem_id_key: str,
                                            global_problem_counts: Dict[str, int],
                                            max_pairs_per_problem: Optional[int] = None,
                                            num_sub_bins: int = 4) -> List[Dict]:
    """档内采样，同时考虑问题ID均衡和speedup分布"""
    if len(items) <= n:
        for item in items:
            pid = item.get(problem_id_key, 'unknown')
            global_problem_counts[pid] = global_problem_counts.get(pid, 0) + 1
        return items.copy()
    
    sampled = []
    
    if distribution == 'uniform':
        speedups = np.array([d['_speedup'] for d in items])
        min_sp, max_sp = speedups.min(), speedups.max()
        
        if max_sp - min_sp < 1e-6:
            sub_bin_items_list = [items]
            targets_per_sub = [n]
        else:
            sub_bin_edges = np.linspace(min_sp, max_sp + 1e-6, num_sub_bins + 1)
            per_sub = n // num_sub_bins
            remainder = n % num_sub_bins
            targets_per_sub = [per_sub + (1 if i < remainder else 0) for i in range(num_sub_bins)]
            
            sub_bin_items_list = []
            for i in range(num_sub_bins):
                mask = (speedups >= sub_bin_edges[i]) & (speedups < sub_bin_edges[i+1])
                sub_items = [items[j] for j in range(len(items)) if mask[j]]
                sub_bin_items_list.append(sub_items)
        
        for sub_items, target in zip(sub_bin_items_list, targets_per_sub):
            if not sub_items or target <= 0:
                continue
            
            problem_items = defaultdict(list)
            for item in sub_items:
                pid = item.get(problem_id_key, 'unknown')
                problem_items[pid].append(item)
            
            for pid in problem_items:
                rng.shuffle(problem_items[pid])
            
            sub_sampled = []
            problem_pointers = {pid: 0 for pid in problem_items}
            all_pids = list(problem_items.keys())
            actual_target = min(target, len(sub_items))
            
            while len(sub_sampled) < actual_target:
                made_progress = False
                rng.shuffle(all_pids)
                sorted_pids = sorted(all_pids, key=lambda pid: global_problem_counts.get(pid, 0))
                
                for pid in sorted_pids:
                    if len(sub_sampled) >= actual_target:
                        break
                    if max_pairs_per_problem is not None and global_problem_counts.get(pid, 0) >= max_pairs_per_problem:
                        continue
                    pointer = problem_pointers[pid]
                    if pointer >= len(problem_items[pid]):
                        continue
                    
                    item = problem_items[pid][pointer]
                    sub_sampled.append(item)
                    problem_pointers[pid] += 1
                    global_problem_counts[pid] = global_problem_counts.get(pid, 0) + 1
                    made_progress = True
                
                if not made_progress:
                    break
            
            sampled.extend(sub_sampled)
        
        if len(sampled) < n:
            sampled_uids = set(d['_pair_uid'] for d in sampled)
            remaining = [d for d in items if d['_pair_uid'] not in sampled_uids]
            
            if remaining:
                problem_items = defaultdict(list)
                for item in remaining:
                    pid = item.get(problem_id_key, 'unknown')
                    problem_items[pid].append(item)
                
                for pid in problem_items:
                    rng.shuffle(problem_items[pid])
                
                problem_pointers = {pid: 0 for pid in problem_items}
                all_pids = list(problem_items.keys())
                
                while len(sampled) < n:
                    made_progress = False
                    rng.shuffle(all_pids)
                    sorted_pids = sorted(all_pids, key=lambda pid: global_problem_counts.get(pid, 0))
                    
                    for pid in sorted_pids:
                        if len(sampled) >= n:
                            break
                        if max_pairs_per_problem is not None and global_problem_counts.get(pid, 0) >= max_pairs_per_problem:
                            continue
                        pointer = problem_pointers[pid]
                        if pointer >= len(problem_items[pid]):
                            continue
                        
                        item = problem_items[pid][pointer]
                        sampled.append(item)
                        problem_pointers[pid] += 1
                        global_problem_counts[pid] = global_problem_counts.get(pid, 0) + 1
                        made_progress = True
                    
                    if not made_progress:
                        break
    
    else:
        problem_items = defaultdict(list)
        for item in items:
            pid = item.get(problem_id_key, 'unknown')
            problem_items[pid].append(item)
        
        if distribution == 'normal':
            all_bin_speedups = np.array([d['_speedup'] for d in items])
            log_speedups = np.log2(all_bin_speedups)
            mean_log = np.mean(log_speedups)
            std_log = np.std(log_speedups) if len(log_speedups) > 1 else 0
            
            for pid in problem_items:
                group = problem_items[pid]
                if not group: continue
                
                if std_log < 1e-6:
                    rng.shuffle(group)
                else:
                    g_speedups = np.array([x['_speedup'] for x in group])
                    g_log = np.log2(g_speedups)
                    weights = np.exp(-0.5 * ((g_log - mean_log) / std_log) ** 2)
                    
                    if weights.sum() < 1e-9:
                         weights[:] = 1.0
                    weights /= weights.sum()
                    
                    try:
                        indices = rng.choice(len(group), len(group), replace=False, p=weights)
                        problem_items[pid] = [group[i] for i in indices]
                    except ValueError:
                        rng.shuffle(problem_items[pid])
        else:
            for pid in problem_items:
                rng.shuffle(problem_items[pid])
        
        problem_pointers = {pid: 0 for pid in problem_items}
        all_pids = list(problem_items.keys())
        
        while len(sampled) < n:
            made_progress = False
            rng.shuffle(all_pids)
            sorted_pids = sorted(all_pids, key=lambda pid: global_problem_counts.get(pid, 0))
            
            for pid in sorted_pids:
                if len(sampled) >= n:
                    break
                if max_pairs_per_problem is not None and global_problem_counts.get(pid, 0) >= max_pairs_per_problem:
                    continue
                pointer = problem_pointers[pid]
                if pointer >= len(problem_items[pid]):
                    continue
                
                item = problem_items[pid][pointer]
                sampled.append(item)
                problem_pointers[pid] += 1
                global_problem_counts[pid] = global_problem_counts.get(pid, 0) + 1
                made_progress = True
            
            if not made_progress:
                break
    
    return sampled


def stratified_sample(binned: Dict[int, List[Dict]], 
                      total: int,
                      bins: List[float],
                      distribution: str,
                      rng: np.random.Generator,
                      balance_problem_id: bool = False,
                      problem_id_key: str = 'problem_id',
                      max_pairs_per_problem: Optional[int] = None,
                      num_sub_bins: int = 4) -> Tuple[List[Dict], Dict, Dict[str, int]]:
    """分层采样，返回采样结果、统计信息和全局problem_id计数"""
    num_bins = len(bins) - 1
    available_bins = [i for i in range(num_bins) if len(binned[i]) > 0]
    
    if len(available_bins) == 0:
        raise ValueError("所有档位均无数据")
    
    per_bin = total // len(available_bins)
    remainder = total % len(available_bins)
    targets = {}
    for i, bin_idx in enumerate(available_bins):
        targets[bin_idx] = per_bin + (1 if i < remainder else 0)
    
    for bin_idx in targets:
        targets[bin_idx] = min(targets[bin_idx], len(binned[bin_idx]))
    
    sampled = []
    global_problem_counts = {}
    sorted_bins = sorted(available_bins, key=lambda b: len(binned[b]))
    
    for bin_idx in sorted_bins:
        target = targets[bin_idx]
        items = binned[bin_idx]
        
        if balance_problem_id:
            bin_sampled = sample_within_bin_with_problem_balance(
                items, target, distribution, rng, 
                problem_id_key, global_problem_counts, max_pairs_per_problem, num_sub_bins
            )
        else:
            if distribution == 'uniform':
                bin_sampled = sample_within_bin_uniform(items, target, rng, num_sub_bins)
            elif distribution == 'normal':
                bin_sampled = sample_within_bin_normal(items, target, rng)
            else:
                bin_sampled = sample_within_bin_natural(items, target, rng)
        
        sampled.extend(bin_sampled)
    
    stats = {'bins': [], 'original': [], 'sampled': [], 'target': []}
    for i in range(num_bins):
        bin_label = f"[{bins[i]}x, {bins[i+1]}x)" if bins[i+1] != float('inf') else f"[{bins[i]}x, inf)"
        stats['bins'].append(bin_label)
        stats['original'].append(len(binned[i]))
        stats['sampled'].append(len([d for d in sampled if d.get('_bin_idx') == i]))
        stats['target'].append(targets.get(i, 0))
    
    if balance_problem_id:
        stats['problem_id_counts'] = global_problem_counts
    
    rng.shuffle(sampled)
    return sampled, stats, global_problem_counts


# ============================================================
# 去重函数
# ============================================================

def normalize_code(code: str) -> str:
    """规范化代码"""
    if not code:
        return ""
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'[ \t]+', ' ', code)
    code = re.sub(r'\n\s*\n', '\n', code)
    code = code.strip()
    return code


def tokenize_code(code: str) -> List[str]:
    """将代码分词"""
    normalized = normalize_code(code)
    tokens = re.findall(r'\b\w+\b', normalized)
    return tokens


def code_hash(code: str) -> str:
    """计算代码哈希"""
    normalized = normalize_code(code)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """计算Jaccard相似度"""
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def edit_similarity(code1: str, code2: str) -> float:
    """计算编辑距离相似度"""
    normalized1 = normalize_code(code1)
    normalized2 = normalize_code(code2)
    if not normalized1 and not normalized2:
        return 1.0
    if not normalized1 or not normalized2:
        return 0.0
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def compute_similarity(code1: str, code2: str, method: str) -> float:
    """根据指定方法计算相似度"""
    if method == 'hash':
        return 1.0 if code_hash(code1) == code_hash(code2) else 0.0
    elif method == 'jaccard':
        tokens1 = tokenize_code(code1)
        tokens2 = tokenize_code(code2)
        return jaccard_similarity(tokens1, tokens2)
    elif method == 'edit':
        return edit_similarity(code1, code2)
    elif method == 'combined':
        if code_hash(code1) == code_hash(code2):
            return 1.0
        tokens1 = tokenize_code(code1)
        tokens2 = tokenize_code(code2)
        return jaccard_similarity(tokens1, tokens2)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_pair_similarity(item1: Dict, item2: Dict,
                             src_key: str, tgt_key: str,
                             compare_mode: str, method: str) -> float:
    """计算两个pair的相似度"""
    src_sim = compute_similarity(item1.get(src_key, ''), item2.get(src_key, ''), method)
    tgt_sim = compute_similarity(item1.get(tgt_key, ''), item2.get(tgt_key, ''), method)
    
    if compare_mode == 'src':
        return src_sim
    elif compare_mode == 'tgt':
        return tgt_sim
    elif compare_mode == 'both':
        return min(src_sim, tgt_sim)
    elif compare_mode == 'either':
        return max(src_sim, tgt_sim)
    else:
        return src_sim


def find_duplicates_in_group(items: List[Dict], 
                              src_key: str,
                              tgt_key: str,
                              threshold: float,
                              method: str,
                              compare_mode: str = 'both') -> List[Set[int]]:
    """在一组items中找出重复的集合"""
    n = len(items)
    if n <= 1:
        return []
    
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_pair_similarity(items[i], items[j], src_key, tgt_key, compare_mode, method)
            if sim >= threshold:
                union(i, j)
    
    groups = defaultdict(set)
    for i in range(n):
        root = find(i)
        groups[root].add(i)
    
    return [g for g in groups.values() if len(g) > 1]


def select_best_from_group(items: List[Dict], 
                            indices: Set[int],
                            strategy: str,
                            rng: np.random.Generator) -> int:
    """从重复组中选择保留的项"""
    idx_list = list(indices)
    
    if strategy == 'first':
        return min(idx_list)
    elif strategy == 'random':
        return rng.choice(idx_list)
    elif strategy == 'max-speedup':
        best_idx = idx_list[0]
        best_speedup = items[best_idx].get('speedup', items[best_idx].get('_speedup', 0))
        for idx in idx_list[1:]:
            speedup = items[idx].get('speedup', items[idx].get('_speedup', 0))
            if speedup > best_speedup:
                best_speedup = speedup
                best_idx = idx
        return best_idx
    elif strategy == 'min-speedup':
        best_idx = idx_list[0]
        best_speedup = items[best_idx].get('speedup', items[best_idx].get('_speedup', float('inf')))
        for idx in idx_list[1:]:
            speedup = items[idx].get('speedup', items[idx].get('_speedup', float('inf')))
            if speedup < best_speedup:
                best_speedup = speedup
                best_idx = idx
        return best_idx
    else:
        return idx_list[0]


def deduplicate_dataset(data: List[Dict],
                         problem_id_key: str,
                         bin_key: str,
                         src_key: str,
                         tgt_key: str,
                         threshold: float,
                         method: str,
                         keep_strategy: str,
                         compare_mode: str = 'both',
                         verbose: bool = False,
                         rng: np.random.Generator = None) -> Tuple[List[Dict], Dict]:
    """对数据集进行去重"""
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    groups = defaultdict(list)
    for i, item in enumerate(data):
        pid = item.get(problem_id_key, 'unknown')
        sbin = item.get(bin_key, item.get('_bin_label', 'unknown'))
        groups[(pid, sbin)].append((i, item))
    
    stats = {
        'total_items': len(data),
        'total_groups': len(groups),
        'groups_with_duplicates': 0,
        'total_duplicates_removed': 0,
        'duplicate_details': [],
    }
    
    keep_indices = set(range(len(data)))
    removed_items = []
    
    for (pid, sbin), group_items in groups.items():
        if len(group_items) <= 1:
            continue
        
        items = [item for _, item in group_items]
        original_indices = [idx for idx, _ in group_items]
        
        dup_groups = find_duplicates_in_group(items, src_key, tgt_key, threshold, method, compare_mode)
        
        if not dup_groups:
            continue
        
        stats['groups_with_duplicates'] += 1
        
        removed_in_group_count = 0
        for dup_set in dup_groups:
            local_best = select_best_from_group(items, dup_set, keep_strategy, rng)
            global_best = original_indices[local_best]
            
            for local_idx in dup_set:
                global_idx = original_indices[local_idx]
                if global_idx != global_best:
                    if global_idx in keep_indices:
                        keep_indices.discard(global_idx)
                        removed_items.append(data[global_idx])
                        stats['total_duplicates_removed'] += 1
                        removed_in_group_count += 1
        
        if removed_in_group_count > 0:
            stats['duplicate_details'].append({
                'problem_id': pid,
                'speedup_bin': sbin,
                'original_count': len(group_items),
                'kept_count': len(group_items) - removed_in_group_count,
                'removed_count': removed_in_group_count,
            })
            
            if verbose:
                print(f"  [{pid}][{sbin}]: {len(group_items)} -> {len(group_items) - removed_in_group_count} "
                      f"(removed {removed_in_group_count})")
    
    deduped = [data[i] for i in sorted(keep_indices)]
    stats['final_count'] = len(deduped)
    stats['removed_items'] = removed_items
    
    return deduped, stats


# ============================================================
# 补充函数 (修复版)
# ============================================================

def replenish_dataset(deduped: List[Dict],
                      removed_items: List[Dict],
                      original_data: List[Dict],
                      target_count: int,
                      bins: List[float],
                      problem_id_key: str,
                      bin_key: str,
                      src_key: str,
                      tgt_key: str,
                      rng: np.random.Generator,
                      balance_problem_id: bool = False,
                      global_problem_counts: Optional[Dict[str, int]] = None,
                      max_pairs_per_problem: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    从原始数据补充，保持目标数量
    
    修复：
    1. 补充时遵循 problem_id 均衡策略
    2. 补充候选随机打乱，避免同一 problem_id 被连续选中
    3. 应用 max_pairs_per_problem 限制
    """
    
    if len(deduped) >= target_count:
        return deduped, []
    
    need_count = target_count - len(deduped)
    print(f"\n需要补充: {need_count} 条")
    
    # 初始化或使用传入的 problem_id 计数
    if global_problem_counts is None:
        global_problem_counts = defaultdict(int)
        for item in deduped:
            pid = item.get(problem_id_key, 'unknown')
            global_problem_counts[pid] += 1
    else:
        # 复制一份，避免修改原始计数
        global_problem_counts = defaultdict(int, global_problem_counts)
    
    # 收集已有的内容哈希
    existing_hashes = set()
    for item in deduped:
        s_hash = code_hash(item.get(src_key, ''))
        t_hash = code_hash(item.get(tgt_key, ''))
        existing_hashes.add((s_hash, t_hash))
    
    # 排除被移除的项
    for item in removed_items:
        s_hash = code_hash(item.get(src_key, ''))
        t_hash = code_hash(item.get(tgt_key, ''))
        existing_hashes.add((s_hash, t_hash))
    
    def compute_speedup_bin(item):
        speedup = item.get('speedup', item.get('_speedup'))
        if speedup is None:
            # 支持两种字段名格式: unified (slow_time/fast_time) 和 legacy (src_agg_runtime/tgt_agg_runtime)
            src_time = item.get('slow_time', item.get('src_agg_runtime', 0))
            tgt_time = item.get('fast_time', item.get('tgt_agg_runtime', 0))
            try:
                speedup = float(src_time) / float(tgt_time) if float(tgt_time) > 0 else 1.0
            except (ValueError, TypeError, ZeroDivisionError):
                speedup = 1.0
        
        for i in range(len(bins) - 1):
            if bins[i] <= speedup < bins[i + 1]:
                bin_label = f"[{bins[i]}x, {bins[i + 1]}x)" if bins[i + 1] != float('inf') else f"[{bins[i]}x, inf)"
                return speedup, bin_label, i
        return None, None, None
    
    # 找候选并按 problem_id 分组
    candidates_by_pid = defaultdict(list)
    for item in original_data:
        if bin_key not in item and '_bin_label' not in item:
            speedup, bin_label, bin_idx = compute_speedup_bin(item)
            if speedup is None:
                continue
            item['speedup'] = speedup
            item[bin_key] = bin_label
            item['speedup_bin_idx'] = bin_idx
            item['_bin_label'] = bin_label
            item['_bin_idx'] = bin_idx
        
        s_hash = code_hash(item.get(src_key, ''))
        t_hash = code_hash(item.get(tgt_key, ''))
        if (s_hash, t_hash) not in existing_hashes:
            pid = item.get(problem_id_key, 'unknown')
            candidates_by_pid[pid].append(item)
    
    total_candidates = sum(len(v) for v in candidates_by_pid.values())
    print(f"找到候选: {total_candidates} 条 (来自 {len(candidates_by_pid)} 个 problem_id)")
    
    # 随机打乱每个 problem_id 内的候选
    for pid in candidates_by_pid:
        rng.shuffle(candidates_by_pid[pid])
    
    replenished = []
    
    if balance_problem_id:
        # ========== 均衡补充模式 ==========
        # 使用轮询方式，优先选择当前计数最少的 problem_id
        
        # 统计当前各bin的数量，用于优先补充数量少的bin
        bin_counts = defaultdict(int)
        for item in deduped:
            sbin = item.get(bin_key, item.get('_bin_label', 'unknown'))
            bin_counts[sbin] += 1
        
        # 按 bin 分组候选
        bin_candidates_by_pid = defaultdict(lambda: defaultdict(list))
        for pid, items in candidates_by_pid.items():
            for item in items:
                sbin = item.get(bin_key, item.get('_bin_label', 'unknown'))
                bin_candidates_by_pid[sbin][pid].append(item)
        
        # 记录每个 pid 在每个 bin 的指针
        pid_pointers = defaultdict(lambda: defaultdict(int))
        
        while len(replenished) < need_count:
            # 找到数量最少的 bin
            if not bin_counts:
                break
            min_bin = min(bin_counts.keys(), key=lambda b: bin_counts[b])
            
            # 在这个 bin 中，找到计数最少的 problem_id
            available_pids = [pid for pid in bin_candidates_by_pid[min_bin] 
                             if pid_pointers[min_bin][pid] < len(bin_candidates_by_pid[min_bin][pid])]
            
            if not available_pids:
                # 这个 bin 没有候选了，从其他 bin 找
                found = False
                for sbin in sorted(bin_counts.keys(), key=lambda b: bin_counts[b]):
                    if sbin == min_bin:
                        continue
                    available_pids = [pid for pid in bin_candidates_by_pid[sbin] 
                                     if pid_pointers[sbin][pid] < len(bin_candidates_by_pid[sbin][pid])]
                    if available_pids:
                        min_bin = sbin
                        found = True
                        break
                
                if not found:
                    print(f"⚠ 候选不足，只能补充 {len(replenished)} 条")
                    break
            
            # 过滤掉已达上限的 pid
            if max_pairs_per_problem is not None:
                available_pids = [pid for pid in available_pids 
                                 if global_problem_counts[pid] < max_pairs_per_problem]
            
            if not available_pids:
                # 所有 pid 都达上限，直接从任意 bin 任意 pid 取
                found = False
                for sbin in bin_candidates_by_pid:
                    for pid in bin_candidates_by_pid[sbin]:
                        if pid_pointers[sbin][pid] < len(bin_candidates_by_pid[sbin][pid]):
                            if max_pairs_per_problem is None or global_problem_counts[pid] < max_pairs_per_problem:
                                available_pids = [pid]
                                min_bin = sbin
                                found = True
                                break
                    if found:
                        break
                
                if not found:
                    print(f"⚠ 所有候选已用尽或达到上限，只能补充 {len(replenished)} 条")
                    break
            
            # 选择计数最少的 pid
            selected_pid = min(available_pids, key=lambda p: global_problem_counts[p])
            
            # 取一个候选
            pointer = pid_pointers[min_bin][selected_pid]
            item = bin_candidates_by_pid[min_bin][selected_pid][pointer]
            
            # 动态检查，防止补充进来的样本之间重复
            s_hash = code_hash(item.get(src_key, ''))
            t_hash = code_hash(item.get(tgt_key, ''))
            if (s_hash, t_hash) in existing_hashes:
                pid_pointers[min_bin][selected_pid] += 1
                continue

            replenished.append(item)
            existing_hashes.add((s_hash, t_hash)) # 记录已选中的哈希
            pid_pointers[min_bin][selected_pid] += 1
            global_problem_counts[selected_pid] += 1
            bin_counts[min_bin] += 1
        
        print(f"实际补充: {len(replenished)} 条 (均衡模式)")
    
    else:
        # ========== 非均衡补充模式 (原始逻辑，但随机打乱) ==========
        # 按bin分组候选
        bin_candidates = defaultdict(list)
        for pid, items in candidates_by_pid.items():
            for item in items:
                sbin = item.get(bin_key, item.get('_bin_label', 'unknown'))
                bin_candidates[sbin].append(item)
        
        # 随机打乱每个 bin 内的候选
        for sbin in bin_candidates:
            rng.shuffle(bin_candidates[sbin])
        
        # 统计当前各bin的数量
        bin_counts = defaultdict(int)
        for item in deduped:
            sbin = item.get(bin_key, item.get('_bin_label', 'unknown'))
            bin_counts[sbin] += 1
        
        # 优先补充数量少的bin
        while len(replenished) < need_count:
            if not bin_counts:
                break
            min_bin = min(bin_counts.keys(), key=lambda b: bin_counts[b])
            
            if min_bin and bin_candidates[min_bin]:
                item = bin_candidates[min_bin].pop(0)
                
                # 动态哈希检查
                s_hash = code_hash(item.get(src_key, ''))
                t_hash = code_hash(item.get(tgt_key, ''))
                if (s_hash, t_hash) in existing_hashes:
                    continue
                    
                replenished.append(item)
                existing_hashes.add((s_hash, t_hash))
                bin_counts[min_bin] += 1
            else:
                found = False
                for sbin, cands in bin_candidates.items():
                    if cands:
                        item = cands.pop(0)
                        
                        s_hash = code_hash(item.get(src_key, ''))
                        t_hash = code_hash(item.get(tgt_key, ''))
                        if (s_hash, t_hash) in existing_hashes:
                            continue
                            
                        replenished.append(item)
                        existing_hashes.add((s_hash, t_hash))
                        bin_counts[sbin] = bin_counts.get(sbin, 0) + 1
                        found = True
                        break
                
                if not found:
                    print(f"⚠ 候选不足，只能补充 {len(replenished)} 条")
                    break
        
        print(f"实际补充: {len(replenished)} 条 (随机模式)")
    
    final_data = deduped + replenished
    return final_data, replenished


# ============================================================
# 绘图函数
# ============================================================

def plot_speedup_distribution(sampled: List[Dict],
                               stats: Dict,
                               bins: List[float],
                               all_data: List[Dict],
                               output_path: str,
                               fig_format: str,
                               dpi: int,
                               problem_id_key: str = 'problem_id',
                               raw_data_for_cdf: List[Dict] = None,
                               time_v0_key: str = 'slow_time',
                               time_v1_key: str = 'fast_time'):
    """绘制speedup分布图 (7图整合版，改进版支持超大speedup值的压缩显示)
    
    Args:
        raw_data_for_cdf: 用于 CDF 图的原始数据（包含 speedup >= 1.0 的完整数据）
    """
    
    # 创建 3x3 网格（实际使用 7 个子图）
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    num_bins = len(stats['bins'])
    colors = plt.cm.Set2(np.linspace(0, 1, num_bins))
    
    speedups = sorted([d.get('_speedup', d.get('speedup', 1)) for d in sampled])
    log_speedups = np.log2(speedups)
    n = len(speedups)
    
    min_sp_plot = max(1.0, min(speedups))
    max_sp_plot = max(speedups)
    
    # 检测是否有超大speedup值（用于自适应调整x轴显示）
    # 阈值设为64，如果最大值超过64，则使用压缩显示
    has_extreme_speedup = max_sp_plot > 64
    compress_threshold = 64 if has_extreme_speedup else None
    
    # ============================================================
    # 第1行: 分布概览
    # ============================================================
    
    # [0, 0] 左上：各档数量柱状图
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(num_bins)
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, stats['original'], width, label='Original', 
                    color='lightgray', edgecolor='black', alpha=0.6)
    bars2 = ax1.bar(x + width/2, stats['sampled'], width, label='Sampled',
                    color='steelblue', edgecolor='black', alpha=0.8)
    
    ax1.set_xlabel('Speedup Bin', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('Sample Distribution by Speedup Bin', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats['bins'], rotation=30, ha='right', fontsize=8)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars2, stats['sampled']):
        if count > 0:
            ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=7)
    
    # [0, 1] 中上：Speedup直方图（改进版：对超大值范围进行压缩显示）
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 如果有超大speedup，裁剪显示范围
    if has_extreme_speedup:
        # 只显示合理范围内的数据
        filtered_log_speedups = [ls for ls, s in zip(log_speedups, speedups) if s <= compress_threshold]
        extreme_count = len(log_speedups) - len(filtered_log_speedups)
        
        ax2.hist(filtered_log_speedups, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        
        # 添加超大值的提示
        if extreme_count > 0:
            ax2.text(0.98, 0.95, f'{extreme_count} outliers\n>{compress_threshold}× hidden', 
                    transform=ax2.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=8)
    else:
        ax2.hist(log_speedups, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    ax2.set_xlabel('Speedup (log2 scale)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title(f'Speedup Distribution (n={n})', fontsize=11, fontweight='bold')
    
    # 设置bin边界线
    bin_edges = [np.log2(b) if b != float('inf') else np.log2(max(speedups) * 2) for b in bins]
    for edge in bin_edges[:-1]:
        # 只在可见范围内显示边界线
        if not has_extreme_speedup or edge <= np.log2(compress_threshold):
            ax2.axvline(x=edge, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # 如果有超大值，限制x轴范围
    if has_extreme_speedup:
        ax2.set_xlim(0, np.log2(compress_threshold))
    
    ax2.grid(axis='y', alpha=0.3)
    
    # [0, 2] 右上：Problem ID 分布
    ax3 = fig.add_subplot(gs[0, 2])
    problem_counts = defaultdict(int)
    for d in sampled:
        pid = d.get(problem_id_key, 'unknown')
        problem_counts[pid] += 1
    
    counts = sorted(problem_counts.values(), reverse=True)
    n_problems = len(counts)
    
    ax3.plot(range(n_problems), counts, 'b-', linewidth=1.5)
    ax3.axhline(y=np.mean(counts), color='red', linestyle='--', 
                label=f'Mean: {np.mean(counts):.1f}', linewidth=1.5)
    ax3.set_xlabel('Problem ID (sorted)', fontsize=10)
    ax3.set_ylabel('Sample Count', fontsize=10)
    ax3.set_title(f'Problem ID Distribution (n={n_problems}, std={np.std(counts):.2f})',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # 第2行: CDF 对比与散点图
    # ============================================================
    
    # [1, 0] 左中：Original vs Sampled CDF
    ax4 = fig.add_subplot(gs[1, 0])
    
    # 使用完整原始数据（包含 speedup >= 1.0）用于 CDF 显示
    if raw_data_for_cdf is not None:
        orig_speedups = []
        for d in raw_data_for_cdf:
            if '_speedup' in d:
                orig_speedups.append(d['_speedup'])
            elif 'speedup' in d:
                orig_speedups.append(d['speedup'])
            else:
                try:
                    t0, t1 = float(d.get(time_v0_key, 0)), float(d.get(time_v1_key, 0))
                    if t1 > 0:
                        orig_speedups.append(t0 / t1)
                except (ValueError, TypeError):
                    pass
        orig_speedups = sorted([s for s in orig_speedups if s >= 1.0])
    else:
        orig_speedups = sorted([d['_speedup'] for d in all_data])
    orig_n = len(orig_speedups)
    
    # Original 数据的 CDF（从 speedup=1.0 开始）
    ax4.plot(orig_speedups, np.arange(1, orig_n+1) / orig_n, 'gray', 
             linewidth=1, alpha=0.7, label=f'Original (n={orig_n})')
    # Sampled 数据的 CDF
    ax4.plot(speedups, np.arange(1, n+1) / n, 'steelblue', 
             linewidth=2, label=f'Sampled (n={n})')
    
    # Ideal 线：从 (1, 0) 开始到最大值，表示理想的 log-uniform 分布
    # 这条线不受采样影响，表示如果 speedup 在 [1, max] 区间上 log-均匀分布的理想情况
    ideal_max = max(max_sp_plot, orig_speedups[-1] if orig_speedups else max_sp_plot)
    ax4.plot([1, ideal_max], [0, 1], 'k--', alpha=0.3, label='Ideal')
    
    ax4.set_xlabel('Speedup', fontsize=10)
    ax4.set_ylabel('Cumulative Probability', fontsize=10)
    ax4.set_title('Original vs Sampled CDF', fontsize=11, fontweight='bold')
    ax4.set_xscale('log', base=2)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    # [1, 1] 中中：CDF (Linear Scale)（改进版：对超大值进行压缩显示）
    ax5 = fig.add_subplot(gs[1, 1])
    
    if has_extreme_speedup:
        # 只显示合理范围内的数据
        filtered_speedups = [s for s in speedups if s <= compress_threshold]
        filtered_n = len(filtered_speedups)
        extreme_count = n - filtered_n
        
        ax5.plot(filtered_speedups, np.arange(1, filtered_n+1) / n, 'steelblue', 
                linewidth=2, label='Sampled (visible range)')
        
        # Ideal Log-Uniform CDF on Linear Scale (只在可见范围)
        x_ideal = np.linspace(min_sp_plot, min(compress_threshold, max_sp_plot), 200)
        denom = np.log2(max_sp_plot) - np.log2(min_sp_plot)
        if denom > 1e-9:
            y_ideal = (np.log2(x_ideal) - np.log2(min_sp_plot)) / denom
            ax5.plot(x_ideal, y_ideal, 'r--', linewidth=1.5, alpha=0.6, label='Ideal (Log-Uniform)')
        
        # 添加超大值提示
        if extreme_count > 0:
            ax5.text(0.98, 0.5, f'{extreme_count} outliers\n>{compress_threshold}× hidden', 
                    transform=ax5.transAxes, ha='right', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=8)
        
        ax5.set_xlim(1, compress_threshold)
    else:
        ax5.plot(speedups, np.arange(1, n+1) / n, 'steelblue', linewidth=2, label='Sampled')
        
        # Ideal Log-Uniform CDF on Linear Scale
        x_ideal = np.linspace(min_sp_plot, max_sp_plot, 200)
        denom = np.log2(max_sp_plot) - np.log2(min_sp_plot)
        if denom > 1e-9:
            y_ideal = (np.log2(x_ideal) - np.log2(min_sp_plot)) / denom
            ax5.plot(x_ideal, y_ideal, 'r--', linewidth=1.5, alpha=0.6, label='Ideal (Log-Uniform)')
    
    ax5.set_xlabel('Speedup (Linear Scale)', fontsize=10)
    ax5.set_ylabel('Cumulative Probability', fontsize=10)
    ax5.set_title('CDF (Linear Scale)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='lower right', fontsize=8)
    
    # [1, 2] 右中：Original Data Scatter（改进版：支持超大值压缩显示）
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 为原始数据生成散点图
    orig_scatter_x = []
    orig_scatter_y = []
    orig_scatter_colors = []
    rng_plot = np.random.default_rng(42)
    
    for i in range(num_bins):
        bin_items = [d for d in all_data if d.get('_bin_idx', -1) == i]
        if not bin_items: continue
        
        for item in bin_items:
            sp = item.get('_speedup', 1)
            jitter = rng_plot.uniform(-0.35, 0.35)
            orig_scatter_x.append(sp)
            orig_scatter_y.append(i + jitter)
            orig_scatter_colors.append(colors[i % len(colors)])
    
    if orig_scatter_x:
        ax6.scatter(orig_scatter_x, orig_scatter_y, s=3, c=orig_scatter_colors, 
                   alpha=0.3, edgecolors='none')
    
    ax6.set_yticks(range(num_bins))
    ax6.set_yticklabels(stats['bins'], fontsize=8)
    ax6.set_ylabel('Speedup Bin', fontsize=10)
    ax6.set_xlabel('Speedup', fontsize=10)
    ax6.set_title(f'Original Data (n={orig_n})', fontsize=11, fontweight='bold')
    ax6.set_xscale('log', base=2)
    
    # 如果有超大speedup值，限制x轴范围并添加标注
    if has_extreme_speedup:
        ax6.set_xlim(1, min(compress_threshold * 2, max_sp_plot))
        ax6.text(0.98, 0.02, f'Max: {max_sp_plot:.0f}×', 
                transform=ax6.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
    
    ax6.grid(True, alpha=0.3, which='both')
    
    for i in range(num_bins - 1):
        ax6.axhline(y=i + 0.5, color='gray', linestyle='--', alpha=0.2)
    
    # ============================================================
    # 第3行: 采样散点图
    # ============================================================
    
    # [2, 0] 左下：Sampled Data Scatter（改进版：支持超大值压缩显示）
    ax7 = fig.add_subplot(gs[2, 0])
    
    sampled_scatter_x = []
    sampled_scatter_y = []
    sampled_scatter_colors = []
    rng_plot2 = np.random.default_rng(42)
    
    for i in range(num_bins):
        bin_items = [d for d in sampled if d.get('_bin_idx', -1) == i]
        if not bin_items: continue
        
        for item in bin_items:
            sp = item.get('_speedup', item.get('speedup', 1))
            jitter = rng_plot2.uniform(-0.35, 0.35)
            sampled_scatter_x.append(sp)
            sampled_scatter_y.append(i + jitter)
            sampled_scatter_colors.append(colors[i % len(colors)])
    
    if sampled_scatter_x:
        ax7.scatter(sampled_scatter_x, sampled_scatter_y, s=10, c=sampled_scatter_colors, 
                   alpha=0.6, edgecolors='none')
    
    ax7.set_yticks(range(num_bins))
    ax7.set_yticklabels(stats['bins'], fontsize=8)
    ax7.set_ylabel('Speedup Bin', fontsize=10)
    ax7.set_xlabel('Speedup', fontsize=10)
    ax7.set_title(f'Sampled Data (n={n})', fontsize=11, fontweight='bold')
    ax7.set_xscale('log', base=2)
    
    # 如果有超大speedup值，限制x轴范围并添加标注
    if has_extreme_speedup:
        ax7.set_xlim(1, min(compress_threshold * 2, max_sp_plot))
        ax7.text(0.98, 0.02, f'Max: {max_sp_plot:.0f}×', 
                transform=ax7.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
    
    ax7.grid(True, alpha=0.3, which='both')
    
    for i in range(num_bins - 1):
        ax7.axhline(y=i + 0.5, color='gray', linestyle='--', alpha=0.2)
    
    plt.savefig(output_path, format=fig_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Speedup分布图已保存: {output_path}")


def plot_problem_id_distribution(sampled: List[Dict], 
                                  problem_id_key: str,
                                  bins: List[float],
                                  output_path: str,
                                  fig_format: str,
                                  dpi: int):
    """绘制问题ID分布图 (已整合到主分布图中，此函数保留以防万一)"""
    
    print(f"问题ID分布已整合到主分布图中，跳过单独绘制。")
    # 如需单独绘制，取消下方注释
    # problem_counts = defaultdict(int)
    # for d in sampled:
    #     pid = d.get(problem_id_key, 'unknown')
    #     problem_counts[pid] += 1
    # 
    # counts = sorted(problem_counts.values(), reverse=True)
    # n_problems = len(counts)
    # 
    # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # ax.plot(range(n_problems), counts, 'b-', linewidth=1)
    # ax.axhline(y=np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    # ax.set_xlabel('Problem ID (sorted by count)', fontsize=11)
    # ax.set_ylabel('Sample Count', fontsize=11)
    # ax.set_title(f'Global Problem ID Distribution (n={n_problems}, std={np.std(counts):.2f})',
    #              fontsize=12, fontweight='bold')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(output_path, format=fig_format, dpi=dpi, bbox_inches='tight')
    # plt.close()


def plot_problem_id_per_bin(sampled: List[Dict],
                             problem_id_key: str,
                             bins: List[float],
                             output_path: str,
                             fig_format: str,
                             dpi: int):
    """绘制每个bin的问题ID分布图"""
    
    num_bins = len(bins) - 1
    bin_labels = []
    for i in range(num_bins):
        if bins[i+1] == float('inf'):
            bin_labels.append(f"[{bins[i]}x, inf)")
        else:
            bin_labels.append(f"[{bins[i]}x, {bins[i+1]}x)")
    
    fig, axes = plt.subplots(2, (num_bins + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i in range(num_bins):
        ax = axes[i]
        bin_items = [d for d in sampled if d.get('_bin_idx') == i]
        
        if not bin_items:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.set_title(f'{bin_labels[i]}', fontsize=11, fontweight='bold')
            continue
        
        problem_counts = defaultdict(int)
        for d in bin_items:
            pid = d.get(problem_id_key, 'unknown')
            problem_counts[pid] += 1
        
        counts = sorted(problem_counts.values(), reverse=True)
        n_problems = len(counts)
        
        ax.plot(range(n_problems), counts, 'b-', linewidth=1)
        ax.axhline(y=np.mean(counts), color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Problem ID', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{bin_labels[i]} (n={len(bin_items)}, pids={n_problems})', 
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    for i in range(num_bins, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, format=fig_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"各Bin问题ID分布图已保存: {output_path}")


def plot_dedup_comparison(sampled: List[Dict],
                           final_data: List[Dict],
                           replenished: List[Dict],
                           problem_id_key: str,
                           bin_key: str,
                           output_path: str,
                           fig_format: str,
                           dpi: int):
    """绘制去重补充对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：采样 vs 最终 的 problem_id 分布对比
    ax1 = axes[0, 0]
    
    sampled_counts = defaultdict(int)
    for d in sampled:
        pid = d.get(problem_id_key, 'unknown')
        sampled_counts[pid] += 1
    
    final_counts = defaultdict(int)
    for d in final_data:
        pid = d.get(problem_id_key, 'unknown')
        final_counts[pid] += 1
    
    sampled_sorted = sorted(sampled_counts.values(), reverse=True)
    final_sorted = sorted(final_counts.values(), reverse=True)
    
    ax1.plot(range(len(sampled_sorted)), sampled_sorted, 'b-', linewidth=1, alpha=0.7, label=f'Sampled (n={len(sampled)})')
    ax1.plot(range(len(final_sorted)), final_sorted, 'g-', linewidth=1, label=f'Final (n={len(final_data)})')
    ax1.axhline(y=np.mean(final_sorted), color='red', linestyle='--', label=f'Final Mean: {np.mean(final_sorted):.1f}')
    ax1.set_xlabel('Problem ID (sorted by count)', fontsize=11)
    ax1.set_ylabel('Sample Count', fontsize=11)
    ax1.set_title('Problem ID Distribution: Sampled vs Final', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右上：补充数据的 problem_id 分布
    ax2 = axes[0, 1]
    if replenished:
        replenished_counts = defaultdict(int)
        for d in replenished:
            pid = d.get(problem_id_key, 'unknown')
            replenished_counts[pid] += 1
        
        rep_sorted = sorted(replenished_counts.values(), reverse=True)
        ax2.bar(range(len(rep_sorted)), rep_sorted, color='orange', alpha=0.7)
        ax2.axhline(y=np.mean(rep_sorted), color='red', linestyle='--', label=f'Mean: {np.mean(rep_sorted):.1f}')
        ax2.set_xlabel('Problem ID (sorted by count)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title(f'Replenished Problem ID Distribution (n={len(replenished)}, pids={len(replenished_counts)})', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No Replenishment', ha='center', va='center', fontsize=12)
        ax2.set_title('Replenished Problem ID Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 左下：各 bin 的样本数对比
    ax3 = axes[1, 0]
    
    sampled_bin_counts = defaultdict(int)
    final_bin_counts = defaultdict(int)
    
    for d in sampled:
        sbin = d.get(bin_key, d.get('_bin_label', 'unknown'))
        sampled_bin_counts[sbin] += 1
    
    for d in final_data:
        sbin = d.get(bin_key, d.get('_bin_label', 'unknown'))
        final_bin_counts[sbin] += 1
    
    all_bins = sorted(set(sampled_bin_counts.keys()) | set(final_bin_counts.keys()))
    x = np.arange(len(all_bins))
    width = 0.35
    
    sampled_vals = [sampled_bin_counts[b] for b in all_bins]
    final_vals = [final_bin_counts[b] for b in all_bins]
    
    ax3.bar(x - width/2, sampled_vals, width, label='Sampled', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, final_vals, width, label='Final', color='forestgreen', alpha=0.7)
    ax3.set_xlabel('Speedup Bin', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Bin Distribution: Sampled vs Final', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_bins, rotation=30, ha='right', fontsize=9)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 右下：补充数据的 bin 分布
    ax4 = axes[1, 1]
    if replenished:
        rep_bin_counts = defaultdict(int)
        for d in replenished:
            sbin = d.get(bin_key, d.get('_bin_label', 'unknown'))
            rep_bin_counts[sbin] += 1
        
        rep_bins = sorted(rep_bin_counts.keys())
        rep_vals = [rep_bin_counts[b] for b in rep_bins]
        
        ax4.bar(range(len(rep_bins)), rep_vals, color='orange', alpha=0.7)
        ax4.set_xlabel('Speedup Bin', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title(f'Replenished Bin Distribution (n={len(replenished)})', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(rep_bins)))
        ax4.set_xticklabels(rep_bins, rotation=30, ha='right', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No Replenishment', ha='center', va='center', fontsize=12)
        ax4.set_title('Replenished Bin Distribution', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format=fig_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"去重补充对比图已保存: {output_path}")


# ============================================================
# 输出函数
# ============================================================

def save_dataset(data: List[Dict], output_path: str, language: str = None):
    """保存数据集
    
    Args:
        data: 数据列表
        output_path: 输出文件路径
        language: 数据集语言 (如 'cpp', 'python'，会注入到每条数据的 language 字段)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for d in data:
            clean_d = {k: v for k, v in d.items() if not k.startswith('_')}
            # 注入 language 字段（如果提供）
            if language:
                clean_d['language'] = language.lower()
            f.write(json.dumps(clean_d, ensure_ascii=False) + '\n')
    print(f"数据集已保存: {output_path} ({len(data)} 条)")


def generate_report(sampled_count: int, deduped_count: int, final_count: int,
                    removed_count: int, replenished_count: int,
                    stats: Dict, dedup_stats: Dict, output_path: str):
    """生成处理报告"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("PIE Dataset Builder - 处理报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【流程摘要】\n")
        f.write("-" * 70 + "\n")
        f.write(f"1. 采样后数量: {sampled_count}\n")
        f.write(f"2. 去重后数量: {deduped_count}\n")
        f.write(f"3. 补充数量: {replenished_count}\n")
        f.write(f"4. 最终数量: {final_count}\n")
        f.write(f"   去重移除: {removed_count}\n")
        f.write(f"   校验: {sampled_count} - {removed_count} + {replenished_count} = {final_count}\n\n")
        
        f.write("【各Bin统计】\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Bin':<20} {'Original':>10} {'Sampled':>10}\n")
        f.write("-" * 70 + "\n")
        for i, bin_label in enumerate(stats['bins']):
            f.write(f"{bin_label:<20} {stats['original'][i]:>10} {stats['sampled'][i]:>10}\n")
        f.write("-" * 70 + "\n\n")
        
        if dedup_stats:
            f.write("【去重统计】\n")
            f.write("-" * 70 + "\n")
            f.write(f"存在重复的分组数: {dedup_stats.get('groups_with_duplicates', 0)}\n")
            f.write(f"移除的重复数: {dedup_stats.get('total_duplicates_removed', 0)}\n")
            if dedup_stats.get('duplicate_details'):
                f.write("\n详细信息 (top 20):\n")
                for detail in dedup_stats['duplicate_details'][:20]:
                    f.write(f"  [{detail['problem_id']}][{detail['speedup_bin']}]: "
                            f"{detail['original_count']} -> {detail['kept_count']}\n")
            f.write("-" * 70 + "\n")
    
    print(f"报告已保存: {output_path}")


def print_summary(stats: Dict, dedup_stats: Dict, final_count: int,
                  replenished_count: int, distribution: str,
                  balance_problem_id: bool = False, problem_id_key: str = None):
    """打印摘要"""
    print("\n" + "=" * 70)
    print("数据集构建摘要")
    print("=" * 70)
    print(f"分布方式: {distribution}")
    print(f"采样后数量: {sum(stats['sampled'])}")
    if dedup_stats:
        print(f"去重移除: {dedup_stats.get('total_duplicates_removed', 0)}")
    print(f"补充数量: {replenished_count}")
    print(f"最终数量: {final_count}")
    if balance_problem_id:
        print(f"问题ID均衡: 启用 (字段: {problem_id_key})")
    print("-" * 70)
    print(f"{'Bin':<20} {'Original':>10} {'Sampled':>10}")
    print("-" * 70)
    
    for i, bin_label in enumerate(stats['bins']):
        print(f"{bin_label:<20} {stats['original'][i]:>10} {stats['sampled'][i]:>10}")
    
    print("-" * 70)
    print(f"{'Total':<20} {sum(stats['original']):>10} {sum(stats['sampled']):>10}")
    print("=" * 70)


# ============================================================
# 主函数
# ============================================================

def main():
    args = parse_args()
    
    # ========== 设置所有随机种子，确保完全可复现 ==========
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)  # 同时设置 Python 内置的 random
    
    bins = parse_bins(args.bins)
    print(f"分档边界: {bins}")
    print(f"随机种子: {args.seed} (完全可复现)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 1. 加载数据 ====================
    print(f"\n[1/5] 加载数据: {args.input}")
    raw_data = load_data(args.input)
    print(f"原始数据量: {len(raw_data)}")
    
    print("\n计算speedup...")
    data = compute_speedup(raw_data, args.speedup_key, args.time_v0_key, 
                           args.time_v1_key, args.problem_id_key)
    print(f"有效数据量: {len(data)}")
    
    if len(data) == 0:
        print("Error: 无有效数据")
        sys.exit(1)
    
    # ==================== 2. 分档采样 ====================
    print(f"\n[2/5] 分档采样...")
    binned = assign_bins(data, bins)
    
    total_available = sum(len(binned[i]) for i in range(len(bins) - 1))
    actual_total = min(args.total, total_available)
    
    print(f"  目标数量: {args.total}")
    print(f"  可用数量: {total_available}")
    print(f"  实际采样: {actual_total}")
    print(f"  分布方式: {args.distribution}")
    print(f"  问题ID均衡: {args.balance_problem_id}")
    
    sampled, stats, global_problem_counts = stratified_sample(
        binned, actual_total, bins, args.distribution, rng,
        balance_problem_id=args.balance_problem_id,
        problem_id_key=args.problem_id_key,
        max_pairs_per_problem=args.max_pairs_per_problem,
        num_sub_bins=args.sub_bins
    )
    
    sampled_count = len(sampled)
    print(f"  采样完成: {sampled_count} 条")
    
    # ==================== 3. 去重 ====================
    dedup_stats = None
    removed_items = []
    
    if not args.no_dedup:
        print(f"\n[3/5] 去重...")
        print(f"  方法: {args.dedup_method}")
        print(f"  阈值: {args.dedup_threshold}")
        print(f"  比较模式: {args.compare_mode}")
        print(f"  保留策略: {args.keep_strategy}")
        
        deduped, dedup_stats = deduplicate_dataset(
            sampled,
            args.problem_id_key,
            '_bin_label',
            args.src_code_key,
            args.tgt_code_key,
            args.dedup_threshold,
            args.dedup_method,
            args.keep_strategy,
            args.compare_mode,
            args.verbose,
            rng  # 传入 rng 以保证可复现
        )
        
        removed_items = dedup_stats.get('removed_items', [])
        print(f"  去重移除: {dedup_stats['total_duplicates_removed']} 条")
        print(f"  去重后: {len(deduped)} 条")
        
        # 更新 global_problem_counts（去重后的计数）
        global_problem_counts = defaultdict(int)
        for item in deduped:
            pid = item.get(args.problem_id_key, 'unknown')
            global_problem_counts[pid] += 1
    else:
        print(f"\n[3/5] 跳过去重")
        deduped = sampled
    
    # ==================== 4. 补充 ====================
    replenished = []
    if not args.no_replenish:
        print(f"\n[4/5] 补充...")
        
        final_data, replenished = replenish_dataset(
            deduped, removed_items, data, args.total,
            bins, args.problem_id_key, '_bin_label',
            args.src_code_key, args.tgt_code_key, rng,
            balance_problem_id=args.balance_problem_id,  # 传递均衡参数
            global_problem_counts=global_problem_counts,  # 传递当前计数
            max_pairs_per_problem=args.max_pairs_per_problem  # 传递上限
        )
    else:
        print(f"\n[4/5] 跳过补充")
        final_data = deduped
    
    print(f"  最终数量: {len(final_data)} 条")
    
    # 更新stats中的sampled为最终数据的分布
    final_stats = {'bins': stats['bins'], 'original': stats['original'], 'sampled': [], 'target': stats['target']}
    for i in range(len(bins) - 1):
        count = len([d for d in final_data if d.get('_bin_idx', d.get('speedup_bin_idx', -1)) == i])
        final_stats['sampled'].append(count)
    
    # ==================== 5. 保存和输出 ====================
    print(f"\n[5/5] 保存结果...")
    output_base = f"{args.output_name}_seed{args.seed}"
    dataset_path = output_dir / f"{output_base}.jsonl"
    save_dataset(final_data, str(dataset_path), language=args.language)
    
    report_path = output_dir / f"{output_base}_report.txt"
    generate_report(
        sampled_count, len(deduped), len(final_data),
        dedup_stats['total_duplicates_removed'] if dedup_stats else 0,
        len(replenished), stats, dedup_stats, str(report_path)
    )
    
    print_summary(final_stats, dedup_stats, len(final_data), len(replenished),
                  args.distribution, args.balance_problem_id, args.problem_id_key)
    
    # ==================== 生成图表（基于最终数据） ====================
    if not args.no_plot:
        print("\n生成分布图（基于最终数据集）...")
        
        # 用户要求：相当于对于新数据集在输出一次只不过名字加了个_dedup后缀
        suffix = "_dedup"
        
        speedup_plot_path = output_dir / f"{output_base}{suffix}_speedup.{args.fig_format}"
        plot_speedup_distribution(final_data, final_stats, bins, data, str(speedup_plot_path),
                                   args.fig_format, args.fig_dpi, args.problem_id_key,
                                   raw_data_for_cdf=raw_data,
                                   time_v0_key=args.time_v0_key,
                                   time_v1_key=args.time_v1_key)
        
        if args.balance_problem_id:
            problem_plot_path = output_dir / f"{output_base}{suffix}_problem_id.{args.fig_format}"
            plot_problem_id_distribution(final_data, args.problem_id_key, bins,
                                          str(problem_plot_path), args.fig_format, args.fig_dpi)
            
            bin_problem_plot_path = output_dir / f"{output_base}{suffix}_problem_per_bin.{args.fig_format}"
            plot_problem_id_per_bin(final_data, args.problem_id_key, bins,
                                     str(bin_problem_plot_path), args.fig_format, args.fig_dpi)
    
    print(f"\n✓ 完成! (seed={args.seed})")


if __name__ == '__main__':
    main()