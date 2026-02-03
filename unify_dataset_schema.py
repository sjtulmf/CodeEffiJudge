#!/usr/bin/env python3
"""
统一数据集Schema工具
将不同来源的数据集统一为相同的字段格式

支持的输入格式:
1. PIE 原始格式 (src_code/tgt_code)
2. Sampled 格式 (slow_code/fast_code)  
3. Java benchmark 格式 (单solution, 需要配对)

统一后的输出格式:
{
    "pair_id": str,           # 配对唯一标识
    "problem_id": str,        # 问题ID
    "language": str,          # 编程语言 (cpp/python/java)
    "slow_code": str,         # 慢速代码
    "fast_code": str,         # 快速代码
    "slow_time": float,       # 慢速代码执行时间
    "fast_time": float,       # 快速代码执行时间
    "speedup": float,         # 加速比 (slow_time / fast_time)
    "slow_submission_id": str,# 慢速提交ID (可选)
    "fast_submission_id": str,# 快速提交ID (可选)
    "source": str             # 数据来源 (pie/sampled/java_benchmark)
}

Usage:
    # 转换 PIE 原始格式
    python unify_dataset_schema.py -i pie_data.jsonl -o unified.jsonl --format pie --language cpp
    
    # 转换 Sampled 格式 (已有统一字段)
    python unify_dataset_schema.py -i sampled.jsonl -o unified.jsonl --format sampled
    
    # 转换 Java benchmark 格式 (需要配对)
    python unify_dataset_schema.py -i java_results.jsonl -o unified.jsonl --format java --min-speedup 1.5
    
    # 合并多个数据集
    python unify_dataset_schema.py -i cpp.jsonl python.jsonl java.jsonl -o merged.jsonl --merge
"""

import argparse
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 统一的输出Schema
# ============================================================

UNIFIED_SCHEMA = {
    "required": [
        "pair_id",
        "problem_id", 
        "language",
        "slow_code",
        "fast_code",
        "slow_time",
        "fast_time",
        "speedup"
    ],
    "optional": [
        "slow_submission_id",
        "fast_submission_id",
        "source",
        "slow_tier",
        "fast_tier"
    ]
}


def create_unified_record(
    pair_id: str,
    problem_id: str,
    language: str,
    slow_code: str,
    fast_code: str,
    slow_time: float,
    fast_time: float,
    speedup: float = None,
    slow_submission_id: str = None,
    fast_submission_id: str = None,
    source: str = None,
    **extra_fields
) -> Dict[str, Any]:
    """创建统一格式的记录"""
    if speedup is None and fast_time > 0:
        speedup = slow_time / fast_time
    
    record = {
        "pair_id": pair_id,
        "problem_id": problem_id,
        "language": language,
        "slow_code": slow_code,
        "fast_code": fast_code,
        "slow_time": round(slow_time, 6) if slow_time else 0.0,
        "fast_time": round(fast_time, 6) if fast_time else 0.0,
        "speedup": round(speedup, 4) if speedup else 0.0,
        "source": source or "unknown"
    }
    
    # 添加可选字段
    if slow_submission_id:
        record["slow_submission_id"] = slow_submission_id
    if fast_submission_id:
        record["fast_submission_id"] = fast_submission_id
    
    # 添加额外字段
    for key, value in extra_fields.items():
        if key not in record and value is not None:
            record[key] = value
    
    return record


# ============================================================
# 格式转换器
# ============================================================

class PIEConverter:
    """PIE 原始格式转换器 (src_code/tgt_code -> slow_code/fast_code)"""
    
    def __init__(self, language: str = "cpp"):
        self.language = language
        self.pair_counter = 0
    
    def convert(self, record: Dict) -> Optional[Dict]:
        """转换单条记录"""
        self.pair_counter += 1
        
        # PIE 格式: src = slow, tgt = fast
        slow_code = record.get("src_code", "")
        fast_code = record.get("tgt_code", "")
        slow_time = record.get("src_agg_runtime", 0)
        fast_time = record.get("tgt_agg_runtime", 0)
        
        if not slow_code or not fast_code:
            logger.warning(f"跳过空代码记录")
            return None
        
        if slow_time <= 0 or fast_time <= 0:
            logger.warning(f"跳过时间为0的记录")
            return None
        
        return create_unified_record(
            pair_id=f"pie_{self.pair_counter}",
            problem_id=record.get("problem_id", f"unknown_{self.pair_counter}"),
            language=self.language,
            slow_code=slow_code,
            fast_code=fast_code,
            slow_time=slow_time,
            fast_time=fast_time,
            speedup=record.get("speedup"),
            slow_submission_id=record.get("src_id"),
            fast_submission_id=record.get("tgt_id"),
            source="pie"
        )


class SampledConverter:
    """Sampled 格式转换器 (已有 slow_code/fast_code)"""
    
    def __init__(self):
        self.pair_counter = 0
    
    def convert(self, record: Dict) -> Optional[Dict]:
        """转换单条记录"""
        self.pair_counter += 1
        
        slow_code = record.get("slow_code", "")
        fast_code = record.get("fast_code", "")
        slow_time = record.get("slow_time", 0)
        fast_time = record.get("fast_time", 0)
        
        if not slow_code or not fast_code:
            return None
        
        return create_unified_record(
            pair_id=record.get("pair_id", f"sampled_{self.pair_counter}"),
            problem_id=record.get("problem_id", "unknown"),
            language=record.get("language", "unknown"),
            slow_code=slow_code,
            fast_code=fast_code,
            slow_time=slow_time,
            fast_time=fast_time,
            speedup=record.get("speedup"),
            slow_submission_id=record.get("slow_submission_id"),
            fast_submission_id=record.get("fast_submission_id"),
            source="sampled",
            slow_tier=record.get("slow_tier"),
            fast_tier=record.get("fast_tier")
        )


class JavaBenchmarkConverter:
    """Java Benchmark 格式转换器 (单solution -> 配对)"""
    
    def __init__(self, min_speedup: float = 1.0, metric_key: str = "avg_time_ns"):
        self.min_speedup = min_speedup
        self.metric_key = metric_key
        self.pair_counter = 0
    
    def convert_batch(self, records: List[Dict]) -> List[Dict]:
        """将多条单solution记录配对"""
        # 按 problem_id 分组
        by_problem = defaultdict(list)
        for r in records:
            if r.get("status") == "success" and r.get(self.metric_key, 0) > 0:
                by_problem[r.get("problem_id", "unknown")].append(r)
        
        pairs = []
        for problem_id, solutions in by_problem.items():
            # 按时间排序 (快 -> 慢)
            solutions.sort(key=lambda x: x.get(self.metric_key, float('inf')))
            
            if len(solutions) < 2:
                continue
            
            # 构造所有满足条件的配对
            for i, fast_sol in enumerate(solutions):
                for slow_sol in solutions[i+1:]:
                    fast_time = fast_sol.get(self.metric_key, 0)
                    slow_time = slow_sol.get(self.metric_key, 0)
                    
                    if fast_time <= 0:
                        continue
                    
                    speedup = slow_time / fast_time
                    if speedup < self.min_speedup:
                        continue
                    
                    self.pair_counter += 1
                    pair = create_unified_record(
                        pair_id=f"java_{self.pair_counter}",
                        problem_id=problem_id,
                        language="java",
                        slow_code=slow_sol.get("code", ""),
                        fast_code=fast_sol.get("code", ""),
                        slow_time=slow_time,  # 保持 ns 单位
                        fast_time=fast_time,
                        speedup=speedup,
                        slow_submission_id=slow_sol.get("solution_id"),
                        fast_submission_id=fast_sol.get("solution_id"),
                        source="java_benchmark"
                    )
                    pairs.append(pair)
        
        return pairs


class PythonBenchmarkConverter:
    """Python Benchmark 格式转换器 (problem级别 with results数组 -> 配对)"""
    
    def __init__(self, min_speedup: float = 1.0, metric_key: str = "avg_cpu_instruction"):
        self.min_speedup = min_speedup
        self.metric_key = metric_key
        self.pair_counter = 0
    
    def convert_batch(self, records: List[Dict]) -> List[Dict]:
        """将problem级别记录转换为配对"""
        pairs = []
        
        for problem_record in records:
            problem_id = problem_record.get("problem_id", "unknown")
            results = problem_record.get("results", [])
            
            # 过滤有效的solution
            valid_solutions = []
            for r in results:
                if r.get("status") == "success" and r.get(self.metric_key, 0) > 0:
                    valid_solutions.append(r)
            
            if len(valid_solutions) < 2:
                continue
            
            # 按指标排序 (快 -> 慢，CPU指令数越少越快)
            valid_solutions.sort(key=lambda x: x.get(self.metric_key, float('inf')))
            
            # 构造所有满足条件的配对
            for i, fast_sol in enumerate(valid_solutions):
                for slow_sol in valid_solutions[i+1:]:
                    fast_metric = fast_sol.get(self.metric_key, 0)
                    slow_metric = slow_sol.get(self.metric_key, 0)
                    
                    if fast_metric <= 0:
                        continue
                    
                    speedup = slow_metric / fast_metric
                    if speedup < self.min_speedup:
                        continue
                    
                    self.pair_counter += 1
                    pair = create_unified_record(
                        pair_id=f"python_{self.pair_counter}",
                        problem_id=str(problem_id),
                        language="python",
                        slow_code=slow_sol.get("code", ""),
                        fast_code=fast_sol.get("code", ""),
                        slow_time=slow_metric,  # CPU instructions
                        fast_time=fast_metric,
                        speedup=speedup,
                        slow_submission_id=f"sol_{slow_sol.get('solution_idx', '')}",
                        fast_submission_id=f"sol_{fast_sol.get('solution_idx', '')}",
                        source="python_benchmark"
                    )
                    pairs.append(pair)
        
        return pairs


# ============================================================
# 主函数
# ============================================================

def detect_format(record: Dict) -> str:
    """自动检测数据格式"""
    if "src_code" in record and "tgt_code" in record:
        return "pie"
    elif "slow_code" in record and "fast_code" in record:
        return "sampled"
    elif "code" in record and "avg_time_ns" in record:
        return "java"
    elif "results" in record and isinstance(record.get("results"), list):
        # Python benchmark 格式: problem 级别，包含 results 数组
        return "python"
    else:
        return "unknown"


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"{path}:{line_num} 解析失败: {e}")
    return records


def save_jsonl(records: List[Dict], path: str):
    """保存 JSONL 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    logger.info(f"保存 {len(records)} 条记录到 {path}")


def convert_file(
    input_paths: List[str],  # 改为支持列表
    output_path: str,
    format_type: str = "auto",
    language: str = None,
    min_speedup: float = 1.0
) -> int:
    """转换单个或多个输入文件"""
    records = []
    for path in input_paths:
        logger.info(f"正在加载: {path}")
        records.extend(load_jsonl(path))
        
    if not records:
        logger.warning(f"输入文件均为空")
        return 0
    
    # 自动检测格式
    if format_type == "auto":
        format_type = detect_format(records[0])
        logger.info(f"检测到格式: {format_type}")
    
    # 选择转换器
    unified = []
    if format_type == "pie":
        lang = language or "cpp"
        converter = PIEConverter(language=lang)
        for r in records:
            converted = converter.convert(r)
            if converted:
                unified.append(converted)
    
    elif format_type == "sampled":
        converter = SampledConverter()
        for r in records:
            converted = converter.convert(r)
            if converted:
                unified.append(converted)
    
    elif format_type == "java":
        converter = JavaBenchmarkConverter(min_speedup=min_speedup)
        unified = converter.convert_batch(records)
    
    elif format_type == "python":
        converter = PythonBenchmarkConverter(min_speedup=min_speedup)
        unified = converter.convert_batch(records)
    
    else:
        logger.error(f"未知格式: {format_type}")
        return 0
    
    save_jsonl(unified, output_path)
    return len(unified)


def merge_files(input_paths: List[str], output_path: str):
    """合并多个已统一的文件"""
    all_records = []
    for path in input_paths:
        records = load_jsonl(path)
        logger.info(f"加载 {len(records)} 条记录从 {path}")
        all_records.extend(records)
    
    # 重新分配 pair_id 确保唯一
    for i, r in enumerate(all_records, 1):
        lang = r.get("language", "unknown")
        r["pair_id"] = f"{lang}_{i}"
    
    save_jsonl(all_records, output_path)
    return len(all_records)


def print_schema_comparison(records: List[Dict], name: str = "Dataset"):
    """打印数据集schema信息"""
    if not records:
        print(f"{name}: 空数据集")
        return
    
    sample = records[0]
    print(f"\n{'='*60}")
    print(f"{name} Schema Analysis ({len(records)} records)")
    print(f"{'='*60}")
    
    print("\n字段列表:")
    for key in sorted(sample.keys()):
        value = sample[key]
        value_type = type(value).__name__
        preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        print(f"  - {key}: {value_type} = {preview}")
    
    # 检查必需字段
    print("\n统一Schema检查:")
    for field in UNIFIED_SCHEMA["required"]:
        status = "✓" if field in sample else "✗"
        print(f"  {status} {field}")


def main():
    parser = argparse.ArgumentParser(
        description='统一数据集Schema工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换 PIE 格式 (cpp)
  python unify_dataset_schema.py -i pie_train.jsonl -o unified_cpp.jsonl --format pie --language cpp
  
  # 转换 PIE 格式 (python)
  python unify_dataset_schema.py -i pie_train.jsonl -o unified_python.jsonl --format pie --language python
  
  # 转换 Sampled 格式
  python unify_dataset_schema.py -i cross_tier_pairs.jsonl -o unified.jsonl --format sampled
  
  # 转换 Java benchmark 格式
  python unify_dataset_schema.py -i java_results.jsonl -o unified_java.jsonl --format java --min-speedup 1.5
  
  # 合并多个数据集
  python unify_dataset_schema.py --merge -i cpp.jsonl python.jsonl java.jsonl -o merged.jsonl
  
  # 分析数据集schema
  python unify_dataset_schema.py --analyze -i data.jsonl
        """
    )
    
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='输入文件路径 (支持多个文件用于合并)')
    parser.add_argument('-o', '--output', 
                        help='输出文件路径')
    parser.add_argument('--format', choices=['auto', 'pie', 'sampled', 'java', 'python'],
                        default='auto', help='输入数据格式 (default: auto)')
    parser.add_argument('--language', choices=['cpp', 'python', 'java'],
                        help='编程语言 (用于 PIE 格式)')
    parser.add_argument('--min-speedup', type=float, default=1.0,
                        help='最小加速比阈值 (default: 1.0)')
    parser.add_argument('--merge', action='store_true',
                        help='合并多个已统一的数据集')
    parser.add_argument('--analyze', action='store_true',
                        help='分析数据集schema (不转换)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细信息')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 分析模式
    if args.analyze:
        for path in args.input:
            records = load_jsonl(path)
            print_schema_comparison(records, Path(path).name)
        return
    
    # 检查输出路径
    if not args.output:
        parser.error("需要指定输出路径 -o/--output")
    
    # 合并模式
    if args.merge:
        total = merge_files(args.input, args.output)
        print(f"\n合并完成: {total} 条记录")
        return
    
    # 转换模式
    total = convert_file(
        args.input,
        args.output,
        format_type=args.format,
        language=args.language,
        min_speedup=args.min_speedup
    )
    
    print(f"\n转换完成: {total} 条记录")
    
    # 显示转换后的schema
    if total > 0:
        records = load_jsonl(args.output)
        print_schema_comparison(records[:1], "转换后")


if __name__ == "__main__":
    main()
