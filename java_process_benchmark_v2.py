#!/usr/bin/env python3
"""
Java 进程级基准测试器 V2

适配原始数据集格式（problem_id, solutions[], testcases[]），
类似 java_benchmark.py 的处理流程，但使用简单的进程级测量替代 JMH。

与 coffe_benchmark.py 的对齐：
    - 进程隔离：每次测量都是独立 JVM 进程
    - 多次采样：多次运行取平均值，去除异常值
    - 内部计时：使用 System.nanoTime() 在 Java 代码内部精确计时
    - CPU 绑定：通过 taskset 绑定 CPU 核心减少调度噪声
    - 预热：执行正式测量前先预热 JVM

使用方法:
    python java_process_benchmark_v2.py -i problems.jsonl -o results.jsonl --cpu-core 0
    
    # 多 worker 并行模式
    python java_process_benchmark_v2.py -i problems.jsonl -o results.jsonl --workers 4 --cpu-start 0
"""

import json
import os
import sys
import subprocess
import tempfile
import shutil
import statistics
import argparse
import signal
import time
import hashlib
import logging
import re
import random
import atexit
import gc
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any, Set
from multiprocessing import Process, Queue, Manager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from java_verifier import JavaVerifier, find_common_valid_testcases

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 配置 ====================

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 测量参数
    warmup_runs: int = 3          # 预热运行次数
    measurement_runs: int = 10    # 正式测量次数
    trim_count: int = 2           # 去掉最高/最低各几次
    
    # JVM 参数
    heap_size_gb: int = 2         # 堆大小 (GB)
    use_g1gc: bool = True         # 使用 G1 GC
    disable_tiered: bool = True   # 禁用分层编译
    
    # 超时
    compile_timeout: float = 60.0   # 编译超时 (秒)
    run_timeout: float = 30.0       # 单次运行超时 (秒)
    
    # CPU 绑定
    cpu_core: Optional[int] = None  # 绑定的 CPU 核心
    
    # RSD 阈值
    rsd_threshold: float = 0.15     # 相对标准差阈值 (15%)
    
    # 解法限制
    max_solutions: int = 25         # 每个问题最多测量多少解法
    max_testcases: int = 30         # 每个问题最多使用多少测试用例
    min_solutions: int = 2          # 最少解法数
    min_testcases: int = 2          # 最少测试用例数
    
    def get_jvm_args(self) -> List[str]:
        """生成 JVM 参数"""
        args = [
            f"-Xms{self.heap_size_gb}G",
            f"-Xmx{self.heap_size_gb}G",
            "-XX:+AlwaysPreTouch",
        ]
        if self.use_g1gc:
            args.append("-XX:+UseG1GC")
        if self.disable_tiered:
            args.append("-XX:-TieredCompilation")
        return args


# ==================== 数据类 ====================

@dataclass
class SolutionMetrics:
    """单个解法的测量结果"""
    problem_id: str = ""
    solution_id: str = ""
    code: str = ""
    language: str = "java"
    
    # 时间统计
    total_time_ns: float = 0
    avg_time_ns: float = 0
    rsd_percent: float = 0
    rsd_per_testcase: List[float] = field(default_factory=list)
    
    # 测试用例信息
    testcase_stats: Dict[str, Any] = field(default_factory=dict)
    valid_testcases: int = 0
    total_testcases: int = 0
    
    # 状态
    status: str = ""  # success, compile_failed, runtime_error, unstable, etc.
    is_stable: bool = True  # RSD <= 5% 为稳定
    error_message: str = ""


# ==================== TimingWrapper 模板 ====================

TIMING_WRAPPER_TEMPLATE = '''
import java.io.*;
import java.util.*;

public class TimingWrapper {{
    public static void main(String[] args) throws Exception {{
        if (args.length < 2) {{
            System.err.println("Usage: TimingWrapper <testcase_index> <run_mode>");
            System.exit(1);
        }}
        
        int tcIndex = Integer.parseInt(args[0]);
        int runMode = Integer.parseInt(args[1]);  // 0=warmup, 1=measure
        
        // 读取测试用例
        String testcasesJson = readFile("{testcases_file}");
        List<String> testcases = parseInputs(testcasesJson);
        
        if (tcIndex < 0 || tcIndex >= testcases.size()) {{
            System.out.println("{{");
            System.out.println("  \\"testcase\\": " + tcIndex + ",");
            System.out.println("  \\"error\\": \\"Invalid testcase index\\",");
            System.out.println("  \\"success\\": false");
            System.out.println("}}");
            System.exit(1);
        }}
        
        String input = testcases.get(tcIndex);
        
        // 准备输入流
        InputStream originalIn = System.in;
        PrintStream originalOut = System.out;
        
        try {{
            ByteArrayInputStream bais = new ByteArrayInputStream(input.getBytes("UTF-8"));
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            PrintStream ps = new PrintStream(baos, true, "UTF-8");
            
            System.setIn(bais);
            System.setOut(ps);
            
            // 计时执行
            long startNs = System.nanoTime();
            Solution.main(new String[0]);
            long endNs = System.nanoTime();
            long elapsedNs = endNs - startNs;
            
            // 恢复输出流
            System.setIn(originalIn);
            System.setOut(originalOut);
            
            // 输出 JSON 结果
            System.out.println("{{");
            System.out.println("  \\"testcase\\": " + tcIndex + ",");
            System.out.println("  \\"time_ns\\": " + elapsedNs + ",");
            System.out.println("  \\"success\\": true");
            System.out.println("}}");
            
        }} catch (Exception e) {{
            System.setIn(originalIn);
            System.setOut(originalOut);
            
            String errMsg = e.getMessage();
            if (errMsg == null) errMsg = "null";
            errMsg = errMsg.replace("\\\\", "\\\\\\\\").replace("\\"", "\\\\\\"").replace("\\n", "\\\\n").replace("\\r", "\\\\r");
            
            System.out.println("{{");
            System.out.println("  \\"testcase\\": " + tcIndex + ",");
            System.out.println("  \\"error\\": \\"" + e.getClass().getSimpleName() + ": " + errMsg + "\\",");
            System.out.println("  \\"success\\": false");
            System.out.println("}}");
            System.exit(1);
        }}
    }}
    
    private static String readFile(String path) throws IOException {{
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {{
            String line;
            while ((line = br.readLine()) != null) {{
                sb.append(line).append("\\n");
            }}
        }}
        return sb.toString();
    }}
    
    private static List<String> parseInputs(String json) {{
        List<String> inputs = new ArrayList<>();
        int start = json.indexOf('[');
        int end = json.lastIndexOf(']');
        if (start < 0 || end < 0) return inputs;
        
        String arrayContent = json.substring(start + 1, end);
        
        int depth = 0;
        boolean inString = false;
        boolean escaped = false;
        StringBuilder current = new StringBuilder();
        
        for (int i = 0; i < arrayContent.length(); i++) {{
            char c = arrayContent.charAt(i);
            
            if (escaped) {{
                if (c == 'n') current.append('\\n');
                else if (c == 't') current.append('\\t');
                else if (c == 'r') current.append('\\r');
                else current.append(c);
                escaped = false;
                continue;
            }}
            
            if (c == '\\\\' && inString) {{
                escaped = true;
                continue;
            }}
            
            if (c == '"' && depth == 0) {{
                inString = !inString;
                continue;
            }}
            
            if (inString) {{
                current.append(c);
            }} else {{
                if (c == '[' || c == '{{') depth++;
                else if (c == ']' || c == '}}') depth--;
                else if (c == ',' && depth == 0) {{
                    String val = current.toString().trim();
                    if (!val.isEmpty()) {{
                        inputs.add(val);
                    }}
                    current = new StringBuilder();
                    continue;
                }}
                if (depth > 0 || (c != ' ' && c != '\\n' && c != '\\r' && c != '\\t')) {{
                    current.append(c);
                }}
            }}
        }}
        
        String val = current.toString().trim();
        if (!val.isEmpty()) {{
            inputs.add(val);
        }}
        
        return inputs;
    }}
}}
'''


# ==================== 辅助函数 ====================

def rename_main_class(code: str, new_name: str = "Solution") -> str:
    """将 Java 代码中的主类重命名为指定名称"""
    pattern = r'public\s+class\s+(\w+)'
    match = re.search(pattern, code)
    if match:
        old_name = match.group(1)
        if old_name != new_name:
            code = re.sub(r'\bclass\s+' + old_name + r'\b', f'class {new_name}', code)
    else:
        pattern = r'\bclass\s+(\w+)'
        match = re.search(pattern, code)
        if match:
            old_name = match.group(1)
            if old_name != new_name:
                code = re.sub(r'\bclass\s+' + old_name + r'\b', f'class {new_name}', code)
    return code


def get_available_memory_gb() -> float:
    """获取当前系统可用内存(GB)"""
    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            return mem.available / (1024 ** 3)
        except:
            pass
    return float('inf')


def kill_process_tree(pid: int):
    """杀死进程及其所有子进程"""
    if not HAS_PSUTIL:
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass
        return
    
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        psutil.wait_procs(children + [parent], timeout=5)
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.warning(f"清理进程树失败 (PID={pid}): {e}")


# ==================== 基准测试器 ====================

class JavaProcessBenchmark:
    """Java 进程级基准测试器"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.java_cmd = "java"
        self.javac_cmd = "javac"
        self.verifier = JavaVerifier()
    
    def compile_code(self, work_dir: Path, files: List[str]) -> Tuple[bool, str]:
        """编译 Java 代码"""
        cmd = [self.javac_cmd, "-encoding", "UTF-8"] + files
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=self.config.compile_timeout
            )
            if result.returncode != 0:
                return False, result.stderr[:1000]
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except Exception as e:
            return False, str(e)
    
    def run_single(self, work_dir: Path, testcase_idx: int) -> Tuple[bool, float, str]:
        """
        执行单次测量
        
        Returns:
            (success, time_ns, error_message)
        """
        cmd = [self.java_cmd]
        cmd.extend(self.config.get_jvm_args())
        cmd.extend(["-cp", ".", "TimingWrapper", str(testcase_idx), "1"])
        
        # CPU 绑定
        if self.config.cpu_core is not None:
            cmd = ["taskset", "-c", str(self.config.cpu_core)] + cmd
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=self.config.run_timeout
            )
            
            if result.returncode != 0:
                return False, 0, result.stderr[:500] if result.stderr else "Unknown error"
            
            # 解析输出中的 JSON 结果
            output = result.stdout.strip()
            json_start = output.rfind('{')
            json_end = output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                try:
                    data = json.loads(json_str)
                    if data.get('success', False):
                        return True, data.get('time_ns', 0), ""
                    else:
                        return False, 0, data.get('error', 'Unknown error')
                except json.JSONDecodeError:
                    pass
            
            return False, 0, f"Failed to parse output: {output[:200]}"
            
        except subprocess.TimeoutExpired:
            return False, 0, "Execution timeout"
        except Exception as e:
            return False, 0, str(e)
    
    def measure_testcase(self, work_dir: Path, testcase_idx: int) -> Dict:
        """
        对单个测试用例进行完整测量（预热 + 多次测量）
        """
        # 预热阶段
        for i in range(self.config.warmup_runs):
            success, _, error = self.run_single(work_dir, testcase_idx)
            if not success:
                return {
                    "testcase_idx": testcase_idx,
                    "success": False,
                    "error": f"Warmup failed: {error}"
                }
        
        # 正式测量
        times = []
        for i in range(self.config.measurement_runs):
            success, time_ns, error = self.run_single(work_dir, testcase_idx)
            if success and time_ns > 0:
                times.append(time_ns)
        
        if len(times) < 3:
            return {
                "testcase_idx": testcase_idx,
                "success": False,
                "error": f"Too few valid runs: {len(times)}/{self.config.measurement_runs}"
            }
        
        # 去除异常值
        if len(times) > 2 * self.config.trim_count:
            times_sorted = sorted(times)
            times_trimmed = times_sorted[self.config.trim_count:-self.config.trim_count]
        else:
            times_trimmed = times
        
        # 计算统计量
        mean_ns = statistics.mean(times_trimmed)
        std_ns = statistics.stdev(times_trimmed) if len(times_trimmed) > 1 else 0
        rsd = (std_ns / mean_ns * 100) if mean_ns > 0 else 0
        
        return {
            "testcase_idx": testcase_idx,
            "success": True,
            "mean_time_ns": mean_ns,
            "std_time_ns": std_ns,
            "rsd_percent": rsd,
            "valid_runs": len(times),
            "raw_times_ns": times
        }
    
    def benchmark_solution(
        self,
        solution_code: str,
        testcases: List,
        solution_id: str = ""
    ) -> Dict[str, Any]:
        """
        对一个解法进行完整基准测试
        
        Args:
            solution_code: Java 源代码
            testcases: 测试用例列表，可以是字符串列表或字典列表 [{"input": "...", "expected": ...}]
            solution_id: 解法 ID
        """
        work_dir = Path(tempfile.mkdtemp(prefix="java_bench_"))
        
        try:
            # 1. 准备 Solution.java
            solution_code = rename_main_class(solution_code, "Solution")
            solution_file = work_dir / "Solution.java"
            with open(solution_file, 'w', encoding='utf-8') as f:
                f.write(solution_code)
            
            # 2. 准备测试用例文件 - 提取 input 字段（如果是字典格式）
            tc_inputs = []
            for tc in testcases:
                if isinstance(tc, dict):
                    tc_inputs.append(tc.get('input', ''))
                else:
                    tc_inputs.append(str(tc))
            
            testcases_file = work_dir / "testcases.json"
            with open(testcases_file, 'w', encoding='utf-8') as f:
                json.dump(tc_inputs, f, ensure_ascii=False)
            
            # 3. 生成 TimingWrapper.java
            wrapper_code = TIMING_WRAPPER_TEMPLATE.format(
                testcases_file=str(testcases_file).replace('\\', '/')
            )
            wrapper_file = work_dir / "TimingWrapper.java"
            with open(wrapper_file, 'w', encoding='utf-8') as f:
                f.write(wrapper_code)
            
            # 4. 编译
            success, error = self.compile_code(work_dir, ["Solution.java", "TimingWrapper.java"])
            if not success:
                return {
                    "solution_id": solution_id,
                    "success": False,
                    "status": "compile_failed",
                    "error": error
                }
            
            # 5. 对每个测试用例进行测量
            testcase_results = []
            total_time_ns = 0
            valid_testcases = 0
            rsd_list = []
            
            for tc_idx in range(len(testcases)):
                tc_result = self.measure_testcase(work_dir, tc_idx)
                testcase_results.append(tc_result)
                
                if tc_result.get('success', False):
                    total_time_ns += tc_result.get('mean_time_ns', 0)
                    valid_testcases += 1
                    rsd_list.append(tc_result.get('rsd_percent', 0))
            
            # 6. 汇总结果
            avg_rsd = statistics.mean(rsd_list) if rsd_list else 0
            
            return {
                "solution_id": solution_id,
                "success": valid_testcases > 0,
                "status": "success" if valid_testcases > 0 else "all_testcases_failed",
                "total_time_ns": total_time_ns,
                "avg_time_ns": total_time_ns / valid_testcases if valid_testcases > 0 else 0,
                "avg_rsd_percent": avg_rsd,
                "valid_testcases": valid_testcases,
                "total_testcases": len(testcases),
                "rsd_per_testcase": rsd_list,
                "testcase_stats": testcase_results
            }
            
        finally:
            try:
                shutil.rmtree(work_dir)
            except:
                pass
    
    def measure_problem(
        self,
        problem: Dict,
        completed_pairs: Set[Tuple[str, str]] = None
    ) -> List[SolutionMetrics]:
        """
        测量单个问题的所有解法
        
        与 java_benchmark.py 保持相同的处理流程
        """
        problem_id = problem.get('problem_id', 'unknown')
        solutions = problem.get('solutions', [])
        testcases = problem.get('testcases', [])
        
        logger.info(f"问题: {problem_id}, 解法数: {len(solutions)}, 测试用例数: {len(testcases)}")
        
        if len(solutions) < self.config.min_solutions:
            logger.warning(f"解法数不足 ({len(solutions)} < {self.config.min_solutions})，跳过")
            return []
        
        if len(testcases) < self.config.min_testcases:
            logger.warning(f"测试用例数不足 ({len(testcases)} < {self.config.min_testcases})，跳过")
            return []
        
        completed_pairs = completed_pairs or set()
        
        # 确定性选择（用 problem_id 作为种子）
        seed = int(hashlib.md5(problem_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        # 确定性选择测试用例
        if len(testcases) > self.config.max_testcases:
            tc_indices = list(range(len(testcases)))
            rng.shuffle(tc_indices)
            selected_tc_indices = sorted(tc_indices[:self.config.max_testcases])
            testcases = [testcases[i] for i in selected_tc_indices]
            logger.info(f"  确定性选择测试用例: {len(testcases)} 个")
        
        # 确定性排序解法
        solutions_with_order = [(s, rng.random()) for s in solutions]
        solutions_with_order.sort(key=lambda x: x[1])
        solutions = [s for s, _ in solutions_with_order]
        
        # 限制解法数量
        if self.config.max_solutions and len(solutions) > self.config.max_solutions:
            solutions = solutions[:self.config.max_solutions]
            logger.info(f"  限制解法数: {len(solutions)} 个")
        
        # 过滤已完成的解法
        pending_solutions = []
        skipped_count = 0
        for sol in solutions:
            sol_id = sol.get('id', '')
            if (problem_id, sol_id) in completed_pairs:
                skipped_count += 1
            else:
                pending_solutions.append(sol)
        
        if skipped_count > 0:
            logger.info(f"  跳过已完成: {skipped_count} 个，剩余: {len(pending_solutions)} 个")
        
        if len(pending_solutions) == 0:
            logger.info(f"  所有解法已完成，跳过此问题")
            return []
        
        # 预验证
        logger.info(f"  预验证解法...")
        verification_results = self.verifier.batch_verify_solutions(pending_solutions, testcases)
        
        # 筛选公共有效组合
        valid_tc_indices, valid_sol_indices = find_common_valid_testcases(
            verification_results,
            testcases,
            min_testcases=self.config.min_testcases,
            min_solutions=self.config.min_solutions
        )
        
        if not valid_tc_indices or not valid_sol_indices:
            logger.warning(f"  无有效公共组合，跳过")
            return []
        
        common_testcases = [testcases[i] for i in valid_tc_indices]
        valid_solutions = [pending_solutions[i] for i in valid_sol_indices]
        
        logger.info(f"  筛选后: {len(valid_solutions)} 解法, {len(common_testcases)} 测试用例")
        
        # 测量每个解法
        results = []
        for sol_idx, sol in enumerate(valid_solutions):
            sol_id = sol.get('id', '')
            sol_code = sol.get('code', '')
            
            logger.info(f"  [{sol_idx+1}/{len(valid_solutions)}] 测量解法 {sol_id}")
            
            bench_result = self.benchmark_solution(sol_code, common_testcases, sol_id)
            
            # 判断是否稳定 (RSD <= 5%)
            avg_rsd = bench_result.get('avg_rsd_percent', 0)
            is_stable = avg_rsd <= 5.0
            
            metrics = SolutionMetrics(
                problem_id=problem_id,
                solution_id=sol_id,
                code=sol_code,
                language="java",
                total_time_ns=bench_result.get('total_time_ns', 0),
                avg_time_ns=bench_result.get('avg_time_ns', 0),
                rsd_percent=avg_rsd,
                rsd_per_testcase=bench_result.get('rsd_per_testcase', []),
                testcase_stats=bench_result.get('testcase_stats', {}),
                valid_testcases=bench_result.get('valid_testcases', 0),
                total_testcases=bench_result.get('total_testcases', 0),
                status=bench_result.get('status', 'unknown'),
                is_stable=is_stable,
                error_message=bench_result.get('error', '')
            )
            results.append(metrics)
            
            if metrics.status == "success":
                stable_mark = "✓" if is_stable else "⚠"
                logger.info(f"    {stable_mark} 时间: {metrics.avg_time_ns/1e6:.2f}ms, RSD: {metrics.rsd_percent:.1f}%{'' if is_stable else ' [不稳定]'}")
            else:
                logger.warning(f"    ✗ {metrics.status}: {metrics.error_message[:100]}")
        
        return results


# ==================== 断点续跑 ====================

class ResumeManager:
    """断点续跑管理器"""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.completed_pairs: Set[Tuple[str, str]] = set()
        self.load_completed()
    
    def load_completed(self):
        """加载已完成的 (problem_id, solution_id)"""
        if not os.path.exists(self.output_file):
            return
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        pid = record.get('problem_id', '')
                        sid = record.get('solution_id', '')
                        if pid and sid:
                            self.completed_pairs.add((pid, sid))
            logger.info(f"已加载 {len(self.completed_pairs)} 条已完成记录")
        except Exception as e:
            logger.warning(f"加载断点失败: {e}")
    
    def is_completed(self, problem_id: str, solution_id: str) -> bool:
        return (problem_id, solution_id) in self.completed_pairs
    
    def mark_completed(self, problem_id: str, solution_id: str):
        self.completed_pairs.add((problem_id, solution_id))


# ==================== Worker 函数 ====================

def worker_func(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    config_dict: Dict,
    completed_pairs_list: List,
    cpu_core: Optional[int] = None
):
    """Worker 进程主函数"""
    # 重建配置
    config = BenchmarkConfig(**config_dict)
    config.cpu_core = cpu_core
    
    # 重建 completed_pairs
    completed_pairs = set(tuple(x) for x in completed_pairs_list)
    
    benchmark = JavaProcessBenchmark(config)
    
    while True:
        try:
            task = task_queue.get(timeout=1)
        except:
            continue
        
        if task is None:  # 终止信号
            break
        
        problem_idx, problem = task
        
        try:
            results = benchmark.measure_problem(problem, completed_pairs)
            for r in results:
                result_queue.put(asdict(r))
        except Exception as e:
            logger.error(f"Worker {worker_id}: 处理问题失败: {e}")
            import traceback
            traceback.print_exc()


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description='Java 进程级基准测试器 V2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 输入输出
    parser.add_argument('-i', '--input', required=True,
                        help='输入数据集 (jsonl 格式)')
    parser.add_argument('-o', '--output', required=True,
                        help='输出结果文件 (jsonl 格式)')
    
    # 测量参数
    parser.add_argument('--warmup', type=int, default=3,
                        help='预热运行次数 (default: 3)')
    parser.add_argument('--runs', type=int, default=10,
                        help='正式测量次数 (default: 10)')
    parser.add_argument('--trim', type=int, default=2,
                        help='去掉最高/最低各几次 (default: 2)')
    parser.add_argument('--heap', type=int, default=2,
                        help='JVM 堆大小 GB (default: 2)')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='单次运行超时秒数 (default: 30)')
    
    # CPU 绑定
    parser.add_argument('--cpu-core', type=int, default=None,
                        help='绑定的 CPU 核心 (单核模式)')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行 worker 数 (default: 1)')
    parser.add_argument('--cpu-start', type=int, default=0,
                        help='多 worker 模式起始 CPU 核心')
    
    # 解法限制
    parser.add_argument('--max-solutions', type=int, default=25,
                        help='每个问题最多测量多少解法 (default: 25)')
    parser.add_argument('--max-testcases', type=int, default=30,
                        help='每个问题最多使用多少测试用例 (default: 30)')
    parser.add_argument('--min-testcases', type=int, default=2,
                        help='每个问题最少使用多少测试用例 (default: 2)')
    
    # 其他
    parser.add_argument('--no-resume', action='store_true',
                        help='不使用断点续跑')
    parser.add_argument('--limit', type=int, default=None,
                        help='只处理前 N 个问题')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建配置
    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        measurement_runs=args.runs,
        trim_count=args.trim,
        heap_size_gb=args.heap,
        run_timeout=args.timeout,
        cpu_core=args.cpu_core,
        max_solutions=args.max_solutions,
        max_testcases=args.max_testcases,
        min_testcases=args.min_testcases
    )
    
    # 加载数据集
    problems = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    if args.limit:
        problems = problems[:args.limit]
    
    logger.info(f"加载 {len(problems)} 个问题")
    
    # 断点续跑
    resume_manager = None
    if not args.no_resume:
        resume_manager = ResumeManager(args.output)
    
    # 输出文件 (追加模式)
    output_mode = 'a' if not args.no_resume and os.path.exists(args.output) else 'w'
    
    if args.workers <= 1:
        # 单进程模式
        benchmark = JavaProcessBenchmark(config)
        completed_pairs = resume_manager.completed_pairs if resume_manager else set()
        
        total_results = 0
        skipped_errors = 0
        for prob_idx, problem in enumerate(problems):
            logger.info(f"\n{'='*60}")
            logger.info(f"问题 {prob_idx + 1}/{len(problems)}")
            
            try:
                results = benchmark.measure_problem(problem, completed_pairs)
                
                with open(args.output, 'a', encoding='utf-8') as f:
                    for r in results:
                        # 只保存成功的结果
                        if r.status == "success":
                            f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
                            total_results += 1
                        else:
                            skipped_errors += 1
                            logger.debug(f"跳过错误结果: {r.solution_id} - {r.status}")
                        if resume_manager:
                            resume_manager.mark_completed(r.problem_id, r.solution_id)
                
            except Exception as e:
                logger.error(f"处理问题失败: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n完成: 共 {total_results} 条成功结果, 跳过 {skipped_errors} 条错误")
    
    else:
        # 多进程模式
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        
        # 准备 completed_pairs
        completed_pairs_list = list(
            resume_manager.completed_pairs if resume_manager else []
        )
        
        # 启动 workers
        workers = []
        for i in range(args.workers):
            cpu_core = args.cpu_start + i if args.cpu_start is not None else None
            p = Process(
                target=worker_func,
                args=(i, task_queue, result_queue, asdict(config),
                      completed_pairs_list, cpu_core)
            )
            p.start()
            workers.append(p)
        
        # 放入任务
        for prob_idx, problem in enumerate(problems):
            task_queue.put((prob_idx, problem))
        
        # 放入终止信号
        for _ in workers:
            task_queue.put(None)
        
        # 收集结果
        total_results = 0
        skipped_errors = 0
        finished_workers = 0
        
        with open(args.output, output_mode, encoding='utf-8') as f:
            while finished_workers < args.workers or not result_queue.empty():
                try:
                    result = result_queue.get(timeout=1)
                    # 只保存成功的结果
                    if result.get('status') == 'success':
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        total_results += 1
                    else:
                        skipped_errors += 1
                except:
                    # 检查 workers 状态
                    for p in workers:
                        if not p.is_alive():
                            finished_workers += 1
                            workers.remove(p)
                            break
        
        # 等待所有 workers
        for p in workers:
            p.join()
        
        logger.info(f"\n完成: 共 {total_results} 条成功结果, 跳过 {skipped_errors} 条错误")


if __name__ == '__main__':
    main()
