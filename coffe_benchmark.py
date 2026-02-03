#!/usr/bin/env python3
"""
Mercury/Coffe Python CPU指令数基准测试 
- 与 Coffe 作者保持一致的测量方法
- 每次测量在独立子进程中执行（关键差异）
- 支持两种代码类型：io（stdin/stdout）和函数调用
- 使用 time_limit 进行超时保护
- 两阶段测量：先验证，后测量
- V6 改进：
  - 修复 CPU affinity 继承问题，在 Measurement 进程中显式设置
  - 支持多数据集合并输出到同一文件
  - 添加数据集标签字段
  - 优化日志输出
"""

import json
import os
import sys
import gc
import io
import time
import random
import signal
import argparse
import contextlib
import statistics
import copy
import tempfile
import shutil
import builtins
import platform
import faulthandler
import multiprocessing
from multiprocessing import Process, Queue, Value, Array, Manager
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from io import StringIO
from unittest.mock import patch, mock_open

# cirron 用于测量 CPU 指令数
try:
    from cirron import Collector
    CIRRON_AVAILABLE = True
except ImportError:
    CIRRON_AVAILABLE = False
    print("错误: 请安装 cirron: pip install cirron")
    sys.exit(1)

import numpy as np


# ==================== 配置 ====================
NUM_RUNS = 12           # 每个测试用例运行次数
TRIM_COUNT = 1          # 去掉最高/最低各几次
TIMEOUT_PER_TESTCASE = 10    # 单个测试用例超时(秒)
MAX_TEST_CASES = 50    # 最多使用的测试用例数

# 常用导入
COMMON_IMPORTS = """
import sys
import math
import collections
from collections import defaultdict, Counter, deque, OrderedDict
from typing import List, Optional, Tuple, Dict, Set
from functools import lru_cache, reduce
from itertools import combinations, permutations, product, accumulate
from heapq import heappush, heappop, heapify, heapreplace, nlargest, nsmallest
from bisect import bisect_left, bisect_right, insort_left, insort_right
import string
import re
"""

# 状态常量
SUCCEED = 1
FAILED = -1
TIMEOUT = -2
UNKNOWN = -3

INF = 9999999999999999


# ==================== 来自 Coffe 的辅助类和函数 ====================

class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


class Capturing(list):
    """捕获 stdout 输出"""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda: None  # 修复 Python 3.13 兼容性
        return self
    
    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


@contextlib.contextmanager
def swallow_io():
    """抑制所有 IO"""
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    """超时限制"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    """创建临时目录并切换到该目录"""
    with tempfile.TemporaryDirectory() as dirname:
        cwd = os.getcwd()
        os.chdir(dirname)
        try:
            yield dirname
        finally:
            os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    禁用危险函数，防止代码干扰测试环境
    """
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )
    
    faulthandler.disable()
    
    builtins.exit = None
    builtins.quit = None
    
    os.environ["OMP_NUM_THREADS"] = "1"
    
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.chdir = None
    
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    
    import subprocess
    subprocess.Popen = None
    
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


# ==================== 辅助函数 ====================

def check_perf_available() -> Tuple[bool, str]:
    """检查 perf_event 是否可用"""
    try:
        with open('/proc/sys/kernel/perf_event_paranoid', 'r') as f:
            level = int(f.read().strip())
        if level <= 0:
            return True, f"perf_event_paranoid={level}"
        else:
            return False, f"perf_event_paranoid={level}，需要设置为-1或0"
    except:
        return False, "无法读取perf_event_paranoid"


def format_instructions(n: int) -> str:
    """格式化指令数显示"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(n)


def run_stdin_code(code: str, exec_globals: dict, inputs: str, measure_time: bool = False):
    """运行 stdin 类型的代码"""
    inputs_line_iterator = iter(inputs.split("\n"))
    
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    def inner_call():
        if measure_time:
            start_time = time.time()
        exec(code, exec_globals)
        if measure_time:
            return time.time() - start_time
    
    return inner_call()


def run_stdin_code_for_instr(code: str, exec_globals: dict, inputs: str):
    """运行 stdin 类型代码并测量指令数"""
    inputs_line_iterator = iter(inputs.split("\n"))
    
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    def inner_call():
        with Collector() as collector:
            exec(code, exec_globals)
        return collector.counters.instruction_count
    
    return inner_call()


def detect_code_type(problem: Dict) -> str:
    """
    检测代码类型：'io' (stdin/stdout) 或 'function' (函数调用)
    """
    # 检查是否有 entry_point（通常表示函数调用类型）
    entry_point = problem.get('entry_point', '')
    
    # 检查 solutions
    solutions = problem.get('solutions', [])
    if solutions:
        first_solution = solutions[0] if isinstance(solutions[0], str) else solutions[0].get('code', '')
        
        # 如果代码中有 input() 或 sys.stdin，很可能是 io 类型
        if 'input()' in first_solution or 'sys.stdin' in first_solution:
            return 'io'
        
        # 如果有 def solution( 或 class Solution，是函数类型
        if 'def solution(' in first_solution or 'class Solution' in first_solution:
            return 'function'
    
    # 检查 input_output 格式
    input_output = problem.get('input_output', {})
    if isinstance(input_output, str):
        try:
            input_output = json.loads(input_output)
        except:
            input_output = {}
    
    inputs = input_output.get('inputs', [])
    if inputs and isinstance(inputs[0], str) and '\n' in inputs[0]:
        return 'io'
    
    # 默认使用函数类型
    return 'function' if entry_point else 'io'


def extract_entry_point(prompt: str) -> str:
    """从 prompt 中提取函数名"""
    import re
    # 匹配 def xxx( 的模式
    match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if match:
        return match.group(1)
    return ''


def fix_generator_code(code: str) -> str:
    """
    尝试自动修复生成器代码中的常见错误
    
    Args:
        code: 原始生成器代码
        
    Returns:
        修复后的代码
    """
    # 修复1: 在 random.randint 调用中将普通除法替换为整数除法
    # 这是一个启发式方法：查找 random.randint(...) 中的 / 并替换为 //
    # 注意：这可能会有副作用，但对于随机数生成器来说通常是正确的
    
    import re
    
    # 找到所有 random.randint(...) 调用
    def fix_randint_call(match):
        full_call = match.group(0)
        # 在这个调用内部，将 / 替换为 //（但不替换 //）
        fixed_call = re.sub(r'(?<!/)/(?!/)', r'//', full_call)
        return fixed_call
    
    # 匹配 random.randint(...) 包括嵌套括号
    # 使用递归正则（Python不支持），所以用简单方法
    lines = code.split('\n')
    fixed_lines = []
    
    for line in lines:
        if 'random.randint' in line and ' / ' in line:
            # 简单替换：将 " / " 替换为 " // "
            # 只在包含 random.randint 的行中进行
            line = line.replace(' / ', ' // ')
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def generate_test_cases_from_generator(
    generator_code: str,
    num_cases: int,
    existing_cases: List[Dict],
    timeout: float = 5.0
) -> List[Dict]:
    """
    使用 generator_code 生成测试用例，补充到指定数目
    
    Args:
        generator_code: 生成器代码，包含 generate_test_case() 函数
        num_cases: 目标测试用例数
        existing_cases: 已有的测试用例
        timeout: 生成单个用例的超时时间
    
    Returns:
        合并后的测试用例列表
    """
    cases = list(existing_cases)  # 复制已有用例
    
    if len(cases) >= num_cases:
        return cases[:num_cases]
    
    if not generator_code or generator_code == 'None':
        return cases
    
    # Mercury 格式: input 是参数列表，如 [[matrix]] 或 [num1, num2]
    # generator 生成的是单个参数值，需要包装成 [value]
    # 但要确保格式与已有用例一致
    
    # 自动修复生成器代码中的常见错误
    generator_code = fix_generator_code(generator_code)
    
    # 解析生成器代码
    try:
        exec_globals = {}
        # 添加常用导入
        exec(PRELUDE_CODE, exec_globals)
        random.seed(1024)  # 固定种子保证可复现
        exec(generator_code, exec_globals)
        
        # 查找生成函数
        gen_fn = None
        for name in ['generate_test_case', 'generate_input', 'gen_test_case', 'gen_input']:
            if name in exec_globals and callable(exec_globals[name]):
                gen_fn = exec_globals[name]
                break
        
        if gen_fn is None:
            return cases
        
        # 生成测试用例直到达到目标数目
        max_attempts = num_cases * 3  # 最多尝试次数
        attempts = 0
        failures = 0
        
        while len(cases) < num_cases and attempts < max_attempts:
            attempts += 1
            try:
                with time_limit(timeout):
                    input_val = gen_fn()
                    # Mercury 格式: input 是参数列表
                    # generator 可能返回:
                    # - 单个值: matrix -> 包装成 [matrix]
                    # - 元组: (s, t) -> 转成列表 [s, t]
                    if isinstance(input_val, tuple):
                        # 多参数情况，元组转列表
                        new_case = {'input': list(input_val)}
                    else:
                        # 单参数情况，包装成列表
                        new_case = {'input': [input_val]}
                    cases.append(new_case)
            except Exception as e:
                failures += 1
                # 连续失败太多次则放弃
                if failures > 10:
                    break
                continue
        
    except Exception as e:
        pass
    
    return cases[:num_cases]


def parse_test_cases(problem: Dict, max_cases: int) -> Tuple[List[Dict], str]:
    """
    解析测试用例，返回 (test_cases, code_type)
    支持从已有数据和 generator_code 动态生成
    """
    code_type = detect_code_type(problem)
    cases = []
    
    # 从 input_output 获取测试用例
    input_output = problem.get('input_output', {})
    if isinstance(input_output, str):
        try:
            input_output = json.loads(input_output)
        except:
            input_output = {}
    
    # 兼容两种格式: inputs/outputs (复数) 和 input/output (单数)
    inputs = input_output.get('inputs', []) or input_output.get('input', [])
    outputs = input_output.get('outputs', []) or input_output.get('output', [])
    
    if inputs:
        for i, inp in enumerate(inputs):
            case = {'input': inp}
            if i < len(outputs):
                case['output'] = outputs[i]
            cases.append(case)
    
    # 从 test_cases 获取
    test_cases = problem.get('test_cases', [])
    if isinstance(test_cases, str):
        # Mercury 数据集中 test_cases 可能是 "None" 字符串或 JSON 字符串
        if test_cases == 'None' or test_cases == 'null':
            test_cases = []
        else:
            try:
                test_cases = json.loads(test_cases)
            except:
                test_cases = []
    
    if test_cases:
        for tc in test_cases:
            if tc not in cases:
                cases.append(tc)
    
    # 从 base_input 获取（某些数据集格式）
    base_input = problem.get('base_input', [])
    if base_input:
        for inp in base_input:
            cases.append({'input': inp})
    
    # 如果用例不足，使用 generator_code 动态生成
    generator_code = problem.get('generator_code', '')
    if generator_code and generator_code != 'None' and len(cases) < max_cases:
        cases = generate_test_cases_from_generator(
            generator_code=generator_code,
            num_cases=max_cases,
            existing_cases=cases
        )
    
    return cases[:max_cases], code_type


# ==================== 核心测量函数（与 Coffe 一致） ====================

# 预编译的通用导入
PRELUDE_CODE = """
import sys
import math
import collections
import string
import re
import json
import heapq
import bisect
import functools
import itertools
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from collections import defaultdict, Counter, deque, OrderedDict
from functools import lru_cache, reduce, cmp_to_key, cache
from itertools import combinations, permutations, product, accumulate, chain
from heapq import heappush, heappop, heapify, heapreplace, nlargest, nsmallest
from bisect import bisect_left, bisect_right, insort_left, insort_right
"""


def unsafe_single_execution(
    io_type: bool,
    code: str,
    testcase: Dict,
    time_lmt: float,
    entry_point: str,
    results: list,
    measure_instr: bool = True,
    cpu_core: int = None  # V6新增：显式传递CPU核心
):
    """
    在子进程中执行单次测量
    与 Coffe 的 unsafe_runtime_execute 保持一致
    
    V6改进：显式设置 CPU affinity，确保不发生核心漂移
    """
    # V6: 显式设置 CPU affinity（即使继承了也再设一次，确保万无一失）
    if cpu_core is not None:
        try:
            os.sched_setaffinity(0, {cpu_core})
        except (OSError, AttributeError):
            # Windows 或权限不足时忽略
            pass
    
    sys.set_int_max_str_digits(1000000)
    sys.setrecursionlimit(1000000)
    
    with create_tempdir():
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        
        maximum_memory_bytes = 4 * 1024 * 1024 * 1024
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
        random.seed(1024)
        
        try:
            if io_type:
                # stdin/stdout 类型
                new_code = code.replace("if __name__", "if 1 or __name__")
                
                # 获取输入
                if isinstance(testcase.get('input'), str):
                    testcase_input = testcase['input']
                elif isinstance(testcase.get('input'), list) and len(testcase['input']) > 0:
                    testcase_input = testcase['input'][0] if isinstance(testcase['input'][0], str) else str(testcase['input'][0])
                else:
                    testcase_input = str(testcase.get('input', ''))
                
                exec_globals = {}
                with time_limit(time_lmt):
                    with Capturing() as out:
                        if measure_instr:
                            duration = run_stdin_code_for_instr(new_code, exec_globals, testcase_input)
                        else:
                            duration = run_stdin_code(new_code, exec_globals, testcase_input, measure_time=True)
                results.append(duration)
            else:
                # 函数调用类型
                with swallow_io():
                    # 先执行通用导入
                    exec_globals = {}
                    exec(PRELUDE_CODE, exec_globals)
                    
                    # 获取输入参数
                    testcase_input = testcase.get('input', [])
                    if isinstance(testcase_input, str):
                        try:
                            testcase_input = eval(testcase_input)
                        except:
                            testcase_input = [testcase_input]
                    if not isinstance(testcase_input, (list, tuple)):
                        testcase_input = [testcase_input]
                    
                    exec(code, exec_globals)
                    
                    # 获取函数
                    fn = None
                    if 'Solution' in exec_globals and entry_point:
                        try:
                            sol_obj = exec_globals['Solution']()
                            fn = getattr(sol_obj, entry_point)
                        except:
                            pass
                    elif 'solution' in exec_globals:
                        fn = exec_globals['solution']
                    elif entry_point and entry_point in exec_globals:
                        fn = exec_globals[entry_point]
                    
                    if fn is None:
                        raise RuntimeError("Cannot find entry point function")
                    
                    with time_limit(time_lmt):
                        if measure_instr:
                            with Collector() as collector:
                                fn(*testcase_input)
                            duration = collector.counters.instruction_count
                        else:
                            start_time = time.time()
                            fn(*testcase_input)
                            duration = time.time() - start_time
                results.append(duration)
        except BaseException as e:
            # 异常情况不添加结果
            pass
        
        # 恢复函数
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def execute_single_measurement(
    io_type: bool,
    code: str,
    testcase: Dict,
    time_lmt: float,
    timeout: float,
    entry_point: str,
    measure_instr: bool = True,
    cpu_core: int = None  # V6新增
) -> Optional[float]:
    """
    执行单次测量，在独立子进程中运行
    
    V6改进：传递 cpu_core 给子进程
    """
    manager = multiprocessing.Manager()
    results = manager.list()
    
    p = multiprocessing.Process(
        target=unsafe_single_execution,
        args=(io_type, code, testcase, time_lmt, entry_point, results, measure_instr, cpu_core)
    )
    p.start()
    p.join(timeout=timeout + 1)
    
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)
    
    if len(results) > 0:
        return results[0]
    else:
        return None


def measure_solution_instructions(
    code: str,
    testcases: List[Dict],
    entry_point: str,
    code_type: str,
    num_runs: int = 12,
    time_limit_per_tc: float = 10.0,
    cpu_core: int = None  # V6新增
) -> Tuple[Optional[float], Optional[float], int, str]:
    """
    测量单个解法的 CPU 指令数
    
    返回: (avg_instructions, rsd_percent, valid_tc_count, status)
    """
    io_type = (code_type == 'io')
    timeout = time_limit_per_tc + 2
    
    all_tc_means = []
    all_tc_stds = []
    
    print(f"    测量 {len(testcases)} 个测试用例，每个运行 {num_runs} 次...")
    
    for tc_idx, testcase in enumerate(testcases):
        if tc_idx % 10 == 0:  # 每10个测试用例打印一次
            print(f"      进度: 测试用例 {tc_idx+1}/{len(testcases)}")
        tc_measurements = []
        
        for run in range(num_runs):
            result = execute_single_measurement(
                io_type=io_type,
                code=code,
                testcase=testcase,
                time_lmt=time_limit_per_tc,
                timeout=timeout,
                entry_point=entry_point,
                measure_instr=True,
                cpu_core=cpu_core  # V6: 传递CPU核心
            )
            if result is not None:
                tc_measurements.append(result)
        
        # 处理该测试用例的结果
        # 修复2: 优化去极值逻辑，直接使用切片
        if len(tc_measurements) >= 3:
            tc_measurements.sort()
            tc_measurements = tc_measurements[1:-1]  # 去掉最大最小值，直接切片
            
            tc_mean = statistics.mean(tc_measurements)
            tc_std = statistics.stdev(tc_measurements) if len(tc_measurements) > 1 else 0
            all_tc_means.append(tc_mean)
            all_tc_stds.append(tc_std)
        elif tc_measurements:
            tc_mean = statistics.mean(tc_measurements)
            tc_std = statistics.stdev(tc_measurements) if len(tc_measurements) > 1 else 0
            all_tc_means.append(tc_mean)
            all_tc_stds.append(tc_std)
    
    if not all_tc_means:
        return None, None, 0, 'all_tc_failed'
    
    # 汇总统计
    total_instr = sum(all_tc_means)
    avg_per_tc = statistics.mean(all_tc_means)
    avg_tc_std = statistics.mean(all_tc_stds) if all_tc_stds else 0
    rsd_percent = (avg_tc_std / avg_per_tc * 100) if avg_per_tc > 0 else 0
    
    return int(total_instr), round(rsd_percent, 4), len(all_tc_means), 'success'


def verify_solution(
    code: str,
    testcases: List[Dict],
    entry_point: str,
    code_type: str,
    time_limit_per_tc: float = 5.0,
    cpu_core: int = None  # V6新增
) -> Tuple[List[int], str]:
    """
    验证解法在测试用例上能否成功执行
    
    返回: (valid_tc_indices, status)
    """
    io_type = (code_type == 'io')
    timeout = time_limit_per_tc + 2
    valid_indices = []
    
    # 进度输出已在调用方处理，这里静默执行
    for tc_idx, testcase in enumerate(testcases):
        result = execute_single_measurement(
            io_type=io_type,
            code=code,
            testcase=testcase,
            time_lmt=time_limit_per_tc,
            timeout=timeout,
            entry_point=entry_point,
            measure_instr=False,  # 验证阶段只需要时间
            cpu_core=cpu_core  # V6: 传递CPU核心
        )
        if result is not None:
            valid_indices.append(tc_idx)
    
    status = 'success' if valid_indices else 'no_valid_tc'
    return valid_indices, status


# ==================== 批量验证函数（优化验证阶段效率） ====================

def unsafe_batch_verify(
    io_type: bool,
    code: str,
    testcases: List[Dict],
    time_lmt: float,
    entry_point: str,
    results: list,
    cpu_core: int = None
):
    """
    在单个子进程中批量验证一个解法在所有测试用例上的执行情况
    比每个TC都启动新进程快得多
    
    results: 传入一个共享列表，验证成功的TC索引会被添加到其中
    """
    if cpu_core is not None:
        try:
            os.sched_setaffinity(0, {cpu_core})
        except (OSError, AttributeError):
            pass
    
    sys.set_int_max_str_digits(1000000)
    sys.setrecursionlimit(1000000)
    
    with create_tempdir():
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        
        maximum_memory_bytes = 4 * 1024 * 1024 * 1024
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
        random.seed(1024)
        
        valid_indices = []
        
        try:
            if io_type:
                new_code = code.replace("if __name__", "if 1 or __name__")
                
                for tc_idx, testcase in enumerate(testcases):
                    try:
                        # 获取输入
                        if isinstance(testcase.get('input'), str):
                            testcase_input = testcase['input']
                        elif isinstance(testcase.get('input'), list) and len(testcase['input']) > 0:
                            testcase_input = testcase['input'][0] if isinstance(testcase['input'][0], str) else str(testcase['input'][0])
                        else:
                            testcase_input = str(testcase.get('input', ''))
                        
                        exec_globals = {}
                        with time_limit(time_lmt):
                            with Capturing() as out:
                                run_stdin_code(new_code, exec_globals, testcase_input, measure_time=False)
                        valid_indices.append(tc_idx)
                    except:
                        pass
            else:
                # 函数调用类型 - 只编译一次代码
                exec_globals = {}
                exec(PRELUDE_CODE, exec_globals)
                exec(code, exec_globals)
                
                # 获取函数
                fn = None
                if 'Solution' in exec_globals and entry_point:
                    try:
                        sol_obj = exec_globals['Solution']()
                        fn = getattr(sol_obj, entry_point)
                    except:
                        pass
                elif 'solution' in exec_globals:
                    fn = exec_globals['solution']
                elif entry_point and entry_point in exec_globals:
                    fn = exec_globals[entry_point]
                
                if fn is None:
                    # 无法找到函数，所有TC都失败
                    results.extend([])
                    return
                
                for tc_idx, testcase in enumerate(testcases):
                    try:
                        # 获取输入参数
                        testcase_input = testcase.get('input', [])
                        if isinstance(testcase_input, str):
                            try:
                                testcase_input = eval(testcase_input)
                            except:
                                testcase_input = [testcase_input]
                        if not isinstance(testcase_input, (list, tuple)):
                            testcase_input = [testcase_input]
                        
                        with swallow_io():
                            with time_limit(time_lmt):
                                fn(*testcase_input)
                        valid_indices.append(tc_idx)
                    except:
                        pass
        except:
            pass
        
        # 恢复函数
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        
        # 将结果写入共享列表
        results.extend(valid_indices)


def batch_verify_solution(
    code: str,
    testcases: List[Dict],
    entry_point: str,
    code_type: str,
    time_limit_per_tc: float = 5.0,
    total_timeout: float = 300.0,
    cpu_core: int = None
) -> List[int]:
    """
    批量验证一个解法在所有测试用例上的执行情况
    在单个子进程中执行，效率更高
    
    返回: 验证通过的测试用例索引列表
    """
    io_type = (code_type == 'io')
    
    manager = multiprocessing.Manager()
    results = manager.list()
    
    p = multiprocessing.Process(
        target=unsafe_batch_verify,
        args=(io_type, code, testcases, time_limit_per_tc, entry_point, results, cpu_core)
    )
    p.start()
    p.join(timeout=total_timeout)
    
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)
    
    return list(results)


def find_common_valid_testcases(
    solutions: List[str],
    testcases: List[Dict],
    entry_point: str,
    code_type: str,
    time_limit_per_tc: float = 5.0,
    min_testcases: int = 50,
    cpu_core: int = None  # V6新增
) -> Tuple[List[int], List[int]]:
    """
    找出公共有效的测试用例和解法组合
    
    策略（V7改进）：
    1. 先验证所有解法在所有测试用例上的执行情况
    2. 如果测试用例数 >= min_testcases，使用严格筛选（所有解法都通过所有用例）
    3. 如果测试用例数 < min_testcases，使用宽松筛选，优先保留解法：
       - 保留至少能通过 80% 测试用例的解法
       - 保留至少被 80% 解法通过的测试用例
       - 目标：max(测试用例数 × 解法数)
    
    Args:
        solutions: 所有解法代码列表
        testcases: 测试用例列表
        entry_point: 函数入口点
        code_type: 代码类型
        time_limit_per_tc: 单个测试用例超时
        min_testcases: 最少需要的测试用例数
        cpu_core: CPU核心编号 (V6新增)
    
    Returns:
        (valid_tc_indices, valid_sol_indices)
        - valid_tc_indices: 有效测试用例索引
        - valid_sol_indices: 有效解法索引
    """
    if not solutions or not testcases:
        return [], []
    
    io_type = (code_type == 'io')
    timeout = time_limit_per_tc + 2
    
    num_solutions = len(solutions)
    num_testcases = len(testcases)
    
    # 构建验证矩阵: pass_matrix[sol_idx][tc_idx] = True/False
    pass_matrix = [[False] * num_testcases for _ in range(num_solutions)]
    
    print(f"    验证阶段: {num_solutions} 个解法 × {num_testcases} 个测试用例...")
    
    # 修复5: 使用批量验证，在单个进程中验证所有TC，效率更高
    for sol_idx, code in enumerate(solutions):
        if not code:
            continue
        
        if sol_idx % 5 == 0:  # 每5个解法输出一次进度
            print(f"      验证解法 {sol_idx+1}/{num_solutions}...")
        
        # 使用批量验证函数
        valid_tc_indices = batch_verify_solution(
            code=code,
            testcases=testcases,
            entry_point=entry_point,
            code_type=code_type,
            time_limit_per_tc=time_limit_per_tc,
            total_timeout=time_limit_per_tc * num_testcases + 60,  # 总超时
            cpu_core=cpu_core
        )
        
        for tc_idx in valid_tc_indices:
            pass_matrix[sol_idx][tc_idx] = True
    
    # 当前有效的解法和测试用例集合
    valid_sols = set(range(num_solutions))
    valid_tcs = set(range(num_testcases))
    
    print(f"    开始智能筛选 (测试用例数={num_testcases}, 目标≥{min_testcases})...")
    
    # V8: 平衡策略 - 最大化 TC数×解法数，同时保证 TC数 >= min_testcases
    print(f"      使用平衡策略（最大化 TC×解法，TC≥{min_testcases}）")
    
    # 第一步：找出有效的解法（至少能通过一些TC的解法）
    candidate_sols = set()
    for sol_idx in range(num_solutions):
        if not solutions[sol_idx]:
            continue
        passed_count = sum(1 for tc_idx in range(num_testcases) if pass_matrix[sol_idx][tc_idx])
        if passed_count > 0:
            candidate_sols.add(sol_idx)
    
    if not candidate_sols:
        print(f"      没有有效解法")
        return [], []
    
    print(f"      候选解法: {len(candidate_sols)}/{num_solutions} 个")
    
    # 辅助函数：计算给定解法集合的安全TC
    def get_safe_tcs(sol_set):
        if not sol_set:
            return set(range(num_testcases))
        safe = set()
        for tc_idx in range(num_testcases):
            if all(pass_matrix[sol_idx][tc_idx] for sol_idx in sol_set):
                safe.add(tc_idx)
        return safe
    
    # 计算全部解法时的安全TC
    all_safe_tcs = get_safe_tcs(candidate_sols)
    print(f"      全部解法的安全TC: {len(all_safe_tcs)} 个")
    
    if len(all_safe_tcs) >= min_testcases:
        # 情况1: 全部解法的安全TC已经足够，直接使用
        valid_sols = candidate_sols
        valid_tcs = all_safe_tcs
        score = len(valid_tcs) * len(valid_sols)
        print(f"      ✓ 全部保留: {len(valid_sols)} 解法 × {len(valid_tcs)} TC = {score}")
    else:
        # 情况2: 需要权衡，寻找最优的 TC×解法 组合
        print(f"      安全TC不足 ({len(all_safe_tcs)} < {min_testcases})，开始搜索最优组合...")
        
        # 计算每个解法的"破坏力"（移除它能增加多少TC）
        sol_impact = []
        for sol_idx in candidate_sols:
            others = candidate_sols - {sol_idx}
            tcs_without = get_safe_tcs(others)
            impact = len(tcs_without) - len(all_safe_tcs)  # 移除后能增加的TC数
            sol_impact.append((sol_idx, impact))
        
        # 按破坏力从高到低排序（破坏力高的解法可能需要移除）
        sol_impact.sort(key=lambda x: x[1], reverse=True)
        
        # 贪心搜索：从全部解法开始，逐步移除破坏力最高的解法
        # 记录每一步的 TC×解法 值，找最优
        best_score = 0
        best_sols = set()
        best_tcs = set()
        
        current_sols = set(candidate_sols)
        
        for sol_idx, impact in sol_impact:
            # 计算当前组合的分数
            current_tcs = get_safe_tcs(current_sols)
            if len(current_tcs) >= min_testcases:
                score = len(current_tcs) * len(current_sols)
                if score > best_score:
                    best_score = score
                    best_sols = set(current_sols)
                    best_tcs = set(current_tcs)
            
            # 尝试移除这个解法
            if sol_idx in current_sols and len(current_sols) > 1:
                current_sols.remove(sol_idx)
        
        # 检查最终状态
        final_tcs = get_safe_tcs(current_sols)
        if len(final_tcs) >= min_testcases:
            score = len(final_tcs) * len(current_sols)
            if score > best_score:
                best_score = score
                best_sols = current_sols
                best_tcs = final_tcs
        
        valid_sols = best_sols
        valid_tcs = best_tcs
        
        if valid_sols:
            print(f"      ✓ 最优组合: {len(valid_sols)} 解法 × {len(valid_tcs)} TC = {best_score}")
    
    # 修复3: 筛选后检查TC数量是否足够，不足则返回空（跳过该问题）
    if len(valid_tcs) < min_testcases:
        print(f"      筛选失败: 有效测试用例数 ({len(valid_tcs)}) < 最小要求 ({min_testcases})，跳过该问题")
        return [], []
    
    if len(valid_sols) == 0:
        print(f"      筛选失败: 没有有效解法，跳过该问题")
        return [], []
    
    return sorted(list(valid_tcs)), sorted(list(valid_sols))


# ==================== 单问题测量函数 ====================

def measure_single_problem(
    problem: Dict,
    num_runs: int,
    max_test_cases: int,
    problem_idx: int,
    total_problems: int,
    cpu_core: int = None  # V6新增
) -> Dict:
    """
    测量单个问题的所有解法
    
    流程：
    1. 解析/生成测试用例（达到指定数目）
    2. 预验证：找出所有解法都能通过的测试用例
    3. 只使用公共有效用例进行测量
    4. 计算各解法的平均指令数
    """
    problem_id = problem.get('problem_id', problem.get('task_id', problem.get('id', problem.get('slug_name', f'unknown_{problem_idx}'))))
    solutions = problem.get('solutions', [])
    
    # 处理 entry_point
    entry_point = problem.get('entry_point', '')
    if entry_point == 'None' or entry_point is None or entry_point == '':
        # 从 prompt 中解析
        prompt = problem.get('prompt', '')
        entry_point = extract_entry_point(prompt)
    
    if not solutions:
        return {
            'problem_id': problem_id,
            'results': [],
            'status': 'no_solutions'
        }
    
    # 解析/生成测试用例（达到指定数目）
    test_cases, code_type = parse_test_cases(problem, max_test_cases)
    
    if not test_cases:
        return {
            'problem_id': problem_id,
            'num_solutions': len(solutions),
            'results': [],
            'status': 'no_test_cases'
        }
    
    # V7: 检查测试用例数量是否足够（默认要求 >= 30）
    min_required_tc = problem.get('_min_testcases', 30)  # 可以通过参数传递
    if len(test_cases) < min_required_tc:
        print(f"  问题 {problem_idx+1}/{total_problems} [ID: {problem_id}]: 跳过 - 测试用例不足 ({len(test_cases)} < {min_required_tc})")
        return {
            'problem_id': problem_id,
            'num_solutions': len(solutions),
            'num_test_cases_generated': len(test_cases),
            'results': [],
            'status': 'insufficient_test_cases',
            'message': f'Only {len(test_cases)} test cases, need at least {min_required_tc}'
        }
    
    # 提取所有解法代码
    solution_codes = []
    for solution in solutions:
        if isinstance(solution, dict):
            code = solution.get('solution', solution.get('code', ''))
        else:
            code = solution
        solution_codes.append(code if code else '')
    
    # 预验证：找出有效测试用例
    # 修复1: 使用用户设置的 min_required_tc 而不是硬编码的 50
    common_valid_indices, valid_sol_indices = find_common_valid_testcases(
        solutions=solution_codes,
        testcases=test_cases,
        entry_point=entry_point,
        code_type=code_type,
        min_testcases=min_required_tc,  # 修复1: 使用用户设置的值
        cpu_core=cpu_core  # V6: 传递CPU核心
    )
    
    if not common_valid_indices or not valid_sol_indices:
        # 没有公共有效用例或没有有效解法，不保存任何解法
        return {
            'problem_id': problem_id,
            'num_solutions': len(solutions),
            'num_test_cases_generated': len(test_cases),
            'num_common_valid_tc': 0,
            'num_valid_solutions': 0,
            'common_testcases': [],  # 没有有效测试用例
            'code_type': code_type,
            'entry_point': entry_point,
            'results': [],  # 不保存失败的解法
            'status': 'no_common_valid_tc'
        }
    
    # 使用公共有效测试用例
    common_testcases = [test_cases[i] for i in common_valid_indices]
    valid_sol_set = set(valid_sol_indices)
    
    results = []
    
    print(f"  问题 {problem_idx+1}/{total_problems} [ID: {problem_id}]: {len(solution_codes)} 个解法, {len(common_testcases)} 个有效测试用例")
    
    for sol_idx, code in enumerate(solution_codes):
        if not code:
            results.append({
                'solution_idx': sol_idx,
                'code': '',
                'status': 'no_code',
                'avg_cpu_instruction': None
            })
            continue
        
        # 只测量有效解法（能通过所有公共测试用例的解法）
        if sol_idx not in valid_sol_set:
            results.append({
                'solution_idx': sol_idx,
                'code': code,
                'status': 'excluded_invalid',
                'avg_cpu_instruction': None
            })
            continue
        
        print(f"  测量解法 {sol_idx+1}/{len(solution_codes)} (有效解法 {len([r for r in results if r.get('status') not in ['no_code', 'excluded_invalid']])+1}/{len(valid_sol_set)})...")
        
        # 使用公共有效测试用例进行测量
        avg_instr, rsd, valid_count, status = measure_solution_instructions(
            code=code,
            testcases=common_testcases,
            entry_point=entry_point,
            code_type=code_type,
            num_runs=num_runs,
            cpu_core=cpu_core  # V6: 传递CPU核心
        )
        
        results.append({
            'solution_idx': sol_idx,
            'code': code,
            'status': status,
            'avg_cpu_instruction': avg_instr,
            'num_test_cases': valid_count,
            'rsd_percent': rsd
        })
        
        # 输出该解法的测量结果
        if status == 'success':
            print(f"    ✓ 解法 {sol_idx+1}: {avg_instr:,} 指令 (RSD: {rsd:.2f}%, {valid_count} 个用例)")
        else:
            print(f"    ✗ 解法 {sol_idx+1}: {status}")
    
    # 总结
    successful = [r for r in results if r.get('status') == 'success']
    print(f"  完成: {len(successful)}/{len(solution_codes)} 个解法测量成功")
    print(f"-" * 70)
    
    return {
        'problem_id': problem_id,
        'num_solutions': len(solutions),
        'num_test_cases_generated': len(test_cases),
        'num_common_valid_tc': len(common_testcases),
        'num_valid_solutions': len(valid_sol_indices),
        'common_tc_indices': common_valid_indices,
        'valid_sol_indices': valid_sol_indices,
        'common_testcases': common_testcases,  # 保存过滤后的公共有效测试用例
        'code_type': code_type,
        'entry_point': entry_point,
        'results': successful,  # 只保存成功的解法
        'status': 'success' if successful else 'no_successful_solutions'
    }


# ==================== Worker 进程 ====================

def worker_process(
    task_queue: Queue,
    result_queue: Queue,
    num_runs: int,
    max_test_cases: int,
    min_test_cases: int,
    cpu_core: int,
    worker_id: int
):
    """
    Worker 进程
    
    V6改进：将 cpu_core 传递给所有测量函数
    """
    # 设置 Worker 进程的 CPU affinity
    try:
        os.sched_setaffinity(0, {cpu_core})
    except (OSError, AttributeError):
        pass
    
    while True:
        try:
            task = task_queue.get(timeout=1)
        except:
            if task_queue.empty():
                break
            continue
        
        if task is None:
            break
        
        problem, problem_idx, total_problems = task
        
        # V7: 将 min_test_cases 传递给 problem 供 measure_single_problem 使用
        problem['_min_testcases'] = min_test_cases
        
        # V6: 传递 cpu_core 给测量函数
        result = measure_single_problem(
            problem=problem,
            num_runs=num_runs,
            max_test_cases=max_test_cases,
            problem_idx=problem_idx,
            total_problems=total_problems,
            cpu_core=cpu_core  # V6新增
        )
        result['problem_idx'] = problem_idx
        result['worker_id'] = worker_id
        result['cpu_core'] = cpu_core  # V6: 记录使用的CPU核心
        
        result_queue.put(result)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Coffe CPU指令数基准测试 V7 (增加最小测试用例数检查)')
    parser.add_argument('-i', '--input', required=True, help='输入JSONL文件')
    parser.add_argument('-o', '--output', required=True, help='输出JSONL文件')
    parser.add_argument('-n', '--num-runs', type=int, default=NUM_RUNS, help=f'每个测试用例运行次数 (默认{NUM_RUNS})')
    parser.add_argument('-c', '--max-cases', type=int, default=MAX_TEST_CASES, help=f'最多测试用例数 (默认{MAX_TEST_CASES})')
    parser.add_argument('-m', '--min-cases', type=int, default=50, help='最少测试用例数，不足则跳过 (默认50)')
    parser.add_argument('-w', '--workers', type=int, default=None, help='并行Worker数 (默认=CPU核心数-2)')
    parser.add_argument('--cpu-start', type=int, default=0, help='起始CPU核心编号')
    parser.add_argument('--limit', type=int, default=None, help='只处理前N个问题')
    parser.add_argument('--skip', type=int, default=0, help='跳过前N个问题')
    parser.add_argument('--dataset', type=str, default=None, help='数据集标签（用于合并输出）')
    parser.add_argument('--append', action='store_true', help='追加到输出文件而非覆盖')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Coffe CPU指令数基准测试 V7 (增加最小测试用例数检查)")
    print("=" * 70)
    print()
    
    print("【环境检查】")
    print(f"  cirron: {'✓ OK' if CIRRON_AVAILABLE else '✗ 未安装'}")
    perf_ok, perf_msg = check_perf_available()
    print(f"  perf:   {'✓' if perf_ok else '✗'} {perf_msg}")
    
    if not perf_ok:
        print("\n请运行: sudo sysctl -w kernel.perf_event_paranoid=-1")
        sys.exit(1)
    
    cpu_count = os.cpu_count() or 4
    num_workers = args.workers if args.workers else max(1, cpu_count - 2)
    num_workers = min(num_workers, cpu_count)
    
    print()
    print("【配置】")
    print(f"  测量次数: {args.num_runs}")
    print(f"  去除异常: 最高1次 + 最低1次")
    print(f"  测试用例: 最多{args.max_cases}个, 最少{args.min_cases}个")
    print(f"  并行Worker: {num_workers}")
    print(f"  CPU核心: {args.cpu_start} ~ {args.cpu_start + num_workers - 1}")
    if args.dataset:
        print(f"  数据集标签: {args.dataset}")
    if args.append:
        print(f"  输出模式: 追加")
    print()
    print("【V7 改进】")
    print("  - 每次测量在独立子进程中执行（与Coffe作者一致）")
    print("  - 修复 CPU affinity 继承：子进程显式设置 affinity")
    print("  - 支持 --dataset 标签和 --append 追加模式")
    print("  - 使用 reliability_guard 沙箱保护")
    print("  - 测试用例不足时自动跳过问题（保证数据质量）")
    print("  - 验证阶段使用批量验证，效率更高")
    
    # 加载数据
    print()
    print(f"【加载数据】{args.input}")
    
    problems = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    if args.skip > 0:
        problems = problems[args.skip:]
        print(f"  跳过前 {args.skip} 个问题")
    
    if args.limit:
        problems = problems[:args.limit]
        print(f"  限制处理 {args.limit} 个问题")
    
    total_solutions = sum(len(p.get('solutions', [])) for p in problems)
    print(f"  共 {len(problems)} 个问题, {total_solutions} 个解法")
    
    # 估算时间（每次测量都是独立进程，会比 v4 慢很多）
    est_time = total_solutions * args.max_cases * args.num_runs * 0.5 / num_workers / 60
    print()
    print(f"【预估时间】约 {est_time:.0f} 分钟 (独立进程模式较慢)")
    
    print()
    print("【开始测量】")
    print("-" * 70)
    
    start_time = time.time()
    
    # 创建队列
    task_queue = Queue()
    result_queue = Queue()
    
    for idx, problem in enumerate(problems):
        task_queue.put((problem, idx, len(problems)))
    
    for _ in range(num_workers):
        task_queue.put(None)
    
    # 启动 Worker
    workers = []
    for i in range(num_workers):
        cpu_core = args.cpu_start + i
        p = Process(
            target=worker_process,
            args=(task_queue, result_queue, args.num_runs, args.max_cases, args.min_cases, cpu_core, i)
        )
        p.start()
        workers.append(p)
    
    print(f"  已启动 {num_workers} 个 Worker 进程")
    print()
    
    # 收集结果
    completed = 0
    success_count = 0
    all_instructions = []
    all_rsds = []
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # V6: 支持追加模式
    write_mode = 'a' if args.append else 'w'
    if not args.append:
        with open(args.output, 'w') as f:
            pass
    
    while completed < len(problems):
        try:
            result = result_queue.get(timeout=600)
        except:
            print("\n警告: 等待结果超时...")
            alive_workers = sum(1 for w in workers if w.is_alive())
            print(f"  存活 Worker: {alive_workers}/{num_workers}")
            if alive_workers == 0:
                break
            continue
        
        completed += 1
        problem_id = result['problem_id']
        worker_id = result['worker_id']
        code_type = result.get('code_type', 'unknown')
        
        # V6: 添加数据集标签
        if args.dataset:
            result['dataset'] = args.dataset
        
        for r in result.get('results', []):
            if r.get('status') == 'success' and r.get('avg_cpu_instruction'):
                success_count += 1
                all_instructions.append(r['avg_cpu_instruction'])
                if r.get('rsd_percent') is not None:
                    all_rsds.append(r['rsd_percent'])
        
        # V7: 只保存有成功测量结果的问题（过滤掉失败的记录）
        has_success_results = any(r.get('status') == 'success' for r in result.get('results', []))
        if has_success_results and result.get('num_common_valid_tc', 0) > 0:
            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        elapsed = time.time() - start_time
        speed = completed / elapsed * 60 if elapsed > 0 else 0
        eta = (len(problems) - completed) / speed if speed > 0 else 0
        
        num_solutions = result.get('num_solutions', 0)
        num_valid_sols = result.get('num_valid_solutions', 0)
        success_in_this = sum(1 for r in result.get('results', []) if r.get('status') == 'success')
        num_generated = result.get('num_test_cases_generated', 0)
        num_common = result.get('num_common_valid_tc', 0)
        
        # 标注是否被跳过（未保存）
        skipped_mark = "" if has_success_results and num_common > 0 else " [SKIPPED]"
        
        print(f"  [{completed}/{len(problems)}] {problem_id} [{code_type}]: "
              f"{success_in_this}/{num_valid_sols}({num_solutions}) 成功, TC:{num_common}/{num_generated} "
              f"(W{worker_id}, {speed:.1f}问/分, ETA {eta:.0f}分){skipped_mark}")
    
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("【完成】")
    print(f"  输出文件: {args.output}")
    print(f"  总问题数: {len(problems)}")
    print(f"  总耗时:   {elapsed/3600:.2f} 小时 ({elapsed/60:.1f} 分钟)")
    print(f"  成功测量: {success_count} 个解法")
    
    if all_instructions:
        print(f"  平均指令: {int(statistics.mean(all_instructions)):,}")
        print(f"  指令范围: {min(all_instructions):,} ~ {max(all_instructions):,}")
    
    if all_rsds:
        print(f"  平均RSD:  {statistics.mean(all_rsds):.4f}%")
    
    print("=" * 70)


if __name__ == '__main__':
    main()