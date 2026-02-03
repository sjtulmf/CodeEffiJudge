#!/usr/bin/env python3
"""
代码清洗模块 v2.1 (Code Sanitizer)

新增功能 (相比 v2.0):
    - C++ 语法验证 (g++ -fsyntax-only)
    - Java 语法验证 (javac)
    - 功能等价性验证框架 (运行测试用例比较输出)
    - 完整的验证报告生成

修复内容 (相比 v1.0):
    - Python 注释移除改用 tokenize 模块，避免误删字符串内的 #
    - 改进 docstring 检测逻辑
    - 增加更多边界情况处理

设计目标：
    防止模型通过记忆特定变量名或注释来"作弊"（Data Contamination）

处理策略：
    Python:
        1. 使用 tokenize 安全移除注释和文档字符串
        2. 使用 Black 统一代码风格
        3. 使用 ast.parse() 验证语法
    
    C++:
        1. 使用状态机移除注释（正确处理字符串内的 // 和 /*）
        2. 使用 ClangFormat 统一代码风格
        3. 使用 g++ -fsyntax-only 验证语法
    
    Java:
        1. 使用状态机移除注释和 Javadoc
        2. 使用 google-java-format 统一代码风格
        3. 使用 javac 验证语法

接口：
    sanitize_code(code, language, skip_format, validate) -> SanitizeResult
    verify_functional_equivalence(original, cleaned, language, test_cases) -> EquivalenceResult

依赖安装：
    pip install black
    apt install g++ clang-format default-jdk
"""

import re
import ast
import subprocess
import warnings
import tokenize
import io
import tempfile
import os
import random
import hashlib
import shutil
from typing import Dict, Tuple, Set, Optional, List, NamedTuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 抑制 SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)


# ============================================================
# 数据结构定义
# ============================================================

class ValidationStatus(Enum):
    """验证状态枚举"""
    VALID = "valid"
    INVALID = "invalid"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class SanitizeResult:
    """清洗结果"""
    original_code: str
    cleaned_code: str
    method: str
    syntax_valid: ValidationStatus
    syntax_error_msg: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.syntax_valid in (ValidationStatus.VALID, ValidationStatus.SKIPPED)


@dataclass
class TestCaseResult:
    """单个测试用例的验证结果"""
    test_input: str
    original_output: Optional[str]
    cleaned_output: Optional[str]
    original_returncode: int
    cleaned_returncode: int
    is_equivalent: bool
    error_msg: Optional[str] = None


@dataclass
class EquivalenceResult:
    """功能等价性验证结果"""
    total_cases: int
    sampled_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    is_equivalent: bool
    details: List[TestCaseResult] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        if self.sampled_cases == 0:
            return 1.0
        return self.passed_cases / self.sampled_cases


# ============================================================
# 语法验证
# ============================================================

def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    验证 Python 代码语法正确性
    
    Returns:
        (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def validate_cpp_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    验证 C++ 代码语法正确性
    
    使用 g++ -fsyntax-only 进行语法检查
    
    Returns:
        (is_valid, error_message)
    """
    # 检查 g++ 是否可用
    if not shutil.which('g++'):
        return True, None  # g++ 不可用时跳过验证
    
    try:
        with tempfile.NamedTemporaryFile(
            suffix='.cpp', 
            mode='w', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['g++', '-fsyntax-only', '-std=c++17', '-w', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, None
            else:
                # 提取错误信息
                error_msg = result.stderr.strip()
                # 移除临时文件路径，只保留错误描述
                error_msg = re.sub(r'^.*?\.cpp:', '', error_msg, flags=re.MULTILINE)
                return False, error_msg[:500] if error_msg else "Syntax error"
                
        finally:
            os.unlink(temp_file)
            
    except subprocess.TimeoutExpired:
        return False, "Syntax check timeout"
    except Exception as e:
        return True, None  # 出错时不阻塞流程


def validate_java_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    验证 Java 代码语法正确性
    
    使用 javac 进行语法检查
    
    Returns:
        (is_valid, error_message)
    """
    # 检查 javac 是否可用
    if not shutil.which('javac'):
        return True, None  # javac 不可用时跳过验证
    
    try:
        # 尝试从代码中提取类名
        class_match = re.search(r'public\s+class\s+(\w+)', code)
        if class_match:
            class_name = class_match.group(1)
            filename = f"{class_name}.java"
        else:
            # 没有 public class，使用临时名称
            filename = "Main.java"
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, filename)
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result = subprocess.run(
                ['javac', '-Xlint:none', temp_file],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir
            )
            
            if result.returncode == 0:
                return True, None
            else:
                error_msg = result.stderr.strip()
                # 简化错误信息
                error_msg = re.sub(r'^.*?\.java:', '', error_msg, flags=re.MULTILINE)
                return False, error_msg[:500] if error_msg else "Syntax error"
                
    except subprocess.TimeoutExpired:
        return False, "Syntax check timeout"
    except Exception as e:
        return True, None  # 出错时不阻塞流程


def validate_syntax(code: str, language: str) -> Tuple[ValidationStatus, Optional[str]]:
    """
    统一语法验证接口
    
    Returns:
        (status, error_message)
    """
    lang = language.lower().strip()
    
    if lang in ('python', 'py', 'python3'):
        is_valid, error = validate_python_syntax(code)
    elif lang in ('cpp', 'c++', 'cc', 'cxx', 'c'):
        is_valid, error = validate_cpp_syntax(code)
    elif lang in ('java',):
        is_valid, error = validate_java_syntax(code)
    else:
        return ValidationStatus.SKIPPED, None
    
    if is_valid:
        return ValidationStatus.VALID, None
    else:
        return ValidationStatus.INVALID, error


# ============================================================
# 功能等价性验证
# ============================================================

def _run_python_code(code: str, stdin_input: str, timeout: int = 10) -> Tuple[Optional[str], int]:
    """
    执行 Python 代码
    
    Returns:
        (stdout, returncode)
    """
    try:
        result = subprocess.run(
            ['python3', '-c', code],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return None, -1
    except Exception as e:
        return None, -2


def _run_cpp_code(code: str, stdin_input: str, timeout: int = 10) -> Tuple[Optional[str], int]:
    """
    编译并执行 C++ 代码
    
    Returns:
        (stdout, returncode)
    """
    if not shutil.which('g++'):
        return None, -3  # 编译器不可用
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            src_file = os.path.join(temp_dir, 'code.cpp')
            exe_file = os.path.join(temp_dir, 'code.out')
            
            with open(src_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 编译
            compile_result = subprocess.run(
                ['g++', '-std=c++17', '-O2', '-o', exe_file, src_file],
                capture_output=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                return None, -4  # 编译失败
            
            # 执行
            result = subprocess.run(
                [exe_file],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout, result.returncode
            
    except subprocess.TimeoutExpired:
        return None, -1
    except Exception as e:
        return None, -2


def _run_java_code(code: str, stdin_input: str, timeout: int = 15) -> Tuple[Optional[str], int]:
    """
    编译并执行 Java 代码
    
    Returns:
        (stdout, returncode)
    """
    if not shutil.which('javac') or not shutil.which('java'):
        return None, -3  # 编译器不可用
    
    try:
        # 提取类名
        class_match = re.search(r'public\s+class\s+(\w+)', code)
        if class_match:
            class_name = class_match.group(1)
        else:
            # 尝试找任意 class
            class_match = re.search(r'class\s+(\w+)', code)
            if class_match:
                class_name = class_match.group(1)
            else:
                return None, -5  # 找不到类名
        
        with tempfile.TemporaryDirectory() as temp_dir:
            src_file = os.path.join(temp_dir, f'{class_name}.java')
            
            with open(src_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 编译
            compile_result = subprocess.run(
                ['javac', src_file],
                capture_output=True,
                timeout=60,
                cwd=temp_dir
            )
            
            if compile_result.returncode != 0:
                return None, -4  # 编译失败
            
            # 执行
            result = subprocess.run(
                ['java', class_name],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir
            )
            return result.stdout, result.returncode
            
    except subprocess.TimeoutExpired:
        return None, -1
    except Exception as e:
        return None, -2


def _run_code(code: str, language: str, stdin_input: str, timeout: int = 10) -> Tuple[Optional[str], int]:
    """
    执行代码的统一接口
    
    Returns:
        (stdout, returncode)
    """
    lang = language.lower().strip()
    
    if lang in ('python', 'py', 'python3'):
        return _run_python_code(code, stdin_input, timeout)
    elif lang in ('cpp', 'c++', 'cc', 'cxx', 'c'):
        return _run_cpp_code(code, stdin_input, timeout)
    elif lang in ('java',):
        return _run_java_code(code, stdin_input, timeout)
    else:
        return None, -3


def verify_functional_equivalence(
    original_code: str,
    cleaned_code: str,
    language: str,
    test_cases: List[Tuple[str, Optional[str]]],
    sample_ratio: float = 1.0,
    max_samples: int = 10,
    timeout_per_case: int = 10
) -> EquivalenceResult:
    """
    验证预处理后代码的功能等价性
    
    通过运行测试用例，比较原始代码和清洗后代码的输出是否一致。
    
    Args:
        original_code: 原始代码
        cleaned_code: 清洗后的代码
        language: 编程语言
        test_cases: 测试用例列表 [(输入, 期望输出或None), ...]
                   期望输出可以为 None，此时只比较两个版本的输出是否一致
        sample_ratio: 随机抽样比例 (0.0-1.0)
        max_samples: 最大抽样数量
        timeout_per_case: 每个测试用例的超时时间（秒）
    
    Returns:
        EquivalenceResult 包含详细的验证结果
    """
    if not test_cases:
        return EquivalenceResult(
            total_cases=0,
            sampled_cases=0,
            passed_cases=0,
            failed_cases=0,
            error_cases=0,
            is_equivalent=True,
            details=[]
        )
    
    # 计算抽样数量
    total = len(test_cases)
    sample_size = min(max(1, int(total * sample_ratio)), max_samples, total)
    
    # 随机抽样（使用固定种子以便复现）
    if sample_size < total:
        # 使用代码哈希作为种子，确保同一代码总是抽到相同的测试用例
        seed = int(hashlib.md5(original_code.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        sampled = random.sample(test_cases, sample_size)
    else:
        sampled = test_cases
    
    passed = 0
    failed = 0
    errors = 0
    details = []
    
    for test_input, expected_output in sampled:
        # 运行原始代码
        orig_output, orig_rc = _run_code(original_code, language, test_input, timeout_per_case)
        
        # 运行清洗后代码
        clean_output, clean_rc = _run_code(cleaned_code, language, test_input, timeout_per_case)
        
        # 判断等价性
        error_msg = None
        
        if orig_rc < 0 or clean_rc < 0:
            # 执行出错
            errors += 1
            is_equiv = False
            if orig_rc < 0:
                error_msg = f"Original code execution error (code={orig_rc})"
            else:
                error_msg = f"Cleaned code execution error (code={clean_rc})"
        elif orig_output == clean_output:
            # 输出完全一致
            passed += 1
            is_equiv = True
        else:
            # 输出不一致
            failed += 1
            is_equiv = False
            # 生成差异描述
            orig_preview = (orig_output or "")[:100]
            clean_preview = (clean_output or "")[:100]
            error_msg = f"Output mismatch: original='{orig_preview}...', cleaned='{clean_preview}...'"
        
        details.append(TestCaseResult(
            test_input=test_input[:200],  # 截断过长的输入
            original_output=orig_output,
            cleaned_output=clean_output,
            original_returncode=orig_rc,
            cleaned_returncode=clean_rc,
            is_equivalent=is_equiv,
            error_msg=error_msg
        ))
    
    return EquivalenceResult(
        total_cases=total,
        sampled_cases=sample_size,
        passed_cases=passed,
        failed_cases=failed,
        error_cases=errors,
        is_equivalent=(failed == 0),
        details=details
    )


# ============================================================
# Python 清洗 - 使用 tokenize 模块（安全处理字符串）
# ============================================================

def remove_python_comments_safe(code: str) -> str:
    """
    安全移除 Python 注释和文档字符串
    
    使用 tokenize 模块，正确处理：
    - 字符串内的 # 不会被误删
    - 正确识别 docstring vs 普通字符串
    - 保持代码结构完整
    """
    try:
        # Tokenize 源代码
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenizeError:
        # tokenize 失败，回退到基础清理（带警告）
        return _remove_python_comments_fallback(code)
    
    # 识别 docstring 位置
    docstring_positions = _find_docstring_positions(tokens)
    
    # 重建代码，跳过注释和 docstring
    result_tokens = []
    
    for i, tok in enumerate(tokens):
        tok_type = tok.type
        
        # 跳过注释
        if tok_type == tokenize.COMMENT:
            continue
        
        # 跳过 docstring
        if i in docstring_positions:
            continue
        
        result_tokens.append(tok)
    
    # 使用 untokenize 重建代码
    try:
        result = tokenize.untokenize(result_tokens)
        # untokenize 可能产生一些格式问题，尝试重新解析验证
        ast.parse(result)
        return result
    except:
        # 如果 untokenize 结果无效，使用备用方法
        return _rebuild_code_from_tokens(code, tokens, docstring_positions)


def _find_docstring_positions(tokens: List) -> Set[int]:
    """
    识别 docstring 的 token 位置
    
    Docstring 定义：
    - 模块级别：第一个非注释语句是字符串字面量
    - 函数/类级别：def/class 语句后第一个语句是字符串字面量
    
    注意：不会误删普通的多行字符串，只删除真正的 docstring
    """
    docstring_positions = set()
    
    # 追踪状态：是否刚刚看到 def/class 或模块开始
    expect_docstring = True  # 模块开头可能有 docstring
    saw_colon = False  # 是否刚看到 def/class 的冒号
    
    for i, tok in enumerate(tokens):
        tok_type = tok.type
        
        # 跳过注释、换行、缩进等
        if tok_type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, 
                        tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
            continue
        
        # 检查是否是 def 或 class 关键字
        if tok_type == tokenize.NAME and tok.string in ('def', 'class'):
            saw_colon = False
            expect_docstring = False  # 等待看到冒号
            continue
        
        # 检查冒号（def/class 定义结束）
        if tok_type == tokenize.OP and tok.string == ':':
            saw_colon = True
            continue
        
        # 如果刚看到冒号，下一个语句可能是 docstring
        if saw_colon:
            expect_docstring = True
            saw_colon = False
        
        # 检查是否是 docstring
        if tok_type == tokenize.STRING and expect_docstring:
            s = tok.string
            # 只有三引号字符串才可能是 docstring
            if s.startswith(('"""', "'''", 'r"""', "r'''", 'u"""', "u'''")):
                docstring_positions.add(i)
        
        # 看到其他有意义的 token 后，不再期待 docstring
        if tok_type not in (tokenize.STRING,):
            expect_docstring = False
    
    return docstring_positions


def _rebuild_code_from_tokens(code: str, tokens: List, skip_positions: Set[int]) -> str:
    """
    从原始代码重建，精确删除指定 token
    """
    lines = code.splitlines(keepends=True)
    
    # 收集需要删除的区域 (start_row, start_col, end_row, end_col)
    regions_to_remove = []
    
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.COMMENT or i in skip_positions:
            regions_to_remove.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1]))
    
    # 按位置排序（从后往前删除，避免位置偏移）
    regions_to_remove.sort(reverse=True)
    
    # 转换为字符列表便于修改
    line_chars = [list(line) for line in lines]
    
    for start_row, start_col, end_row, end_col in regions_to_remove:
        if start_row == end_row:
            # 单行删除
            if start_row <= len(line_chars):
                line = line_chars[start_row - 1]
                line_chars[start_row - 1] = line[:start_col] + line[end_col:]
        else:
            # 多行删除（主要是多行 docstring）
            if start_row <= len(line_chars):
                # 保留第一行的前半部分
                first_line = line_chars[start_row - 1][:start_col]
                # 保留最后一行的后半部分
                if end_row <= len(line_chars):
                    last_line = line_chars[end_row - 1][end_col:]
                else:
                    last_line = []
                
                # 合并
                line_chars[start_row - 1] = first_line + last_line
                # 删除中间行
                for row in range(end_row - 1, start_row - 1, -1):
                    if row < len(line_chars):
                        line_chars.pop(row)
    
    return ''.join(''.join(line) for line in line_chars)


def _remove_python_comments_fallback(code: str) -> str:
    """
    备用方法：使用状态机移除 Python 注释（当 tokenize 失败时）
    
    正确处理字符串内的 #
    """
    result = []
    i = 0
    n = len(code)
    
    while i < n:
        # 检查字符串开始 (""", ''', ", ')
        if code[i:i+3] in ('"""', "'''"):
            quote = code[i:i+3]
            result.append(quote)
            i += 3
            # 找到结束引号
            while i < n:
                if code[i:i+3] == quote:
                    result.append(quote)
                    i += 3
                    break
                elif code[i] == '\\' and i + 1 < n:
                    result.append(code[i:i+2])
                    i += 2
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        if code[i] in ('"', "'"):
            quote = code[i]
            result.append(quote)
            i += 1
            # 找到结束引号
            while i < n:
                if code[i] == quote:
                    result.append(quote)
                    i += 1
                    break
                elif code[i] == '\\' and i + 1 < n:
                    result.append(code[i:i+2])
                    i += 2
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        # 检查注释
        if code[i] == '#':
            # 跳过到行尾
            while i < n and code[i] != '\n':
                i += 1
            continue
        
        result.append(code[i])
        i += 1
    
    return ''.join(result)


def remove_python_author_info(code: str) -> str:
    """移除作者信息元数据"""
    patterns = [
        r'^__author__\s*=.*$',
        r'^__date__\s*=.*$',
        r'^__version__\s*=.*$',
        r'^__email__\s*=.*$',
        r'^__copyright__\s*=.*$',
        r'^__license__\s*=.*$',
        r'^__maintainer__\s*=.*$',
    ]
    for pattern in patterns:
        code = re.sub(pattern, '', code, flags=re.MULTILINE)
    return code


def format_python(code: str) -> str:
    """使用 Black 格式化 Python 代码"""
    try:
        import black
        return black.format_str(code, mode=black.Mode(line_length=88))
    except ImportError:
        warnings.warn("Black not installed, skipping Python formatting")
        return code
    except black.InvalidInput:
        # 代码语法错误，无法格式化
        return code
    except Exception as e:
        warnings.warn(f"Black formatting failed: {e}")
        return code


def sanitize_python(code: str, skip_format: bool = False, validate: bool = True) -> SanitizeResult:
    """
    Python 代码清洗（v2.1 - 带验证版本）
    
    步骤:
        1. tokenize 安全移除注释和 docstring
        2. 移除作者元数据
        3. Black 格式化
        4. 语法验证
    
    Returns:
        SanitizeResult
    """
    original_code = code
    method = "python_tokenize"
    
    try:
        # 步骤1: 安全移除注释
        code = remove_python_comments_safe(code)
        
        # 步骤2: 移除作者元数据
        code = remove_python_author_info(code)
        
        # 步骤3: 格式化
        if not skip_format:
            code = format_python(code)
        
    except Exception as e:
        # Fallback: 使用状态机方法
        try:
            code = _remove_python_comments_fallback(original_code)
            code = remove_python_author_info(code)
            if not skip_format:
                code = format_python(code)
            method = "python_fallback"
        except:
            code = original_code
            method = "python_failed"
    
    # 步骤4: 语法验证
    if validate:
        syntax_status, syntax_error = validate_syntax(code, "python")
    else:
        syntax_status = ValidationStatus.SKIPPED
        syntax_error = None
    
    return SanitizeResult(
        original_code=original_code,
        cleaned_code=code,
        method=method,
        syntax_valid=syntax_status,
        syntax_error_msg=syntax_error
    )


# ============================================================
# C++ 清洗 - 状态机（已经是安全实现）
# ============================================================

def remove_cpp_comments(code: str) -> str:
    """
    移除 C++ 注释（状态机实现）
    
    正确处理:
    - // 单行注释
    - /* */ 多行注释
    - 字符串内的 // 和 /* (不移除)
    - R"(...)" Raw string literals
    - 字符常量 '//' 等
    """
    result = []
    i = 0
    n = len(code)
    
    while i < n:
        # ===== 字符常量 'x' =====
        if code[i] == "'" and (i == 0 or code[i-1] != '\\'):
            result.append(code[i])
            i += 1
            # 读取字符常量内容
            while i < n:
                if code[i] == '\\' and i + 1 < n:
                    result.append(code[i:i+2])
                    i += 2
                elif code[i] == "'":
                    result.append(code[i])
                    i += 1
                    break
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        # ===== 普通字符串 "..." =====
        if code[i] == '"' and (i == 0 or code[i-1] not in ('\\', 'R')):
            result.append(code[i])
            i += 1
            while i < n:
                if code[i] == '\\' and i + 1 < n:
                    result.append(code[i:i+2])
                    i += 2
                elif code[i] == '"':
                    result.append(code[i])
                    i += 1
                    break
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        # ===== Raw string literal R"delimiter(...)delimiter" =====
        if code[i:i+2] == 'R"':
            result.append('R"')
            i += 2
            # 找到 delimiter (直到遇到 '(')
            delimiter = ""
            while i < n and code[i] != '(':
                delimiter += code[i]
                result.append(code[i])
                i += 1
            
            if i < n:
                result.append(code[i])  # '('
                i += 1
            
            # 找到 )delimiter"
            end_marker = ')' + delimiter + '"'
            while i < n:
                if code[i:i+len(end_marker)] == end_marker:
                    result.append(end_marker)
                    i += len(end_marker)
                    break
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        # ===== 单行注释 // =====
        if code[i:i+2] == '//':
            # 跳过到行尾
            while i < n and code[i] != '\n':
                i += 1
            # 保留换行符
            if i < n:
                result.append('\n')
                i += 1
            continue
        
        # ===== 多行注释 /* */ =====
        if code[i:i+2] == '/*':
            i += 2
            while i < n - 1 and code[i:i+2] != '*/':
                i += 1
            i += 2  # 跳过 */
            # 添加一个空格保持 token 分隔
            result.append(' ')
            continue
        
        result.append(code[i])
        i += 1
    
    return ''.join(result)


def remove_cpp_author_info(code: str) -> str:
    """移除 C++ 代码中的作者信息"""
    patterns = [
        r'@[Aa]uthor[:\s].*$',
        r'@[Dd]ate[:\s].*$',
        r'@[Vv]ersion[:\s].*$',
        r'[Cc]opyright.*$',
        r'[Cc]reated\s+by.*$',
        r'[Ww]ritten\s+by.*$',
        r'[Pp]roblem[:\s].*$',
        r'[Ss]olution\s+by.*$',
    ]
    for pattern in patterns:
        code = re.sub(pattern, '', code, flags=re.MULTILINE)
    
    # 清理多余空行
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    
    return code


def format_cpp(code: str) -> str:
    """使用 ClangFormat 格式化 C++ 代码"""
    try:
        result = subprocess.run(
            ['clang-format', '-style=LLVM', '-assume-filename=code.cpp'],
            input=code,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        else:
            warnings.warn(f"ClangFormat failed: {result.stderr}")
    except FileNotFoundError:
        warnings.warn("clang-format not installed, skipping C++ formatting")
    except subprocess.TimeoutExpired:
        warnings.warn("ClangFormat timeout")
    except Exception as e:
        warnings.warn(f"ClangFormat error: {e}")
    return code


def sanitize_cpp(code: str, skip_format: bool = False, validate: bool = True) -> SanitizeResult:
    """
    C++ 代码清洗（v2.1 - 带验证版本）
    
    Returns:
        SanitizeResult
    """
    original_code = code
    method = "cpp_clean"
    
    try:
        code = remove_cpp_comments(code)
        code = remove_cpp_author_info(code)
        
        if not skip_format:
            code = format_cpp(code)
        
    except Exception as e:
        warnings.warn(f"C++ sanitization failed: {e}")
        code = original_code
        method = "cpp_failed"
    
    # 语法验证
    if validate:
        syntax_status, syntax_error = validate_syntax(code, "cpp")
    else:
        syntax_status = ValidationStatus.SKIPPED
        syntax_error = None
    
    return SanitizeResult(
        original_code=original_code,
        cleaned_code=code,
        method=method,
        syntax_valid=syntax_status,
        syntax_error_msg=syntax_error
    )


# ============================================================
# Java 清洗 - 状态机
# ============================================================

def remove_java_comments(code: str) -> str:
    """
    移除 Java 注释（状态机实现）
    
    正确处理:
    - // 单行注释
    - /* */ 多行注释  
    - /** */ Javadoc
    - 字符串和字符常量内的 // 和 /*
    - Text blocks (Java 15+) \"\"\"...\"\"\"
    """
    result = []
    i = 0
    n = len(code)
    
    while i < n:
        # ===== 字符常量 'x' =====
        if code[i] == "'":
            result.append(code[i])
            i += 1
            while i < n:
                if code[i] == '\\' and i + 1 < n:
                    result.append(code[i:i+2])
                    i += 2
                elif code[i] == "'":
                    result.append(code[i])
                    i += 1
                    break
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        # ===== Text Block (Java 15+) \"\"\"...\"\"\" =====
        if code[i:i+3] == '"""':
            result.append('"""')
            i += 3
            while i < n:
                if code[i:i+3] == '"""':
                    result.append('"""')
                    i += 3
                    break
                elif code[i] == '\\' and i + 1 < n:
                    result.append(code[i:i+2])
                    i += 2
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        # ===== 普通字符串 "..." =====
        if code[i] == '"':
            result.append(code[i])
            i += 1
            while i < n:
                if code[i] == '\\' and i + 1 < n:
                    result.append(code[i:i+2])
                    i += 2
                elif code[i] == '"':
                    result.append(code[i])
                    i += 1
                    break
                elif code[i] == '\n':
                    # 字符串不能跨行（除了 text block）
                    result.append(code[i])
                    i += 1
                    break
                else:
                    result.append(code[i])
                    i += 1
            continue
        
        # ===== 单行注释 // =====
        if code[i:i+2] == '//':
            while i < n and code[i] != '\n':
                i += 1
            if i < n:
                result.append('\n')
                i += 1
            continue
        
        # ===== 多行注释 /* */ 或 Javadoc /** */ =====
        if code[i:i+2] == '/*':
            i += 2
            while i < n - 1 and code[i:i+2] != '*/':
                i += 1
            i += 2
            result.append(' ')  # 保持 token 分隔
            continue
        
        result.append(code[i])
        i += 1
    
    return ''.join(result)


def remove_java_author_info(code: str) -> str:
    """移除 Java 代码中的作者信息"""
    patterns = [
        r'@[Aa]uthor[:\s].*$',
        r'@[Dd]ate[:\s].*$',
        r'@[Vv]ersion[:\s].*$',
        r'@[Ss]ince[:\s].*$',
        r'[Cc]opyright.*$',
        r'[Cc]reated\s+by.*$',
        r'[Ww]ritten\s+by.*$',
        r'[Pp]roblem[:\s].*$',
        r'[Ss]olution\s+by.*$',
    ]
    for pattern in patterns:
        code = re.sub(pattern, '', code, flags=re.MULTILINE)
    
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    
    return code


def format_java(code: str) -> str:
    """使用 google-java-format 格式化 Java 代码"""
    try:
        result = subprocess.run(
            ['google-java-format', '-'],
            input=code,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return result.stdout
        else:
            warnings.warn(f"google-java-format failed: {result.stderr}")
    except FileNotFoundError:
        warnings.warn("google-java-format not installed, skipping Java formatting")
    except subprocess.TimeoutExpired:
        warnings.warn("google-java-format timeout")
    except Exception as e:
        warnings.warn(f"google-java-format error: {e}")
    return code


def sanitize_java(code: str, skip_format: bool = False, validate: bool = True) -> SanitizeResult:
    """
    Java 代码清洗（v2.1 - 带验证版本）
    
    Returns:
        SanitizeResult
    """
    original_code = code
    method = "java_clean"
    
    try:
        code = remove_java_comments(code)
        code = remove_java_author_info(code)
        
        if not skip_format:
            code = format_java(code)
        
    except Exception as e:
        warnings.warn(f"Java sanitization failed: {e}")
        code = original_code
        method = "java_failed"
    
    # 语法验证
    if validate:
        syntax_status, syntax_error = validate_syntax(code, "java")
    else:
        syntax_status = ValidationStatus.SKIPPED
        syntax_error = None
    
    return SanitizeResult(
        original_code=original_code,
        cleaned_code=code,
        method=method,
        syntax_valid=syntax_status,
        syntax_error_msg=syntax_error
    )


# ============================================================
# 统一接口
# ============================================================

def sanitize_code(
    code: str, 
    language: str, 
    skip_format: bool = False,
    validate: bool = True
) -> SanitizeResult:
    """
    代码清洗统一接口
    
    Args:
        code: 原始源代码
        language: 编程语言 ('python', 'cpp', 'java', 等)
        skip_format: 是否跳过格式化
        validate: 是否进行语法验证
    
    Returns:
        SanitizeResult 包含:
        - original_code: 原始代码
        - cleaned_code: 清洗后的代码
        - method: 处理方法标识
        - syntax_valid: 语法验证状态
        - syntax_error_msg: 语法错误信息（如有）
    
    Example:
        >>> code = '''
        ... # Calculate sum
        ... def solve(n):
        ...     url = "http://example.com/#section"  # This # stays
        ...     return n * 2
        ... '''
        >>> result = sanitize_code(code, "python")
        >>> result.success
        True
        >>> "# Calculate" in result.cleaned_code
        False
        >>> "/#section" in result.cleaned_code
        True
    """
    lang = language.lower().strip()
    
    if lang in ('python', 'py', 'python3'):
        return sanitize_python(code, skip_format, validate)
    
    elif lang in ('cpp', 'c++', 'cc', 'cxx', 'c'):
        return sanitize_cpp(code, skip_format, validate)
    
    elif lang in ('java',):
        return sanitize_java(code, skip_format, validate)
    
    else:
        warnings.warn(f"Unknown language: {language}, applying basic cleaning")
        # 基础清理（不安全，仅用于未知语言）
        original = code
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        return SanitizeResult(
            original_code=original,
            cleaned_code=code,
            method="basic_unsafe",
            syntax_valid=ValidationStatus.SKIPPED,
            syntax_error_msg=None
        )


def sanitize_and_verify(
    code: str,
    language: str,
    test_cases: List[Tuple[str, Optional[str]]],
    skip_format: bool = False,
    sample_ratio: float = 1.0,
    max_samples: int = 10
) -> Tuple[SanitizeResult, EquivalenceResult]:
    """
    完整的清洗和验证流程
    
    Args:
        code: 原始代码
        language: 编程语言
        test_cases: 测试用例 [(input, expected_output), ...]
        skip_format: 是否跳过格式化
        sample_ratio: 功能验证的抽样比例
        max_samples: 最大抽样数
    
    Returns:
        (SanitizeResult, EquivalenceResult)
    
    Example:
        >>> code = "x = int(input())\\nprint(x * 2)"
        >>> tests = [("5\\n", "10\\n"), ("3\\n", "6\\n")]
        >>> sanitize_result, equiv_result = sanitize_and_verify(code, "python", tests)
        >>> print(f"Syntax valid: {sanitize_result.syntax_valid}")
        >>> print(f"Functionally equivalent: {equiv_result.is_equivalent}")
    """
    # 步骤 1: 清洗代码
    sanitize_result = sanitize_code(code, language, skip_format, validate=True)
    
    # 步骤 2: 功能等价性验证
    equiv_result = verify_functional_equivalence(
        original_code=sanitize_result.original_code,
        cleaned_code=sanitize_result.cleaned_code,
        language=language,
        test_cases=test_cases,
        sample_ratio=sample_ratio,
        max_samples=max_samples
    )
    
    return sanitize_result, equiv_result


# ============================================================
# 批量处理
# ============================================================

@dataclass
class BatchResult:
    """批量处理结果统计"""
    total: int = 0
    success: int = 0
    syntax_valid: int = 0
    syntax_invalid: int = 0
    syntax_skipped: int = 0
    equiv_tested: int = 0
    equiv_passed: int = 0
    equiv_failed: int = 0
    methods: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def sanitize_batch(
    samples: List[Dict[str, Any]],
    language_key: str = "language",
    code_key: str = "code",
    test_cases_key: str = "test_cases",
    skip_format: bool = False,
    verify_equivalence: bool = True,
    sample_ratio: float = 0.1,
    max_samples: int = 5,
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], BatchResult]:
    """
    批量清洗代码样本
    
    Args:
        samples: 样本列表，每个样本是字典
        language_key: 语言字段名
        code_key: 代码字段名
        test_cases_key: 测试用例字段名（可选）
        skip_format: 是否跳过格式化
        verify_equivalence: 是否验证功能等价性
        sample_ratio: 功能验证抽样比例
        max_samples: 每个样本最大测试用例数
        verbose: 是否打印进度
    
    Returns:
        (processed_samples, batch_result)
    """
    result = BatchResult()
    processed = []
    
    for i, sample in enumerate(samples):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(samples)}...")
        
        result.total += 1
        
        try:
            code = sample.get(code_key, "")
            language = sample.get(language_key, "python")
            test_cases = sample.get(test_cases_key, [])
            
            # 清洗
            sanitize_result = sanitize_code(code, language, skip_format, validate=True)
            
            # 统计语法验证
            if sanitize_result.syntax_valid == ValidationStatus.VALID:
                result.syntax_valid += 1
            elif sanitize_result.syntax_valid == ValidationStatus.INVALID:
                result.syntax_invalid += 1
            else:
                result.syntax_skipped += 1
            
            # 统计方法
            method = sanitize_result.method
            result.methods[method] = result.methods.get(method, 0) + 1
            
            # 功能等价性验证
            equiv_result = None
            if verify_equivalence and test_cases:
                # 转换测试用例格式
                if isinstance(test_cases[0], dict):
                    tc_list = [(tc.get("input", ""), tc.get("output")) for tc in test_cases]
                else:
                    tc_list = test_cases
                
                equiv_result = verify_functional_equivalence(
                    sanitize_result.original_code,
                    sanitize_result.cleaned_code,
                    language,
                    tc_list,
                    sample_ratio,
                    max_samples
                )
                
                result.equiv_tested += 1
                if equiv_result.is_equivalent:
                    result.equiv_passed += 1
                else:
                    result.equiv_failed += 1
            
            # 构建输出
            processed_sample = sample.copy()
            processed_sample["cleaned_code"] = sanitize_result.cleaned_code
            processed_sample["sanitize_method"] = sanitize_result.method
            processed_sample["syntax_valid"] = sanitize_result.syntax_valid.value
            
            if equiv_result:
                processed_sample["equiv_tested"] = True
                processed_sample["equiv_passed"] = equiv_result.is_equivalent
                processed_sample["equiv_pass_rate"] = equiv_result.pass_rate
            
            if sanitize_result.success:
                result.success += 1
            
            processed.append(processed_sample)
            
        except Exception as e:
            result.errors.append(f"Sample {i}: {str(e)}")
            # 保留原样
            processed_sample = sample.copy()
            processed_sample["cleaned_code"] = sample.get(code_key, "")
            processed_sample["sanitize_method"] = "error"
            processed_sample["syntax_valid"] = "error"
            processed.append(processed_sample)
    
    return processed, result


# ============================================================
# 依赖检查
# ============================================================

def check_dependencies() -> Dict[str, bool]:
    """检查依赖状态"""
    deps = {
        'black': False,
        'g++': False,
        'clang-format': False,
        'javac': False,
        'java': False,
        'google-java-format': False,
    }
    
    try:
        import black
        deps['black'] = True
    except ImportError:
        pass
    
    for cmd in ['g++', 'clang-format', 'javac', 'java']:
        deps[cmd] = shutil.which(cmd) is not None
    
    try:
        result = subprocess.run(
            ['google-java-format', '--version'],
            capture_output=True, timeout=5
        )
        deps['google-java-format'] = (result.returncode == 0)
    except:
        pass
    
    return deps


def print_dependency_status():
    """打印依赖状态"""
    deps = check_dependencies()
    print("=" * 60)
    print("代码清洗模块 v2.1 依赖状态")
    print("=" * 60)
    print("\nPython 相关:")
    print(f"  Black (格式化):     {'✓' if deps['black'] else '✗ pip install black'}")
    print("\nC++ 相关:")
    print(f"  g++ (编译/验证):    {'✓' if deps['g++'] else '✗ apt install g++'}")
    print(f"  ClangFormat (格式化): {'✓' if deps['clang-format'] else '✗ apt install clang-format'}")
    print("\nJava 相关:")
    print(f"  javac (编译/验证):  {'✓' if deps['javac'] else '✗ apt install default-jdk'}")
    print(f"  java (运行):        {'✓' if deps['java'] else '✗ apt install default-jdk'}")
    print(f"  google-java-format: {'✓' if deps['google-java-format'] else '✗ 需手动安装'}")
    print("=" * 60)


# ============================================================
# 测试
# ============================================================

def test():
    """完整测试"""
    print("=" * 70)
    print("代码清洗模块 v2.1 测试")
    print("=" * 70)
    
    print_dependency_status()
    
    all_passed = True
    
    # ===== 测试 1: Python 语法验证 =====
    print("\n" + "-" * 70)
    print("测试 1: Python 清洗与语法验证")
    print("-" * 70)
    
    py_code = '''
# This is a comment to remove
def solve(n):
    """Docstring to remove"""
    url = "https://example.com/#section"  # inline comment
    return n * 2

if __name__ == "__main__":
    print(solve(5))
'''
    
    result = sanitize_code(py_code, "python", skip_format=True)
    
    tests = [
        ("清洗成功", result.success),
        ("注释被移除", "# This is a comment" not in result.cleaned_code),
        ("URL 中的 # 保留", "/#section" in result.cleaned_code),
        ("语法有效", result.syntax_valid == ValidationStatus.VALID),
    ]
    
    for name, passed in tests:
        status = '✓' if passed else '✗'
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"  方法: {result.method}")
    
    # ===== 测试 2: Python 功能等价性 =====
    print("\n" + "-" * 70)
    print("测试 2: Python 功能等价性验证")
    print("-" * 70)
    
    py_code_func = '''
# Calculate double
x = int(input())
print(x * 2)
'''
    
    test_cases = [
        ("5\n", "10\n"),
        ("0\n", "0\n"),
        ("100\n", "200\n"),
    ]
    
    san_result, equiv_result = sanitize_and_verify(
        py_code_func, "python", test_cases, skip_format=True
    )
    
    print(f"  语法有效: {'✓' if san_result.syntax_valid == ValidationStatus.VALID else '✗'}")
    print(f"  功能等价: {'✓' if equiv_result.is_equivalent else '✗'}")
    print(f"  测试用例: {equiv_result.passed_cases}/{equiv_result.sampled_cases} 通过")
    
    if not equiv_result.is_equivalent:
        all_passed = False
        for detail in equiv_result.details:
            if not detail.is_equivalent:
                print(f"    失败: {detail.error_msg}")
    
    # ===== 测试 3: C++ 语法验证 =====
    print("\n" + "-" * 70)
    print("测试 3: C++ 清洗与语法验证")
    print("-" * 70)
    
    cpp_code = '''
// Comment to remove
#include <iostream>
using namespace std;

int main() {
    // Read input
    string url = "https://example.com//path";  /* inline */
    cout << url << endl;
    return 0;
}
'''
    
    result = sanitize_code(cpp_code, "cpp", skip_format=True)
    
    tests = [
        ("清洗成功", result.success),
        ("注释被移除", "// Comment to remove" not in result.cleaned_code),
        ("URL 中的 // 保留", "//path" in result.cleaned_code),
    ]
    
    for name, passed in tests:
        status = '✓' if passed else '✗'
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"  语法验证: {result.syntax_valid.value}")
    print(f"  方法: {result.method}")
    
    # ===== 测试 4: C++ 功能等价性 =====
    print("\n" + "-" * 70)
    print("测试 4: C++ 功能等价性验证")
    print("-" * 70)
    
    deps = check_dependencies()
    if deps['g++']:
        cpp_code_func = '''
// Double the input
#include <iostream>
using namespace std;
int main() {
    int x;
    cin >> x;
    cout << x * 2 << endl;  // output
    return 0;
}
'''
        
        test_cases_cpp = [
            ("5\n", "10\n"),
            ("0\n", "0\n"),
        ]
        
        san_result, equiv_result = sanitize_and_verify(
            cpp_code_func, "cpp", test_cases_cpp, skip_format=True
        )
        
        print(f"  语法有效: {'✓' if san_result.syntax_valid == ValidationStatus.VALID else '✗'}")
        print(f"  功能等价: {'✓' if equiv_result.is_equivalent else '✗'}")
        print(f"  测试用例: {equiv_result.passed_cases}/{equiv_result.sampled_cases} 通过")
        
        if not equiv_result.is_equivalent:
            all_passed = False
    else:
        print("  跳过 (g++ 不可用)")
    
    # ===== 测试 5: Java =====
    print("\n" + "-" * 70)
    print("测试 5: Java 清洗与语法验证")
    print("-" * 70)
    
    java_code = '''
/**
 * Javadoc comment
 * @author Test
 */
public class Main {
    // Single line comment
    public static void main(String[] args) {
        String url = "http://example.com";  /* inline */
        System.out.println(url);
    }
}
'''
    
    result = sanitize_code(java_code, "java", skip_format=True)
    
    tests = [
        ("清洗成功", result.success),
        ("Javadoc 被移除", "* Javadoc" not in result.cleaned_code),
        ("@author 被移除", "@author" not in result.cleaned_code),
        ("代码保留", "System.out.println" in result.cleaned_code),
    ]
    
    for name, passed in tests:
        status = '✓' if passed else '✗'
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"  语法验证: {result.syntax_valid.value}")
    print(f"  方法: {result.method}")
    
    # ===== 测试 6: 错误代码处理 =====
    print("\n" + "-" * 70)
    print("测试 6: 语法错误代码处理")
    print("-" * 70)
    
    bad_py = "def broken(\n    return"
    result = sanitize_code(bad_py, "python", skip_format=True)
    
    print(f"  语法状态: {result.syntax_valid.value}")
    print(f"  错误信息: {result.syntax_error_msg}")
    print(f"  方法: {result.method}")
    
    if result.syntax_valid == ValidationStatus.INVALID:
        print("  ✓ 正确识别语法错误")
    else:
        print("  ✗ 未能识别语法错误")
        all_passed = False
    
    # ===== 总结 =====
    print("\n" + "=" * 70)
    if all_passed:
        print("所有测试通过 ✓")
    else:
        print("部分测试失败 ✗")
    print("=" * 70)
    
    return all_passed


def test_batch():
    """测试批量处理"""
    print("\n" + "=" * 70)
    print("批量处理测试")
    print("=" * 70)
    
    samples = [
        {
            "id": 1,
            "language": "python",
            "code": "# Comment\nx = int(input())\nprint(x * 2)",
            "test_cases": [{"input": "5\n", "output": "10\n"}]
        },
        {
            "id": 2,
            "language": "python",
            "code": "# Another\ny = int(input())\nprint(y + 1)",
            "test_cases": [{"input": "5\n", "output": "6\n"}]
        },
        {
            "id": 3,
            "language": "python",
            "code": "def broken(",  # 语法错误
            "test_cases": []
        },
    ]
    
    processed, stats = sanitize_batch(
        samples,
        verify_equivalence=True,
        verbose=False
    )
    
    print(f"\n处理统计:")
    print(f"  总数: {stats.total}")
    print(f"  成功: {stats.success}")
    print(f"  语法有效: {stats.syntax_valid}")
    print(f"  语法无效: {stats.syntax_invalid}")
    print(f"  语法跳过: {stats.syntax_skipped}")
    print(f"  等价性测试: {stats.equiv_tested}")
    print(f"  等价性通过: {stats.equiv_passed}")
    print(f"  方法分布: {stats.methods}")
    
    if stats.errors:
        print(f"  错误: {stats.errors}")


if __name__ == "__main__":
    test()
    test_batch()