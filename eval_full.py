#!/usr/bin/env python3
"""
LLM 代码性能判断评估 v9 - 简化为3种Prompt策略

支持 3 种 Prompt 策略，所有示例均为语言特定:

| Prompt策略          | 示例   | CoT推理 | 语言特定示例 |
|---------------------|--------|---------|--------------|  
| zero-shot (ZS)      |   ❌   |   ❌    |      -       |
| few-shot (FS)       |   ✅   |   ❌    |   ✅(匹配)   |
| few-shot-cot (FS-CoT)| ✅   |   ✅    |   ✅(匹配)   |

只支持成对比较 (pair-wise comparison):
- 双代码对比，输出 A 或 B (表示哪个代码更快)
"""

import json
import time
import re
import argparse
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, field, asdict
from openai import OpenAI

# ==================== 配置 ====================
API_KEY = ""
BASE_URL = ""
MODEL = ""

# ==================== 模型定价表 (USD per 1M tokens) ====================
MODEL_PRICING = {
    # DeepSeek 系列
    "deepseek-v3": {"input": 0.27, "output": 1.10},
    "deepseek-v3.2": {"input": 0.27, "output": 1.10},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    
    # OpenAI GPT 系列
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-5-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.2": {"input": 5.00, "output": 15.00},
    
    # Anthropic Claude 系列
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    
    # Qwen 系列
    "qwen-turbo": {"input": 0.30, "output": 0.60},
    "qwen-plus": {"input": 0.80, "output": 2.00},
    "qwen-max": {"input": 2.00, "output": 6.00},
    "qwen3-235b": {"input": 2.00, "output": 6.00},
    "qwen3-30b": {"input": 0.50, "output": 1.50},
    "qwen3-coder-480b": {"input": 2.50, "output": 7.50},
    "qwen3-coder-30b": {"input": 0.50, "output": 1.50},
    
    # 默认定价（如果模型未在列表中）
    "default": {"input": 5, "output": 15}
}


def get_model_pricing(model_name: str) -> dict:
    """
    获取模型定价信息
    
    Args:
        model_name: 模型名称
    
    Returns:
        包含 input 和 output 价格的字典 (USD per 1M tokens)
    """
    model_lower = model_name.lower()
    
    # 精确匹配
    for key, pricing in MODEL_PRICING.items():
        if key == "default":
            continue
        if key in model_lower:
            return pricing
    
    # 使用默认定价
    print(f"[WARN] 模型 '{model_name}' 未在定价表中，使用默认定价")
    return MODEL_PRICING["default"]


def calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
    """
    计算API调用成本
    
    Args:
        prompt_tokens: 输入token数
        completion_tokens: 输出token数
        model_name: 模型名称
    
    Returns:
        成本（美元）
    """
    pricing = get_model_pricing(model_name)
    
    # 价格是 per 1M tokens，所以需要除以 1,000,000
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


# ==================== 辅助函数：模型名称处理 ====================
def sanitize_model_name_for_filename(model_name: str) -> str:
    """
    将模型名称清理为适合作为文件名的字符串
    
    Args:
        model_name: 原始模型名称，如 "deepseek-v3.2", "gpt-4-turbo", "claude-3-opus"
    
    Returns:
        清理后的字符串，只保留字母、数字、下划线和连字符
    
    Examples:
        "deepseek-v3.2" -> "deepseek-v3.2"
        "gpt-4-turbo-2024-04-09" -> "gpt-4-turbo-2024-04-09"
        "claude/3/opus" -> "claude_3_opus"
        "model:latest" -> "model_latest"
    """
    if not model_name:
        return ""
    
    import re as re_module
    
    # 将路径分隔符和冒号替换为下划线
    cleaned = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # 只保留字母、数字、下划线、连字符和点
    cleaned = re_module.sub(r'[^a-zA-Z0-9_\-.]', '_', cleaned)
    
    # 移除连续的下划线
    cleaned = re_module.sub(r'_+', '_', cleaned)
    
    # 移除首尾的下划线和点
    cleaned = cleaned.strip('_.')
    
    return cleaned


def get_model_suffix(model_name: str) -> str:
    """
    获取用于文件名的模型后缀
    
    Args:
        model_name: 模型名称
    
    Returns:
        带下划线前缀的模型后缀，如 "_deepseek-v3.2"，若模型名为空则返回空字符串
    """
    cleaned = sanitize_model_name_for_filename(model_name)
    if cleaned:
        return f"_{cleaned}"
    return ""


def generate_default_output_filename(input_path: str, prompt_type: str, model_name: str = None) -> str:
    """
    根据输入文件路径、prompt类型和模型名称生成默认输出文件名
    
    Args:
        input_path: 输入数据文件路径
        prompt_type: prompt类型 (zero-shot, few-shot)
        model_name: 模型名称（可选）
    
    Returns:
        输出文件路径，格式为: {input_stem}_results_{prompt_type}_{model}.json
        例如: cpp_natural_seed42_sanitized_results_zero-shot_deepseek-v3.2.json
    """
    input_p = Path(input_path)
    model_suffix = get_model_suffix(model_name) if model_name else ""
    output_name = f"{input_p.stem}_results_{prompt_type}{model_suffix}.json"
    return str(input_p.parent / output_name)


# ==================== 数据集类型检测 ====================
def detect_dataset_type(data: dict) -> str:
    """检测数据集类型"""
    scheme = data.get("scheme", "")
    if scheme:
        return scheme
    
    if "medium_code" in data or "medium_time" in data:
        return "triple"
    elif "raw_code" in data and "clean_code" in data:
        return "clean_vs_raw"
    elif "tier" in data and data.get("tier") in ["fast", "medium", "slow"]:
        return "same_tier"
    elif "pair_type" in data:
        return "cross_tier"
    elif "slow_code" in data and "fast_code" in data:
        return "cross_tier"
    elif "src_code" in data and "tgt_code" in data:
        return "effibench_pair"
    
    return "unknown"


def get_expected_answer_type(data: dict) -> str:
    """获取期望的答案类型"""
    expected = data.get("expected_answer", "")
    if expected:
        if expected.lower() == "similar":
            return "similar"
        else:
            return "fast"
    
    dataset_type = detect_dataset_type(data)
    if dataset_type in ("same_tier", "clean_vs_raw"):
        return "similar"
    else:
        return "fast"


# ==================== 改进的 Prompt 模板 ====================
class PromptTemplates:
    """改进的Prompt模板 - v8 完整实验设计"""
    
    @staticmethod
    def get_lang_display(lang: str) -> str:
        if lang in ("python", "py", "python3"):
            return "Python"
        elif lang in ("cpp", "c++", "cc"):
            return "C++"
        elif lang in ("java",):
            return "Java"
        return lang.upper()
    
    # ==================== Few-Shot 示例库 ====================
    
    @staticmethod
    def get_examples_for_lang(lang: str, with_cot: bool = True) -> str:
        """获取语言特定的Few-Shot示例"""
        lang_lower = lang.lower()
        
        if lang_lower in ("python", "py", "python3"):
            return PromptTemplates._get_python_examples(with_cot)
        elif lang_lower in ("cpp", "c++", "cc"):
            return PromptTemplates._get_cpp_examples(with_cot)
        elif lang_lower in ("java",):
            return PromptTemplates._get_java_examples(with_cot)
        else:
            # 默认使用C++示例
            return PromptTemplates._get_cpp_examples(with_cot)
    
    @staticmethod
    def _get_cpp_examples(with_cot: bool = True) -> str:
        """C++ Few-Shot 示例"""
        if with_cot:
            return """
### Example 1:

Code A:
```cpp
int sum = 0;
for (int i = 0; i < n; i++) {
    sum += arr[i];
}
```

Code B:
```cpp
int sum = 0;
for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
        if (j == i) sum += arr[i];
    }
}
```

**Code A Analysis:**
- Algorithm/Approach: Direct summation
- Time Complexity: O(n)
- Key Operations: n additions

**Code B Analysis:**
- Algorithm/Approach: Unnecessary nested loop
- Time Complexity: O(n²)
- Key Operations: n²/2 comparisons + n additions

**Answer: A**

### Example 2:

Code A:
```cpp
bool found = false;
for (int i = 0; i < n; i++) {
    if (arr[i] == target) found = true;
}
return found;
```

Code B:
```cpp
for (int i = 0; i < n; i++) {
    if (arr[i] == target) return true;
}
return false;
```

**Code A Analysis:**
- Algorithm/Approach: Linear search without early exit
- Time Complexity: O(n) always
- Key Operations: Always traverses entire array

**Code B Analysis:**
- Algorithm/Approach: Linear search with early exit
- Time Complexity: O(n) worst case, O(1) best case
- Key Operations: Returns immediately when found

**Answer: B**
"""
        else:
            # 无CoT版本 - 只给答案
            return """
### Example 1:

Code A:
```cpp
int sum = 0;
for (int i = 0; i < n; i++) {
    sum += arr[i];
}
```

Code B:
```cpp
int sum = 0;
for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
        if (j == i) sum += arr[i];
    }
}
```

**Answer: A**

### Example 2:

Code A:
```cpp
bool found = false;
for (int i = 0; i < n; i++) {
    if (arr[i] == target) found = true;
}
return found;
```

Code B:
```cpp
for (int i = 0; i < n; i++) {
    if (arr[i] == target) return true;
}
return false;
```

**Answer: B**
"""

    @staticmethod
    def _get_python_examples(with_cot: bool = True) -> str:
        """Python Few-Shot 示例"""
        if with_cot:
            return """
### Example 1:

Code A:
```python
total = 0
for i in range(len(arr)):
    total += arr[i]
```

Code B:
```python
total = 0
for i in range(len(arr)):
    for j in range(i + 1):
        if j == i:
            total += arr[i]
```

**Code A Analysis:**
- Algorithm/Approach: Direct summation with index iteration
- Time Complexity: O(n)
- Key Operations: n additions

**Code B Analysis:**
- Algorithm/Approach: Unnecessary nested loop
- Time Complexity: O(n²)
- Key Operations: n²/2 comparisons + n additions

**Answer: A**

### Example 2:

Code A:
```python
def find_target(arr, target):
    found = False
    for x in arr:
        if x == target:
            found = True
    return found
```

Code B:
```python
def find_target(arr, target):
    for x in arr:
        if x == target:
            return True
    return False
```

**Code A Analysis:**
- Algorithm/Approach: Linear search without early exit
- Time Complexity: O(n) always
- Key Operations: Always traverses entire array

**Code B Analysis:**
- Algorithm/Approach: Linear search with early exit
- Time Complexity: O(n) worst case, O(1) best case
- Key Operations: Returns immediately when found

**Answer: B**
"""
        else:
            return """
### Example 1:

Code A:
```python
total = 0
for i in range(len(arr)):
    total += arr[i]
```

Code B:
```python
total = 0
for i in range(len(arr)):
    for j in range(i + 1):
        if j == i:
            total += arr[i]
```

**Answer: A**

### Example 2:

Code A:
```python
def find_target(arr, target):
    found = False
    for x in arr:
        if x == target:
            found = True
    return found
```

Code B:
```python
def find_target(arr, target):
    for x in arr:
        if x == target:
            return True
    return False
```

**Answer: B**
"""

    @staticmethod
    def _get_java_examples(with_cot: bool = True) -> str:
        """Java Few-Shot 示例"""
        if with_cot:
            return """
### Example 1:

Code A:
```java
int sum = 0;
for (int i = 0; i < arr.length; i++) {
    sum += arr[i];
}
```

Code B:
```java
int sum = 0;
for (int i = 0; i < arr.length; i++) {
    for (int j = 0; j <= i; j++) {
        if (j == i) sum += arr[i];
    }
}
```

**Code A Analysis:**
- Algorithm/Approach: Direct summation
- Time Complexity: O(n)
- Key Operations: n additions

**Code B Analysis:**
- Algorithm/Approach: Unnecessary nested loop
- Time Complexity: O(n²)
- Key Operations: n²/2 comparisons + n additions

**Answer: A**

### Example 2:

Code A:
```java
public boolean findTarget(int[] arr, int target) {
    boolean found = false;
    for (int x : arr) {
        if (x == target) found = true;
    }
    return found;
}
```

Code B:
```java
public boolean findTarget(int[] arr, int target) {
    for (int x : arr) {
        if (x == target) return true;
    }
    return false;
}
```

**Code A Analysis:**
- Algorithm/Approach: Linear search without early exit
- Time Complexity: O(n) always
- Key Operations: Always traverses entire array

**Code B Analysis:**
- Algorithm/Approach: Linear search with early exit
- Time Complexity: O(n) worst case, O(1) best case
- Key Operations: Returns immediately when found

**Answer: B**
"""
        else:
            return """
### Example 1:

Code A:
```java
int sum = 0;
for (int i = 0; i < arr.length; i++) {
    sum += arr[i];
}
```

Code B:
```java
int sum = 0;
for (int i = 0; i < arr.length; i++) {
    for (int j = 0; j <= i; j++) {
        if (j == i) sum += arr[i];
    }
}
```

**Answer: A**

### Example 2:

Code A:
```java
public boolean findTarget(int[] arr, int target) {
    boolean found = false;
    for (int x : arr) {
        if (x == target) found = true;
    }
    return found;
}
```

Code B:
```java
public boolean findTarget(int[] arr, int target) {
    for (int x : arr) {
        if (x == target) return true;
    }
    return false;
}
```

**Answer: B**
"""
    
    # ==================== 双代码模板 (A/B/Similar) ====================
    
    @staticmethod
    def zero_shot_pair(code_a: str, code_b: str, lang: str) -> dict:
        """
        Zero-Shot (ZS): Direct comparison without examples or reasoning
        The model is asked to directly compare two functionally equivalent implementations
        and determine which one is more efficient. Outputs only the identifier (A or B).
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs faster based on algorithmic complexity, data structure efficiency, and implementation details."""
        
        user_prompt = f"""Compare the following two functionally equivalent {lang_display} implementations and determine which one is more efficient.

## Code A:
```{lang}
{code_a}
```

## Code B:
```{lang}
{code_b}
```

Based on algorithm complexity, data structures, loop efficiency, and other performance metrics, determine which implementation runs faster.

You must output ONLY the identifier of the faster implementation:
- Output "A" if Code A is faster
- Output "B" if Code B is faster

Your response must be exactly one character: A or B

No explanation or additional text."""
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": False
        }
    
    @staticmethod
    def zero_shot_cot_pair(code_a: str, code_b: str, lang: str) -> dict:
        """
        Zero-Shot CoT: 无示例，但要求推理过程
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs faster based on algorithmic complexity, data structure efficiency, and implementation details.

CRITICAL OUTPUT FORMAT REQUIREMENT:
You MUST end your response with EXACTLY this format:
**Answer: X**

Where X is either 'A' or 'B' (single letter, no other text).
This line must be the LAST line of your response.
Do NOT add any text after the answer line."""
        
        user_prompt = f"""Determine which of the following two {lang_display} code snippets runs faster.

## Code A:
```{lang}
{code_a}
```

## Code B:
```{lang}
{code_b}
```

These two code snippets have different performance characteristics. Analyze using the following format:

**Code A Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Code B Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Comparison Conclusion:**
Based on the analysis above, determine which code is faster. You must choose one:
- A: Code A is faster
- B: Code B is faster

You MUST end your response with EXACTLY this format on the last line:
**Answer: A or B**"""
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": True
        }
    
    @staticmethod
    def few_shot_pair(code_a: str, code_b: str, lang: str) -> dict:
        """
        Few-Shot (FS): With 2 in-context examples showing only input and output
        The prompt includes two in-context examples that demonstrate correct efficiency
        comparisons between code pairs. Each example presents the input code snippets
        and the final decision, but does not expose intermediate reasoning steps.
        Always uses language-specific examples matching the target language.
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        # 始终使用语言特定的示例
        examples = PromptTemplates.get_examples_for_lang(lang, with_cot=False)
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs faster based on algorithmic complexity, data structure efficiency, and implementation details."""
        
        user_prompt = f"""Compare the following two functionally equivalent {lang_display} implementations and determine which one is more efficient.

Here are two examples of correct efficiency comparisons:
{examples}

### Now analyze the following {lang_display} code:

Code A:
```{lang}
{code_a}
```

Code B:
```{lang}
{code_b}
```

Based on algorithm complexity, data structures, and implementation efficiency, determine which implementation runs faster.

You must output ONLY the identifier:
- Output "A" if Code A is faster
- Output "B" if Code B is faster

Your response must be exactly one character: A or B"""
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": False
        }
    
    @staticmethod
    def few_shot_cot_pair(code_a: str, code_b: str, lang: str) -> dict:
        """
        Few-Shot Chain-of-Thought (FS-CoT): With explicit analytical reasoning
        The prompt includes two in-context examples with explicit analytical reasoning.
        Each example illustrates a step-by-step efficiency analysis, including algorithm
        identification, complexity analysis, and key operation counting, followed by the
        final comparison outcome.
        Always uses language-specific examples matching the target language.
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        # 始终使用语言特定的示例
        examples = PromptTemplates.get_examples_for_lang(lang, with_cot=True)
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs faster based on algorithmic complexity, data structure efficiency, and implementation details.

CRITICAL OUTPUT FORMAT REQUIREMENT:
You MUST end your response with EXACTLY this format:
**Answer: X**

Where X is either 'A' or 'B' (single letter, no other text).
This line must be the LAST line of your response.
Do NOT add any text after the answer line."""
        
        user_prompt = f"""Compare the following two functionally equivalent {lang_display} implementations and determine which one is more efficient.

Here are two examples with step-by-step reasoning:
{examples}

### Now analyze the following {lang_display} code:

Code A:
```{lang}
{code_a}
```

Code B:
```{lang}
{code_b}
```

Provide a step-by-step efficiency analysis using the following format:

**Code A Analysis:**
- Algorithm/Approach: [describe the algorithmic approach]
- Time Complexity: [Big-O notation]
- Key Operations: [count or describe critical operations]

**Code B Analysis:**
- Algorithm/Approach: [describe the algorithmic approach]
- Time Complexity: [Big-O notation]
- Key Operations: [count or describe critical operations]

**Answer: [A or B]**

Remember: Your LAST line must be exactly "**Answer: X**" where X is A or B."""

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": True
        }
    
    # ==================== 三元组模板 (A/B/C) ====================
    
    @staticmethod
    def zero_shot_triple(code_a: str, code_b: str, code_c: str, lang: str) -> dict:
        """
        Zero-Shot v8: Triple comparison
        Determine which code is fastest
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs fastest based on algorithmic complexity, data structure efficiency, and implementation details."""
        
        user_prompt = f"""Determine which of the following three {lang_display} code snippets runs fastest.

## Code A:
```{lang}
{code_a}
```

## Code B:
```{lang}
{code_b}
```

## Code C:
```{lang}
{code_c}
```

These three code snippets have different performance characteristics. Based on algorithm complexity, data structures, and implementation efficiency, determine which one is fastest.

You must choose one:
- A: Code A is fastest
- B: Code B is fastest
- C: Code C is fastest

Respond with only "A", "B", or "C", no explanation."""
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": False
        }
    
    @staticmethod
    def zero_shot_cot_triple(code_a: str, code_b: str, code_c: str, lang: str) -> dict:
        """
        Zero-Shot CoT Triple: 无示例，但要求推理过程
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs fastest based on algorithmic complexity, data structure efficiency, and implementation details.

CRITICAL OUTPUT FORMAT REQUIREMENT:
You MUST end your response with EXACTLY this format:
**Answer: X**

Where X is either 'A', 'B', or 'C' (single letter, no other text).
This line must be the LAST line of your response.
Do NOT add any text after the answer line."""
        
        user_prompt = f"""Determine which of the following three {lang_display} code snippets runs fastest.

## Code A:
```{lang}
{code_a}
```

## Code B:
```{lang}
{code_b}
```

## Code C:
```{lang}
{code_c}
```

These three code snippets have different performance characteristics. Analyze using the following format:

**Code A Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Code B Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Code C Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Comparison Conclusion:**
Based on the analysis above, determine which code is fastest.

You MUST end your response with EXACTLY this format on the last line:
**Answer: A, B, or C**"""
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": True
        }
    
    @staticmethod
    def few_shot_triple(code_a: str, code_b: str, code_c: str, lang: str) -> dict:
        """
        Few-Shot w/o CoT Triple: 有示例，但不要求推理过程
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        # Triple示例（无CoT）
        examples = """
### Example:

Code A:
```cpp
int sum = 0;
for (int i = 0; i < n; i++) sum += arr[i];
```

Code B:
```cpp
int sum = 0;
for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
        if (j == i) sum += arr[i];
    }
}
```

Code C:
```cpp
int sum = 0;
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        if (i == j) sum += arr[i];
    }
}
```

**Answer: A**
"""
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs fastest based on algorithmic complexity, data structure efficiency, and implementation details."""
        
        user_prompt = f"""Determine which of the following three {lang_display} code snippets runs fastest.

Here is an example:
{examples}

### Now analyze the following {lang_display} code:

Code A:
```{lang}
{code_a}
```

Code B:
```{lang}
{code_b}
```

Code C:
```{lang}
{code_c}
```

Based on algorithm complexity, data structures, and implementation efficiency, determine which one is fastest.

You must choose one:
- A: Code A is fastest
- B: Code B is fastest
- C: Code C is fastest

Respond with only "A", "B", or "C", no explanation."""
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": False
        }
    
    @staticmethod
    def few_shot_cot_triple(code_a: str, code_b: str, code_c: str, lang: str) -> dict:
        """
        Few-Shot CoT v8: Triple comparison with reasoning
        Determine which code is fastest through step-by-step analysis
        """
        lang_display = PromptTemplates.get_lang_display(lang)
        
        system_prompt = f"""You are a senior Performance Engineer with 10+ years of experience in {lang_display} optimization.

Your task is to determine which code runs fastest based on algorithmic complexity, data structure efficiency, and implementation details.

CRITICAL OUTPUT FORMAT REQUIREMENT:
You MUST end your response with EXACTLY this format:
**Answer: X**

Where X is either 'A', 'B', or 'C' (single letter, no other text).
This line must be the LAST line of your response.
Do NOT add any text after the answer line."""
        
        user_prompt = f"""You are a code performance analysis expert. Analyze the following three code snippets and determine which one runs fastest.

### Analyze the following {lang_display} code:

Code A:
```{lang}
{code_a}
```

Code B:
```{lang}
{code_b}
```

Code C:
```{lang}
{code_c}
```

These three code snippets solve the same problem but have different performance characteristics. Analyze using the following format:

**Code A Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Code B Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Code C Analysis:**
- Algorithm/Approach:
- Time Complexity:
- Key Operations and their costs:

**Comparison Conclusion:**
Based on the analysis above, determine which code is fastest.

CRITICAL: Your response MUST end with EXACTLY this format on the last line:
**Answer: A**
or
**Answer: B**
or
**Answer: C**

REQUIREMENTS:
1. The answer line must be the LAST line of your response
2. Use exactly "**Answer: A**" or "**Answer: B**" or "**Answer: C**" (replace with your choice)
3. Do NOT write anything after this line
4. Do NOT add explanations, periods, or any other characters after the letter
5. The letter must be A, B, or C only (uppercase, single character)

EXAMPLE OF CORRECT FORMAT:
**Code A Analysis:**
...
**Code B Analysis:**
...
**Code C Analysis:**
...
**Comparison Conclusion:**
...
**Answer: C**

[END OF RESPONSE - Nothing should appear after the Answer line]"""

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "expect_cot": True
        }
    

# ==================== 答案提取 ====================
def extract_answer(response: str, is_triple: bool = False) -> Tuple[str, str]:
    """
    从响应的最后一行提取答案
    
    只接受格式: **Answer: X** 或 Answer: X
    其中 X 为 A, B, C (或 Similar)
    """
    response_clean = response.strip()
    
    if not response_clean:
        return "UNKNOWN", response_clean
    
    # 只检查最后一行
    last_line = response_clean.split('\n')[-1].strip()
    
    # 匹配 **Answer: X** 或 Answer: X 格式
    # 支持中英文，支持带/不带星号
    pattern = r'\*?\*?(?:Answer|答案)[：:\s]*([ABC]|Similar|差不多)\*?\*?'
    match = re.search(pattern, last_line, re.IGNORECASE)
    
    if match:
        answer = match.group(1).upper()
        
        # 中文映射
        if answer == "差不多":
            answer = "SIMILAR"
        
        # 标准化
        valid_answers = ["A", "B", "C", "SIMILAR"]
        if answer in valid_answers:
            # Pair 模式不应该有 C
            if not is_triple and answer == "C":
                return "PARSE_FAILED", response_clean
            
            return answer, response_clean
    
    # 如果最后一行只包含单个字母 A/B/C (去除所有符号后)
    cleaned_last = last_line.upper().replace('*', '').replace('.', '').replace(':', '').replace(' ', '')
    if cleaned_last in ["A", "B", "C"]:
        answer = cleaned_last
        # Pair 模式不应该有 C
        if not is_triple and answer == "C":
            return "PARSE_FAILED", response_clean
        return answer, response_clean
    
    # 未能从最后一行提取答案 - 标记为解析失败
    return "PARSE_FAILED", response_clean


# ==================== 数据类 ====================
@dataclass
class PairEvalResult:
    """双代码评估结果"""
    pair_id: str
    problem_id: str = ""
    language: str = "cpp"
    dataset_type: str = ""
    prompt_type: str = "zero-shot"
    expected_answer_type: str = "fast"
    
    # Speedup bin 信息 (来自数据集构建)
    speedup_bin: str = ""
    speedup_bin_idx: int = -1
    
    original_slow_time: float = 0
    original_fast_time: float = 0
    original_speedup: Optional[float] = None
    
    test1_order: str = "slow_A_fast_B"
    test1_correct_answer: str = "B"
    test1_llm_prediction: str = ""
    test1_llm_raw_response: str = ""
    test1_reasoning_trace: str = ""
    test1_correct: Optional[bool] = None
    test1_prompt_tokens: int = 0
    test1_completion_tokens: int = 0
    test1_total_tokens: int = 0
    
    test2_order: str = "fast_A_slow_B"
    test2_correct_answer: str = "A"
    test2_llm_prediction: str = ""
    test2_llm_raw_response: str = ""
    test2_reasoning_trace: str = ""
    test2_correct: Optional[bool] = None
    test2_prompt_tokens: int = 0
    test2_completion_tokens: int = 0
    test2_total_tokens: int = 0
    
    is_consistent: Optional[bool] = None
    category: str = ""
    
    error: Optional[str] = None


@dataclass
class TripleEvalResult:
    """三元组评估结果"""
    pair_id: str
    problem_id: str = ""
    language: str = "cpp"
    dataset_type: str = "triple"
    prompt_type: str = "zero-shot"
    expected_answer_type: str = "fast"
    
    fast_time: float = 0
    medium_time: float = 0
    slow_time: float = 0
    speedup_fast_vs_slow: Optional[float] = None
    
    test1_order: str = "fast_A_medium_B_slow_C"
    test1_correct_answer: str = "A"
    test1_llm_prediction: str = ""
    test1_llm_raw_response: str = ""
    test1_reasoning_trace: str = ""
    test1_correct: Optional[bool] = None
    test1_prompt_tokens: int = 0
    test1_completion_tokens: int = 0
    test1_total_tokens: int = 0
    
    test2_order: str = "slow_A_fast_B_medium_C"
    test2_correct_answer: str = "B"
    test2_llm_prediction: str = ""
    test2_llm_raw_response: str = ""
    test2_reasoning_trace: str = ""
    test2_correct: Optional[bool] = None
    test2_prompt_tokens: int = 0
    test2_completion_tokens: int = 0
    test2_total_tokens: int = 0
    
    test3_order: str = "medium_A_slow_B_fast_C"
    test3_correct_answer: str = "C"
    test3_llm_prediction: str = ""
    test3_llm_raw_response: str = ""
    test3_reasoning_trace: str = ""
    test3_correct: Optional[bool] = None
    test3_prompt_tokens: int = 0
    test3_completion_tokens: int = 0
    test3_total_tokens: int = 0
    
    num_correct: int = 0
    is_all_correct: bool = False
    is_consistent: Optional[bool] = None
    category: str = ""
    
    error: Optional[str] = None


# ==================== LLM 客户端 ====================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ask_llm(codes: List[str], lang: str, prompt_type: str, 
            is_triple: bool = False, retry: int = 3) -> dict:
    """
    统一的LLM调用接口 - v9 简化版
    
    支持的 prompt_type:
    - zero-shot: 无示例，无CoT
    - few-shot: 有示例（语言特定），无CoT
    - few-shot-cot: 有示例（语言特定），有CoT
    
    只支持 pair-wise 比较（is_triple 必须为 False）
    
    返回值包含:
    - prediction: 预测结果 (A/B/PARSE_FAILED/ERROR)
    - raw_response: 原始响应
    - reasoning_trace: 推理过程
    - prompt_tokens: 输入token数
    - completion_tokens: 输出token数
    - total_tokens: 总token数
    """
    
    if is_triple:
        raise ValueError("Triple comparison is no longer supported in v9")
    
    # ==================== 第一步：准备Prompt ====================
    code_a, code_b = codes[0], codes[1]
    
    # 记录代码长度
    total_len = len(code_a) + len(code_b)
    if total_len > 10000:
        print(f"[INFO] Large code pair ({total_len} chars)")
    
    # 根据prompt类型选择模板
    if prompt_type == "zero-shot":
        prompt_data = PromptTemplates.zero_shot_pair(code_a, code_b, lang)
    elif prompt_type == "few-shot":
        prompt_data = PromptTemplates.few_shot_pair(code_a, code_b, lang)
    elif prompt_type == "few-shot-cot":
        prompt_data = PromptTemplates.few_shot_cot_pair(code_a, code_b, lang)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Supported types: zero-shot, few-shot, few-shot-cot")
    
    # ==================== 第二步：设置参数 ====================
    expect_cot = prompt_data.get("expect_cot", False)
    
    # 根据prompt类型设置max_tokens
    if expect_cot:
        max_tokens = 8000  # Few-shot需要详细分析
    else:
        max_tokens = 8000  # Zero-shot只需要简短回答
    
    # ==================== 第三步：重试循环 ====================
    for attempt in range(retry):
        try:
            # 构建API参数
            api_params = {
                "model": MODEL,
                "messages": prompt_data["messages"],
                "temperature": 0.1,
                "timeout": 180
            }
            
            # 🔥 关键修复1: 根据模型选择正确的参数名
            # GPT-5 系列使用 max_completion_tokens，其他模型使用 max_tokens
            if "gpt-5" in MODEL.lower() or "gpt5" in MODEL.lower():
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens
            
            # 调用API
            resp = client.chat.completions.create(**api_params)
            
            # 🔥 容错保护：检查响应对象类型，防止某些中转站返回字符串导致崩溃
            if isinstance(resp, str):
                raise ValueError(f"API returned a string instead of an object: {resp[:100]}")
            
            if not hasattr(resp, 'choices') or not resp.choices:
                raise ValueError(f"API returned an invalid response object: {type(resp)}")
            
            # 🔥 关键修复3: 检查content是否为None
            message_content = resp.choices[0].message.content
            finish_reason = resp.choices[0].finish_reason
            
            # 处理空响应
            if message_content is None or message_content.strip() == "":
                error_msg = f"Empty content (finish_reason: {finish_reason})"
                print(f"[WARN] Attempt {attempt+1}/{retry}: {error_msg}")
                
                # 如果是内容过滤，直接返回错误，不重试
                if finish_reason == "content_filter":
                    return {
                        "prediction": "ERROR",
                        "raw_response": "Content filtered by API",
                        "reasoning_trace": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                
                # 如果是其他原因的空响应，重试
                if attempt < retry - 1:
                    print(f"[RETRY] Retrying after empty response...")
                    time.sleep(3 * (attempt + 1))  # 递增等待时间
                    continue
                
                # 最后一次重试也失败了
                return {
                    "prediction": "ERROR",
                    "raw_response": error_msg,
                    "reasoning_trace": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            # 获取响应内容
            raw_response = message_content.strip()
            
            # 🔥 关键修复4: 处理响应被截断的情况
            if finish_reason == "length":
                print(f"[WARN] Response truncated at {max_tokens} tokens")
                prediction, reasoning_trace = extract_answer(raw_response, is_triple)
                
                # 如果提取失败且还有重试机会，增加max_tokens重试
                if prediction == "UNKNOWN" and attempt < retry - 1:
                    new_max_tokens = int(max_tokens * 1.5)
                    print(f"[RETRY] Increasing max_tokens from {max_tokens} to {new_max_tokens}")
                    max_tokens = new_max_tokens
                    time.sleep(2)
                    continue
                
                # 如果成功提取或已是最后一次重试，返回结果
                usage = resp.usage if hasattr(resp, 'usage') else None
                return {
                    "prediction": prediction,
                    "raw_response": raw_response + "\n\n[RESPONSE TRUNCATED]",
                    "reasoning_trace": reasoning_trace if expect_cot else "",
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                }
            
            # ==================== 正常情况：提取答案 ====================
            prediction, reasoning_trace = extract_answer(raw_response, is_triple)
            
            # 如果提取失败但响应看起来正常，记录警告
            if prediction == "PARSE_FAILED":
                print(f"[WARN] Answer parse failed - last line: {raw_response.split(chr(10))[-1][:100]}")
            
            # 获取token使用信息
            usage = resp.usage if hasattr(resp, 'usage') else None
            
            return {
                "prediction": prediction,
                "raw_response": raw_response,
                "reasoning_trace": reasoning_trace if expect_cot else "",
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{retry} failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            
            # 如果还有重试机会，继续
            if attempt < retry - 1:
                wait_time = 3 * (attempt + 1)
                print(f"[RETRY] Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            
            # 所有重试都失败了
            return {
                "prediction": "ERROR",
                "raw_response": str(e)[:500],  # 限制错误信息长度
                "reasoning_trace": ""
            }
    
    # 理论上不应该到这里，但以防万一
    return {
        "prediction": "ERROR",
        "raw_response": "Max retries exceeded",
        "reasoning_trace": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

# ==================== 处理函数 ====================
def process_pair(data: dict, prompt_type: str = "zero-shot") -> PairEvalResult:
    """处理双代码配对"""
    
    # 使用预生成的 pair_id，或自行生成
    pair_id = data.get("_generated_pair_id") or data.get("pair_id")
    if not pair_id:
        problem_id = data.get("problem_id", "")
        speedup = data.get("speedup", "")
        pair_id = f"{problem_id}_{speedup}" if problem_id else "unknown"
    problem_id = data.get("problem_id", "")
    dataset_type = detect_dataset_type(data)
    expected_answer_type = get_expected_answer_type(data)
    lang = data.get("language", "cpp").lower()
    
    if dataset_type == "clean_vs_raw":
        code_slow = data.get("raw_code", "")
        code_fast = data.get("clean_code", "")
        slow_time = data.get("cpu_time", 0)
        fast_time = data.get("cpu_time", 0)
    elif dataset_type == "same_tier":
        code_slow = data.get("code_b", "")
        code_fast = data.get("code_a", "")
        slow_time = data.get("time_b", 0)
        fast_time = data.get("time_a", 0)
    elif dataset_type == "effibench_pair":
        code_slow = data.get("src_code", "")
        code_fast = data.get("tgt_code", "")
        try:
            slow_time = float(data.get("src_agg_runtime", 0) or 0)
            fast_time = float(data.get("tgt_agg_runtime", 0) or 0)
        except ValueError:
            slow_time, fast_time = 0, 0
    else:
        code_slow = data.get("slow_code", data.get("code_b", ""))
        code_fast = data.get("fast_code", data.get("code_a", ""))
        slow_time = data.get("slow_time", data.get("time_b", 0))
        fast_time = data.get("fast_time", data.get("time_a", 0))
    
    if expected_answer_type == "similar":
        test1_correct_answer = "SIMILAR"
        test2_correct_answer = "SIMILAR"
    else:
        test1_correct_answer = "B"
        test2_correct_answer = "A"
    
    # 自动计算 speedup_bin（如果数据中没有）
    # 使用统一的难度等级分档: hard=[2,4), medium=[4,8), easy=[8,inf)
    speedup_bin = data.get("speedup_bin", "")
    speedup_bin_idx = data.get("speedup_bin_idx", -1)
    
    if not speedup_bin and fast_time > 0:
        speedup = data.get("speedup") or (slow_time / fast_time)
        bins = [2.0, 4.0, 8.0, float('inf')]
        difficulty_map = {0: 'hard', 1: 'medium', 2: 'easy'}
        
        for i in range(len(bins) - 1):
            if bins[i] <= speedup < bins[i + 1]:
                difficulty = difficulty_map.get(i, f'level_{i}')
                if bins[i + 1] == float('inf'):
                    speedup_bin = f"{difficulty} [≥{bins[i]}x]"
                else:
                    speedup_bin = f"{difficulty} [{bins[i]}x-{bins[i+1]}x)"
                speedup_bin_idx = i
                break
        
        # 如果 speedup < 2，归入 hard
        if not speedup_bin and speedup >= 1:
            speedup_bin = "below_threshold"
            speedup_bin_idx = -1
    
    result = PairEvalResult(
        pair_id=str(pair_id),
        problem_id=str(problem_id),
        language=lang,
        dataset_type=dataset_type,
        prompt_type=prompt_type,
        expected_answer_type=expected_answer_type,
        speedup_bin=speedup_bin,
        speedup_bin_idx=speedup_bin_idx,
        original_slow_time=slow_time,
        original_fast_time=fast_time,
        original_speedup=round(slow_time / fast_time, 2) if fast_time > 0 else None,
        test1_correct_answer=test1_correct_answer,
        test2_correct_answer=test2_correct_answer,
    )
    
    if not code_slow or not code_fast:
        result.error = "missing_code"
        return result
    
    # Test 1: slow=A, fast=B
    llm_result1 = ask_llm([code_slow, code_fast], lang, prompt_type, is_triple=False)
    result.test1_llm_prediction = llm_result1["prediction"]
    result.test1_llm_raw_response = llm_result1["raw_response"]
    result.test1_reasoning_trace = llm_result1["reasoning_trace"]
    result.test1_prompt_tokens = llm_result1.get("prompt_tokens", 0)
    result.test1_completion_tokens = llm_result1.get("completion_tokens", 0)
    result.test1_total_tokens = llm_result1.get("total_tokens", 0)
    
    if llm_result1["prediction"] in ["A", "B"]:
        result.test1_correct = (llm_result1["prediction"] == test1_correct_answer)
    elif llm_result1["prediction"] == "PARSE_FAILED":
        result.error = "test1_answer_parse_failed"
        return result
    else:
        result.error = f"test1_error: {llm_result1['raw_response'][:100]}"
        return result
    
    time.sleep(0.3)
    
    # Test 2: fast=A, slow=B
    llm_result2 = ask_llm([code_fast, code_slow], lang, prompt_type, is_triple=False)
    result.test2_llm_prediction = llm_result2["prediction"]
    result.test2_llm_raw_response = llm_result2["raw_response"]
    result.test2_reasoning_trace = llm_result2["reasoning_trace"]
    result.test2_prompt_tokens = llm_result2.get("prompt_tokens", 0)
    result.test2_completion_tokens = llm_result2.get("completion_tokens", 0)
    result.test2_total_tokens = llm_result2.get("total_tokens", 0)
    
    if llm_result2["prediction"] in ["A", "B"]:
        result.test2_correct = (llm_result2["prediction"] == test2_correct_answer)
    elif llm_result2["prediction"] == "PARSE_FAILED":
        result.error = "test2_answer_parse_failed"
        return result
    else:
        result.error = f"test2_error: {llm_result2['raw_response'][:100]}"
        return result
    
    # 综合分析
    test1_chose_fast = (result.test1_llm_prediction == "B")
    test2_chose_fast = (result.test2_llm_prediction == "A")
    result.is_consistent = (test1_chose_fast == test2_chose_fast)
    
    if result.test1_correct and result.test2_correct:
        result.category = "both_correct"
    elif not result.test1_correct and not result.test2_correct:
        result.category = "both_wrong"
    else:
        result.category = "position_bias"
    
    return result


def process_triple(data: dict, prompt_type: str = "zero-shot") -> TripleEvalResult:
    """处理三元组"""
    
    # 使用预生成的 pair_id，或自行生成
    pair_id = data.get("_generated_pair_id") or data.get("pair_id")
    if not pair_id:
        problem_id = data.get("problem_id", "")
        speedup = data.get("speedup", "")
        pair_id = f"{problem_id}_{speedup}" if problem_id else "unknown"
    problem_id = data.get("problem_id", "")
    lang = data.get("language", "cpp").lower()
    
    fast_code = data.get("fast_code", "")
    medium_code = data.get("medium_code", "")
    slow_code = data.get("slow_code", "")
    
    fast_time = data.get("fast_time", 0)
    medium_time = data.get("medium_time", 0)
    slow_time = data.get("slow_time", 0)
    
    result = TripleEvalResult(
        pair_id=str(pair_id),
        problem_id=str(problem_id),
        language=lang,
        prompt_type=prompt_type,
        expected_answer_type="fast",
        fast_time=fast_time,
        medium_time=medium_time,
        slow_time=slow_time,
        speedup_fast_vs_slow=round(slow_time / fast_time, 2) if fast_time > 0 else None
    )
    
    if not fast_code or not medium_code or not slow_code:
        result.error = "missing_code"
        return result
    
    # Test 1: fast=A, medium=B, slow=C，正确=A
    llm1 = ask_llm([fast_code, medium_code, slow_code], lang, prompt_type, is_triple=True)
    result.test1_llm_prediction = llm1["prediction"]
    result.test1_llm_raw_response = llm1["raw_response"]
    result.test1_reasoning_trace = llm1["reasoning_trace"]
    
    if llm1["prediction"] in ["A", "B", "C"]:
        result.test1_correct = (llm1["prediction"] == "A")
    elif llm1["prediction"] == "PARSE_FAILED":
        result.error = "test1_answer_parse_failed"
        return result
    else:
        result.error = f"test1_error"
        return result
    
    time.sleep(0.3)
    
    # Test 2: slow=A, fast=B, medium=C，正确=B
    llm2 = ask_llm([slow_code, fast_code, medium_code], lang, prompt_type, is_triple=True)
    result.test2_llm_prediction = llm2["prediction"]
    result.test2_llm_raw_response = llm2["raw_response"]
    result.test2_reasoning_trace = llm2["reasoning_trace"]
    
    if llm2["prediction"] in ["A", "B", "C"]:
        result.test2_correct = (llm2["prediction"] == "B")
    elif llm2["prediction"] == "PARSE_FAILED":
        result.error = "test2_answer_parse_failed"
        return result
    else:
        result.error = f"test2_error"
        return result
    
    time.sleep(0.3)
    
    # Test 3: medium=A, slow=B, fast=C，正确=C
    llm3 = ask_llm([medium_code, slow_code, fast_code], lang, prompt_type, is_triple=True)
    result.test3_llm_prediction = llm3["prediction"]
    result.test3_llm_raw_response = llm3["raw_response"]
    result.test3_reasoning_trace = llm3["reasoning_trace"]
    
    if llm3["prediction"] in ["A", "B", "C"]:
        result.test3_correct = (llm3["prediction"] == "C")
    elif llm3["prediction"] == "PARSE_FAILED":
        result.error = "test3_answer_parse_failed"
        return result
    else:
        result.error = f"test3_error"
        return result
    
    # 综合分析
    result.num_correct = sum([
        result.test1_correct or False,
        result.test2_correct or False,
        result.test3_correct or False
    ])
    result.is_all_correct = (result.num_correct == 3)
    
    t1_chose_fast = (result.test1_llm_prediction == "A")
    t2_chose_fast = (result.test2_llm_prediction == "B")
    t3_chose_fast = (result.test3_llm_prediction == "C")
    
    result.is_consistent = (t1_chose_fast == t2_chose_fast == t3_chose_fast)
    
    if result.is_all_correct:
        result.category = "all_correct"
    elif result.num_correct == 0:
        result.category = "all_wrong"
    elif result.is_consistent:
        result.category = "consistent_partial"
    else:
        result.category = "position_bias"
    
    return result


# ==================== 评估器 ====================
class Evaluator:
    """统一评估器"""
    
    def __init__(self, output_path: str, num_workers: int = 8, prompt_type: str = "zero-shot"):
        self.output_path = output_path
        self.num_workers = num_workers
        self.prompt_type = prompt_type
        self.processed_ids: set = set()
        self.pair_results: List[PairEvalResult] = []
        self.triple_results: List[TripleEvalResult] = []
        self.lock = Lock()
        
        self.stats = self._init_stats()
        self._load_processed()
    
    def _init_stats(self):
        return {
            "total": 0, "success": 0, "errors": 0, "parse_errors": 0,
            "pair_both_correct": 0, "pair_both_wrong": 0, 
            "pair_position_bias": 0,
            "triple_all_correct": 0, "triple_all_wrong": 0,
            "triple_partial_correct": 0, "triple_position_bias": 0,
            "consistent": 0, "inconsistent": 0,
            "by_dataset": {},
            "by_speedup_bin": {},  # 按 speedup_bin 统计
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0  # 总成本（美元）
        }
    
    def _load_processed(self):
        """加载已处理的记录"""
        if Path(self.output_path).exists():
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
                    
                    for r in data.get("pair_results", []):
                        self.processed_ids.add(r.get("pair_id", ""))
                        self.pair_results.append(PairEvalResult(**r))
                    
                    for r in data.get("triple_results", []):
                        self.processed_ids.add(r.get("pair_id", ""))
                        self.triple_results.append(TripleEvalResult(**r))
                    
                    self._recalculate_stats()
                    
                print(f"[INFO] 已加载 {len(self.processed_ids)} 条历史记录")
            except Exception as e:
                print(f"[WARN] 加载历史记录失败: {e}")
    
    def _recalculate_stats(self):
        """重新计算统计"""
        self.stats = self._init_stats()
        for r in self.pair_results:
            self._update_pair_stats(r, save=False)
        for r in self.triple_results:
            self._update_triple_stats(r, save=False)
    
    def _update_pair_stats(self, result: PairEvalResult, save: bool = True):
        with self.lock:
            self.stats["total"] += 1
            
            # 汇总 token 使用（兼容旧版本数据）
            test1_prompt = getattr(result, 'test1_prompt_tokens', 0)
            test1_completion = getattr(result, 'test1_completion_tokens', 0)
            test2_prompt = getattr(result, 'test2_prompt_tokens', 0)
            test2_completion = getattr(result, 'test2_completion_tokens', 0)
            
            self.stats["total_prompt_tokens"] += (test1_prompt + test2_prompt)
            self.stats["total_completion_tokens"] += (test1_completion + test2_completion)
            self.stats["total_tokens"] += (
                getattr(result, 'test1_total_tokens', 0) + 
                getattr(result, 'test2_total_tokens', 0)
            )
            
            # 计算成本
            cost1 = calculate_cost(test1_prompt, test1_completion, MODEL)
            cost2 = calculate_cost(test2_prompt, test2_completion, MODEL)
            self.stats["total_cost_usd"] += (cost1 + cost2)
            
            ds_type = result.dataset_type
            if ds_type not in self.stats["by_dataset"]:
                self.stats["by_dataset"][ds_type] = {"total": 0, "correct": 0}
            self.stats["by_dataset"][ds_type]["total"] += 1
            
            # 按 speedup_bin 统计
            sbin = result.speedup_bin
            if sbin:
                if sbin not in self.stats["by_speedup_bin"]:
                    self.stats["by_speedup_bin"][sbin] = {
                        "total": 0, "correct": 0,
                        "both_correct": 0, "both_wrong": 0, "position_bias": 0,
                        "bin_idx": result.speedup_bin_idx
                    }
                self.stats["by_speedup_bin"][sbin]["total"] += 1
            
            if result.error:
                self.stats["errors"] += 1
                if "parse_failed" in str(result.error):
                    self.stats["parse_errors"] += 1
            else:
                self.stats["success"] += 1
                
                if result.category == "both_correct":
                    self.stats["pair_both_correct"] += 1
                    self.stats["by_dataset"][ds_type]["correct"] += 1
                    # 按 bin 统计
                    if sbin and sbin in self.stats["by_speedup_bin"]:
                        self.stats["by_speedup_bin"][sbin]["correct"] += 1
                        self.stats["by_speedup_bin"][sbin]["both_correct"] += 1
                elif result.category == "both_wrong":
                    self.stats["pair_both_wrong"] += 1
                    if sbin and sbin in self.stats["by_speedup_bin"]:
                        self.stats["by_speedup_bin"][sbin]["both_wrong"] += 1
                elif result.category == "position_bias":
                    self.stats["pair_position_bias"] += 1
                    if sbin and sbin in self.stats["by_speedup_bin"]:
                        self.stats["by_speedup_bin"][sbin]["position_bias"] += 1
                
                if result.is_consistent:
                    self.stats["consistent"] += 1
                else:
                    self.stats["inconsistent"] += 1
            
            if save:
                self.pair_results.append(result)
    
    def _update_triple_stats(self, result: TripleEvalResult, save: bool = True):
        with self.lock:
            self.stats["total"] += 1
            
            # 汇总 token 使用（兼容旧版本数据）
            test1_prompt = getattr(result, 'test1_prompt_tokens', 0)
            test1_completion = getattr(result, 'test1_completion_tokens', 0)
            test2_prompt = getattr(result, 'test2_prompt_tokens', 0)
            test2_completion = getattr(result, 'test2_completion_tokens', 0)
            test3_prompt = getattr(result, 'test3_prompt_tokens', 0)
            test3_completion = getattr(result, 'test3_completion_tokens', 0)
            
            self.stats["total_prompt_tokens"] += (test1_prompt + test2_prompt + test3_prompt)
            self.stats["total_completion_tokens"] += (test1_completion + test2_completion + test3_completion)
            self.stats["total_tokens"] += (
                getattr(result, 'test1_total_tokens', 0) + 
                getattr(result, 'test2_total_tokens', 0) + 
                getattr(result, 'test3_total_tokens', 0)
            )
            
            # 计算成本
            cost1 = calculate_cost(test1_prompt, test1_completion, MODEL)
            cost2 = calculate_cost(test2_prompt, test2_completion, MODEL)
            cost3 = calculate_cost(test3_prompt, test3_completion, MODEL)
            self.stats["total_cost_usd"] += (cost1 + cost2 + cost3)
            
            ds_type = result.dataset_type
            if ds_type not in self.stats["by_dataset"]:
                self.stats["by_dataset"][ds_type] = {"total": 0, "correct": 0}
            self.stats["by_dataset"][ds_type]["total"] += 1
            
            if result.error:
                self.stats["errors"] += 1
                if "parse_failed" in str(result.error):
                    self.stats["parse_errors"] += 1
            else:
                self.stats["success"] += 1
                
                if result.category == "all_correct":
                    self.stats["triple_all_correct"] += 1
                    self.stats["by_dataset"][ds_type]["correct"] += 1
                elif result.category == "all_wrong":
                    self.stats["triple_all_wrong"] += 1
                elif result.category == "consistent_partial":
                    self.stats["triple_partial_correct"] += 1
                elif result.category == "position_bias":
                    self.stats["triple_position_bias"] += 1
                
                if result.is_consistent:
                    self.stats["consistent"] += 1
                else:
                    self.stats["inconsistent"] += 1
            
            if save:
                self.triple_results.append(result)
    
    def _calculate_accuracy(self) -> dict:
        """计算各项准确率指标"""
        s = self.stats
        success = s["success"]
        accuracy = {}
        
        if success == 0:
            return accuracy
        
        pair_correct = s["pair_both_correct"]
        triple_correct = s["triple_all_correct"]
        total_correct = pair_correct + triple_correct
        
        accuracy["overall_accuracy"] = round(total_correct / success * 100, 2)
        accuracy["overall_correct"] = total_correct
        accuracy["overall_total"] = success
        
        accuracy["consistency_rate"] = round(s["consistent"] / success * 100, 2)
        accuracy["consistent_count"] = s["consistent"]
        
        accuracy["by_dataset"] = {}
        for ds_type, ds_stats in s.get("by_dataset", {}).items():
            total = ds_stats["total"]
            if total > 0:
                correct = ds_stats["correct"]
                accuracy["by_dataset"][ds_type] = {
                    "accuracy": round(correct / total * 100, 2),
                    "correct": correct,
                    "total": total
                }
        
        pair_total = s['pair_both_correct'] + s['pair_both_wrong'] + s['pair_position_bias']
        if pair_total > 0:
            accuracy["pair"] = {
                "total": pair_total,
                "both_correct": s['pair_both_correct'],
                "both_correct_rate": round(s['pair_both_correct'] / pair_total * 100, 2),
                "both_wrong": s['pair_both_wrong'],
                "both_wrong_rate": round(s['pair_both_wrong'] / pair_total * 100, 2),
                "position_bias": s['pair_position_bias'],
                "position_bias_rate": round(s['pair_position_bias'] / pair_total * 100, 2)
            }
        
        triple_total = s['triple_all_correct'] + s['triple_all_wrong'] + s['triple_partial_correct'] + \
                       s['triple_position_bias']
        if triple_total > 0:
            accuracy["triple"] = {
                "total": triple_total,
                "all_correct": s['triple_all_correct'],
                "all_correct_rate": round(s['triple_all_correct'] / triple_total * 100, 2),
                "all_wrong": s['triple_all_wrong'],
                "all_wrong_rate": round(s['triple_all_wrong'] / triple_total * 100, 2),
                "partial_correct": s['triple_partial_correct'],
                "partial_correct_rate": round(s['triple_partial_correct'] / triple_total * 100, 2),
                "position_bias": s['triple_position_bias'],
                "position_bias_rate": round(s['triple_position_bias'] / triple_total * 100, 2)
            }
        
        # 按 speedup_bin 统计
        accuracy["by_speedup_bin"] = {}
        for bin_name, bin_stats in s.get("by_speedup_bin", {}).items():
            total = bin_stats["total"]
            if total > 0:
                correct = bin_stats["correct"]
                accuracy["by_speedup_bin"][bin_name] = {
                    "bin_idx": bin_stats.get("bin_idx", -1),
                    "total": total,
                    "correct": correct,
                    "accuracy": round(correct / total * 100, 2),
                    "both_correct": bin_stats["both_correct"],
                    "both_wrong": bin_stats["both_wrong"],
                    "position_bias": bin_stats["position_bias"]
                }
        
        return accuracy
    
    def _save_results(self):
        """保存结果"""
        accuracy = self._calculate_accuracy()
        
        output_data = {
            "config": {
                "model": MODEL,
                "prompt_type": self.prompt_type,
                "version": "v8_full_experiment_design"
            },
            "accuracy": accuracy,
            "stats": self.stats,
            "pair_results": [asdict(r) for r in self.pair_results],
            "triple_results": [asdict(r) for r in self.triple_results]
        }
        
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def evaluate(self, data_path: str, limit: int = 50):
        """主评估流程"""
        
        tasks = []
        is_triple_dataset = False
        dataset_types_found = set()
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if limit and len(tasks) >= limit:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # 生成唯一标识: 优先用 pair_id, 否则用 problem_id+speedup, 最后用行号
                pair_id = data.get("pair_id")
                if not pair_id:
                    problem_id = data.get("problem_id", "")
                    speedup = data.get("speedup", "")
                    if problem_id:
                        pair_id = f"{problem_id}_{speedup}" if speedup else f"{problem_id}_line{i}"
                    else:
                        pair_id = f"line_{i}"
                # 注入到 data 中供后续处理使用
                data["_generated_pair_id"] = pair_id
                
                if str(pair_id) in self.processed_ids:
                    continue
                
                dtype = detect_dataset_type(data)
                dataset_types_found.add(dtype)
                if dtype == "triple":
                    is_triple_dataset = True
                
                tasks.append(data)
        
        if not tasks:
            print("[INFO] 没有需要处理的新数据")
            self._print_stats(is_triple_dataset)
            return
        
        effective_workers = min(self.num_workers, 64)  # 增加上限至 64，充分利用多核 
        
        print(f"[INFO] 数据集类型: {', '.join(dataset_types_found)}")
        for dtype in dataset_types_found:
            if dtype in ("same_tier", "clean_vs_raw"):
                print(f"  - {dtype}: 正确答案 = Similar")
            else:
                print(f"  - {dtype}: 正确答案 = fast")
        
        print(f"[INFO] 待处理: {len(tasks)} 条")
        print(f"[INFO] Prompt版本: v8 (完整实验设计)")
        print(f"[INFO] Prompt策略: {self.prompt_type}")
        print(f"[INFO] 并发数: {effective_workers}")
        print("=" * 60)
        
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
        
        def process_item(data):
            dtype = detect_dataset_type(data)
            if dtype == "triple":
                return ("triple", process_triple(data, self.prompt_type))
            else:
                return ("pair", process_pair(data, self.prompt_type))
        
        completed = 0
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_data = {executor.submit(process_item, d): d for d in tasks}
            
            iterator = tqdm(as_completed(future_to_data), total=len(tasks), 
                          desc="评估中") if HAS_TQDM else as_completed(future_to_data)
            
            try:
                for future in iterator:
                    result_type, result = future.result()
                    
                    if result_type == "triple":
                        self._update_triple_stats(result)
                    else:
                        self._update_pair_stats(result)
                    
                    completed += 1
                    
                    if completed % 5 == 0:
                        self._save_results()
                        
            except KeyboardInterrupt:
                print(f"\n[WARN] 用户中断")
            finally:
                self._save_results()
        
        self._print_stats(is_triple_dataset)
    
    def _print_stats(self, is_triple: bool = False):
        """打印统计"""
        print("\n" + "=" * 70)
        print(f"评估报告 - v8完整实验设计版 - {self.prompt_type.upper()}")
        print("=" * 70)
        
        s = self.stats
        success = s["success"]
        
        print(f"\n📊 基础统计")
        print(f"   总数: {s['total']}, 成功: {success}, 错误: {s['errors']}")
        
        print(f"\n💰 Token 使用统计")
        print(f"   Prompt Tokens: {s['total_prompt_tokens']:,}")
        print(f"   Completion Tokens: {s['total_completion_tokens']:,}")
        print(f"   Total Tokens: {s['total_tokens']:,}")
        if success > 0:
            print(f"   平均每样本: {s['total_tokens'] / success:.0f} tokens")
        
        print(f"\n💵 成本统计 (模型: {MODEL})")
        pricing = get_model_pricing(MODEL)
        print(f"   定价: Input=${pricing['input']:.2f}/1M tokens, Output=${pricing['output']:.2f}/1M tokens")
        print(f"   总成本: ${s['total_cost_usd']:.4f} USD")
        if success > 0:
            print(f"   平均每样本: ${s['total_cost_usd'] / success:.6f} USD")
        
        if success == 0:
            return
        
        accuracy = self._calculate_accuracy()
        
        print(f"\n📊 总体准确率")
        print(f"   准确率: {accuracy.get('overall_accuracy', 0):.2f}%")
        print(f"   正确: {accuracy.get('overall_correct', 0)} / {accuracy.get('overall_total', 0)}")
        
        print(f"\n📊 按数据集类型统计")
        for ds_type, ds_acc in accuracy.get("by_dataset", {}).items():
            print(f"   {ds_type}:")
            print(f"     准确率: {ds_acc['accuracy']:.2f}%")
            print(f"     正确: {ds_acc['correct']} / {ds_acc['total']}")
        
        print(f"\n📊 一致性")
        print(f"   一致性率: {accuracy.get('consistency_rate', 0):.2f}%")
        
        if "pair" in accuracy:
            p = accuracy["pair"]
            print(f"\n📊 双代码配对详细分类")
            print(f"   两次都对: {p['both_correct']} ({p['both_correct_rate']:.1f}%)")
            print(f"   两次都错: {p['both_wrong']} ({p['both_wrong_rate']:.1f}%)")
            print(f"   位置偏差: {p['position_bias']} ({p['position_bias_rate']:.1f}%)")
        
        if "triple" in accuracy:
            t = accuracy["triple"]
            print(f"\n📊 三元组详细分类")
            print(f"   全部正确: {t['all_correct']} ({t['all_correct_rate']:.1f}%)")
            print(f"   全部错误: {t['all_wrong']} ({t['all_wrong_rate']:.1f}%)")
            print(f"   部分正确: {t['partial_correct']} ({t['partial_correct_rate']:.1f}%)")
            print(f"   位置偏差: {t['position_bias']} ({t['position_bias_rate']:.1f}%)")
        
        # 按 Speedup Bin 统计
        if accuracy.get("by_speedup_bin"):
            print(f"\n📊 按 Speedup 分档准确率")
            print(f"   {'Bin':<18} {'样本':>6} {'正确':>6} {'准确率':>8}")
            print(f"   {'-'*40}")
            # 按 bin_idx 排序
            sorted_bins = sorted(
                accuracy["by_speedup_bin"].items(),
                key=lambda x: x[1].get("bin_idx", 999)
            )
            for bin_name, bin_acc in sorted_bins:
                print(f"   {bin_name:<18} {bin_acc['total']:>6} {bin_acc['correct']:>6} {bin_acc['accuracy']:>7.1f}%")
        
        print(f"\n结果已保存: {self.output_path}")


# ==================== 入口 ====================
def main():
    global MODEL, API_KEY, BASE_URL, client
    
    parser = argparse.ArgumentParser(
        description="LLM 代码性能判断评估 v9 - 简化为3种Prompt策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v9 简化设计:
  支持 3 种 Prompt 策略，所有示例均为语言特定:
  
  ┌──────────────────┬────────┬─────────┬──────────────┐
  │ Prompt策略       │ 示例   │ CoT推理 │ 语言特定示例 │
  ├──────────────────┼────────┼─────────┼──────────────┤
  │ zero-shot (ZS)   │   ❌   │   ❌    │      -       │
  │ few-shot (FS)    │   ✅   │   ❌    │   ✅(匹配)   │
  │ few-shot-cot     │   ✅   │   ✅    │   ✅(匹配)   │
  │ (FS-CoT)         │        │         │              │
  └──────────────────┴────────┴─────────┴──────────────┘

示例:
  # 1. Zero-Shot (ZS): 直接比较，无示例，无推理
  python eval_full.py data.jsonl --prompt-type zero-shot
  
  # 2. Few-Shot (FS): 有语言特定示例，只展示输入输出
  python eval_full.py data.jsonl --prompt-type few-shot
  
  # 3. Few-Shot CoT (FS-CoT): 有语言特定示例，展示详细推理过程
  python eval_full.py data.jsonl --prompt-type few-shot-cot

批量运行所有策略:
  for pt in zero-shot few-shot few-shot-cot; do
    python eval_full.py data.jsonl --prompt-type $pt -n 100
  done
        """
    )
    
    parser.add_argument("data", help="输入 JSONL 数据路径")
    parser.add_argument("-o", "--output", default=None, 
                       help="输出文件路径 (若不指定，将自动根据输入文件名、prompt类型和模型名称生成)")
    parser.add_argument("-n", "--limit", type=int, default=5)
    parser.add_argument("-j", "--workers", type=int, default=8)
    parser.add_argument("--prompt-type", 
                       choices=["zero-shot", "few-shot", "few-shot-cot"], 
                       default="zero-shot",
                       help="""Prompt策略选择:
  zero-shot    : Zero-Shot (ZS) - 直接比较，无示例，无推理
  few-shot     : Few-Shot (FS) - 有2个语言特定示例，只展示输入输出
  few-shot-cot : Few-Shot CoT (FS-CoT) - 有2个语言特定示例，展示详细推理过程""")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    
    args = parser.parse_args()
    
    if args.model:
        MODEL = args.model
    if args.api_key:
        API_KEY = args.api_key
    if args.base_url:
        BASE_URL = args.base_url
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # 生成输出文件名
    # 如果用户指定了 -o/--output，使用用户指定的路径
    # 否则自动根据输入文件名、prompt类型和模型名称生成
    if args.output:
        output_path = args.output
    else:
        output_path = generate_default_output_filename(args.data, args.prompt_type, MODEL)
    
    print(f"LLM 代码性能评估 v9 (简化为3种Prompt策略)")
    print(f"{'='*60}")
    print(f"输入: {args.data}")
    print(f"输出: {output_path}")
    print(f"模型: {MODEL}")
    print(f"Prompt: {args.prompt_type}")
    print(f"{'='*60}\n")
    
    evaluator = Evaluator(output_path, args.workers, args.prompt_type)
    evaluator.evaluate(args.data, args.limit if args.limit > 0 else None)


if __name__ == "__main__":
    main()