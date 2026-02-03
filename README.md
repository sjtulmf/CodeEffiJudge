CodeEffiJudge: Can LLMs Reason about Code Efficiency?
The dataset is available on Hugging Face: coming soon.

### Dataset Structure

```
efficodebench-dataset/
├── final_benchmark_uniform/
│   ├── cpp_seed42_sanitized.jsonl      # C++ benchmark (900 pairs)
│   ├── python_seed42_sanitized.jsonl   # Python benchmark (900 pairs)
│   └── java_seed42_sanitized_process.jsonl  # Java benchmark (900 pairs)
├── python_ab_test.jsonl                # Python test set (A/B format)
├── python_abdiff_test.jsonl            # Python test set (A/B with diff)
├── python_adiff_test.jsonl             # Python test set (A with diff)
└── python_zeroshot_test.jsonl          # Python test set (zero-shot)
```

### Data Format

Each sample contains a pair of functionally equivalent code snippets with different execution speeds:

```json
{
  "pair_id": "pie_12345",
  "problem_id": "p01234",
  "language": "cpp",
  "slow_code": "...",
  "fast_code": "...",
  "slow_time": 1234.5,
  "fast_time": 123.4,
  "speedup": 10.0,
  "speedup_bin": 2,
  "difficulty": "easy"
}
```

### Difficulty Levels

| Difficulty | Speedup Range | Description |
|------------|---------------|-------------|
| hard       | [2x, 4x)      | Small optimization margin |
| medium     | [4x, 8x)      | Moderate optimization |
| easy       | [8x, inf)     | Large optimization margin |

---

## 🛠️ Project Structure

| File | Description |
|------|-------------|
| `extract.py` | **Dataset Sampler**: Extract balanced code pairs from PIE dataset with deduplication and stratified sampling across difficulty levels |
| `code_sanitizer.py` | **Code Sanitizer**: Semantic-level variable renaming and comment removal using LibCST (Python), Clang (C++), OpenRewrite (Java) |
| `unify_dataset_schema.py` | **Schema Unifier**: Convert various input formats (PIE, sampled, Java benchmark) to unified schema |
| `coffe_benchmark.py` | **Python Benchmark**: CPU instruction counting using cirron (perf_event), process isolation, multi-sampling |
| `java_process_benchmark.py` | **Java Benchmark**: Process-level timing with System.nanoTime(), JVM warmup, CPU pinning via taskset |
| `eval_full.py` | **LLM Evaluation**: Pairwise code efficiency comparison with 3 prompting strategies (Zero-Shot, Few-Shot, Few-Shot-CoT) |
| `analyze.py` | **Result Analyzer**: Per-difficulty accuracy statistics, error case analysis, detailed reports |
| `inference.py` | **Model Inference**: Run trained models for code efficiency prediction |
| `train_unified.py` | **Model Training**: Unified training script for fine-tuning models on efficiency judgment task |

---

## 🚀 Quick Start

### 1. Extract Dataset from PIE

```bash
# Sample 900 pairs per language with balanced difficulty distribution
python extract.py \
    -i PIE_Dataset/train.jsonl \
    -n 900 \
    -o ./output \
    --output-name cpp_balanced \
    --balance-problem-id
```

### 2. Sanitize Code (Remove Comments, Rename Variables)

```bash
python code_sanitizer.py \
    -i cpp_balanced.jsonl \
    -o cpp_sanitized.jsonl \
    --language cpp
```

### 3. Unify Dataset Schema

```bash
python unify_dataset_schema.py \
    -i raw_data.jsonl \
    -o unified.jsonl \
    --format pie \
    --language cpp
```

### 4. Run Benchmark Measurements

```bash
# Python (CPU instruction counting)
python coffe_benchmark.py \
    -i python_sanitized.jsonl \
    -o python_benchmark_results.jsonl \
    --cpu-core 0

# Java (process-level timing)
python java_process_benchmark.py \
    -i java_problems.jsonl \
    -o java_benchmark_results.jsonl \
    --cpu-core 0
```

### 5. Evaluate LLM Performance

```bash
python eval_full.py \
    -i cpp_seed42_sanitized.jsonl \
    -o results.json \
    --prompt-type few-shot-cot \
    --model gpt-4o
```

### 6. Analyze Results

```bash
python analyze.py \
    -e results.json \
    -d dataset.jsonl \
    -o report.txt
```

### 7. Train & Inference (Optional)

```bash
# Train a model
python train_unified.py --config config.yaml

# Run inference
python inference.py \
    -i test_data.jsonl \
    -o predictions.jsonl \
    --model ./trained_model
```

---

## 📊 Prompting Strategies

| Strategy | Examples | CoT Reasoning | Language-Specific |
|----------|----------|---------------|-------------------|
| Zero-Shot (ZS) | ❌ | ❌ | - |
| Few-Shot (FS) | ✅ | ❌ | ✅ |
| Few-Shot-CoT (FS-CoT) | ✅ | ✅ | ✅ |

---

## 🔧 Dependencies

```bash
pip install numpy matplotlib pandas scikit-learn openai huggingface_hub
pip install libcst black  # For Python code sanitization
pip install cirron        # For CPU instruction counting (Python benchmark)
```

---

## 📄 License

MIT License

---

## 📬 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{,
  title={},
  author={},
  year={2026},
  url={}
}
```
