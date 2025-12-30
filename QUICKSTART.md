# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 准备模型 / API

本项目既支持本地 HuggingFace 模型（`backend=local`），也支持任意 **OpenAI 兼容 HTTP 接口**（`backend=openai_compatible`）。

常见两种用法：

- **本地模型（HuggingFace 权重）**：需包含
  - `config.json`
  - `tokenizer.json` 或 `tokenizer_config.json`
  - `pytorch_model.bin` 或 `model.safetensors`
- **本地 HTTP 服务（OpenAI 兼容）**：例如  
  `http://10.10.0.102:7865/v1`，模型名如 `Fusion2-chat-v2.0`

## 3. 准备数据集

### QA数据集示例（多种格式）

创建 `my_qa_data.json`（标准格式）:
```json
[
    {
        "question": "什么是人工智能？",
        "answer": "人工智能是计算机科学的一个分支..."
    }
]
```

也支持以下 JSONL 格式（每行一条）：

1. 标准问答格式：
```jsonl
{"question": "什么是人工智能？", "answer": "人工智能是计算机科学的一个分支..."}
{"question": "机器学习的基本原理是什么？", "answer": "机器学习通过算法从数据中学习模式..."}
```

2. `query/response` 格式（当前 qa_data 使用的格式）：
```jsonl
{"query": "根据描述生成医生主诉：……", "response": "左手烙铁头蛇咬伤致左上肢肿痛4小时"}
```

3. 指令 + 输入 + 输出格式（Alpaca 风格）：
```jsonl
{"instruction": "根据下述病人描述生成病历", "input": "……详细病例描述……", "output": "生成的病历结构化结果"}
```

### MCQ数据集示例

创建 `my_mcq_data.json`:
```json
[
    {
        "question": "以下哪个是机器学习算法？",
        "options": {
            "A": "线性回归",
            "B": "二分查找",
            "C": "冒泡排序",
            "D": "快速排序"
        },
        "answer": "A"
    }
]
```

## 4. 运行评估

### 4.1 使用本地 HTTP 接口做 QA 评估（OpenAI 兼容）
```bash
python evaluate.py \
    --backend openai_compatible \
    --api_base_url "http://10.10.0.102:7865/v1" \
    --api_model_name "Fusion2-chat-v2.0" \
    --api_key "sk-internal" \
    --dataset_path qa_data/swift_datasets_test.jsonl \
    --dataset_type qa \
    --dataset_name swift_qa \
    --max_samples 20 \
    --output local_qa_results
```

`--max_samples` 用于限制只评估前 N 条样本（可选）。评估结果将包含以下指标：
- `mean_EM`: 精确匹配率
- `mean_bleu-1/2/3/4`: BLEU分数
- `mean_Rouge-1-R/P/F`: ROUGE-1的召回率、精确率、F1
- `mean_Rouge-2-R/P/F`: ROUGE-2的召回率、精确率、F1
- `mean_Rouge-L-R/P/F`: ROUGE-L的召回率、精确率、F1
- `mean_BERTScore`: BERT语义相似度
- `mean_Comprehensive_Score`: 综合质量分数
- `mean_Factual_Consistency`: 事实一致性

### 4.2 基本MCQ评估（本地模型示例）
```bash
python evaluate.py \
    --model /path/to/your/model \
    --dataset_path my_mcq_data.json \
    --dataset_type mcq
```

### 4.3 完整评估（QA + CMMLU + 性能）
```bash
python evaluate.py \
    --backend openai_compatible \
    --api_base_url "http://10.10.0.102:7865/v1" \
    --api_model_name "Fusion2-chat-v2.0" \
    --api_key "sk-internal" \
    --dataset_path qa_data/swift_datasets_test.jsonl \
    --dataset_type qa \
    --dataset_name swift_qa \
    --max_samples 20 \
    --benchmarks cmmlu \
    --evaluate_performance \
    --output fusion2_all_eval
```

**注意**：确保CMMLU数据集已放置在 `benchmark_data/cmmlu/test/` 目录下。

## 5. 查看结果

评估完成后，查看生成的结果文件：
- `evaluation_results.json` - JSON格式的详细结果
- `evaluation_results.txt` - 表格格式的可读结果

**QA数据集评估结果示例**：
- `mean_EM`: 0.6500
- `mean_bleu-1`: 0.7200
- `mean_bleu-2`: 0.6800
- `mean_bleu-3`: 0.6500
- `mean_bleu-4`: 0.6200
- `mean_Rouge-1-R`: 0.7500, `mean_Rouge-1-P`: 0.7400, `mean_Rouge-1-F`: 0.7450
- `mean_Rouge-2-R`: 0.7000, `mean_Rouge-2-P`: 0.6900, `mean_Rouge-2-F`: 0.6950
- `mean_Rouge-L-R`: 0.7300, `mean_Rouge-L-P`: 0.7200, `mean_Rouge-L-F`: 0.7250
- `mean_BERTScore`: 0.8100
- `mean_Comprehensive_Score`: 0.7800
- `mean_Factual_Consistency`: 0.8500

## 常见问题

### 问题1: 模型加载失败
**解决方案**: 确保模型路径正确，且模型格式为HuggingFace格式。

### 问题2: 数据集格式错误
**解决方案**: 参考 `examples/` 目录下的示例数据格式。

### 问题3: 内存不足
**解决方案**: 
- 使用更小的批次大小
- 减少测试样本数量
- 使用CPU模式（速度较慢）

### 问题4: 基准数据集（CMMLU）不存在或格式错误
**解决方案**: 
- 确认 `benchmark_data/cmmlu/test/` 目录存在
- 确认目录下有 `.csv` 或 `.jsonl` 文件（文件名为各个领域，如 `agronomy.csv`）
- 参考 README 中的 CMMLU 数据格式说明手动整理数据

