## 大模型评测框架（中文）

一个面向中文大模型的通用评测框架，支持：

- **自定义 QA 数据集评测**
- **自定义 MCQ（多选题）数据集评测**
- **中文基准数据集 CMMLU**
- **基础性能与可靠性评估**
- **本地大模型（含 30B、多 GPU 自动并行）与 OpenAI 兼容 API**

核心入口脚本为 `evaluate.py`，同时提供若干启动脚本（如 `start_eval.sh`、`start_qa_eval.sh`）方便直接在多 GPU 环境下评测本地 30B 模型。

---

## 功能概览

### 自定义 QA 评测

- **自动支持多种数据格式/字段名**（`question/answer`、`query/response`、`instruction/input/output`、`问题/答案` 等）
- **自动中文分词**（`jieba`）
- 主要指标：
  - **mean_EM**：精确匹配率（宽松标准化后比较）
  - **mean_bleu-1/2/3/4**：BLEU 分数
  - **mean_Rouge-1/2/L-R/P/F**：ROUGE 系列指标
  - **mean_BERTScore**：基于中文 BERT 的语义相似度
  - **mean_Comprehensive_Score**：综合质量分（BERTScore × 0.6 + ROUGE-L F1 × 0.4）

### 自定义 MCQ 评测

- 支持单选 / 多选格式
- 主要指标：
  - **Accuracy**：准确率
  - **ECE**：校准误差
  - **Position / Length Bias**：位置与长度偏差分析

### 基准与性能 / 可靠性

- **CMMLU**：从 `benchmark_data/cmmlu/test/` 读取 `.csv` / `.jsonl`
- **性能**：吞吐量、首 Token 延迟、生成速度、端到端延迟
- **可靠性**（可选）：输入扰动鲁棒性、语义一致性、错误率等

---

## 安装

```bash
cd 大模型评测

pip install -r requirements.txt

# 如需 NLTK 相关指标（可选）
python -c "import nltk; nltk.download('punkt')"
```

依赖中已包含：

- `torch`、`transformers`
- `rouge-score`、`bert-score`、`nltk`、`jieba`
- `modelscope`（用于从魔塔社区调用中文 NLI / BERT 模型）

---

## 快速上手：两种典型用法

### 1. 使用本地 HTTP（OpenAI 兼容）做 QA 评测

```bash
python evaluate.py \
  --backend openai_compatible \
  --api_base_url "http://10.10.0.102:7865/v1" \
  --api_model_name "Fusion2-chat-v2.0" \
  --api_key "sk-internal" \
  --qa_dataset qa_data/swift_datasets_test.jsonl \
  --qa_dataset_name swift_qa \
  --max_samples 20 \
  --output local_qa_results
```

### 2. 使用本地 30B 模型 + 多 GPU 做 QA 评测（示例）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python evaluate.py \
  --backend local \
  --model /home/model/Fusion3-30B-A3B-2601-SFT-v1 \
  --qa_dataset /home/project/大模型评测/qa_data_2/test.jsonl \
  --qa_dataset_name qa_test \
  --output /home/model/Fusion3-30B-A3B-2601-SFT-v1/eval_output/qa_eval \
  --device cuda \
  --device_map auto \
  --gpu_ids "4,5,6,7"
```

项目中已经提供了一个精简脚本 `start_qa_eval.sh`，内部就是类似上面的命令。

---

## `evaluate.py` 参数速览

### 模型来源

- **`--backend`**：
  - `local`：本地 HuggingFace / Transformers 模型
  - `openai_compatible` / `internal_api` / `qwen` / `deepseek`：任意 OpenAI 兼容 HTTP 接口
- **本地模型专用**
  - `--model`：本地模型路径或模型名
  - `--device`：`cuda` / `cpu`（默认 `cuda`）
  - `--device_map`：`auto` / `None`（自动模型并行或单卡）
  - `--gpu_ids`：如 `"0,1,2,3"` / `"4,5,6,7"`，会设置 `CUDA_VISIBLE_DEVICES`
  - `--max_memory`：按 GPU 限制显存，如 `"0:20GiB,1:20GiB"`
- **OpenAI 兼容 API**
  - `--api_base_url`、`--api_model_name`、`--api_key`

### 自定义数据集

- **新参数（推荐）**
  - `--qa_dataset` / `--qa_dataset_name`
  - `--mcq_dataset` / `--mcq_dataset_name`
  - `--max_samples`：截断样本数
- **兼容旧参数**
  - `--dataset_path` + `--dataset_type (qa|mcq)` + `--dataset_name`

### 其他

- `--benchmarks cmmlu`
- `--evaluate_performance`
- `--evaluate_reliability` + `--reliability_dataset`
- 预训练评估（困惑度 + 生成质量）：`--pretrain_dataset` 等
- 输出路径：`--output`（不带扩展名）

---

## 数据格式示例（QA / MCQ）

### QA（标准 JSON）

```json
[
  {
    "question": "什么是人工智能？",
    "answer": "人工智能是计算机科学的一个分支..."
  }
]
```

### QA（JSONL，多种字段）

```jsonl
{"question": "什么是人工智能？", "answer": "人工智能是计算机科学的一个分支..."}
{"query": "根据描述生成医生主诉：……", "response": "左手烙铁头蛇咬伤致左上肢肿痛4小时"}
{"instruction": "根据下述病人描述生成病历", "input": "……详细病例描述……", "output": "生成的病历结构化结果"}
{"问题": "什么是人工智能？", "答案": "人工智能是计算机科学的一个分支..."}
```

### MCQ（JSON）

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

---

## 评估结果输出

- `${output}.json`：所有模型 / 数据集 / 指标的结构化结果
- `${output}.txt`：表格形式的汇总，便于人工查看或汇报

表格字段包括：

- `Model`、`Dataset`、`Type`（QA/MCQ/PERF/RELIABILITY/PRETRAIN）
- `Metric`、`Num`（样本数）、`Score`、`Domain`（对 CMMLU 等有用）

---

## 项目结构（简要）

更多细节见 `PROJECT_STRUCTURE.md`。

```text
.
├── evaluate.py              # 主入口
├── start_eval.sh            # 多 GPU 预训练评测示例脚本
├── start_qa_eval.sh         # 多 GPU QA 评测脚本（本地 30B 示例）
├── evaluators/              # 各类评测逻辑
├── utils/                   # 模型 / 数据加载
├── benchmark_data/cmmlu/    # CMMLU 数据集
├── qa_data/                 # QA 测试数据示例
├── pretrain_data/           # 用于预训练评测的数据
└── *.md                     # 文档
```

---

## 许可证与贡献

- 许可证：MIT
- 欢迎提交 Issue / PR，一般贡献方向：
  - 新的评估指标 / 新的基准数据集
  - 更多数据格式适配
  - 针对特定硬件 / 集群环境的脚本示例

