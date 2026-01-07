## 项目结构概览

```text
大模型评测/
├── evaluate.py                  # 主评估脚本（命令行入口）
├── start_eval.sh                # 多 GPU 预训练评测示例脚本
├── start_qa_eval.sh             # 多 GPU QA 评测脚本（本地 30B 示例）
├── requirements.txt             # Python 依赖
├── README.md                    # 总体说明文档
├── QUICKSTART.md                # 快速开始（常用命令）
├── COMMAND_LINE_EXAMPLES.md     # 命令行示例汇总
├── PROJECT_STRUCTURE.md         # 本文件：项目结构说明
│
├── evaluators/                  # 各类评估器
│   ├── __init__.py
│   ├── qa_evaluator.py          # QA 评估（EM / BLEU / ROUGE / BERTScore 等）
│   ├── mcq_evaluator.py         # MCQ 评估（Accuracy / ECE / 偏差分析）
│   ├── benchmark_evaluator.py   # 基准评测（当前主要是 CMMLU）
│   ├── performance_evaluator.py # 性能评测（吞吐量 / 延迟等）
│   └── reliability_evaluator.py # 可靠性评测（鲁棒性 / 一致性等）
│
├── utils/
│   ├── __init__.py
│   ├── model_loader.py          # 模型加载封装（local / OpenAI 兼容 API）
│   └── data_loader.py           # 数据加载与格式自适应（JSON / JSONL / CSV）
│
├── examples/
│   ├── qa_dataset_example.json          # QA 数据示例
│   ├── mcq_dataset_example.json         # MCQ 数据示例
│   └── reliability_dataset_example.json # 可靠性数据示例
│
├── qa_data/                     # QA 评测数据示例（如 swift_datasets_test.jsonl）
│
├── pretrain_data/               # 预训练评测用数据（如 pre_train_datasets_optimized.jsonl）
│
└── benchmark_data/
    └── cmmlu/
        └── test/                # CMMLU 测试集（多个领域的 .csv / .jsonl）
```

---

## 核心模块说明

### 1. `evaluate.py`

- 统一的命令行入口
- 负责解析参数、创建 `ModelLoader`、调度各类 `Evaluator`
- 支持多种 backend：
  - `local`：本地 HuggingFace / Transformers 模型（含多 GPU）
  - `openai_compatible` / `internal_api` / `qwen` / `deepseek`：HTTP 接口

### 2. `e
valuators/`

- `qa_evaluator.py`：
  - 指标：EM、BLEU-1/2/3/4、ROUGE-1/2/L、BERTScore、综合质量分
  - 中文友好：使用 `jieba` 分词、定制中文 ROUGE Tokenizer
  - BERTScore 默认使用中文 BERT（可通过 ModelScope 调用）
- `mcq_evaluator.py`：
  - 指标：Accuracy、ECE、Position Bias、Length Bias、Distractor Sensitivity
- `benchmark_evaluator.py`：
  - 目前主要封装 CMMLU，读取 `benchmark_data/cmmlu/test/`
- `performance_evaluator.py`：
  - 统计吞吐量、首 Token 延迟、生成速度等基础性能指标
- `reliability_evaluator.py`：
  - 输入扰动、语义一致性、错误率等可靠性维度

### 3. `utils/`

- `model_loader.py`：
  - 封装本地模型加载 / API 调用逻辑
  - 支持：
    - 单卡 / 多卡（`device_map=auto` 自动模型并行）
    - 通过 `gpu_ids` 设置 `CUDA_VISIBLE_DEVICES`
    - 控制每张卡可用显存（`max_memory`）
- `data_loader.py`：
  - 自动识别 QA / MCQ / 可靠性数据集格式
  - 支持 JSON / JSONL / CSV，多种字段名组合（`question/answer`、`query/response` 等）

---

## 典型数据流

```text
命令行参数
   ↓
evaluate.py
   ↓
ModelLoader  (加载模型或初始化 API 客户端)
   ↓
DataLoader   (加载并标准化数据集)
   ↓
各类 Evaluator (QA / MCQ / CMMLU / 性能 / 可靠性)
   ↓
结果汇总到 evaluator.results
   ↓
保存为 <output>.json 与 <output>.txt
```

---

## 扩展建议

- **新增评估指标**：在对应的 `Evaluator` 中新增方法，并在 `evaluate()` 中汇总到 `results`
- **新增数据格式**：在 `data_loader.py` 中扩展解析逻辑，保证输出统一字段
- **新增基准数据集**：仿照 `benchmark_evaluator.py` 的 CMMLU 实现，新建一个基准分支

