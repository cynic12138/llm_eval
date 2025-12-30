# 项目结构

```
大模型评测/
├── evaluate.py                      # 主评估脚本（命令行入口）
├── requirements.txt                 # Python依赖包
├── README.md                        # 详细文档
├── QUICKSTART.md                    # 快速开始指南
├── PROJECT_STRUCTURE.md             # 项目结构说明（本文件）
├── .gitignore                       # Git忽略文件
│
├── evaluators/                      # 评估器模块
│   ├── __init__.py
│   ├── qa_evaluator.py             # QA数据集评估器
│   ├── mcq_evaluator.py            # MCQ数据集评估器
│   ├── benchmark_evaluator.py      # 基准数据集评估器（当前仅支持 CMMLU）
│   ├── performance_evaluator.py    # 性能指标评估器
│   └── reliability_evaluator.py    # 可靠性指标评估器
│
├── utils/                           # 工具模块
│   ├── __init__.py
│   ├── model_loader.py             # 模型加载器
│   └── data_loader.py              # 数据加载器
│
├── examples/                         # 示例数据
│   ├── qa_dataset_example.json     # QA数据集示例
│   ├── mcq_dataset_example.json    # MCQ数据集示例
│   └── reliability_dataset_example.json  # 可靠性测试数据集示例
│
└── benchmark_data/                  # 基准数据集存储目录（自动创建）
    └── cmmlu/                       # CMMLU 数据集根目录（需手动放置）
        └── test/                    # 测试集（支持 .csv / .jsonl）
```

## 核心模块说明

### evaluate.py
主评估脚本，提供命令行接口，协调各个评估器完成评估任务。

### evaluators/
- **qa_evaluator.py**: 实现QA数据集的评估指标
  - **核心指标**：mean_EM、mean_bleu-1/2/3/4、mean_Rouge-1/2/L的R/P/F、mean_BERTScore、mean_Comprehensive_Score、mean_Factual_Consistency
  - **技术特性**：
    - 使用jieba进行中文分词（BLEU、ROUGE）
    - 自定义中文分词器用于ROUGE计算
    - BERTScore自动使用CUDA加速
    - NLI模型（hfl/rbtl3）用于事实一致性检测
    - 所有指标计算整个数据集的平均值（mean_前缀）
- **mcq_evaluator.py**: 实现MCQ数据集的评估指标（准确率、ECE、位置偏差、长度偏差、干扰项敏感性）
- **benchmark_evaluator.py**: 处理 CMMLU 基准数据集的加载和评估（从 `benchmark_data/cmmlu/test` 读取）
- **performance_evaluator.py**: 评估基础性能指标（吞吐量、延迟等）
- **reliability_evaluator.py**: 评估可靠性指标（输入鲁棒性、系统稳定性）

### utils/
- **model_loader.py**: 加载和推理模型，支持HuggingFace格式
- **data_loader.py**: 加载各种格式的数据集（JSON、JSONL、CSV）

## 数据流向

```
用户输入（命令行参数）
    ↓
evaluate.py (主脚本)
    ↓
ModelLoader (加载模型)
    ↓
DataLoader (加载数据)
    ↓
各种Evaluator (执行评估)
    ↓
结果汇总和输出
```

## 扩展指南

### 添加新的评估指标

1. 在对应的评估器类中添加新方法
2. 在 `evaluate()` 方法中调用新方法
3. 将结果添加到返回列表中

### 添加新的数据集格式

1. 在 `utils/data_loader.py` 中添加新的加载方法
2. 在 `load_dataset()` 方法中添加格式识别逻辑

### 添加新的基准数据集

1. 在 `evaluators/benchmark_evaluator.py` 中添加下载和加载方法
2. 在 `evaluate()` 方法中添加新的基准数据集分支

