模型评估框架
一个全面的中文大模型评估框架，支持问答（QA）和多选题（MCQ）两种数据集类型的自动评估，包括自定义数据集评估、基准数据集评估、性能指标评估和可靠性指标评估。

功能特性
1. 自定义数据集评估
QA数据集评估指标
核心自动化指标：
mean_EM：精确匹配率（Exact Match）
mean_bleu-1/2/3/4：BLEU分数（基于1-gram到4-gram的几何平均）
mean_Rouge-1-R/P/F：ROUGE-1的召回率、精确率、F1分数
mean_Rouge-2-R/P/F：ROUGE-2的召回率、精确率、F1分数
mean_Rouge-L-R/P/F：ROUGE-L的召回率、精确率、F1分数
mean_BERTScore：基于bert-base-chinese的语义相似度评估（自动使用CUDA加速）
mean_Comprehensive_Score：综合质量分数（BERTScore × 0.6 + ROUGE-L F1 × 0.4）
mean_Factual_Consistency：事实一致性（基于NLI模型的矛盾检测，1 - 矛盾概率）
注意：所有指标名称前都有 mean_ 前缀，表示整个数据集的平均值。

MCQ数据集评估指标
基础指标：

Accuracy：正确选择率（支持单选和多选）
Expected Calibration Error (ECE)：置信度校准误差
按认知水平分组的准确率
可靠性指标：

Position Bias：位置偏差分析（卡方检验）
Length Bias：长度偏差分析（T检验）
Distractor Sensitivity：干扰项敏感性
2. 基准数据集评估
CMMLU（当前仅支持该基准）：
从本地 benchmark_data/cmmlu/test 读取数据（支持 .csv / .jsonl）
领域细分准确率
总体准确率（CMMLU_Total_Accuracy）
领域衰减率（专业领域 vs 通用领域）
3. 基础性能指标评估
吞吐量：Batch=8, 输入128 tokens，≥45 req/s（A100 80G标准）
首Token延迟：P99, 冷启动，≤600 ms
生成速度：输出100 tokens (greedy)，≥150 tokens/s
端到端延迟：P95, 输入256 + 输出128，≤1.2 s
4. 可靠性指标评估（可选）
输入鲁棒性：

拼写错误鲁棒性（拼音/形似字符）
词序打乱鲁棒性
对抗后缀鲁棒性（5种预定义攻击字符串）
语义一致性：3种同义问题的BERTScore标准差（<0.05为稳定）
系统稳定性：

无效输出率：JSON解析失败/空响应比例
错误率监控
安装
# 克隆或下载项目
cd 大模型评测

# 安装依赖
pip install -r requirements.txt

# 下载NLTK数据（如果需要）
python -c "import nltk; nltk.download('punkt')"

使用方法
命令行接口
python evaluate.py --backend <模型后端> [模型参数] [评测选项]

参数说明
模型来源参数
--backend: 模型调用后端类型：

local：本地 HuggingFace 模型（默认）
internal_api：内部 OpenAI 兼容接口（如截图中的内部模型）
qwen：阿里 Qwen（DashScope 兼容模式）
deepseek：DeepSeek API
openai_compatible：任意 OpenAI 兼容接口
本地模型（backend = local）：

--model: 本地模型路径 / HuggingFace 模型名称（必需）
--device: cuda 或 cpu（默认：cuda）
OpenAI 兼容 API（backend = internal_api / qwen / deepseek / openai_compatible）：

--model_api: 模型 API 标识（仅用于结果展示，可选）
--api_key: API Key
backend=qwen 时可省略，从环境变量 DASHSCOPE_API_KEY 读取
backend=deepseek 时可省略，从环境变量 DEEPSEEK_API_KEY 读取
--api_base_url: OpenAI 兼容接口的 base_url（例如 `http://10.10.0.102:7865/v1`）
backend=qwen 默认：https://dashscope.aliyuncs.com/compatible-mode/v1
backend=deepseek 默认：https://api.deepseek.com/v1
--api_model_name: 模型名称（如 Fusion2-chat-v2.0 / qwen-plus / deepseek-chat）
--temperature: 采样温度，对应配置中的 TEMPERATURE
--max_context: 模型最大上下文长度，对应 MODEL_MAX_CONTEXT
--max_tokens: 默认最大生成 tokens，对应 DEFAULT_MAX_TOKENS
自定义数据集参数
--dataset_path: 自定义数据集路径（支持 .json、.jsonl、.csv 格式）
--dataset_type: 数据集类型，qa 或 mcq
--dataset_name: 数据集名称（默认：custom_dataset）
--max_samples: 自定义数据集评估时最多使用的样本数（可选，默认使用全部样本）
基准数据集参数
--benchmarks: 要评估的基准数据集（当前仅支持 cmmlu）
--max_samples: 自定义数据集评估时最多使用的样本数（可选，默认使用全部样本）
性能评估参数
--evaluate_performance: 是否进行基础性能指标评估
可靠性评估参数
--evaluate_reliability: 是否进行可靠性指标评估
--reliability_dataset: 可靠性测试数据集路径
其他参数
--output: 结果输出路径（不含扩展名，默认：evaluation_results）
--device: 设备类型，cuda 或 cpu（默认：cuda）
使用示例
1. 评估QA数据集
python evaluate.py \
    --model /path/to/model \
    --dataset_path /path/to/qa_dataset.json \
    --dataset_type qa \
    --dataset_name my_qa_dataset
2. 评估MCQ数据集
python evaluate.py \
    --model /path/to/model \
    --dataset_path /path/to/mcq_dataset.json \
    --dataset_type mcq \
    --dataset_name my_mcq_dataset
3. 评估基准数据集（CMMLU）
python evaluate.py \
    --model /path/to/model \
    --benchmarks cmmlu
注意：CMMLU数据集需要手动下载并放置在 benchmark_data/cmmlu/test/ 目录下（支持 .csv 或 .jsonl 格式）。

4. 完整评估（自定义QA数据集 + CMMLU基准 + 性能）
python evaluate.py \
    --backend openai_compatible \
    --api_base_url "http://10.10.0.102:7865/v1" \
    --api_model_name "Fusion2-chat-v2.0" \
    --api_key "sk-internal" \
    --dataset_path qa_data/swift_datasets_test.jsonl \
    --dataset_type qa \
    --dataset_name swift_qa \
    --max_samples 200 \
    --benchmarks cmmlu \
    --evaluate_performance \
    --output full_evaluation_results
数据格式
QA数据集格式
支持多种数据格式和字段名称，系统会自动识别：

1. 标准格式（JSON/JSONL/CSV）
[
    {
        "question": "什么是人工智能？",
        "answer": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
    }
]
2. Alpaca格式（instruction/input/output）
{"instruction": "根据描述生成医生主诉", "input": "患者描述...", "output": "生成的病历结果"}
3. Query/Response格式
{"query": "根据描述生成医生主诉：……", "response": "左手烙铁头蛇咬伤致左上肢肿痛4小时"}
4. 中文字段格式
{"问题": "什么是人工智能？", "答案": "人工智能是计算机科学的一个分支..."}
注意：系统会自动识别以下字段组合：

question/answer（标准格式）
instruction/input/output（Alpaca格式）
query/response（查询响应格式）
问题/答案（中文字段）
MCQ数据集格式
JSON格式
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
    },
    {
        "question": "深度学习的优势包括？",
        "options": {
            "A": "自动特征提取",
            "B": "处理大规模数据",
            "C": "两者都是",
            "D": "两者都不是"
        },
        "answer": "C"
    }
]
JSONL格式
{"question": "以下哪个是机器学习算法？", "options": {"A": "线性回归", "B": "二分查找", "C": "冒泡排序", "D": "快速排序"}, "answer": "A"}
{"question": "深度学习的优势包括？", "options": {"A": "自动特征提取", "B": "处理大规模数据", "C": "两者都是", "D": "两者都不是"}, "answer": "C"}
CSV格式
question,A,B,C,D,answer
"以下哪个是机器学习算法？","线性回归","二分查找","冒泡排序","快速排序","A"
"深度学习的优势包括？","自动特征提取","处理大规模数据","两者都是","两者都不是","C"
可靠性测试数据集格式
标准格式（用于输入鲁棒性测试）
[
    {
        "question": "什么是人工智能？",
        "answer": "人工智能是计算机科学的一个分支..."
    }
]
同义问题格式（用于语义一致性测试）
[
    {
        "paraphrases": [
            "什么是人工智能？",
            "人工智能的定义是什么？",
            "请解释人工智能的概念"
        ]
    },
    {
        "paraphrases": [
            "机器学习的基本原理是什么？",
            "机器学习是如何工作的？",
            "请说明机器学习的核心思想"
        ]
    }
]
输出结果
评估完成后，会生成两个文件：

JSON格式 (evaluation_results.json)：包含所有评估结果的详细数据
表格格式 (evaluation_results.txt)：可读性强的表格格式结果
结果表格示例
+------------------+----------------+----------+----------------------+------+--------+----------+
| Model            | Dataset        | Type     | Metric               | Num  | Score  | Domain   |
+==================+================+==========+======================+======+========+==========+
| my_model         | my_qa_dataset  | QA       | mean_EM              | 100  | 0.6500 | default  |
| my_model         | my_qa_dataset  | QA       | mean_bleu-1          | 100  | 0.7200 | default  |
| my_model         | my_qa_dataset  | QA       | mean_bleu-2          | 100  | 0.6800 | default  |
| my_model         | my_qa_dataset  | QA       | mean_bleu-3          | 100  | 0.6500 | default  |
| my_model         | my_qa_dataset  | QA       | mean_bleu-4          | 100  | 0.6200 | default  |
| my_model         | my_qa_dataset  | QA       | mean_Rouge-1-F       | 100  | 0.7500 | default  |
| my_model         | my_qa_dataset  | QA       | mean_Rouge-2-F       | 100  | 0.7000 | default  |
| my_model         | my_qa_dataset  | QA       | mean_Rouge-L-F       | 100  | 0.7300 | default  |
| my_model         | my_qa_dataset  | QA       | mean_BERTScore       | 100  | 0.8100 | default  |
| my_model         | my_qa_dataset  | QA       | mean_Comprehensive_Score | 100 | 0.7800 | default  |
| my_model         | my_qa_dataset  | QA       | mean_Factual_Consistency | 100 | 0.8500 | default  |
| my_model         | cmmlu          | MCQ      | Accuracy             | 500  | 0.6800 | all      |
+------------------+----------------+----------+----------------------+------+--------+----------+
评估指标说明
QA数据集指标
mean_EM: 精确匹配率，衡量生成答案与参考答案的完全匹配程度（标准化后比较）
mean_bleu-1/2/3/4: BLEU分数，基于n-gram重叠度（使用几何平均，jieba中文分词）
BLEU-1：基于1-gram的词级重叠
BLEU-2：基于2-gram的短语级重叠
BLEU-3：基于3-gram的重叠
BLEU-4：基于4-gram的重叠（最常用，能捕捉更多语义连贯性）
mean_Rouge-1-R/P/F: ROUGE-1的召回率、精确率、F1分数（1-gram重叠）
mean_Rouge-2-R/P/F: ROUGE-2的召回率、精确率、F1分数（2-gram重叠）
mean_Rouge-L-R/P/F: ROUGE-L的召回率、精确率、F1分数（最长公共子序列）
mean_BERTScore: 基于BERT的语义相似度，使用bert-base-chinese模型（自动使用CUDA加速）
mean_Comprehensive_Score: 综合质量分数（BERTScore × 0.6 + ROUGE-L F1 × 0.4）
mean_Factual_Consistency: 事实一致性，基于NLI模型（hfl/rbtl3）检测矛盾概率，一致性分数 = 1 - 矛盾概率
技术细节：

所有ROUGE指标使用自定义中文分词器（jieba），确保中文文本正确分词
BLEU计算使用标准几何平均权重，确保符合BLEU标准
BERTScore自动检测并使用CUDA（如果可用）以加速计算
MCQ数据集指标
Accuracy: 正确选择率
ECE: 期望校准误差，衡量预测概率与实际准确率的一致性
Position Bias: 位置偏差，检测模型是否对特定位置选项有偏好
Length Bias: 长度偏差，检测模型是否对选项长度有偏好
Distractor Sensitivity: 干扰项敏感性，衡量模型对干扰项变化的敏感程度
性能指标
Throughput: 吞吐量，单位时间内处理的请求数
First Token Latency: 首Token延迟，从输入到生成第一个token的时间
Generation Speed: 生成速度，每秒生成的token数
End-to-End Latency: 端到端延迟，从输入到完整输出的时间
可靠性指标
Robustness: 鲁棒性分数，衡量模型在输入扰动下的性能保持能力
Semantic Consistency: 语义一致性，衡量模型对同义问题的回答一致性
Error Rate: 错误率，包括无效输出率和空响应率
注意事项
模型格式：支持HuggingFace格式的模型，需要包含tokenizer和model文件
GPU要求：推荐使用GPU（CUDA）进行推理，CPU模式速度较慢
BERTScore会自动使用CUDA加速（如果可用）
NLI模型（hfl/rbtl3）推理也会使用GPU（如果可用）
内存要求：根据模型大小，需要足够的GPU/CPU内存
数据集准备：CMMLU数据集需要手动下载并放置在 benchmark_data/cmmlu/test/ 目录下（支持 .csv 或 .jsonl 格式）
依赖安装：某些依赖（如bert-score、jieba）可能需要较长时间安装
中文分词：所有文本评估指标（BLEU、ROUGE）使用jieba进行中文分词，确保评估准确性
常见问题
Q: 如何支持自定义模型格式？
A: 修改 utils/model_loader.py 中的 _load_model 方法，添加您的模型加载逻辑。

Q: 如何添加新的评估指标？
A: 在对应的评估器类（qa_evaluator.py 或 mcq_evaluator.py）中添加新的方法，并在 evaluate 方法中调用。

Q: 评估速度很慢怎么办？
A: 可以：

使用GPU加速
减少测试样本数量
调整批次大小
使用更小的模型进行评估
Q: 如何自定义基准数据集的下载源？
A: CMMLU数据集需要手动下载并放置在 benchmark_data/cmmlu/test/ 目录下（支持 .csv 或 .jsonl 格式）。

Q: 评估指标中的 mean_ 前缀是什么意思？
A: 所有指标名称前的 mean_ 前缀表示该指标是计算整个数据集所有样本的平均值。例如 mean_bleu-1 表示所有样本的BLEU-1分数的平均值。

Q: 为什么ROUGE和BLEU分数使用jieba分词？
A: 标准的分词工具（如NLTK的word_tokenize）是为英文设计的，对中文会按字符分割，导致评估不准确。使用jieba进行中文分词可以确保评估结果的准确性。

许可证
本项目采用 MIT 许可证。

贡献
欢迎提交Issue和Pull Request！
