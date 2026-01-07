## 快速开始

本节只保留你**最可能用到的几种命令**，详细参数说明见 `README.md`。

---

## 1. 安装与环境

```bash
pip install -r requirements.txt

# 如需 NLTK 相关功能（可选）
python -c "import nltk; nltk.download('punkt')"
```

确认 Python 能导入本项目：

```bash
cd 大模型评测
python -c "import evaluate"
```

---

## 2. 准备本地模型与数据

- 本地大模型示例：`/home/model/Fusion3-30B-A3B-2601-SFT-v1`
- QA 数据集示例：`/home/project/大模型评测/qa_data_2/test.jsonl`

QA 数据格式（JSONL 任意一种即可）：

```jsonl
{"question": "什么是人工智能？", "answer": "人工智能是计算机科学的一个分支..."}
{"query": "根据描述生成医生主诉：……", "response": "左手烙铁头蛇咬伤致左上肢肿痛4小时"}
{"instruction": "根据描述生成医生主诉", "input": "……病人原文描述……", "output": "生成的主诉文本"}
```

---

## 3. 用 OpenAI 兼容接口做 QA 评测

当你有一个本地 HTTP 服务（OpenAI 兼容）时，可以这样跑 QA：

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

结果会输出：

- `local_qa_results.json`
- `local_qa_results.txt`

---

## 4. 用本地 30B 模型 + 多 GPU 做 QA 评测

这是你当前最常用的场景：**在 4 张卡上评测本地 30B 模型**。

### 4.1 直接命令行

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python evaluate.py \
  --backend local \
  --model /home/model/Fusion3-30B-A3B-2601-SFT-v1 \
  --qa_dataset /home/project/大模型评测/qa_data_2/test.jsonl \
  --qa_dataset_name qa_test \
  --output /home/model/Fusion3-30B-A3B-2601-SFT-v1/eval_output/qa_eval \
  --device cuda \
  --device_map auto \
  --gpu_ids "0,1,2,3"
```

说明：

- 物理 GPU 用的是 4,5,6,7 四张卡
- 因为设置了 `CUDA_VISIBLE_DEVICES=4,5,6,7`，在程序内部这四张卡会被映射为逻辑 GPU `0,1,2,3`
- `--device_map auto` 会自动把 30B 模型切到这几张卡上

### 4.2 使用脚本

仓库里已经有一个精简脚本 `start_qa_eval.sh`，内容就是上面这条命令，你可以直接：

```bash
bash start_qa_eval.sh
```

如需改模型 / 数据路径，只要编辑脚本里的路径即可。

---

## 5. 预训练数据评测（PPL + 生成质量）

如果你要评测一个**预训练/微调模型的困惑度**和生成质量，可以用：

```bash
python evaluate.py \
  --backend local \
  --model /home/model/Fusion3-30B-A3B-2507-v1 \
  --pretrain_dataset /home/project/大模型评测/pretrain_datasets/pre_test.jsonl \
  --pretrain_dataset_name pre_test \
  --max_samples 100 \
  --output /home/model/Fusion3-30B-A3B-2507-v1/eval_output/pretrain_eval \
  --device cuda \
  --device_map auto \
  --gpu_ids "0,1,2,3,4,5,6,7"
```

---

## 6. CMMLU 与其他评测

### 6.1 CMMLU

```bash
python evaluate.py \
  --backend local \
  --model /path/to/your/model \
  --benchmarks cmmlu \
  --output cmmlu_eval
```

要求：

- `benchmark_data/cmmlu/test/` 目录下放置 CMMLU 的 `.csv` 或 `.jsonl` 文件（按领域划分）

### 6.2 只做基础性能测试

```bash
python evaluate.py \
  --backend openai_compatible \
  --api_base_url "http://10.10.0.102:7865/v1" \
  --api_model_name "Fusion2-chat-v2.0" \
  --api_key "sk-internal" \
  --evaluate_performance \
  --output performance_only
```

---

## 7. 结果查看技巧

- 直接查看表格：

  ```bash
  less /path/to/output.txt
  ```

- 如果想自己画图 / 做报告，用 JSON：

  ```bash
  python -m json.tool /path/to/output.json | head
  ```

