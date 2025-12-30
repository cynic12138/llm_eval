# -*- coding: utf-8 -*-
"""
基准数据集评估器
仅支持 CMMLU 数据集的评估。
从 benchmark_data/cmmlu/test 读取本地数据（支持 .jsonl 或 .csv 格式）。
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

from evaluators.mcq_evaluator import MCQEvaluator


class BenchmarkEvaluator:
    """基准数据集评估器"""
    
    def __init__(self, model_loader):
        """
        初始化基准数据集评估器
        
        Args:
            model_loader: 模型加载器
        """
        self.model_loader = model_loader
        self.mcq_evaluator = MCQEvaluator(model_loader)
        self.data_dir = Path("benchmark_data")
        self.data_dir.mkdir(exist_ok=True)
    
    def get_cmmlu_dir(self):
        """
        获取 CMMLU 数据集目录。
        
        从 benchmark_data/cmmlu/test 读取本地数据（支持 .jsonl 或 .csv 格式）。
        如果数据不存在，会给出提示。
        """
        cmmlu_dir = self.data_dir / "cmmlu"
        test_dir = cmmlu_dir / "test"
        
        if not test_dir.exists():
            print(f"错误: 未找到 CMMLU 测试集目录: {test_dir}")
            print("请确保已将 CMMLU 数据集放置在 benchmark_data/cmmlu/test 目录下")
            return cmmlu_dir
        
        jsonl_files = list(test_dir.glob("*.jsonl"))
        csv_files = list(test_dir.glob("*.csv"))
        
        if not jsonl_files and not csv_files:
            print(f"警告: CMMLU 测试集目录 {test_dir} 中未找到任何 .jsonl 或 .csv 文件")
        else:
            file_type = "jsonl" if jsonl_files else "csv"
            file_count = len(jsonl_files) if jsonl_files else len(csv_files)
            print(f"找到 CMMLU 数据集: {file_count} 个 {file_type} 文件")
        
        return cmmlu_dir
    
    def load_cmmlu_data(self, data_dir: Path) -> Dict[str, List[Dict]]:
        """加载CMMLU数据（支持 jsonl 或 csv，本地离线使用）"""
        data: Dict[str, List[Dict]] = {}
        test_dir = data_dir / "test"

        if not test_dir.exists():
            print("警告: 未找到 CMMLU 测试集目录 benchmark_data/cmmlu/test")
            return data

        jsonl_files = list(test_dir.glob("*.jsonl"))
        csv_files = list(test_dir.glob("*.csv"))

        if not jsonl_files and not csv_files:
            print("警告: CMMLU 测试集目录中未找到任何 .jsonl 或 .csv 文件")
            return data

        # 1) 先读取 jsonl（如果有），兼容原始格式
        for jsonl_file in jsonl_files:
            domain = jsonl_file.stem
            domain_data: List[Dict] = []

            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    mcq_item = {
                        "question": item.get("input", ""),
                        "options": {
                            "A": item.get("A", ""),
                            "B": item.get("B", ""),
                            "C": item.get("C", ""),
                            "D": item.get("D", ""),
                        },
                        "answer": item.get("target", ""),
                    }
                    if "E" in item:
                        mcq_item["options"]["E"] = item.get("E", "")
                    domain_data.append(mcq_item)

            data[domain] = domain_data

        # 2) 再读取 csv（你当前手动下载的数据就是 csv）
        #    假定表头为: index,Question,A,B,C,D,Answer 或 Question,A,B,C,D,Answer
        import csv  # 局部导入，避免顶层无用依赖

        for csv_file in csv_files:
            domain = csv_file.stem

            # 如果同名 domain 已经由 jsonl 加载，则跳过 csv，避免重复
            if domain in data:
                continue

            domain_data: List[Dict] = []
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = row.get("Question", "") or row.get("question", "")
                    answer = row.get("Answer", "") or row.get("answer", "")
                    mcq_item = {
                        "question": question,
                        "options": {
                            "A": row.get("A", ""),
                            "B": row.get("B", ""),
                            "C": row.get("C", ""),
                            "D": row.get("D", ""),
                        },
                        "answer": answer,
                    }
                    domain_data.append(mcq_item)

            data[domain] = domain_data

        return data
    
    def evaluate_cmmlu(self) -> List[Tuple[str, float, int, str]]:
        """评估CMMLU数据集"""
        print("\n评估CMMLU数据集...")
        
        # 获取数据目录
        data_dir = self.get_cmmlu_dir()
        
        # 加载数据
        domain_data = self.load_cmmlu_data(data_dir)
        
        if not domain_data:
            print("警告: 未找到CMMLU数据")
            return []
        
        results = []
        domain_accuracies = {}
        
        # 按领域评估
        for domain, data in tqdm(domain_data.items(), desc="领域评估"):
            print(f"\n评估领域: {domain} ({len(data)} 样本)")
            domain_results = self.mcq_evaluator.evaluate(data, f"cmmlu_{domain}")
            
            # 提取准确率
            for metric_name, score, num_samples in domain_results:
                if metric_name == 'Accuracy':
                    domain_accuracies[domain] = score
                    results.append((f'CMMLU_{domain}_Accuracy', score, num_samples, domain))
                else:
                    results.append((f'CMMLU_{domain}_{metric_name}', score, num_samples, domain))
        
        # 计算总体准确率
        if domain_accuracies:
            total_accuracy = np.mean(list(domain_accuracies.values()))
            total_samples = sum(len(data) for data in domain_data.values())
            results.append(('CMMLU_Total_Accuracy', total_accuracy, total_samples, 'all'))
        
        # 计算领域衰减率（专业领域 vs 通用领域）
        professional_domains = ['anatomy', 'agronomy', 'medicine']
        general_domains = ['general', 'common']
        
        prof_acc = np.mean([domain_accuracies.get(d, 0) for d in professional_domains if d in domain_accuracies])
        gen_acc = np.mean([domain_accuracies.get(d, 0) for d in general_domains if d in domain_accuracies])
        
        if prof_acc > 0 and gen_acc > 0:
            domain_decay = prof_acc - gen_acc
            results.append(('CMMLU_Domain_Decay_Rate', domain_decay, len(domain_data), 'all'))
        
        return results
    
    def evaluate(self, benchmark: str) -> List[Tuple[str, float, int, str]]:
        """
        评估基准数据集
        
        Args:
            benchmark: 基准数据集名称（仅支持 'cmmlu'）
        
        Returns:
            评估结果列表
        """
        if benchmark.lower() == 'cmmlu':
            return self.evaluate_cmmlu()
        else:
            raise ValueError(f"不支持的基准数据集: {benchmark}。当前仅支持 'cmmlu'")

