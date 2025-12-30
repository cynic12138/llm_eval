#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估主脚本
支持QA和MCQ两种数据集类型的自动评估
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from tabulate import tabulate
from tqdm import tqdm

from evaluators.qa_evaluator import QAEvaluator
from evaluators.mcq_evaluator import MCQEvaluator
from evaluators.benchmark_evaluator import BenchmarkEvaluator
from evaluators.performance_evaluator import PerformanceEvaluator
from evaluators.reliability_evaluator import ReliabilityEvaluator
from utils.model_loader import ModelLoader
from utils.data_loader import DataLoader


class ModelEvaluator:
    """模型评估主类"""
    
    def __init__(self, model_loader: ModelLoader, max_samples: Optional[int] = None):
        """
        初始化评估器
        
        Args:
            model_loader: 已配置好的模型加载器 / 大模型客户端
            max_samples: 自定义数据集评估时最多使用的样本数（None 表示使用全部）
        """
        self.model_loader = model_loader
        self.max_samples = max_samples
        # 为了在结果表格中展示模型标识，这里尽量从 loader 中推断
        model_id = getattr(model_loader, "model_path", None) or getattr(
            model_loader, "model_name", "unknown_model"
        )
        self.model_id = os.path.basename(str(model_id))
        self.results = []
        
    def evaluate_custom_dataset(self, dataset_path: str, dataset_type: str, 
                                 dataset_name: str = "custom_dataset"):
        """
        评估自定义数据集
        
        Args:
            dataset_path: 数据集路径
            dataset_type: 数据集类型 ('qa' 或 'mcq')
            dataset_name: 数据集名称
        """
        print(f"\n{'='*60}")
        print(f"开始评估自定义数据集: {dataset_name}")
        print(f"数据集类型: {dataset_type.upper()}")
        print(f"{'='*60}\n")
        
        # 加载数据集
        data_loader = DataLoader()
        data = data_loader.load_dataset(dataset_path, dataset_type)
        
        # 如果设置了 max_samples，则仅取前 max_samples 条样本
        if self.max_samples is not None and self.max_samples > 0:
            original_len = len(data)
            data = data[: self.max_samples]
            print(f"提示: 已将自定义数据集从 {original_len} 条截断为前 {len(data)} 条用于评估（max_samples={self.max_samples}）")
        
        if dataset_type.lower() == 'qa':
            evaluator = QAEvaluator(self.model_loader)
            metrics = evaluator.evaluate(data, dataset_name)
        elif dataset_type.lower() == 'mcq':
            evaluator = MCQEvaluator(self.model_loader)
            metrics = evaluator.evaluate(data, dataset_name)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        # 保存结果
        for metric_name, score, num_samples in metrics:
            self.results.append({
                'model': self.model_id,
                'dataset': dataset_name,
                'dataset_type': dataset_type.upper(),
                'metric': metric_name,
                'num': num_samples,
                'score': score
            })
        
        return metrics
    
    def evaluate_benchmarks(self, benchmarks: List[str] = None):
        """
        评估基准数据集
        
        Args:
            benchmarks: 要评估的基准数据集列表，None表示评估所有（当前仅支持 'cmmlu'）
        """
        if benchmarks is None:
            benchmarks = ['cmmlu']
        
        print(f"\n{'='*60}")
        print(f"开始评估基准数据集: {', '.join(benchmarks)}")
        print(f"{'='*60}\n")
        
        benchmark_evaluator = BenchmarkEvaluator(self.model_loader)
        
        for benchmark in benchmarks:
            if benchmark.lower() != 'cmmlu':
                print(f"警告: 跳过不支持的基准数据集: {benchmark}（当前仅支持 'cmmlu'）")
                continue
                
            print(f"\n评估基准数据集: {benchmark.upper()}")
            metrics = benchmark_evaluator.evaluate(benchmark)
            
            for metric_name, score, num_samples, domain in metrics:
                self.results.append({
                    'model': self.model_id,
                    'dataset': benchmark,
                    'dataset_type': 'MCQ',
                    'metric': metric_name,
                    'num': num_samples,
                    'score': score,
                    'domain': domain
                })
    
    def evaluate_performance(self):
        """评估基础性能指标"""
        print(f"\n{'='*60}")
        print("开始评估基础性能指标")
        print(f"{'='*60}\n")
        
        perf_evaluator = PerformanceEvaluator(self.model_loader)
        metrics = perf_evaluator.evaluate()
        
        for metric_name, score, num_samples in metrics:
            self.results.append({
                'model': self.model_id,
                'dataset': 'performance',
                'dataset_type': 'PERF',
                'metric': metric_name,
                'num': num_samples,
                'score': score
            })
        
        return metrics
    
    def evaluate_reliability(self, reliability_dataset_path: str = None):
        """
        评估可靠性指标
        
        Args:
            reliability_dataset_path: 可靠性测试数据集路径
        """
        print(f"\n{'='*60}")
        print("开始评估可靠性指标")
        print(f"{'='*60}\n")
        
        rel_evaluator = ReliabilityEvaluator(self.model_loader)
        metrics = rel_evaluator.evaluate(reliability_dataset_path)
        
        for metric_name, score, num_samples in metrics:
            self.results.append({
                'model': self.model_id,
                'dataset': 'reliability',
                'dataset_type': 'RELIABILITY',
                'metric': metric_name,
                'num': num_samples,
                'score': score
            })
        
        return metrics
    
    def generate_results_table(self) -> str:
        """生成结果表格"""
        if not self.results:
            return "暂无评估结果"
        
        # 准备表格数据
        table_data = []
        for result in self.results:
            table_data.append([
                result.get('model', 'N/A'),
                result.get('dataset', 'N/A'),
                result.get('dataset_type', 'N/A'),
                result.get('metric', 'N/A'),
                result.get('num', 'N/A'),
                f"{result.get('score', 0):.4f}" if isinstance(result.get('score'), (int, float)) else result.get('score', 'N/A'),
                result.get('domain', 'default')
            ])
        
        headers = ['Model', 'Dataset', 'Type', 'Metric', 'Num', 'Score', 'Domain']
        table = tabulate(table_data, headers=headers, tablefmt='grid')
        
        return table
    
    def save_results(self, output_path: str):
        """保存结果到文件"""
        # 保存JSON格式
        json_path = output_path.replace('.txt', '.json') if output_path.endswith('.txt') else output_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存表格格式
        table = self.generate_results_table()
        txt_path = output_path if output_path.endswith('.txt') else output_path + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(table)
        
        print(f"\n结果已保存到: {json_path} 和 {txt_path}")


def main():
    parser = argparse.ArgumentParser(description='模型评估脚本')
    
    # 模型来源相关参数
    parser.add_argument(
        '--backend',
        type=str,
        default='local',
        choices=['local', 'internal_api', 'qwen', 'deepseek', 'openai_compatible'],
        help='模型调用后端类型：local（本地模型）、internal_api（内部OpenAI兼容接口）、qwen、deepseek、openai_compatible'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='本地模型路径 / HuggingFace模型名称（backend=local 时必填）'
    )
    
    # OpenAI 兼容 API 参数（internal_api / qwen / deepseek / openai_compatible）
    parser.add_argument('--model_api', type=str, default=None,
                        help='模型API标识（可选，仅用于日志标注，不参与实际调用）')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI兼容API的api_key。qwen/deepseek模式下可不填，默认从环境变量DASHSCOPE_API_KEY/DEEPSEEK_API_KEY读取')
    parser.add_argument('--api_base_url', type=str, default=None,
                        help='OpenAI兼容API的base_url，例如 http://10.10.0.102:7865/v1')
    parser.add_argument('--api_model_name', type=str, default=None,
                        help='OpenAI兼容API的模型名称，例如 Fusion2-chat-v2.0 / qwen-plus / deepseek-chat')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度 TEMPERATURE')
    parser.add_argument('--max_context', type=int, default=8192,
                        help='模型最大上下文长度 MODEL_MAX_CONTEXT')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='默认最大生成tokens DEFAULT_MAX_TOKENS')
    
    # 自定义数据集参数
    parser.add_argument('--dataset_path', type=str, default=None, 
                       help='自定义数据集路径')
    parser.add_argument('--dataset_type', type=str, choices=['qa', 'mcq'], 
                       default=None, help='数据集类型: qa 或 mcq')
    parser.add_argument('--dataset_name', type=str, default='custom_dataset',
                       help='数据集名称')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='自定义数据集评估时最多使用的样本数（默认使用全部样本）')
    
    # 基准数据集参数
    parser.add_argument('--benchmarks', type=str, nargs='+', 
                       choices=['cmmlu'], default=None,
                       help='要评估的基准数据集（当前仅支持 cmmlu）')
    
    # 性能评估参数
    parser.add_argument('--evaluate_performance', action='store_true',
                       help='是否进行基础性能指标评估')
    
    # 可靠性评估参数
    parser.add_argument('--evaluate_reliability', action='store_true',
                       help='是否进行可靠性指标评估')
    parser.add_argument('--reliability_dataset', type=str, default=None,
                       help='可靠性测试数据集路径')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='结果输出路径（不含扩展名）')
    
    # 设备参数（仅本地模型有效）
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='设备类型（仅backend=local时生效）')
    
    args = parser.parse_args()
    
    # 验证至少选择了一种评估方式
    has_custom_dataset = args.dataset_path and args.dataset_type
    has_benchmarks = args.benchmarks is not None and len(args.benchmarks) > 0
    has_performance = args.evaluate_performance
    has_reliability = args.evaluate_reliability
    
    if not (has_custom_dataset or has_benchmarks or has_performance or has_reliability):
        parser.error("请至少选择一种评估方式：\n"
                    "  - 自定义数据集: --dataset_path 和 --dataset_type\n"
                    "  - 基准数据集: --benchmarks\n"
                    "  - 性能指标: --evaluate_performance\n"
                    "  - 可靠性指标: --evaluate_reliability")
    
    # 验证模型来源参数
    if args.backend == 'local':
        if not args.model:
            parser.error("backend=local 模式下必须通过 --model 提供本地模型路径或名称")
    else:
        # API 模式下建议提供 base_url 和 model_name（qwen/deepseek 可以有默认值）
        if args.backend in ['internal_api', 'openai_compatible']:
            if not args.api_base_url or not args.api_model_name:
                parser.error("internal_api/openai_compatible 模式需要提供 --api_base_url 和 --api_model_name")
    
    # 验证自定义数据集参数
    if args.dataset_path and not args.dataset_type:
        parser.error("使用 --dataset_path 时必须指定 --dataset_type (qa 或 mcq)")
    if args.dataset_type and not args.dataset_path:
        parser.error("使用 --dataset_type 时必须指定 --dataset_path")
    
    # 验证可靠性数据集参数
    if args.evaluate_reliability and not args.reliability_dataset:
        print("警告: 进行可靠性评估但未提供可靠性数据集，将使用默认测试数据")
    
    try:
        # 创建模型加载器 / API 客户端
        model_loader = ModelLoader(
            backend=args.backend,
            model_path=args.model,
            device=args.device,
            api_key=args.api_key,
            base_url=args.api_base_url,
            model_name=args.api_model_name,
            temperature=args.temperature,
            max_context=args.max_context,
            max_tokens=args.max_tokens,
        )

        # 创建评估器
        evaluator = ModelEvaluator(model_loader, max_samples=args.max_samples)
        
        # 评估自定义数据集
        if has_custom_dataset:
            try:
                evaluator.evaluate_custom_dataset(
                    args.dataset_path, 
                    args.dataset_type,
                    args.dataset_name
                )
            except Exception as e:
                print(f"错误: 自定义数据集评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 评估基准数据集
        if has_benchmarks:
            try:
                evaluator.evaluate_benchmarks(args.benchmarks)
            except Exception as e:
                print(f"错误: 基准数据集评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 评估性能指标
        if has_performance:
            try:
                evaluator.evaluate_performance()
            except Exception as e:
                print(f"错误: 性能指标评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 评估可靠性指标
        if has_reliability:
            try:
                evaluator.evaluate_reliability(args.reliability_dataset)
            except Exception as e:
                print(f"错误: 可靠性指标评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 显示结果
        if evaluator.results:
            print("\n" + "="*60)
            print("评估结果总览")
            print("="*60)
            print(evaluator.generate_results_table())
            
            # 保存结果
            evaluator.save_results(args.output)
        else:
            print("\n警告: 没有生成任何评估结果")
    
    except Exception as e:
        print(f"错误: 评估过程失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

