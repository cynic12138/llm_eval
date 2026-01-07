# -*- coding: utf-8 -*-
"""
预训练模型评估器
用于评估经过LoRA等微调后的预训练模型性能

支持：
1. 困惑度（Perplexity）评估：在预训练数据集上计算困惑度
2. 生成质量评估：评估模型在预训练数据上的生成质量
3. 知识保留测试：对比微调前后的性能
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch
from pathlib import Path


class PretrainEvaluator:
    """预训练模型评估器"""
    
    def __init__(self, model_loader):
        """
        初始化预训练评估器
        
        Args:
            model_loader: 模型加载器
        """
        self.model_loader = model_loader
        
    def load_pretrain_dataset(self, dataset_path: str) -> List[Dict]:
        """
        加载预训练数据集
        
        支持多种格式：
        1. 纯文本格式：{"text": "..."}
        2. 指令格式：{"instruction": "...", "input": "...", "output": "..."}
        3. 对话格式：{"conversations": [...]}
        4. 其他格式：自动识别
        
        Args:
            dataset_path: 数据集路径（支持 .jsonl 或 .json）
        
        Returns:
            数据列表
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"预训练数据集文件不存在: {dataset_path}")
        
        data = []
        
        if dataset_path.suffix == '.jsonl':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        data.append(item)
        elif dataset_path.suffix == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict):
                    if 'data' in content:
                        data = content['data']
                    else:
                        data = [content]
        else:
            raise ValueError(f"不支持的文件格式: {dataset_path.suffix}，仅支持 .json 或 .jsonl")
        
        return data
    
    def extract_text_from_item(self, item: Dict) -> str:
        """
        从数据项中提取文本内容
        
        Args:
            item: 数据项
        
        Returns:
            提取的文本
        """
        # 尝试多种格式
        if 'text' in item:
            return item['text']
        elif 'content' in item:
            return item['content']
        elif 'instruction' in item and 'output' in item:
            # 指令格式：组合 instruction + input + output
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            if input_text:
                return f"{instruction}\n{input_text}\n{output}"
            else:
                return f"{instruction}\n{output}"
        elif 'query' in item and 'response' in item:
            # Query/Response 格式
            return f"{item['query']}\n{item['response']}"
        elif 'conversations' in item:
            # 对话格式：拼接所有对话
            conversations = item['conversations']
            if isinstance(conversations, list):
                texts = []
                for conv in conversations:
                    if isinstance(conv, dict):
                        role = conv.get('from', conv.get('role', ''))
                        content = conv.get('value', conv.get('content', ''))
                        if content:
                            texts.append(f"{role}: {content}")
                    elif isinstance(conv, str):
                        texts.append(conv)
                return "\n".join(texts)
        elif 'prompt' in item:
            return item['prompt']
        
        # 如果都没有，尝试将整个字典转换为字符串（排除元数据字段）
        exclude_fields = {'id', 'source', 'meta', 'metadata'}
        text_parts = []
        for key, value in item.items():
            if key not in exclude_fields and isinstance(value, str):
                text_parts.append(value)
        
        if text_parts:
            return "\n".join(text_parts)
        
        # 最后兜底：返回空字符串
        return ""
    
    def calculate_perplexity(self, texts: List[str], max_samples: Optional[int] = None) -> float:
        """
        计算困惑度（Perplexity）
        
        注意：这需要模型支持计算对数概率。对于API模式，可能无法直接计算。
        
        Args:
            texts: 文本列表
            max_samples: 最大样本数（None表示使用全部）
        
        Returns:
            平均困惑度
        """
        if max_samples and max_samples > 0:
            texts = texts[:max_samples]
        
        if not texts:
            return 0.0
        
        # 检查是否是本地模型
        if self.model_loader.backend != "local":
            print("警告: 困惑度计算仅支持本地模型，API模式无法计算。返回0.0")
            return 0.0
        
        if not hasattr(self.model_loader, 'model') or self.model_loader.model is None:
            print("警告: 模型未加载，无法计算困惑度。返回0.0")
            return 0.0
        
        if not hasattr(self.model_loader, 'tokenizer') or self.model_loader.tokenizer is None:
            print("警告: Tokenizer未加载，无法计算困惑度。返回0.0")
            return 0.0
        
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        device = self.model_loader.device
        
        total_loss = 0.0
        total_tokens = 0
        
        print(f"计算困惑度（共 {len(texts)} 个样本）...")
        
        for text in tqdm(texts, desc="困惑度计算"):
            if not text or not text.strip():
                continue
            
            try:
                # 编码文本
                inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                                 max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 计算损失
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    # 计算token数
                    num_tokens = inputs['input_ids'].shape[1]
                    if num_tokens > 0:
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
            except Exception as e:
                print(f"警告: 计算困惑度时出错: {e}")
                continue
        
        if total_tokens == 0:
            print("警告: 没有有效的token，无法计算困惑度。返回0.0")
            return 0.0
        
        # 平均损失
        avg_loss = total_loss / total_tokens
        
        # 困惑度 = exp(平均损失)
        perplexity = np.exp(avg_loss)
        
        return float(perplexity)
    
    def evaluate_generation_quality(self, data: List[Dict], max_samples: Optional[int] = None) -> List[Tuple[str, float, int]]:
        """
        评估生成质量
        
        对于预训练数据集中的每个样本，使用模型生成，然后评估生成质量。
        这可以用于评估模型是否"记住"了训练数据。
        
        Args:
            data: 预训练数据集
            max_samples: 最大样本数（None表示使用全部）
        
        Returns:
            评估结果列表，每个元素为 (指标名称, 分数, 样本数)
        """
        if max_samples and max_samples > 0:
            data = data[:max_samples]
        
        results = []
        
        print(f"评估生成质量（共 {len(data)} 个样本）...")
        
        # 提取文本和提示
        prompts = []
        references = []
        
        for item in data:
            text = self.extract_text_from_item(item)
            if not text or not text.strip():
                continue
            
            # 对于预训练数据，通常使用前半部分作为提示，后半部分作为参考
            # 或者使用 instruction 作为提示，output 作为参考
            if 'instruction' in item and 'output' in item:
                prompt = item.get('instruction', '')
                if item.get('input'):
                    prompt = f"{prompt}\n{item['input']}"
                reference = item.get('output', '')
            elif 'query' in item and 'response' in item:
                prompt = item['query']
                reference = item['response']
            else:
                # 对于纯文本，使用前50%作为提示，后50%作为参考
                text_len = len(text)
                split_point = text_len // 2
                prompt = text[:split_point]
                reference = text[split_point:]
            
            if prompt and reference:
                prompts.append(prompt)
                references.append(reference)
        
        if not prompts:
            print("警告: 没有有效的提示-参考对，无法评估生成质量")
            return results
        
        # 生成预测
        predictions = []
        print("生成模型预测...")
        for prompt in tqdm(prompts, desc="生成预测"):
            try:
                pred = self.model_loader.generate(prompt, max_length=512)
                predictions.append(pred if pred else "")
            except Exception as e:
                print(f"警告: 生成预测时出错: {e}")
                predictions.append("")
        
        # 计算指标（使用QA评估器的指标）
        from evaluators.qa_evaluator import QAEvaluator
        qa_evaluator = QAEvaluator(self.model_loader)
        
        # 计算BLEU分数
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            if pred and ref:
                bleu = qa_evaluator.bleu_score(pred, ref, n=4)
                bleu_scores.append(bleu)
        
        if bleu_scores:
            results.append(('mean_bleu-4', np.mean(bleu_scores), len(bleu_scores)))
        
        # 计算ROUGE分数
        rouge1_f = []
        rouge2_f = []
        rougeL_f = []
        
        for pred, ref in zip(predictions, references):
            if pred and ref:
                rouge_scores = qa_evaluator.rouge_scores(pred, ref)
                rouge1_f.append(rouge_scores['rouge1']['fmeasure'])
                rouge2_f.append(rouge_scores['rouge2']['fmeasure'])
                rougeL_f.append(rouge_scores['rougeL']['fmeasure'])
        
        if rouge1_f:
            results.append(('mean_Rouge-1-F', np.mean(rouge1_f), len(rouge1_f)))
        if rouge2_f:
            results.append(('mean_Rouge-2-F', np.mean(rouge2_f), len(rouge2_f)))
        if rougeL_f:
            results.append(('mean_Rouge-L-F', np.mean(rougeL_f), len(rougeL_f)))
        
        # 计算BERTScore
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
        if valid_pairs:
            preds = [p for p, r in valid_pairs]
            refs = [r for p, r in valid_pairs]
            bert_score_value = qa_evaluator.bert_score(preds, refs)
            results.append(('mean_BERTScore', bert_score_value, len(valid_pairs)))
        
        return results
    
    def evaluate(self, pretrain_dataset_path: str, dataset_name: str = "pretrain_dataset",
                 calculate_perplexity: bool = True, 
                 evaluate_generation: bool = True,
                 max_samples: Optional[int] = None) -> List[Tuple[str, float, int]]:
        """
        评估预训练模型
        
        Args:
            pretrain_dataset_path: 预训练数据集路径
            dataset_name: 数据集名称
            calculate_perplexity: 是否计算困惑度
            evaluate_generation: 是否评估生成质量
            max_samples: 最大样本数（None表示使用全部）
        
        Returns:
            评估结果列表，每个元素为 (指标名称, 分数, 样本数)
        """
        print(f"\n{'='*60}")
        print(f"开始评估预训练模型: {dataset_name}")
        print(f"{'='*60}\n")
        
        # 加载数据集
        data = self.load_pretrain_dataset(pretrain_dataset_path)
        print(f"加载了 {len(data)} 个样本")
        
        if max_samples and max_samples > 0:
            original_len = len(data)
            data = data[:max_samples]
            print(f"提示: 已将数据集从 {original_len} 条截断为前 {len(data)} 条用于评估")
        
        results = []
        
        # 1. 计算困惑度
        if calculate_perplexity:
            try:
                print("\n计算困惑度...")
                texts = [self.extract_text_from_item(item) for item in data]
                texts = [t for t in texts if t and t.strip()]
                
                if texts:
                    perplexity = self.calculate_perplexity(texts, max_samples=None)
                    results.append(('Perplexity', perplexity, len(texts)))
                    print(f"困惑度: {perplexity:.4f}")
                else:
                    print("警告: 没有有效的文本，跳过困惑度计算")
            except Exception as e:
                print(f"错误: 计算困惑度失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. 评估生成质量
        if evaluate_generation:
            try:
                print("\n评估生成质量...")
                gen_results = self.evaluate_generation_quality(data, max_samples=None)
                results.extend(gen_results)
            except Exception as e:
                print(f"错误: 评估生成质量失败: {e}")
                import traceback
                traceback.print_exc()
        
        return results
