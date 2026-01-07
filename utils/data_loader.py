# -*- coding: utf-8 -*-
"""
数据加载器
支持多种数据格式的加载
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional


class DataLoader:
    """数据加载器"""
    
    def load_dataset(self, dataset_path: str, dataset_type: str) -> List[Dict]:
        """
        加载数据集
        
        Args:
            dataset_path: 数据集路径
            dataset_type: 数据集类型 ('qa' 或 'mcq')
        
        Returns:
            数据列表
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        # 根据文件扩展名选择加载方式
        if dataset_path.suffix == '.json':
            return self._load_json(dataset_path, dataset_type)
        elif dataset_path.suffix == '.jsonl':
            return self._load_jsonl(dataset_path, dataset_type)
        elif dataset_path.suffix == '.csv':
            return self._load_csv(dataset_path, dataset_type)
        else:
            raise ValueError(f"不支持的文件格式: {dataset_path.suffix}")
    
    def _load_json(self, file_path: Path, dataset_type: str) -> List[Dict]:
        """加载JSON格式数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果数据是字典，尝试提取列表
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'questions' in data:
                data = data['questions']
            else:
                data = [data]
        
        # 验证和转换数据格式
        return self._validate_and_convert(data, dataset_type)
    
    def _load_jsonl(self, file_path: Path, dataset_type: str) -> List[Dict]:
        """加载JSONL格式数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    data.append(item)
        
        return self._validate_and_convert(data, dataset_type)
    
    def _load_csv(self, file_path: Path, dataset_type: str) -> List[Dict]:
        """加载CSV格式数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        
        return self._validate_and_convert(data, dataset_type)
    
    def _validate_and_convert(self, data: List[Dict], dataset_type: str) -> List[Dict]:
        """
        验证和转换数据格式
        
        Args:
            data: 原始数据
            dataset_type: 数据集类型
        
        Returns:
            转换后的数据
        """
        if dataset_type.lower() == 'qa':
            return self._convert_to_qa_format(data)
        elif dataset_type.lower() == 'mcq':
            return self._convert_to_mcq_format(data)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    def _convert_to_qa_format(self, data: List[Dict]) -> List[Dict]:
        """转换为QA格式（支持多种格式）"""
        converted = []
        skipped_count = 0
        skipped_reasons = {'no_question': 0, 'no_answer': 0, 'both_missing': 0}
        
        for idx, item in enumerate(data):
            # 支持多种字段名格式（按优先级顺序）
            # 首先尝试标准格式
            question = (
                item.get('question') or 
                item.get('q') or 
                item.get('问题') or
                item.get('query') or
                None
            )
            
            # 如果没有找到标准格式，尝试 instruction-following 格式
            if not question:
                instruction = item.get('instruction') or item.get('指令') or ''
                input_text = item.get('input') or item.get('输入') or ''
                
                # 组合 instruction 和 input
                if instruction and input_text:
                    question = f"{instruction}\n{input_text}"
                elif instruction:
                    question = instruction
                elif input_text:
                    question = input_text
                else:
                    question = ''
            
            # 支持多种答案字段名格式（按优先级顺序）
            answer = (
                # 标准格式
                item.get('answer') or 
                item.get('a') or 
                item.get('答案') or
                # query/response 格式
                item.get('response') or
                # instruction-following 格式的 output
                item.get('output') or
                item.get('输出') or
                ''
            )
            
            if not question or not answer:
                skipped_count += 1
                if not question and not answer:
                    skipped_reasons['both_missing'] += 1
                elif not question:
                    skipped_reasons['no_question'] += 1
                elif not answer:
                    skipped_reasons['no_answer'] += 1
                continue
            
            qa_item = {
                'question': question,
                'answer': answer
            }
            
            converted.append(qa_item)
        
        # 输出统计信息
        if skipped_count > 0:
            print(f"警告: 在QA格式转换过程中，跳过了 {skipped_count}/{len(data)} 条样本")
            print(f"  原因统计: 缺少问题 {skipped_reasons['no_question']} 条, "
                  f"缺少答案 {skipped_reasons['no_answer']} 条, "
                  f"两者都缺少 {skipped_reasons['both_missing']} 条")
            print(f"  有效样本数: {len(converted)}/{len(data)}")
        
        return converted
    
    def _convert_to_mcq_format(self, data: List[Dict]) -> List[Dict]:
        """转换为MCQ格式"""
        converted = []
        
        for item in data:
            # 处理选项
            options = {}
            if 'options' in item:
                options = item['options']
            else:
                # 尝试从其他字段提取选项
                for key in ['A', 'B', 'C', 'D', 'E', 'F']:
                    if key in item:
                        options[key] = item[key]
            
            mcq_item = {
                'question': item.get('question', item.get('q', item.get('问题', ''))),
                'options': options,
                'answer': item.get('answer', item.get('correct_answer', item.get('答案', '')))
            }
            
            if not mcq_item['question'] or not mcq_item['options'] or not mcq_item['answer']:
                continue
            
            converted.append(mcq_item)
        
        return converted
    
    def load_reliability_dataset(self, dataset_path: str) -> List[Dict]:
        """
        加载可靠性测试数据集
        
        Args:
            dataset_path: 数据集路径
        
        Returns:
            数据列表
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        # 加载JSON格式
        if dataset_path.suffix == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                else:
                    data = [data]
        
        # 加载JSONL格式
        elif dataset_path.suffix == '.jsonl':
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        data.append(item)
        else:
            raise ValueError(f"不支持的文件格式: {dataset_path.suffix}")
        
        return data

