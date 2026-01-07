# -*- coding: utf-8 -*-
"""
MCQ数据集评估器
实现多选题格式的评估指标
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from scipy import stats
import re


class MCQEvaluator:
    """MCQ数据集评估器"""
    
    def __init__(self, model_loader):
        """
        初始化MCQ评估器
        
        Args:
            model_loader: 模型加载器
        """
        self.model_loader = model_loader
    
    def extract_option(self, text: str) -> str:
        """从文本中提取选项（A, B, C, D等）"""
        if not text:
            return None
        
        text = text.strip()
        
        # 匹配选项模式（按优先级排序）
        patterns = [
            # 答案：A / 答案是A / 选择：B / 选项为C
            r'(?:答案|选择|选项)[是|为|：|:]\s*([A-Z])',
            # 开头就是单个字母：A. / A、/ A)
            r'^([A-Z])[\.、\)）]',
            # 括号中的字母：(A) / （B）
            r'[\(（]([A-Z])[\)）]',
            # 行尾的字母：...A / ...B
            r'([A-Z])\s*$',
            # 单独的字母（前后有空格或标点）
            r'\s([A-Z])\s',
            # 任何大写字母（最后兜底）
            r'([A-Z])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                option = match.group(1).upper()
                # 只返回有效的选项字母
                if option in ['A', 'B', 'C', 'D', 'E', 'F']:
                    return option
        
        return None
    
    def accuracy(self, predictions: List[str], references: List[str]) -> float:
        """准确率计算"""
        correct = 0
        for pred, ref in zip(predictions, references):
            if pred and ref and pred.upper() == ref.upper():
                correct += 1
        return correct / len(predictions) if len(predictions) > 0 else 0.0
    
    def expected_calibration_error(self, predictions: List[str], 
                                   probabilities: List[float],
                                   references: List[str]) -> float:
        """期望校准误差（ECE）"""
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在这个置信度区间的样本
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # 计算这个区间内的准确率
                accuracy_in_bin = np.mean([
                    1.0 if pred == ref else 0.0
                    for pred, ref, in_bin_flag in zip(predictions, references, in_bin)
                    if in_bin_flag
                ])
                
                # 计算这个区间内的平均置信度
                avg_confidence_in_bin = np.mean([
                    prob for prob, in_bin_flag in zip(probabilities, in_bin)
                    if in_bin_flag
                ])
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def position_bias(self, data: List[Dict], predictions: List[str]) -> float:
        """位置偏差分析（卡方检验）"""
        # 统计每个位置的正确率
        position_correct = {pos: {'correct': 0, 'total': 0} 
                           for pos in ['A', 'B', 'C', 'D', 'E', 'F']}
        
        for item, pred in zip(data, predictions):
            correct_answer = item.get('answer', '').upper()
            position = correct_answer
            
            if position in position_correct:
                position_correct[position]['total'] += 1
                if pred and pred.upper() == position:
                    position_correct[position]['correct'] += 1
        
        # 计算每个位置的正确率
        correct_rates = []
        totals = []
        for pos in ['A', 'B', 'C', 'D', 'E', 'F']:
            if position_correct[pos]['total'] > 0:
                rate = position_correct[pos]['correct'] / position_correct[pos]['total']
                correct_rates.append(rate)
                totals.append(position_correct[pos]['total'])
        
        if len(correct_rates) < 2:
            return 0.0
        
        # 卡方检验
        try:
            # 构建观察频数
            observed = []
            for pos in ['A', 'B', 'C', 'D', 'E', 'F']:
                if position_correct[pos]['total'] > 0:
                    observed.append([
                        position_correct[pos]['correct'],
                        position_correct[pos]['total'] - position_correct[pos]['correct']
                    ])
            
            if len(observed) >= 2:
                observed_array = np.array(observed)
                chi2, p_value = stats.chi2_contingency(observed_array)[:2]
                return p_value  # 返回p值，p值越小表示偏差越大
        except:
            pass
        
        return 1.0
    
    def length_bias(self, data: List[Dict]) -> float:
        """长度偏差分析（T检验）"""
        correct_lengths = []
        incorrect_lengths = []
        
        for item in data:
            correct_answer = item.get('answer', '').upper()
            options = item.get('options', {})
            
            # 处理 options 可能是列表或字典的情况
            if isinstance(options, list):
                # 如果是列表，转换为字典
                options_dict = {}
                option_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                for idx, opt_value in enumerate(options):
                    if idx < len(option_keys):
                        options_dict[option_keys[idx]] = str(opt_value)
                options = options_dict
            elif not isinstance(options, dict):
                # 如果既不是列表也不是字典，跳过该项
                continue
            
            if correct_answer in options:
                correct_length = len(str(options[correct_answer]))
                correct_lengths.append(correct_length)
            
            # 收集错误选项的长度
            for opt_key, opt_value in options.items():
                if opt_key.upper() != correct_answer:
                    incorrect_lengths.append(len(str(opt_value)))
        
        if len(correct_lengths) == 0 or len(incorrect_lengths) == 0:
            return 1.0
        
        # T检验
        try:
            t_stat, p_value = stats.ttest_ind(correct_lengths, incorrect_lengths)
            return p_value  # 返回p值
        except:
            return 1.0
    
    def distractor_sensitivity(self, data: List[Dict], predictions: List[str]) -> float:
        """干扰项敏感性"""
        # 这里简化实现：计算替换干扰项后的准确率波动
        # 实际应用中需要提供替换后的数据集
        
        # 模拟：基于原始准确率计算敏感性
        original_accuracy = self.accuracy(
            predictions,
            [item.get('answer', '') for item in data]
        )
        
        # 这里返回一个模拟值，实际需要替换干扰项后重新评估
        # 敏感性 = 准确率的标准差（需要多次替换测试）
        return 0.0  # 占位符，需要实际数据集支持
    
    def get_cognitive_level(self, question: str) -> str:
        """根据问题自动标注认知水平"""
        # 简化的动词词典
        verb_dict = {
            '记忆': ['是什么', '定义', '列出', '回忆', '识别'],
            '理解': ['解释', '说明', '描述', '比较', '区分'],
            '应用': ['应用', '使用', '计算', '解决', '演示'],
            '分析': ['分析', '比较', '对比', '分类', '区分'],
            '评价': ['评价', '判断', '评估', '批评', '选择'],
            '创造': ['设计', '创造', '构建', '提出', '发明']
        }
        
        question_lower = question.lower()
        for level, verbs in verb_dict.items():
            for verb in verbs:
                if verb in question_lower:
                    return level
        
        return 'general'
    
    def evaluate(self, data: List[Dict], dataset_name: str = "mcq_dataset") -> List[Tuple[str, float, int]]:
        """
        评估MCQ数据集
        
        Args:
            data: 数据集列表，每个元素包含 'question', 'options', 'answer' 字段
            dataset_name: 数据集名称
        
        Returns:
            评估结果列表，每个元素为 (指标名称, 分数, 样本数)
        """
        print(f"开始评估MCQ数据集: {dataset_name}")
        print(f"样本数量: {len(data)}")
        
        results = []
        predictions = []
        probabilities = []
        
        # 生成预测
        print("\n生成模型预测...")
        failed_count = 0
        for item in tqdm(data, desc="预测进度"):
            question = item['question']
            options = item.get('options', {})
            
            # 处理 options 可能是列表或字典的情况
            if isinstance(options, list):
                # 如果是列表，转换为字典（假设列表项是选项值，使用 A, B, C, D... 作为键）
                options_dict = {}
                option_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                for idx, opt_value in enumerate(options):
                    if idx < len(option_keys):
                        options_dict[option_keys[idx]] = str(opt_value)
                options = options_dict
            elif not isinstance(options, dict):
                # 如果既不是列表也不是字典，尝试转换为字典
                options = {}
            
            # 构建选项文本
            options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
            prompt = f"{question}\n{options_text}\n请选择正确答案（只需回答选项字母，如A、B、C或D）："
            
            # 使用模型生成答案
            try:
                response = self.model_loader.generate(prompt)
                if not response:
                    failed_count += 1
                    predictions.append(None)
                    probabilities.append(0.0)
                    continue
                
                pred_option = self.extract_option(response)
                if pred_option is None:
                    # 如果提取失败，尝试从响应中查找选项字母
                    # 有时模型会返回完整答案文本，需要匹配选项内容
                    for opt_key, opt_value in options.items():
                        if opt_value and str(opt_value).strip() in response:
                            pred_option = opt_key.upper()
                            break
                
                predictions.append(pred_option)
                # 获取置信度（简化实现，实际需要模型输出概率）
                probabilities.append(0.8 if pred_option else 0.0)
            except Exception as e:
                print(f"\n警告: 生成预测时出错: {e}")
                failed_count += 1
                predictions.append(None)
                probabilities.append(0.0)
        
        if failed_count > 0:
            print(f"\n警告: {failed_count}/{len(data)} 个样本的预测生成失败")
        
        references = [item.get('answer', '').upper() for item in data]
        
        # 计算基础指标
        print("\n计算基础指标...")
        
        # 准确率
        acc = self.accuracy(predictions, references)
        results.append(('Accuracy', acc, len(data)))
        
        # 按认知水平分组计算准确率
        cognitive_levels = {}
        for item, pred, ref in zip(data, predictions, references):
            level = self.get_cognitive_level(item['question'])
            if level not in cognitive_levels:
                cognitive_levels[level] = {'correct': 0, 'total': 0}
            cognitive_levels[level]['total'] += 1
            if pred and pred.upper() == ref:
                cognitive_levels[level]['correct'] += 1
        
        for level, stats in cognitive_levels.items():
            if stats['total'] > 0:
                level_acc = stats['correct'] / stats['total']
                results.append((f'Accuracy_{level}', level_acc, stats['total']))
        
        # 置信度校准
        ece = self.expected_calibration_error(predictions, probabilities, references)
        results.append(('ECE', ece, len(data)))
        
        # 可靠性指标
        print("\n计算可靠性指标...")
        
        # 位置偏差
        position_bias_p = self.position_bias(data, predictions)
        results.append(('Position_Bias_PValue', position_bias_p, len(data)))
        
        # 长度偏差
        length_bias_p = self.length_bias(data)
        results.append(('Length_Bias_PValue', length_bias_p, len(data)))
        
        # 干扰项敏感性
        distractor_sens = self.distractor_sensitivity(data, predictions)
        results.append(('Distractor_Sensitivity', distractor_sens, len(data)))
        
        return results

