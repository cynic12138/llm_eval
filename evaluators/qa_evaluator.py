# -*- coding: utf-8 -*-
"""
QA数据集评估器
实现问答格式的评估指标
"""

import re
import unicodedata
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import jieba
from rouge_score import rouge_scorer
from rouge_score import tokenizers
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 初始化 jieba 分词器
try:
    jieba.initialize()
except:
    pass


class ChineseTokenizer(tokenizers.Tokenizer):
    """中文分词器，使用 jieba 进行分词"""
    
    def __init__(self):
        """初始化中文分词器"""
        pass
    
    def tokenize(self, text: str) -> List[str]:
        """
        使用 jieba 对中文文本进行分词
        
        Args:
            text: 输入文本
        
        Returns:
            分词后的token列表
        """
        if not text:
            return []
        
        try:
            # 使用 jieba 进行中文分词
            tokens = list(jieba.cut(text.lower()))
            # 过滤空字符串和只包含空白字符的token
            tokens = [token.strip() for token in tokens if token.strip()]
            return tokens
        except Exception as e:
            # 如果 jieba 分词失败，使用简单分词（按空格和标点分割）
            return re.findall(r'\w+', text.lower())


class QAEvaluator:
    """QA数据集评估器"""
    
    def __init__(self, model_loader):
        """
        初始化QA评估器
        
        Args:
            model_loader: 模型加载器
        """
        self.model_loader = model_loader
        # 初始化 ROUGE scorer，支持 rouge1, rouge2, rougeL
        # 使用自定义的中文分词器，确保中文文本能正确分词
        chinese_tokenizer = ChineseTokenizer()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=False,
            tokenizer=chinese_tokenizer
        )
        # BLEU 平滑函数
        self.bleu_smoothing = SmoothingFunction().method1
        
        # 加载NLI模型用于矛盾检测
        # 使用 hfl/rbtl3 模型（中文 RoBERTa-wwm-ext-large 3层模型）
        # 注意：该模型是通用中文BERT模型，主要用于Fill-Mask任务，不是专门为NLI任务微调的
        # 如果模型不支持序列分类，将回退到专门的NLI模型
        try:
            # 尝试加载 hfl/rbtl3 模型
            self.nli_tokenizer = AutoTokenizer.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
            self.nli_model = AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
            self.nli_model.eval()
            # 标签顺序未知，需要根据实际测试确定
            # 默认假设：0=蕴含, 1=中性, 2=矛盾（需要验证）
            self.nli_label_order = 'entailment_neutral_contradiction'
            print("已加载模型: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        except Exception as e1:
            try:
                # 回退到原来的中文NLI模型
                self.nli_tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-deberta-v3-base-nli')
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                    'hfl/chinese-deberta-v3-base-nli'
                )
                self.nli_model.eval()
                self.nli_label_order = 'entailment_neutral_contradiction'
                print(f"错误详情: {e1}")
            except Exception as e2:
                try:
                    # 最后回退到英文模型
                    self.nli_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
                    self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                        'microsoft/deberta-v3-base'
                    )
                    self.nli_model.eval()
                    self.nli_label_order = 'contradiction_neutral_entailment'
                    print(f"警告: 无法加载中文模型，已回退到英文模型: {e2}")
                except Exception as e3:
                    print(f"警告: 无法加载NLI模型: {e3}")
                    self.nli_model = None
                    self.nli_label_order = None
    
    def normalize_text(self, text: str) -> str:
        """EM专用文本标准化（更宽松的精确匹配）
        
        设计目标：
        - 对大小写、全角/半角差异不敏感
        - 忽略所有中英文标点
        - 忽略所有空白符（空格、制表符、换行等）
        
        示例：
        - \"北京 是 中国 的 首都。\" → \"北京是中国的首都\"
        - \"北京是中国的首都\"      → \"北京是中国的首都\"
        - 两者标准化后相同，EM 判为 1
        """
        if not text:
            return ""
        
        # 1) 统一全角/半角、兼容等价字符（如全角标点、全角数字等）
        text = unicodedata.normalize('NFKC', text)
        
        # 2) 保留：字母数字、下划线、空白符、CJK 中文；删除其它符号（标点等）
        #    即使某些环境下 \\w 不包含中文，\\u4e00-\\u9fff 也能保证中文不被删
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text, flags=re.UNICODE)
        
        # 3) 全部转小写（对英文不区分大小写，对中文无影响）
        text = text.lower()
        
        # 4) 删除所有空白符，使 EM 对空格/换行位置完全不敏感
        text = re.sub(r'\s+', '', text)
        
        return text
    
    def exact_match(self, pred: str, ref: str) -> float:
        """精确匹配率"""
        pred_norm = self.normalize_text(pred)
        ref_norm = self.normalize_text(ref)
        return 1.0 if pred_norm == ref_norm else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """分词函数，使用 jieba 进行中文分词"""
        if not text:
            return []
        try:
            return list(jieba.cut(text.lower()))
        except:
            # 如果 jieba 分词失败，使用简单分词
            return re.findall(r'\w+', text.lower())
    
    def bleu_score(self, pred: str, ref: str, n: int = 4) -> float:
        """
        计算 BLEU-n 分数
        
        Args:
            pred: 预测文本
            ref: 参考文本
            n: n-gram 的 n (1, 2, 3, 或 4)
        
        Returns:
            BLEU-n 分数
        """
        if not pred or not ref:
            return 0.0
        
        try:
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                return 0.0
            
            # 构建权重：标准BLEU使用几何平均，只计算到 n-gram，其余为0
            # BLEU-1: (1.0, 0.0, 0.0, 0.0)
            # BLEU-2: (0.5, 0.5, 0.0, 0.0)
            # BLEU-3: (1/3, 1/3, 1/3, 0.0)
            # BLEU-4: (0.25, 0.25, 0.25, 0.25)
            weights = [1.0 / n] * n + [0.0] * (4 - n)
            
            # 计算 BLEU 分数
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=tuple(weights),  # 转换为元组
                smoothing_function=self.bleu_smoothing
            )
            return float(score)
        except Exception as e:
            print(f"警告: BLEU-{n}计算失败: {e}")
            return 0.0
    
    def rouge_scores(self, pred: str, ref: str) -> Dict[str, Dict[str, float]]:
        """
        计算 ROUGE-1, ROUGE-2, ROUGE-L 的 R/P/F 分数
        
        Args:
            pred: 预测文本
            ref: 参考文本
        
        Returns:
            包含 rouge1, rouge2, rougeL 的字典，每个包含 recall, precision, fmeasure
        """
        if not pred or not ref:
            return {
                'rouge1': {'recall': 0.0, 'precision': 0.0, 'fmeasure': 0.0},
                'rouge2': {'recall': 0.0, 'precision': 0.0, 'fmeasure': 0.0},
                'rougeL': {'recall': 0.0, 'precision': 0.0, 'fmeasure': 0.0}
            }
        
        try:
            scores = self.rouge_scorer.score(ref, pred)
            return {
                'rouge1': {
                    'recall': scores['rouge1'].recall,
                    'precision': scores['rouge1'].precision,
                    'fmeasure': scores['rouge1'].fmeasure
                },
                'rouge2': {
                    'recall': scores['rouge2'].recall,
                    'precision': scores['rouge2'].precision,
                    'fmeasure': scores['rouge2'].fmeasure
                },
                'rougeL': {
                    'recall': scores['rougeL'].recall,
                    'precision': scores['rougeL'].precision,
                    'fmeasure': scores['rougeL'].fmeasure
                }
            }
        except Exception as e:
            print(f"警告: ROUGE计算失败: {e}")
            return {
                'rouge1': {'recall': 0.0, 'precision': 0.0, 'fmeasure': 0.0},
                'rouge2': {'recall': 0.0, 'precision': 0.0, 'fmeasure': 0.0},
                'rougeL': {'recall': 0.0, 'precision': 0.0, 'fmeasure': 0.0}
            }
    
    def bert_score(self, preds: List[str], refs: List[str]) -> float:
        """BERTScore评估"""
        try:
            # 自动检测并使用CUDA（如果可用）
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            P, R, F1 = bert_score(
                preds, refs, 
                lang='zh', 
                model_type='bert-base-chinese',
                device=device,  # 指定设备以加速计算
                verbose=False
            )
            return float(F1.mean().item())
        except Exception as e:
            print(f"警告: BERTScore计算失败: {e}")
            return 0.0
    
    def nli_contradiction_detection(self, question: str, answer: str) -> float:
        """NLI矛盾检测"""
        if self.nli_model is None:
            return 0.0
        
        try:
            # 构建输入
            inputs = self.nli_tokenizer(
                question, answer,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # 根据模型类型确定标签顺序
                if self.nli_label_order == 'entailment_neutral_contradiction':
                    # 中文模型：0=蕴含, 1=中性, 2=矛盾
                    contradiction_prob = probs[0][2].item()
                elif self.nli_label_order == 'contradiction_neutral_entailment':
                    # 英文模型（假设）：0=矛盾, 1=中性, 2=蕴含
                    contradiction_prob = probs[0][0].item()
                else:
                    # 未知标签顺序，尝试使用索引2（通常矛盾在最后）
                    contradiction_prob = probs[0][2].item()
            
            return contradiction_prob
        except Exception as e:
            print(f"警告: NLI检测失败: {e}")
            return 0.0
    
    def evaluate(self, data: List[Dict], dataset_name: str = "qa_dataset") -> List[Tuple[str, float, int]]:
        """
        评估QA数据集
        
        Args:
            data: 数据集列表，每个元素包含 'question' 和 'answer' 字段
            dataset_name: 数据集名称
        
        Returns:
            评估结果列表，每个元素为 (指标名称, 分数, 样本数)，所有指标名称前都有 "mean_" 前缀
        """
        print(f"开始评估QA数据集: {dataset_name}")
        print(f"样本数量: {len(data)}")
        
        results = []
        predictions = []
        references = []
        
        # 生成预测
        print("\n生成模型预测...")
        empty_output_count = 0
        for item in tqdm(data, desc="预测进度"):
            question = item['question']
            reference = item['answer']
            
            # 使用模型生成答案
            prediction = self.model_loader.generate(question)
            
            # 检查空输出
            if not prediction or len(prediction.strip()) == 0:
                empty_output_count += 1
                # 空输出时使用空字符串，后续计算会自动处理
                prediction = ""
            
            predictions.append(prediction)
            references.append(reference)
        
        if empty_output_count > 0:
            print(f"警告: 发现 {empty_output_count}/{len(data)} 个空输出样本")
        
        # 计算核心自动化指标
        print("\n计算核心自动化指标...")
        
        # Exact Match
        em_scores = []
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="EM计算"):
            # 空输出时直接赋值 EM=0
            if not pred or len(pred.strip()) == 0:
                em_scores.append(0.0)
            else:
                em_scores.append(self.exact_match(pred, ref))
        em_score = np.mean(em_scores)
        results.append(('mean_EM', em_score, len(data)))
        
        # BLEU-1/2/3/4
        print("计算BLEU分数...")
        for n in [1, 2, 3, 4]:
            bleu_scores = []
            for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc=f"BLEU-{n}计算"):
                # 空输出时直接赋值 BLEU=0
                if not pred or len(pred.strip()) == 0:
                    bleu_scores.append(0.0)
                else:
                    bleu_scores.append(self.bleu_score(pred, ref, n=n))
            bleu_mean = np.mean(bleu_scores)
            results.append((f'mean_bleu-{n}', bleu_mean, len(data)))
        
        # ROUGE-1/2/L 的 R/P/F
        print("计算ROUGE分数...")
        rouge1_recalls = []
        rouge1_precisions = []
        rouge1_fmeasures = []
        rouge2_recalls = []
        rouge2_precisions = []
        rouge2_fmeasures = []
        rougeL_recalls = []
        rougeL_precisions = []
        rougeL_fmeasures = []
        
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="ROUGE计算"):
            # 空输出时直接赋值 ROUGE=0
            if not pred or len(pred.strip()) == 0:
                rouge1_recalls.append(0.0)
                rouge1_precisions.append(0.0)
                rouge1_fmeasures.append(0.0)
                rouge2_recalls.append(0.0)
                rouge2_precisions.append(0.0)
                rouge2_fmeasures.append(0.0)
                rougeL_recalls.append(0.0)
                rougeL_precisions.append(0.0)
                rougeL_fmeasures.append(0.0)
            else:
                rouge_scores = self.rouge_scores(pred, ref)
                rouge1_recalls.append(rouge_scores['rouge1']['recall'])
                rouge1_precisions.append(rouge_scores['rouge1']['precision'])
                rouge1_fmeasures.append(rouge_scores['rouge1']['fmeasure'])
                rouge2_recalls.append(rouge_scores['rouge2']['recall'])
                rouge2_precisions.append(rouge_scores['rouge2']['precision'])
                rouge2_fmeasures.append(rouge_scores['rouge2']['fmeasure'])
                rougeL_recalls.append(rouge_scores['rougeL']['recall'])
                rougeL_precisions.append(rouge_scores['rougeL']['precision'])
                rougeL_fmeasures.append(rouge_scores['rougeL']['fmeasure'])
        
        # ROUGE-1 R/P/F
        results.append(('mean_Rouge-1-R', np.mean(rouge1_recalls), len(data)))
        results.append(('mean_Rouge-1-P', np.mean(rouge1_precisions), len(data)))
        results.append(('mean_Rouge-1-F', np.mean(rouge1_fmeasures), len(data)))
        
        # ROUGE-2 R/P/F
        results.append(('mean_Rouge-2-R', np.mean(rouge2_recalls), len(data)))
        results.append(('mean_Rouge-2-P', np.mean(rouge2_precisions), len(data)))
        results.append(('mean_Rouge-2-F', np.mean(rouge2_fmeasures), len(data)))
        
        # ROUGE-L R/P/F
        results.append(('mean_Rouge-L-R', np.mean(rougeL_recalls), len(data)))
        results.append(('mean_Rouge-L-P', np.mean(rougeL_precisions), len(data)))
        results.append(('mean_Rouge-L-F', np.mean(rougeL_fmeasures), len(data)))
        
        # BERTScore
        print("计算BERTScore...")
        bert_score_value = self.bert_score(predictions, references)
        results.append(('mean_BERTScore', bert_score_value, len(data)))
        
        # 综合质量分数 (使用 ROUGE-L F1)
        rouge_l_f1 = np.mean(rougeL_fmeasures)
        comprehensive_score = bert_score_value * 0.6 + rouge_l_f1 * 0.4
        results.append(('mean_Comprehensive_Score', comprehensive_score, len(data)))
        
        # 事实一致性 - NLI矛盾检测
        print("计算事实一致性...")
        contradiction_probs = []
        for item, pred in tqdm(zip(data, predictions), total=len(data), desc="矛盾检测"):
            contradiction_prob = self.nli_contradiction_detection(item['question'], pred)
            contradiction_probs.append(contradiction_prob)
        
        avg_contradiction = np.mean(contradiction_probs)
        consistency_score = 1.0 - avg_contradiction  # 一致性分数 = 1 - 矛盾概率
        results.append(('mean_Factual_Consistency', consistency_score, len(data)))
        
        return results
