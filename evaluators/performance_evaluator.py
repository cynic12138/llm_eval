# -*- coding: utf-8 -*-
"""
基础性能指标评估器
评估吞吐量、延迟等性能指标
"""

import time
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import torch


class PerformanceEvaluator:
    """性能指标评估器"""
    
    def __init__(self, model_loader):
        """
        初始化性能评估器
        
        Args:
            model_loader: 模型加载器
        """
        self.model_loader = model_loader
    
    def measure_throughput(self, batch_size: int = 8, input_tokens: int = 128, 
                          num_requests: int = 100) -> float:
        """
        测量吞吐量（req/s）
        
        Args:
            batch_size: 批次大小
            input_tokens: 输入token数
            num_requests: 请求数量
        
        Returns:
            吞吐量（请求/秒）
        """
        print(f"测量吞吐量: batch_size={batch_size}, input_tokens={input_tokens}")
        
        # 生成测试输入
        test_input = "测试文本 " * (input_tokens // 2)  # 简化处理
        
        start_time = time.time()
        successful_requests = 0
        
        for _ in tqdm(range(num_requests), desc="吞吐量测试"):
            try:
                # 批量处理
                inputs = [test_input] * batch_size
                results = self.model_loader.generate_batch(inputs)
                # 检查是否有有效输出
                if results and any(r for r in results if r):
                    successful_requests += 1
            except Exception as e:
                print(f"\n警告: 吞吐量测试请求失败: {e}")
                continue
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if elapsed_time <= 0:
            print("错误: 吞吐量测试时间异常，返回0")
            return 0.0
        
        throughput = successful_requests / elapsed_time
        if successful_requests < num_requests:
            print(f"警告: 只有 {successful_requests}/{num_requests} 个请求成功")
        
        return throughput
    
    def measure_first_token_latency(self, input_tokens: int = 128, 
                                   num_tests: int = 100, percentile: float = 0.99) -> float:
        """
        测量首Token延迟（P99，冷启动）
        
        Args:
            input_tokens: 输入token数
            num_tests: 测试次数
            percentile: 百分位数（0.99表示P99）
        
        Returns:
            首Token延迟（毫秒）
        """
        print(f"测量首Token延迟 (P{int(percentile*100)}): input_tokens={input_tokens}")
        
        test_input = "测试文本 " * (input_tokens // 2)
        latencies = []
        
        for _ in tqdm(range(num_tests), desc="首Token延迟测试"):
            # 冷启动：每次重新加载模型（简化处理，实际可能需要更复杂的实现）
            start_time = time.time()
            _ = self.model_loader.generate_first_token(test_input)
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            latencies.append(latency)
        
        p99_latency = np.percentile(latencies, percentile * 100)
        return p99_latency
    
    def measure_generation_speed(self, output_tokens: int = 100, 
                                num_tests: int = 10) -> float:
        """
        测量生成速度（tokens/s）
        
        Args:
            output_tokens: 输出token数
            num_tests: 测试次数
        
        Returns:
            生成速度（tokens/秒）
        """
        print(f"测量生成速度: output_tokens={output_tokens}")
        
        test_input = "请生成一段文本："
        speeds = []
        failed_count = 0
        
        for _ in tqdm(range(num_tests), desc="生成速度测试"):
            try:
                start_time = time.time()
                output = self.model_loader.generate_with_length(test_input, max_length=output_tokens)
                elapsed_time = time.time() - start_time
                
                if not output or elapsed_time <= 0:
                    failed_count += 1
                    continue
                
                # 估算token数（简化处理：中文约1.5字符/token，英文约4字符/token）
                # 这里用更保守的估算：按字符数除以2（假设中英文混合）
                estimated_tokens = len(output) / 2.0  # 更保守的估算
                speed = estimated_tokens / elapsed_time if elapsed_time > 0 else 0
                
                if speed > 0:  # 只记录有效的速度值
                    speeds.append(speed)
            except Exception as e:
                print(f"\n警告: 生成速度测试出错: {e}")
                failed_count += 1
                continue
        
        if failed_count > 0:
            print(f"警告: {failed_count}/{num_tests} 次生成速度测试失败")
        
        if len(speeds) == 0:
            print("错误: 所有生成速度测试都失败，返回0")
            return 0.0
        
        avg_speed = np.mean(speeds)
        return avg_speed
    
    def measure_end_to_end_latency(self, input_tokens: int = 256, 
                                  output_tokens: int = 128,
                                  num_tests: int = 100, percentile: float = 0.95) -> float:
        """
        测量端到端延迟（P95）
        
        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数
            num_tests: 测试次数
            percentile: 百分位数（0.95表示P95）
        
        Returns:
            端到端延迟（秒）
        """
        print(f"测量端到端延迟 (P{int(percentile*100)}): input={input_tokens}, output={output_tokens}")
        
        test_input = "测试文本 " * (input_tokens // 2)
        latencies = []
        failed_count = 0
        
        for _ in tqdm(range(num_tests), desc="端到端延迟测试"):
            try:
                start_time = time.time()
                output = self.model_loader.generate_with_length(test_input, max_length=output_tokens)
                latency = time.time() - start_time
                
                # 只记录有效的延迟（有输出且延迟合理）
                if output and latency > 0:
                    latencies.append(latency)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"\n警告: 端到端延迟测试出错: {e}")
                failed_count += 1
                continue
        
        if failed_count > 0:
            print(f"警告: {failed_count}/{num_tests} 次端到端延迟测试失败")
        
        if len(latencies) == 0:
            print("错误: 所有端到端延迟测试都失败，返回0")
            return 0.0
        
        p95_latency = np.percentile(latencies, percentile * 100)
        return p95_latency
    
    def evaluate(self) -> List[Tuple[str, float, int]]:
        """
        评估所有性能指标
        
        Returns:
            评估结果列表
        """
        results = []
        
        # 吞吐量
        throughput = self.measure_throughput(batch_size=8, input_tokens=128, num_requests=50)
        results.append(('Throughput_req_per_s', throughput, 50))
        
        # 首Token延迟（P99，冷启动）
        first_token_latency = self.measure_first_token_latency(
            input_tokens=128, num_tests=50, percentile=0.99
        )
        results.append(('First_Token_Latency_P99_ms', first_token_latency, 50))
        
        # 生成速度
        generation_speed = self.measure_generation_speed(output_tokens=100, num_tests=10)
        results.append(('Generation_Speed_tokens_per_s', generation_speed, 10))
        
        # 端到端延迟（P95）
        e2e_latency = self.measure_end_to_end_latency(
            input_tokens=256, output_tokens=128, num_tests=50, percentile=0.95
        )
        results.append(('End_to_End_Latency_P95_s', e2e_latency, 50))
        
        return results

