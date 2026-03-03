import torch
import numpy as np
import torch.nn as nn
from functools import partial
from typing import Dict, List, Union, Optional, Callable

class BaseEvaluator:
    def __init__(self, device):
        self.results = []
        self.device = device

    def prepare(self):
        """用于初始化模型或预处理"""
        pass

    def evaluate(self, prompts, images):
        """
        :param prompts: List[str]
        :param images: List[PIL.Image]
        :return: Tensor or Dict
        """
        raise NotImplementedError()

    def aggregate(self):
        """
        对多个 batch 的结果进行汇总（默认是 cat）
        """
        return torch.cat(self.results, dim=0)

    def clear(self):
        """清理缓存"""
        self.results.clear()

# 简化的evaluator类
class SimpleFeatureEvaluator_Raw:
    """简化的特征评估器"""
    def __init__(self):
        self.results_all = []
    
    def evaluate(self, feature_scores: List[torch.Tensor]):
        if feature_scores:
            results = torch.stack(feature_scores, dim=1)  # [batch, steps]
            self.results_all.append(results)
    
    def aggregate(self):
        if self.results_all:
            return torch.cat(self.results_all, dim=0)
        else:
            return torch.tensor([])
