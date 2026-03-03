import torch
import numpy as np
import torch.nn as nn
from functools import partial
from typing import Dict, List, Union
import matplotlib.pyplot as plt

def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    香农熵：H(p) = - Σ p log p.
    返回 shape = (batch, head, pixel) 的熵，方便后续聚合.
    """
    p_safe = p.clamp_min(eps)      # 避免 log(0)
    return -(p_safe * p_safe.log()).nansum(dim=-1)

class AttentionMapAnalyzer:
    def __init__(self, min_early_steps=5, max_early_steps=50):
        self.min_early_steps = min_early_steps
        self.max_early_steps = max_early_steps
        self.results_all = {}
        
    def prepare(self, attention_maps: Dict[str, List[torch.Tensor]]):
        self.attention_maps = attention_maps
        self.num_layers = len(attention_maps)
        self.num_steps = len(next(iter(attention_maps.values())))  # 假设所有 map 都有相同的 step 数
        self.batch_size = next(iter(attention_maps.values()))[0].shape[0]  # 假设所有 map 的 batch size 相同

    def compute_score_for_single_tensor(self, inputs: torch.Tensor):
        '''
        return shape: [bs]
        '''
        pass

    def compute_score_for_single_step_t(self, t: int=0):
        results = torch.zeros(self.batch_size)
        for name, maps in self.attention_maps.items():
            results += self.compute_score_for_single_tensor(maps[t])
        return results / self.num_layers
    
    @torch.no_grad()
    def evaluate_prem(self):
        results = []
        for t in range(self.num_steps):
            results.append(self.compute_score_for_single_step_t(t))
        self.results = torch.stack(results, dim=1) # (batch, num_steps)

    def compute_quality_score(self, start: int=0, end: int=5):
        times = end - start
        if self.results.shape[1] < times:
            raise ValueError("Not enough steps for quality score")

        early_results = self.results[:, start:end]  # N x steps
        min_result = early_results.min(dim=1, keepdim=True).values  # 找到每个样本的最小 entropy
        early_results = early_results - min_result  # 减去最小值，使最小值为 0
        base = early_results[:, 0:1] * times  # 初始熵乘以步数作为基准面积
        actual_area = early_results.sum(dim=1, keepdim=True)
        drop_area = base - actual_area

        score = drop_area / (base + 1e-6)  # 归一化处理
        return score.squeeze()

    def evaluate(self, *args, **kwargs):
        self.evaluate_prem()
        for t in range(self.min_early_steps, self.max_early_steps):
            score_t = self.compute_quality_score(start=0, end=t)
            self.results_all.setdefault(t, []).append(score_t.cpu())

    def aggregate(self):
        for k, v in self.results_all.items():
            self.results_all[k] = torch.cat(v, dim=0)
        return self.results_all
    
    def plot_results_trend(self):
        """
        绘制 entropy 随 step 的变化曲线
        """
        import matplotlib.pyplot as plt
        plt.plot(self.results[0], label="Entropy per Step")
        plt.xlabel("Sampling Step")
        plt.ylabel("Average Entropy")
        plt.title("Attention Map Entropy over Sampling Steps")
        plt.legend()
        plt.show()

class AttentionMap_Ent_Analyzer_AUC(AttentionMapAnalyzer):
    def __init__(self):
        super(AttentionMap_Ent_Analyzer_AUC, self).__init__()

    def compute_score_for_single_tensor(self, attn_tensor):
        """
        对单个 step 的 attention map 计算平均 entropy
        """
        batch, heads, num_pix, num_token = attn_tensor.shape

        # reshape to [batch * heads * num_pix, num_token]
        flat_attn = attn_tensor.view(-1, num_token)

        # 计算 entropy (注意输入是概率分布)
        ent = torch.distributions.Categorical(probs=flat_attn).entropy().view(batch, heads * num_pix)
        avg_ent = ent.mean(1) # batch
        return avg_ent

    def plot_single_layer_entropy_trend(self, prompt=None):
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
        axes = axes.ravel()         # 变成一维列表，方便循环

        for i, (ax, k) in enumerate(zip(axes, self.attention_maps.keys())):
            v = [t.float() for t in self.attention_maps[k]]
            entropys = []
            for i in range(len(v) - 1):
                # 计算熵
                ent = _entropy(v[i])
                entropys.append(ent.mean().item())
            entropys = np.array(entropys)
            ax.plot(entropys, linewidth=2, label="entropy")
            # ax.legend()
            ax.set_title(k.split('attentions')[0] + 'attentions', fontsize=12)

        fig.tight_layout()
        if prompt:
            fig.savefig(f'step_images/{prompt[:10]}/SD2_attnmap_entropy.png')

class AttentionMap_ERank_Analyzer_AUC(AttentionMapAnalyzer):
    def __init__(self):
        super(AttentionMap_ERank_Analyzer_AUC, self).__init__()

    def compute_score_for_single_tensor(self, attn_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the Effective Rank for each matrix in a 4D tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor of shape (bs, n_head, num_pix, num_token)
            
        Returns:
            torch.Tensor: Effective ranks of shape (bs, n_head)
        """
        bs, n_head, num_pix, num_token = attn_tensor.shape
        
        # 展平前两个维度，变成 [bs * n_head, num_pix, num_token]
        flat_tensor = attn_tensor.view(bs * n_head, num_pix, num_token)
        
        # SVD 分解：U, S, V = torch.svd(flat_tensor)
        # 只需要奇异值 S
        s = torch.linalg.svdvals(flat_tensor)  # 更高效，只求奇异值
        
        # 归一化奇异值
        s_normalized = s / (s.sum(dim=1, keepdim=True) + 1e-12)  # 加小量防止除零
        
        # 香农熵
        entropy = -torch.sum(s_normalized * torch.log2(s_normalized + 1e-12), dim=1)
        
        # Effective Rank
        erank = torch.exp(entropy)
        
        # 恢复成 [bs, n_head]
        return erank.view(bs, n_head).mean(1)


class AttentionScoreAnalyzer_AUC:
    def __init__(self, min_early_steps=5, max_early_steps=50):
        self.min_early_steps = min_early_steps
        self.max_early_steps = max_early_steps
        self.results_all = {}
        
    def prepare(self, attention_scores: Dict[str, List[torch.Tensor]]):
        self.attention_scores = attention_scores
        self.num_layers = len(attention_scores)
        self.num_steps = len(next(iter(attention_scores.values())))  # 假设所有 map 都有相同的 step 数
        self.device = next(iter(attention_scores.values()))[0].device
        self.batch_size = next(iter(attention_scores.values()))[0].shape[0]  # 假设所有 map 的 batch size 相同
    
    @torch.no_grad()
    def evaluate_prem(self):
        self.results = torch.zeros(self.batch_size, self.num_steps, device=self.device)
        for layer, score in self.attention_scores.items():
            self.results += torch.stack(score, dim=1)
        self.results /= self.num_layers

    def compute_quality_score(self, start: int=0, end: int=5):
        times = end - start
        if self.results.shape[1] < times:
            raise ValueError("Not enough steps for quality score")

        early_results = self.results[:, start:end]  # N x steps
        min_result = early_results.min(dim=1, keepdim=True).values  # 找到每个样本的最小 entropy
        early_results = early_results - min_result  # 减去最小值，使最小值为 0
        base = early_results[:, 0:1] * times  # 初始熵乘以步数作为基准面积
        actual_area = early_results.sum(dim=1, keepdim=True)
        drop_area = base - actual_area

        score = drop_area / (base + 1e-6)  # 归一化处理
        return score.squeeze()

    def evaluate(self, *args, **kwargs):
        self.evaluate_prem()
        for t in range(self.min_early_steps, self.max_early_steps):
            score_t = self.compute_quality_score(start=0, end=t)
            self.results_all.setdefault(t, []).append(score_t.cpu())

    def aggregate(self):
        for k, v in self.results_all.items():
            self.results_all[k] = torch.cat(v, dim=0)
        return self.results_all
    
    def plot_results_trend(self):
        """
        绘制 entropy 随 step 的变化曲线
        """
        import matplotlib.pyplot as plt
        plt.plot(self.results[0], label="Entropy per Step")
        plt.xlabel("Sampling Step")
        plt.ylabel("Average Entropy")
        plt.title("Attention Map Entropy over Sampling Steps")
        plt.legend()
        plt.show()

class AttentionScoreAnalyzer_Diff:
    def __init__(self, min_early_steps=5, max_early_steps=50):
        self.min_early_steps = min_early_steps
        self.max_early_steps = max_early_steps
        self.results_all = {}
        
    def prepare(self, attention_scores: Dict[str, List[torch.Tensor]]):
        self.attention_scores = attention_scores
        self.num_layers = len(attention_scores)
        self.num_steps = len(next(iter(attention_scores.values())))  # 假设所有 map 都有相同的 step 数
        self.device = next(iter(attention_scores.values()))[0].device
        self.batch_size = next(iter(attention_scores.values()))[0].shape[0]  # 假设所有 map 的 batch size 相同
    
    @torch.no_grad()
    def evaluate_prem(self):
        self.results = torch.zeros(self.batch_size, self.num_steps, device=self.device)
        for layer, score in self.attention_scores.items():
            self.results += torch.stack(score, dim=1)
        self.results /= self.num_layers

    def compute_quality_score(self, start: int=0, end: int=5):
        times = end - start
        if self.results.shape[1] < times:
            raise ValueError("Not enough steps for quality score")

        score = (self.results[:, end] - self.results[:, start]) / self.results[:, start]
        return score.flatten()

    def evaluate(self, *args, **kwargs):
        self.evaluate_prem()
        for t in range(self.min_early_steps, self.max_early_steps):
            score_t = self.compute_quality_score(start=0, end=t)
            self.results_all.setdefault(t, []).append(score_t.cpu())

    def aggregate(self):
        for k, v in self.results_all.items():
            self.results_all[k] = torch.cat(v, dim=0)
        return self.results_all
    
    def plot_results_trend(self):
        """
        绘制 entropy 随 step 的变化曲线
        """
        import matplotlib.pyplot as plt
        plt.plot(self.results[0], label="Entropy per Step")
        plt.xlabel("Sampling Step")
        plt.ylabel("Average Entropy")
        plt.title("Attention Map Entropy over Sampling Steps")
        plt.legend()
        plt.show()

class OutputAnalyzer_AUC:
    def __init__(self, min_early_steps=5, max_early_steps=50):
        self.min_early_steps = min_early_steps
        self.max_early_steps = max_early_steps
        self.results_all = {}
    
    def compute_quality_score(self, start: int=0, end: int=5):
        times = end - start
        if self.results.shape[1] < times:
            raise ValueError("Not enough steps for quality score")

        early_results = self.results[:, start:end]  # N x steps
        # min_result = early_results.min(dim=1, keepdim=True).values  # 找到每个样本的最小 entropy
        # early_results = early_results - min_result  # 减去最小值，使最小值为 0
        base = early_results[:, 0:1] * times  # 初始熵乘以步数作为基准面积
        actual_area = early_results.sum(dim=1, keepdim=True)
        drop_area = base - actual_area

        score = drop_area / (base + 1e-6)  # 归一化处理
        return score.squeeze()

    @torch.no_grad()
    def evaluate(self, output_scores: List[torch.Tensor]):
        self.output_scores = output_scores
        self.results = torch.stack(output_scores, dim=1)
        for t in range(self.min_early_steps, self.max_early_steps):
            score_t = self.compute_quality_score(start=0, end=t)
            self.results_all.setdefault(t, []).append(score_t.cpu())

    def aggregate(self):
        for k, v in self.results_all.items():
            self.results_all[k] = torch.cat(v, dim=0)
        return self.results_all
    
    def plot_results_trend(self):
        """
        绘制 entropy 随 step 的变化曲线
        """
        import matplotlib.pyplot as plt
        plt.plot(self.results[0], label="Entropy per Step")
        plt.xlabel("Sampling Step")
        plt.ylabel("Average Entropy")
        plt.title("Attention Map Entropy over Sampling Steps")
        plt.legend()
        plt.show()

class OutputAnalyzer_Diff(OutputAnalyzer_AUC):
    def __init__(self, min_early_steps=5, max_early_steps=50):
        super().__init__(min_early_steps, max_early_steps)
        self.results_all = {}
    
    def compute_quality_score(self, start: int=0, end: int=5):
        times = end - start
        if self.results.shape[1] < times:
            raise ValueError("Not enough steps for quality score")

        score = (self.results[:, end] - self.results[:, start]) / self.results[:, start]
        return score.flatten()

class AttentionScoreAnalyzer_Raw(AttentionScoreAnalyzer_AUC):
    def __init__(self, min_early_steps=5, max_early_steps=50):
        self.min_early_steps = min_early_steps
        self.max_early_steps = max_early_steps
        self.results_all = []

    @torch.no_grad()
    def prepare(self, attention_scores: Dict[str, List[torch.Tensor]]):
        self.attention_scores = attention_scores
        self.num_layers = len(attention_scores)
        assert self.num_layers == 1
        for layer, score in self.attention_scores.items():
            self.results = torch.stack(score, dim=1)

    def evaluate(self, *args, **kwargs):
        self.results_all.append(self.results)

    def aggregate(self):
        return torch.cat(self.results_all, dim=0)
    
class OutputAnalyzer_Raw(OutputAnalyzer_AUC):
    def __init__(self, min_early_steps=5, max_early_steps=50):
        super().__init__(min_early_steps, max_early_steps)
        self.results_all = {}

    @torch.no_grad()
    def evaluate(self, output_scores: List[torch.Tensor]):
        self.output_scores = output_scores
        self.results = torch.stack(output_scores, dim=1)
        self.results_all.append(self.results)
            
    def aggregate(self):
        return torch.cat(self.results_all, dim=0)
    

