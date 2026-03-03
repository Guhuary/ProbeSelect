from cgitb import text
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_logsumexp

class RankingLossMargin_InfoNCE_CORR(nn.Module):
	"""
	Loss = sum_i log( exp(s_i) / sum_{j: y_j > y_i + m} exp(s_j) )
	where s_k = w * x_k / tau
	"""
	def __init__(self, max_temperature: float = 1.0, eps: float = 1e-5, max_margin: float = 0.0, 
			  sim_margin: float = 0.6,
			  lambda_list: float = 1.0, lambda_infonce: float=1.0, lambda_corr: float=1.0):
		super().__init__()
		self.temperature = max_temperature
		self.eps = eps
		self.margin = max_margin
		self.min_val = 0.1
		self.lambda_list = lambda_list
		self.lambda_infonce = lambda_infonce
		self.sim_margin = sim_margin
		self.lambda_corr = lambda_corr

	def corr_loss(self, x, y, eps=1e-8):
		if x.dim() > 1:
			x = x.squeeze()
		if y.dim() > 1:
			y = y.squeeze()
		x = (x - x.mean()) / (x.std(unbiased=False)+eps)
		y = (y - y.mean()) / (y.std(unbiased=False)+eps)
		return 1.0 - (x*y).mean()
	
	def rank_with_text_sim(self, predictions: torch.Tensor, targets: torch.Tensor, 
						text_emb = None, epoch: int=1, max_epoch: int=1) -> torch.Tensor:
		# 1) flatten
		if predictions.dim() > 1:
			predictions = predictions.squeeze()
		if targets.dim() > 1:
			targets = targets.squeeze()
		
		current_temperature = self.temperature * max(1 - epoch / max_epoch, self.min_val)
		predictions = predictions / current_temperature
		current_margin = self.margin * (1 - epoch / max_epoch)

		cond_matrix = targets.view(-1, 1) - targets.view(1, -1) >= current_margin

		N = cond_matrix.shape[0]
		diag_mask = torch.eye(N, dtype=torch.bool, device=cond_matrix.device)

		if text_emb is not None:
			text_emb = text_emb / text_emb.norm(-1, keepdim=True)
			sim_matrix = text_emb @ text_emb.T
			sim_matrix = sim_matrix > self.sim_margin
			cond_matrix = cond_matrix & sim_matrix
		cond_matrix = cond_matrix | diag_mask

		row, col = cond_matrix.nonzero(as_tuple=True)

		log_softmax = scatter_logsumexp(predictions[col], row, dim=0, dim_size=N)
		out = predictions - log_softmax
		return - out.mean()

	def contrastive_loss(self, similarity_matrix, epoch, max_epoch):
		N = similarity_matrix.size(0)
		current_temperature = self.temperature * (epoch / max_epoch * (1 - self.min_val) + self.min_val)
		# 对角线是正样本对：S[i][i]
		logits = similarity_matrix / current_temperature  # (N, N)

		# 构造标签：第 i 行的正样本是第 i 列
		labels = torch.arange(N, device=similarity_matrix.device)

		# 使用交叉熵损失：每行是一个分类任务，有 N 类，只有对角线是正类
		loss = F.cross_entropy(logits, labels, reduction='mean')
		return loss
	
	def forward(self, predictions: torch.Tensor, matrix: torch.Tensor, 
			 text_emb: torch.Tensor, targets: torch.Tensor, 
			 epoch: int, max_epoch: int):
		loss = 0
		# list_loss = self.rank_with_margin(predictions, targets, epoch, max_epoch)
		list_loss = self.rank_with_text_sim(predictions, targets, text_emb=None, epoch=epoch, max_epoch=max_epoch)
		if matrix is not None:
			info_nce = self.contrastive_loss(matrix, epoch, max_epoch)
		else:
			info_nce = 0
		corr = self.corr_loss(predictions, targets)
		out = {
			'loss': self.lambda_list * list_loss + self.lambda_infonce * info_nce + self.lambda_corr * corr, 
		 	'list_loss': list_loss, 
			'info_nce': info_nce,
			'corr': corr
			}
		return out
	
	def rank_with_margin(self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int, max_epoch: int) -> torch.Tensor:
		# 1) flatten
		if predictions.dim() > 1:
			predictions = predictions.squeeze()
		if targets.dim() > 1:
			targets = targets.squeeze()
		# current_temperature = self.temperature * (1 - epoch / max_epoch + self.min_val)
		# current_margin = self.margin * (1 - epoch / max_epoch + self.min_val)
		# In ranklossmargin_corr.py, change:
		current_temperature = self.temperature * (epoch / max_epoch * (1 - self.min_val) + self.min_val)
		current_margin = self.margin * (1 - epoch / max_epoch + self.min_val)
		# current_margin = self.margin * ((max_epoch - epoch) / max_epoch)  # Decrease margin over time

		# 2) 先按目标值从大到小排序（更高的 target 在前面）
		idx = torch.argsort(targets, descending=True)
		t = targets[idx]                                   # [N] sorted targets
		s = (predictions[idx]) / current_temperature  # [N] sorted scores

		a_inc = (-t).contiguous()                          # ascending
		thresh = -(t + current_margin)                        # 需要 a_inc < -thresh_raw
		# count[i] = # { j | -t[j] < -(t[i]+m) } = insertion index
		counts = torch.searchsorted(a_inc, thresh, right=True)  # [N], int64
		valid = counts > 0
		
		# 3) 前缀 logsumexp（稳定版）
		shift = s.max().detach()
		exp_s = torch.exp(s - shift)                       # [N]
		prefix_sum = torch.cumsum(exp_s.flip(dims=[0]), dim=0).flip(dims=[0]) - exp_s.detach()            # prefix sums of exp(s)
		prefix_sum = prefix_sum[(counts - 1).clamp_min(0)] + exp_s
		log_prefix_sum = torch.log(prefix_sum + self.eps) + shift  # [N]

		log_probs = s[valid] - log_prefix_sum[valid]            # 逐样本 log(exp(s_i))/sum_{j in set} exp(s_j)
		loss = -log_probs.mean()
		return loss
