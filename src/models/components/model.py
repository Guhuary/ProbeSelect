# from diffusers.models.unets.unet_2d_blocks import *
from functools import partial
# from typing import Optional, Tuple, Union
from math import log
from pydoc import text
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from zmq import has
from .attention_down_blk import AttnDownBlock2D_With_Time, Timesteps, TimestepEmbedding

from ...utils import RankedLogger
logger = RankedLogger(__name__, rank_zero_only=True)

OUTPUT_DIM = {
	"clip16": 512,
	"clip32": 512,
	"pick": 1024,
	"as": 256, # 1,
	'ICT': 1024,
	'HP': 1,
	'ICT_HP': 1024,
	'BLIP_ITC': 256,
	'BLIP_ITM': 256,
	'ImgReward': 768,
	'HPSv20': 1024,
	'HPSv21': 1024,
}

TEXT_DIM = {
	"clip16": 512,
	"clip32": 512,
	"pick": 1024,
	'as': 256,
	'ICT': 1024,
	'ICT_HP': 1024,
	'BLIP_ITC': 256,
	'BLIP_ITM': 256,
	'ImgReward': 768,
	'HPSv20': 1024,
	'HPSv21': 1024,
}

HAS_TEXT_EMBED = {
	"clip16": True,
	"clip32": True,
	"pick": True,
	"as": True, # False
	'ICT': True,
	'HP': False,
	'ICT_HP': True,
	'BLIP_ITC': True,
	'BLIP_ITM': True,
	'ImgReward': True,
	'HPSv20': True,
	'HPSv21': True,
}

def _build_mlp(
	input_dim: int,
	hidden_dims: List[int],
	output_dim: int,
	use_batch_norm: bool,
	activation: str,
	dropout_rate: float, 
	sigmoid: bool=False
) -> nn.Sequential:
	"""Build MLP layers."""
	layers = []
	in_dim = input_dim
	
	# Hidden layers
	for hidden_dim in hidden_dims:
		layers.append(nn.Linear(in_dim, hidden_dim))
		
		if use_batch_norm:
			layers.append(nn.BatchNorm1d(hidden_dim))
		
		if activation == "relu":
			layers.append(nn.ReLU(inplace=True))
		elif activation == "leaky_relu":
			layers.append(nn.LeakyReLU(0.2, inplace=True))
		elif activation == "gelu":
			layers.append(nn.GELU())
		else:
			raise ValueError(f"Unsupported activation function: {activation}")
		
		if dropout_rate > 0:
			layers.append(nn.Dropout(dropout_rate))
		
		in_dim = hidden_dim
	
	# Output layer
	layers.append(nn.Linear(in_dim, output_dim))
	if sigmoid:
		layers.append(nn.Sigmoid())
	return nn.Sequential(*layers)

class LatentEncoderAttn_withtime(nn.Module):
	def __init__(self, input_channel: int, encoder_channels: List[int], num_layers: int, latent_height: int, latent_width: int, 
				 resnet_eps: float = 1e-5, resnet_act_fn: str = "silu",
				 resnet_groups: int = 32, attention_head_dim: int = 5,
					output_scale_factor: float = 1.0, downsample_type: str = "conv", pool_size: int = 3,
					temb_channels: int = 512, resnet_time_scale_shift: str = "default"):
		super(LatentEncoderAttn_withtime, self).__init__()
		self.encoder_layers = nn.ModuleList()
		h, w = latent_height, latent_width
		self.down_factor = 2 if downsample_type == "conv" else 4

		in_channels = input_channel
		for out_channels in encoder_channels:
			self.encoder_layers.append(
				AttnDownBlock2D_With_Time(
					in_channels=in_channels,
					out_channels=out_channels,
					temb_channels=temb_channels, 
					num_layers=num_layers,
					resnet_eps=resnet_eps,
					resnet_time_scale_shift=resnet_time_scale_shift,
					resnet_act_fn=resnet_act_fn,
					resnet_groups=resnet_groups,
					attention_head_dim=attention_head_dim,
					output_scale_factor=output_scale_factor,
					downsample_type=downsample_type
				)
			)
			in_channels = out_channels
			h = (h) // self.down_factor
			w = (w) // self.down_factor

		assert h == w, "Height and width must be equal after downsampling"
		self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
		self.output_dim = encoder_channels[-1] * h * w // (pool_size ** 2)
		logger.info(f"Encoder output dimension: {self.output_dim}")

	def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through encoder.
		
		Args:
			x: Input tensor [B, C, H, W]
			
		Returns:
			Flattened features [B, output_dim]
		"""
		for i, layer in enumerate(self.encoder_layers):
			# logger.info(f"Encoder layer {i} input min {x.min()}, max {x.max()}, mean {x.mean()}, std {x.std()}")
			x = layer(x, time_emb)
		
		# Flatten
		x = self.pool(x).contiguous().view(x.shape[0], -1)
		# x = x.mean(dim=(2, 3)) # avgpool on H and W
		return x
	
class ProbHead(nn.Module):
	"""Complete model for predicting quality metrics from latent features."""
	
	def __init__(self, latent_channels: int, latent_height: int, num_layers: int,
				 latent_width: int, hidden_dims: List[int], hidden_dims_mlp: List[int], use_batch_norm: bool, 
				 activation: str, dropout_rate: float = 0.0,
				 resnet_eps: float = 1e-5, resnet_act_fn: str = "silu",
				 resnet_groups: int = 32, attention_head_dim: int = 5,
				 output_scale_factor: float = 1.0, downsample_type: str = "conv", 
				 temb_channels: int = 512, resnet_time_scale_shift: str = "default", 
				 num_channels_time: int = 64, text_dim: int = 512, has_text_embed: bool = False, final_output_dim: int = 256):
		super(ProbHead, self).__init__()
		self.latent_channels = latent_channels
		self.latent_height = latent_height
		self.latent_width = latent_width
		self.hidden_dims = hidden_dims
		self.use_batch_norm = use_batch_norm
		self.activation = activation
		self.dropout_rate = dropout_rate

		# Latent encoder
		self.encoder = LatentEncoderAttn_withtime(input_channel=latent_channels, encoder_channels=hidden_dims, 
				num_layers=num_layers, latent_height=latent_height, latent_width=latent_width, 
				 resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
				 resnet_groups=resnet_groups, attention_head_dim=attention_head_dim,
					output_scale_factor=output_scale_factor, downsample_type=downsample_type,
					temb_channels=temb_channels, resnet_time_scale_shift=resnet_time_scale_shift)

		self.output_head = _build_mlp(
			input_dim=self.encoder.output_dim,
			hidden_dims=hidden_dims_mlp,
			output_dim=final_output_dim,  # 4 quality metrics
			use_batch_norm=use_batch_norm,
			activation=activation,
			dropout_rate=dropout_rate
		)

		self.time_proj = Timesteps(num_channels=num_channels_time)
		self.time_emb = TimestepEmbedding(in_channels=num_channels_time, time_embed_dim=temb_channels)
		self.has_text_embed = has_text_embed
		if has_text_embed:
			self.text_encoder = _build_mlp(
				input_dim=text_dim,
				hidden_dims=[(final_output_dim + text_dim) // 2],
				output_dim=final_output_dim,
				use_batch_norm=use_batch_norm,
				activation=activation,
				dropout_rate=dropout_rate
			)
		feat_dim = final_output_dim * 4 if has_text_embed else final_output_dim
		self.final_embed = _build_mlp(
			input_dim=feat_dim,
			hidden_dims=[512, 256],
			output_dim=1,
			use_batch_norm=use_batch_norm,
			activation=activation,
			dropout_rate=dropout_rate, 
			sigmoid=True
		)

	def forward(self, latent, t, text_emb=None):
		t_emb = self.time_proj(t)
		t_emb = self.time_emb(t_emb)
		latent = self.encoder(latent, t_emb)
		latent = self.output_head(latent)
		if text_emb is not None:
			text_emb = self.text_encoder(text_emb)
			matrix = F.cosine_similarity(latent.unsqueeze(1), text_emb.unsqueeze(0), dim=2)
			latent = torch.cat([latent, text_emb, latent * text_emb, (latent - text_emb).abs()], dim=1)
			output = self.final_embed(latent)
		else:
			output = self.final_embed(latent)
			matrix = None
		return output.squeeze(), matrix

class ProbHeads(nn.Module):
	def __init__(self, latent_channels: int, latent_height: int, num_layers: int,
				 latent_width: int, hidden_dims: List[int], hidden_dims_mlp: List[int], use_batch_norm: bool, 
				 activation: str, dropout_rate: float = 0.0, resnet_eps: float = 1e-5, resnet_act_fn: str = "silu",
				 resnet_groups: int = 32, attention_head_dim: int = 5,
				 output_scale_factor: float = 1.0, downsample_type: str = "conv", 
				 temb_channels: int = 512, resnet_time_scale_shift: str = "default", 
				 num_channels_time: int = 64, used_keys: List[str] = ['clip16', 'clip32']
				 ):
		super(ProbHeads, self).__init__()
		self.used_keys = used_keys
		self.probe_heads = nn.ModuleDict()
		for key in used_keys:
			# text_dim = TEXT_DIM.get(key, 0)
			text_dim = OUTPUT_DIM[key]
			has_text_embed = HAS_TEXT_EMBED[key]
			self.probe_heads[key] = ProbHead(latent_channels=latent_channels, latent_height=latent_height, num_layers=num_layers,
				 latent_width=latent_width, hidden_dims=hidden_dims, hidden_dims_mlp=hidden_dims_mlp, use_batch_norm=use_batch_norm, 
				 activation=activation, dropout_rate=dropout_rate,
				 resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
				 resnet_groups=resnet_groups, attention_head_dim=attention_head_dim,
				 output_scale_factor=output_scale_factor, downsample_type=downsample_type, 
				 temb_channels=temb_channels, resnet_time_scale_shift=resnet_time_scale_shift, 
				 num_channels_time=num_channels_time, text_dim=text_dim, has_text_embed=has_text_embed)

	def forward(self, data):
		outputs = {}
		for key in self.used_keys:
			feat = data['records']
			txt = data.get(key + '_t', None)
			t_idx = data['time']
			output, matrix = self.probe_heads[key](feat, t_idx, txt)
			outputs[key] = (output, matrix)
		return outputs
