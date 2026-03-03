import torch
import numpy as np
import torch.nn as nn
from functools import partial

from .base_evaluator import BaseEvaluator

# Aesthetic Score Evaluator
class AestheticScoreEvaluator(BaseEvaluator):
    def __init__(self, encoder_model_name='/mnt/sharedata/ssd_large/users/guohl/pretrained_models/google/siglip-so400m-patch14-384', 
                 device='cuda', model_dtype=torch.float16):
        super().__init__(device)
        self.encoder_model_name = encoder_model_name
        self.model_dtype = model_dtype
        self.model = None
        self.preprocessor = None
        self.prepare()
        self.results = []

    def prepare(self):
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        model, preprocessor = convert_v2_5_from_siglip(
            encoder_model_name=self.encoder_model_name,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = model.to(self.model_dtype).to(self.device)
        self.preprocessor = preprocessor

    @torch.no_grad()
    def evaluate(self, prompts, images):
        pixel_values = self.preprocessor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.model_dtype).to(self.device)
        score = self.model(pixel_values).logits.squeeze()
        self.results.append(score)
        return score
    
    @torch.no_grad()
    def get_score(self, prompts, images):
        pixel_values = self.preprocessor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.model_dtype).to(self.device)
        score = self.model(pixel_values).logits.squeeze()
        return score