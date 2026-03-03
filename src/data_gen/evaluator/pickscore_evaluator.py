import torch
import numpy as np
import torch.nn as nn
from functools import partial

from .base_evaluator import BaseEvaluator

from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

class PickScoreEvaluator(BaseEvaluator):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(device)
        self.model = None
        self.model_dtype = model_dtype
        processor_name_or_path = "/mnt/sharedata/ssd_large/users/guohl/pretrained_models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "/mnt/sharedata/ssd_large/users/guohl/pretrained_models/yuvalkirstain/PickScore_v1"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path, torch_dtype=model_dtype)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path, torch_dtype=model_dtype).eval().to(device)
        self.results = []

    def calc_probs(self, prompt, images):
        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.model_dtype).to(self.device)
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.model_dtype).to(self.device)

        # embed
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(-1)
        
        # get probabilities if you have multiple images to choose from
        # probs = torch.softmax(scores, dim=-1)    
        return scores
    
    @torch.no_grad()
    def evaluate(self, prompts, images):
        scores = self.calc_probs(prompts, images)
        self.results.append(scores)

    @torch.no_grad()
    def get_score(self, prompts, images):
        scores = self.calc_probs(prompts, images)
        return scores
    
    @torch.no_grad()
    def get_embedding(self, prompts):
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.model_dtype).to(self.device)

        # embed
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True).detach()
        return text_embs