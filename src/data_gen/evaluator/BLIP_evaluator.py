import torch
import numpy as np
import torch.nn as nn
from functools import partial

import requests
from .base_evaluator import BaseEvaluator

from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipTextModel

from torch.nn.functional import normalize
from typing import Optional

class BLIP_Evaluator(BaseEvaluator):
    """Original BLIP Evaluator - embedding contains image-text cross-attention information"""
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(device)
        # Load BLIP models

        self.processor = BlipProcessor.from_pretrained('/mnt/sharedata/ssd_large/users/guohl/pretrained_models/Salesforce/blip-itm-base-coco')
        self.model = BlipForImageTextRetrieval.from_pretrained('/mnt/sharedata/ssd_large/users/guohl/pretrained_models/Salesforce/blip-itm-base-coco', 
                                                               torch_dtype=torch.float16).to(device) # type: ignore
        self.model.eval()

    def get_embedding(self, input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        use_itm_head: Optional[bool] = True,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )

        if use_itm_head:
            question_embeds = self.model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        else:
            question_embeds = self.model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        return question_embeds

    @torch.no_grad()
    def get_itm_score_and_embedding(self, prompts, images):
        inputs = self.processor(images, prompts, return_tensors="pt", padding=True, truncation=True, max_length=40).to(self.device, torch.float16) # type: ignore
        outputs = self.model(**inputs, use_itm_head=True)
        score = outputs.itm_score.detach().softmax(dim=-1)[:, 1]

        question_embeds = self.get_embedding(**inputs, use_itm_head=True)
        embedding = question_embeds[:, 0, :].detach()
        embedding = normalize(self.model.text_proj(embedding), dim=1)
        return score, embedding

    @torch.no_grad()
    def get_itc_score_and_embedding(self, prompts, images):
        inputs = self.processor(images, prompts, return_tensors="pt", padding=True, truncation=True, max_length=40).to(self.device, torch.float16) # type: ignore
        outputs = self.model(**inputs, use_itm_head=False)
        score = outputs.itm_score.detach()
        score = torch.diag(score)

        question_embeds = self.get_embedding(**inputs, use_itm_head=False)
        embedding = question_embeds[:, 0, :].detach()
        embedding = normalize(self.model.text_proj(embedding), dim=1)
        return score, embedding



if __name__ == '__main__':
    # Test original BLIP_Evaluator
    print("Testing BLIP_Evaluator (with image-text cross-attention)...")
    processor = BlipProcessor.from_pretrained('/mnt/sharedata/ssd_large/users/guohl/pretrained_models/Salesforce/blip-itm-base-coco')
    model = BlipForImageTextRetrieval.from_pretrained('/mnt/sharedata/ssd_large/users/guohl/pretrained_models/Salesforce/blip-itm-base-coco', 
                                                               torch_dtype=torch.float16).to('cuda')  # type: ignore

    raw_image = Image.open('demo.jpg').convert('RGB')
    raw_image = [raw_image] * 10
    question = ["A woman and a dog sitting together in a beach."] * 10
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)  # type: ignore
    with torch.no_grad():
        outputs = model(**inputs)
    print("ITM mode - itm_score shape:", outputs.itm_score.shape, "question_embeds shape:", outputs.question_embeds.shape)

    with torch.no_grad():
        outputs = model(**inputs, use_itm_head=False)
    print("ITC mode - itm_score shape:", outputs.itm_score.shape, "question_embeds shape:", outputs.question_embeds.shape)
    
    # Test BLIP_TextOnly_Evaluator
    print("\nTesting BLIP_TextOnly_Evaluator (text-only embedding)...")
    evaluator = BLIP_Evaluator(device='cuda')
    score, text_embedding = evaluator.get_itm_score_and_embedding(question, raw_image)
    print("ITM score shape:", score.shape, "Text-only embedding shape:", text_embedding.shape)
    
    score, text_embedding = evaluator.get_itc_score_and_embedding(question, raw_image)
    print("ITC score shape:", score.shape, "Text-only embedding shape:", text_embedding.shape)
