import re
import torch
import numpy as np
import torch.nn as nn
from functools import partial
from typing import Optional, Union

from .base_evaluator import BaseEvaluator

from transformers import AutoProcessor, AutoModel

from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os
from ImageReward import ImageReward

class Modified_ImgReward(ImageReward):
    def __init__(self, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", med_config: Optional[str] = None):
        super().__init__(device=str(device), med_config=med_config)
        self.model_dtype = torch.float16
        
    @torch.no_grad()
    def score_with_embedding(self, prompt, generations_list):
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        # txt_set = []

        image = torch.stack([self.preprocess(img) for img in generations_list], dim=0).to(self.device, dtype=self.model_dtype)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output_with_embedding = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
        txt_set = text_output_with_embedding.last_hidden_state[:,0,:]
        text_embedding = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = None,
                                                encoder_attention_mask = None,
                                                return_dict = True,
                                                mode='textonly',
                                            ).last_hidden_state[:,0,:]
        rewards = self.mlp(txt_set) # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        return rewards, text_embedding


def load(
    name: str = "ImageReward-v1.0",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    med_config: Optional[str] = None,
):
    model_path = name

    print('load checkpoint from %s'%model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    
    model = Modified_ImgReward(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict,strict=False)
    print("checkpoint loaded")
    model.eval()

    return model

class ImgReward_Evaluator(BaseEvaluator):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(device)
        # Load CLIP models

        self.model = load(device=device, 
                        name='/mnt/sharedata/ssd_large/users/guohl/pretrained_models/THUDM/ImageReward/ImageReward.pt', 
                        med_config='/mnt/sharedata/ssd_large/users/guohl/pretrained_models/THUDM/ImageReward/med_config.json')
        self.model = self.model.to(model_dtype)

    @torch.no_grad()
    def get_score_and_embedding(self, prompt, images):
        return self.model.score_with_embedding(prompt, images)

if __name__ == '__main__':
    evaluator = ImgReward_Evaluator(device='cuda')
    imgs = [Image.open('demo.jpg').convert('RGB')] * 5
    prompts = ['a painting of an ocean with clouds and birds, day time, low depth field effect', 'A plate of seasoned chicken, pizza, green bean.', 
            'A bowl of fruit on a table in an apartment.', 'An adult giraffe and a baby giraffe walking', 'An older woman is petting her white horse.'
        ]
    score, text_output = evaluator.get_score_and_embedding(prompts, imgs)
    print(score)
    print(text_output.shape)
