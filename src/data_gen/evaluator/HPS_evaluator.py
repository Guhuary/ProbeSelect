import torch
import numpy as np
import torch.nn as nn
from functools import partial

# from .base_evaluator import BaseEvaluator

from transformers import AutoProcessor, AutoModel

from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import hpsv2
from hpsv2.img_score import get_tokenizer
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import torch.nn as nn
from typing import Union
import os

torch_type = torch.float16
device_default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hps_version_map = {
    'v2.0': '/mnt/sharedata/ssd_large/users/guohl/pretrained_models/xswu/HPSv2/HPS_v2_compressed.pt',
    'v2.1': '/mnt/sharedata/ssd_large/users/guohl/pretrained_models/xswu/HPSv2/HPS_v2.1_compressed.pt',
}

def initialize_model(device='cuda'):
    model_dict = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        '/mnt/sharedata/ssd_large/users/guohl/pretrained_models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
        precision='fp16',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    model_dict['model'] = model
    model_dict['preprocess_val'] = preprocess_val
    return model_dict


class HPS_Evaluator:
    def __init__(self, device='cuda', hps_version: str = "v2.1"):
        self.model_dict = initialize_model(device=device)
        self.model = self.model_dict['model'] # type: ignore
        self.preprocess_val = self.model_dict['preprocess_val'] # type: ignore

        cp = hps_version_map[hps_version]
        self.model.load_state_dict(torch.load(cp, map_location='cpu')['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def get_score_and_embedding(self, prompt: list[str], imgs: Union[list, str, Image.Image]): 
        image = [self.preprocess_val(img) for img in imgs] # type: ignore
        image = torch.stack(image, dim=0).to(device=self.device, non_blocking=True).to(torch_type)
        text = self.tokenizer(prompt).to(device=self.device, non_blocking=True)

        outputs = self.model(image, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T

        hps_score = torch.diagonal(logits_per_image)

        return hps_score, text_features

if __name__ == '__main__':
    evaluator = HPS_Evaluator(device='cuda', hps_version="v2.1")
    imgs = [Image.open('demo.jpg').convert('RGB')] * 5
    prompts = ['a painting of an ocean with clouds and birds, day time, low depth field effect'] * 5
    score, text_embedding = evaluator.get_score_and_embedding(prompts, imgs)
    print(score)
    print(text_embedding.shape)
