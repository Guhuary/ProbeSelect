import torch
import numpy as np
import torch.nn as nn
from functools import partial

from .base_evaluator import BaseEvaluator

from transformers import AutoProcessor, AutoModel

from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from hpsv2.src.open_clip import get_tokenizer
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, x):
        return self.layers(x)


class ICT_Evaluator(BaseEvaluator):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(device)
        # Load CLIP models
        pretrained_model_name_or_path = "/mnt/sharedata/ssd_large/users/guohl/pretrained_models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        
        # Load models and prepare preprocessor
        clip_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
        self.preprocess_val = lambda img: clip_processor(images=img, return_tensors="pt")["pixel_values"]
        
        # Load ICT model
        print("Loading ICT model...")
        self.ict_model = CLIPModel.from_pretrained(pretrained_model_name_or_path)
        ictmodel_path ="/mnt/sharedata/ssd_large/users/guohl/pretrained_models/ICTHP/ICTHP_models/ICT"
        checkpoint_path = f"{ictmodel_path}/pytorch_model.bin"
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.ict_model.load_state_dict(state_dict, strict=False)
        self.ict_model = self.ict_model.half().to(device) # type: ignore
        self.ict_model.eval()

        self.tokenizer = get_tokenizer('ViT-H-14')

    @torch.no_grad()
    def ict_score(self, prompts, images):
        # ICT score calculation
        image_ict_features = self.ict_model.get_image_features(pixel_values=images)
        image_ict_features = image_ict_features / image_ict_features.norm(dim=-1, keepdim=True)
        
        # Process text input
        text_input_ids = self.tokenizer(prompts).to(self.device)
        text_features_ict = self.ict_model.get_text_features(text_input_ids)
        text_features_ict = text_features_ict / text_features_ict.norm(dim=-1, keepdim=True)

        ict_scores = (text_features_ict * image_ict_features).sum(dim=-1)
        return ict_scores

    @torch.no_grad()
    def get_embedding(self, prompts):
        text_input_ids = self.tokenizer(prompts).to(self.device)
        text_features_ict = self.ict_model.get_text_features(text_input_ids)
        text_features_ict = text_features_ict / text_features_ict.norm(dim=-1, keepdim=True)
        return text_features_ict
    
    @torch.no_grad()
    def get_score(self, prompts, images):
        images = self.preprocess_val(images).to(self.device) # type: ignore
        ict_score = self.ict_score(prompts, images)
        return ict_score.cpu().detach()

class HP_Evaluator(BaseEvaluator):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(device)
        pretrained_model_name_or_path = "/mnt/sharedata/ssd_large/users/guohl/pretrained_models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # Load CLIP and HP models
        print("Loading HP model...")
        self.hp_backbone = CLIPModel.from_pretrained(pretrained_model_name_or_path)
        
        clip_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
        self.preprocess_val = lambda img: clip_processor(images=img, return_tensors="pt")["pixel_values"]

        self.hp_scorer = MLP(hidden_dim=1024)
        
        hpmodel_path ="/mnt/sharedata/ssd_large/users/guohl/pretrained_models/ICTHP/ICTHP_models/HP"
        hp_backbone_checkpoint_path = f"{hpmodel_path}/hp_backbone/pytorch_model.bin"
        hp_scorer_checkpoint_path = f"{hpmodel_path}/hp_scorer/mlp_pytorch_model.bin"
        
        hp_backbonestate_dict = torch.load(hp_backbone_checkpoint_path, map_location="cpu")
        hp_scorer_state_dict = torch.load(hp_scorer_checkpoint_path, map_location="cpu")
        
        self.hp_backbone.load_state_dict(hp_backbonestate_dict, strict=False)
        self.hp_scorer.load_state_dict(hp_scorer_state_dict, strict=False)
        
        self.hp_backbone = self.hp_backbone.half().to(device) # type: ignore
        self.hp_scorer = self.hp_scorer.half().to(device)
        self.hp_backbone.eval()
        self.hp_scorer.eval()

    def hp_score(self, prompts, images):
        # HP score calculation
        image_hp_backbone_features = self.hp_backbone.get_image_features(pixel_values=images)
        hp_scores = self.hp_scorer(image_hp_backbone_features)
        hp_scores = torch.sigmoid(hp_scores).squeeze()
        return hp_scores
    
    def get_score(self, prompts, images):
        images = self.preprocess_val(images).to(self.device) # type: ignore
        hp_score = self.hp_score(prompts, images)
        return hp_score.cpu().detach()
    
