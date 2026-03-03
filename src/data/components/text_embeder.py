import torch
import numpy as np
import torch.nn as nn
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Union, cast
import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.multimodal.clip_score import _detect_modality, _process_image_data, _process_text_data, _get_features

class CLIP_Embeder(nn.Module):
    def __init__(self, model_name, device='cuda', model_dtype=torch.float16):
        super().__init__()
        self.device = device
        self.model = None
        self.processor = None
        self.model_dtype = model_dtype
        self.prepare(model_name, model_dtype)
        self.results = []

    def _clip_embedding(self, 
        source,
        model,
        processor,
    ) -> Tensor:
        """Update function for CLIP Score."""
        source_modality = _detect_modality(source)

        source_data = (
            _process_image_data(cast(Union[Tensor, List[Tensor]], source))
            if source_modality == "image"
            else _process_text_data(cast(Union[str, List[str]], source))
        )

        device = (
            source_data[0].device
            if source_modality == "image" and isinstance(source_data[0], Tensor)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = model.to(device)

        source_features = _get_features(
            cast(List[Union[Tensor, str]], source_data), source_modality, device, model, processor
        )

        source_features = source_features / source_features.norm(p=2, dim=-1, keepdim=True)

        return source_features.detach()

    def prepare(self, model_name, model_dtype):
        from torchmetrics.multimodal.clip_score import CLIPScore
        from torchmetrics.functional.multimodal.clip_score import _clip_score_update
        metric = CLIPScore(model_name_or_path=model_name).to(model_dtype)
        self.model = metric.model
        self.processor = metric.processor
        self.clip_embedding = partial(self._clip_embedding, model=metric.model, processor=metric.processor)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, prompts):
        return self.clip_embedding(prompts)

class CLIP16Embeder(CLIP_Embeder):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(model_name = '/data/home/guohl/Gen/pretrained_models/openai/clip-vit-base-patch16', 
                         device=device)

class CLIP32Embeder(CLIP_Embeder):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(model_name = '/data/home/guohl/Gen/pretrained_models/openai/clip-vit-base-patch32', 
                         device=device)


from transformers import AutoProcessor, AutoModel
from PIL import Image


class PickEmbeder(nn.Module):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__()
        self.device = device
        self.model_dtype = model_dtype
        processor_name_or_path = "/data/home/guohl/Gen/pretrained_models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "/data/home/guohl/Gen/pretrained_models/yuvalkirstain/PickScore_v1"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path, torch_dtype=model_dtype)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path, torch_dtype=model_dtype).eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, prompt):
        # preprocess
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.model_dtype).to(self.device)

        # embed
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        return text_embs



if __name__ == '__main__':
    text = ['A man is in a kitchen making pizzas.', 'The dining table near the kitchen has a bowl of fruit on it.']
    clip16 = CLIP16Embeder()
    clip32 = CLIP32Embeder()
    pickemb = PickEmbeder()
    embed16 = clip16(text)
    embed32 = clip32(text)
    pick = pickemb(text)
    print(embed16.shape, embed32.shape, pick.shape)