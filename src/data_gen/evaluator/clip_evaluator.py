import torch
import numpy as np
import torch.nn as nn
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Union, cast
import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.multimodal.clip_score import _detect_modality, _process_image_data, _process_text_data, _get_features

from .base_evaluator import BaseEvaluator

class CLIPSocreEvaluator(BaseEvaluator):
    def __init__(self, model_name, device='cuda', model_dtype=torch.float16):
        super().__init__(device)
        self.model = None
        self.processor = None
        self.model_dtype = model_dtype
        self.prepare(model_name, model_dtype)
        self.results = []

    def prepare(self, model_name, model_dtype):
        from torchmetrics.multimodal.clip_score import CLIPScore
        from torchmetrics.functional.multimodal.clip_score import _clip_score_update
        metric = CLIPScore(model_name_or_path=model_name).to(model_dtype).to(self.device)
        self.model = metric.model
        self.processor = metric.processor
        self._clip_score_update = partial(_clip_score_update, model=metric.model, processor=metric.processor)
        self.clip_embedding = partial(self._clip_embedding, model=metric.model, processor=metric.processor)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def evaluate(self, prompts, images):
        image_tensor = torch.tensor(np.array(images)).permute(0, 3, 1, 2).to(self.model_dtype).to(self.device)
        score, _ = self._clip_score_update(prompts, image_tensor)
        self.results.append(score)

    @torch.no_grad()
    def get_score(self, prompts, images):
        image_tensor = torch.tensor(np.array(images)).permute(0, 3, 1, 2).to(self.model_dtype).to(self.device)
        score, _ = self._clip_score_update(prompts, image_tensor)
        return score

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

    @torch.no_grad()
    def get_embedding(self, prompts):
        return self.clip_embedding(prompts)
    
class CLIP16Evaluator(CLIPSocreEvaluator):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(model_name = '/mnt/sharedata/ssd_large/users/guohl/pretrained_models/openai/clip-vit-base-patch16', 
                         device=device)

class CLIP32Evaluator(CLIPSocreEvaluator):
    def __init__(self, device='cuda', model_dtype=torch.float16):
        super().__init__(model_name = '/mnt/sharedata/ssd_large/users/guohl/pretrained_models/openai/clip-vit-base-patch32', 
                         device=device)
