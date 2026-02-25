import contextlib

import torch
from spa3r import SPA3R

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class Spa3REncoder(BaseGeometryEncoder):

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        embed_dim = config.encoder_kwargs.get('embed_dim', 768)
        num_queries = config.encoder_kwargs.get('num_queries', 256)
        self.embed_dim = embed_dim
        self.spa3r = Spa3R(embed_dim=embed_dim, num_queries=num_queries)

        if self.freeze_encoder:
            for param in self.spa3r.parameters():
                param.requires_grad = False

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.spa3r.eval()

        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        inputs_dict = {"images": images}
        grad_ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with grad_ctx:
            with torch.amp.autocast("cuda", dtype=dtype):
                features = self.spa3r(inputs_dict, mode="predict")

        return features

    def get_feature_dim(self) -> int:
        return self.embed_dim

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        state_dict = {
            k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')
        }

        self.spa3r.load_state_dict(state_dict)
        if self.freeze_encoder:
            for param in self.spa3r.parameters():
                param.requires_grad = False
