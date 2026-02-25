import torch
from PIL import Image
from torchvision.transforms import v2 as T


class ResizeBySide(T.Transform):

    def __init__(self, target_height=None, target_width=None):
        super().__init__()
        assert (target_height is None) ^ (target_width is None)
        self.target_height = target_height
        self.target_width = target_width

    def transform(self, inpt, params):
        return self._transform(inpt, params)

    def _transform(self, inpt, params):
        if isinstance(inpt, Image.Image):
            w, h = inpt.size
        elif isinstance(inpt, torch.Tensor):
            h, w = inpt.shape[-2:]

        if self.target_height is not None:
            new_height = self.target_height
            new_width = int(self.target_height / h * w)
        else:
            new_height = int(self.target_width / w * h)
            new_width = self.target_width

        return T.functional.resize(inpt, (new_height, new_width))


class AdaptiveDivisibleCrop(T.Transform):

    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor

    def transform(self, inpt, params):
        return self._transform(inpt, params)

    def _transform(self, inpt, params):
        if isinstance(inpt, Image.Image):
            w, h = inpt.size
        elif isinstance(inpt, torch.Tensor):
            h, w = inpt.shape[-2:]

        new_h = (h // self.divisor) * self.divisor
        new_w = (w // self.divisor) * self.divisor

        if new_h == h and new_w == w:
            return inpt

        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2

        return T.functional.crop(inpt, start_h, start_w, new_h, new_w)
