import torch_audiomentations
from torch import nn


class Gain(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = torch_audiomentations.Gain(output_type="tensor", **kwargs)

    def forward(self, audio):
        return self.transform(audio.unsqueeze(0)).squeeze(0)
