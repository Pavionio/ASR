import torch_audiomentations
from torch import nn


class AddNoise(nn.Module):
    def __init__(self, sample_rate=16000, **kwargs):
        super().__init__()
        self.transform = torch_audiomentations.AddColoredNoise(
            sample_rate=sample_rate, output_type="tensor", **kwargs
        )

    def forward(self, audio):
        return self.transform(audio.unsqueeze(0)).squeeze(0)
