from torch import nn


class TemporalNormalize(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, spectrogram):
        mean = spectrogram.mean(dim=-1, keepdim=True)
        std = spectrogram.std(dim=-1, keepdim=True, correction=0)
        return (spectrogram - mean) / std.clamp_min(self.eps)
