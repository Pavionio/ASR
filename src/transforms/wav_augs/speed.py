import torch
import torchaudio.transforms as T
from torch import nn


class SpeedPerturbation(nn.Module):
    def __init__(self, factors=(0.9, 1.0, 1.1), sample_rate=16000, p=0.5):
        super().__init__()
        self.transform = T.SpeedPerturbation(orig_freq=sample_rate, factors=factors)
        self.p = p

    def forward(self, audio):
        if torch.rand(()) >= self.p:
            return audio
        waveform, _ = self.transform(audio)
        return waveform
