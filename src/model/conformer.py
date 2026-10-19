import torch
from torch import nn
from torch.nn import functional as F


def get_rope_embeddings(seq_len, head_dim, device):
    freqs = 10000.0 ** (
        -torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = positions[:, None] * freqs[None, :]
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x, cos, sin):
    dtype = x.dtype
    x = x.float()
    even, odd = x[..., 0::2], x[..., 1::2]
    return (
        torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        .flatten(-2)
        .to(dtype)
    )


def sequence_mask(lengths, max_length, device):
    return (
        torch.arange(max_length, device=device)[None, :] < lengths.to(device)[:, None]
    )


class FeedForward(nn.Sequential):
    def __init__(self, d_model, dropout=0.1, expansion_factor=4):
        super().__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion_factor * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * d_model, d_model),
            nn.Dropout(dropout),
        )


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, lengths, cos, sin):
        batch_size, time, dim = x.shape
        q = (
            self.q_proj(x)
            .reshape(batch_size, time, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, time, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, time, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        valid = sequence_mask(lengths, time, x.device)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch_size, time, dim)
        return self.out_proj(out).masked_fill(~valid.unsqueeze(-1), 0.0)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model=256, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
        )
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.pointwise_out = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x, lengths):
        padding = ~sequence_mask(lengths, x.size(1), x.device).unsqueeze(-1)
        x = self.norm(x).masked_fill(padding, 0.0).transpose(1, 2)
        x = self.pointwise_in(x).masked_fill(padding.transpose(1, 2), 0.0)
        x = self.depthwise(x).transpose(1, 2)
        x = self.channel_norm(x).transpose(1, 2)
        return self.pointwise_out(x).transpose(1, 2).masked_fill(padding, 0.0)


class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dropout=0.1, kernel_size=31):
        super().__init__()
        self.ffn1 = FeedForward(d_model, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForward(d_model, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths, cos, sin):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn_dropout(self.attn(self.attn_norm(x), lengths, cos, sin))
        x = x + self.conv(x, lengths)
        x = x + 0.5 * self.ffn2(x)
        padding = ~sequence_mask(lengths, x.size(1), x.device).unsqueeze(-1)
        return self.final_norm(x).masked_fill(padding, 0.0)


class ConformerCTC(nn.Module):
    def __init__(
        self,
        n_feats,
        n_tokens,
        d_model=256,
        num_heads=4,
        num_layers=8,
        kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.subsampling = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                ),
            ]
        )
        self.input_projection = nn.Sequential(
            nn.Linear(d_model * ((n_feats + 3) // 4), d_model),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, num_heads, dropout, kernel_size)
                for _ in range(num_layers)
            ]
        )
        self.head_dim = d_model // num_heads
        self.classifier = nn.Linear(d_model, n_tokens)

    def __str__(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return (
            f"{super().__str__()}\n"
            f"All parameters: {total}\n"
            f"Trainable parameters: {trainable}"
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        lengths = spectrogram_length.to(device=spectrogram.device, dtype=torch.long)
        valid = sequence_mask(lengths, spectrogram.size(-1), spectrogram.device)
        x = spectrogram.masked_fill(~valid[:, None, :], 0.0).unsqueeze(1)
        for layer in self.subsampling:
            x = layer(x)
            lengths = (lengths + 1) // 2
            valid = sequence_mask(lengths, x.size(-1), x.device)
            x = x.masked_fill(~valid[:, None, None, :], 0.0)
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = self.input_projection(x).masked_fill(~valid.unsqueeze(-1), 0.0)
        cos, sin = get_rope_embeddings(x.size(1), self.head_dim, x.device)
        for block in self.blocks:
            x = block(x, lengths, cos, sin)
        return {
            "log_probs": F.log_softmax(self.classifier(x).float(), dim=-1),
            "log_probs_length": (spectrogram_length + 3) // 4,
        }
