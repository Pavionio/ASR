from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def warmup_cosine(optimizer, total_steps, warmup_fraction=0.05, min_lr=3e-6):
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
