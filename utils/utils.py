import torch
import torch.nn as nn
import random
import numpy
import os
import math
import numpy as np


def seed_all(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class ThreePhaseLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(0.05 * total_steps)
        self.maintain_steps = int(0.55 * total_steps)
        self.cosine_steps = total_steps - self.warmup_steps - self.maintain_steps
        super(ThreePhaseLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_steps + self.maintain_steps:
            return self.base_lrs

        else:
            cosine_epoch = self.last_epoch - self.warmup_steps - self.maintain_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epoch / self.cosine_steps))
            return [base_lr * cosine_factor for base_lr in self.base_lrs]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.normalized_shape = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_normalized = x / (rms + self.eps)
        return self.scale * x_normalized


def sequence_mean(output: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    if output.dim() != 3:
        raise ValueError(f"期望输出张量为3维，但得到了{output.dim()}维")
    if sequence_lengths.dim() != 1:
        raise ValueError(f"期望 sequence_lengths 为1维，但得到了{sequence_lengths.dim()}维")
    if output.size(0) != sequence_lengths.size(0):
        raise ValueError(f"batch_size 不匹配: {output.size(0)} vs {sequence_lengths.size(0)}")

    bs, length, embedding = output.size()

    mask = (torch.arange(length, device=output.device)[None, :] < sequence_lengths[:, None]).unsqueeze(-1)

    masked_output = output * mask

    sum_output = masked_output.sum(dim=1)
    safe_lengths = torch.clamp(sequence_lengths.float(), min=1.0).unsqueeze(-1)

    return sum_output / safe_lengths
