from __future__ import annotations

import math

import torch
from torch.nn import functional as F


class FixedProjector:
    """A deterministic, non-trainable dimension adapter for equal downstream capacity."""

    def __init__(self, input_dim: int, output_dim: int, seed: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim > output_dim:
            generator = torch.Generator().manual_seed(seed + input_dim * 1009)
            matrix = torch.randn((input_dim, output_dim), generator=generator)
            self.matrix = matrix / math.sqrt(output_dim)
        else:
            self.matrix = None

    def __call__(self, values: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        values = values.float()
        if self.matrix is None:
            output = F.pad(values, (0, self.output_dim - self.input_dim))
        else:
            pieces = [values[start:start + chunk_size] @ self.matrix for start in range(0, len(values), chunk_size)]
            output = torch.cat(pieces)
        return F.normalize(output, dim=1)
