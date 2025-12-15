import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterPadding(nn.Module):
    """Center padding module to ensure input size is multiple of patch_size"""

    def __init__(self, multiple: int):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size: int) -> tuple[int, int]:
        """Calculate padding for a given dimension"""
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get padding for height and width (reverse order for F.pad)
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        return F.pad(x, pads)
