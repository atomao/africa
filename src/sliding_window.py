import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum


class BlendMode(Enum):
    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"

    @classmethod
    def from_str(cls, mode_str: str) -> "BlendMode":
        """Convert string to BlendMode enum."""
        mode_str = mode_str.lower()
        if mode_str == "constant":
            return cls.CONSTANT
        elif mode_str == "gaussian":
            return cls.GAUSSIAN
        elif mode_str == "linear":
            return cls.LINEAR
        else:
            raise ValueError(f"Unknown blend mode: {mode_str}")


class SlidingWindowInference:
    """
    Sliding window inference for semantic segmentation on large images.
    Handles overlapping windows with weighted averaging in overlap regions.
    """

    def __init__(
        self,
        window_size: tuple[int, int] = (512, 512),
        overlap: float = 0.25,
        blend_mode: BlendMode = BlendMode.CONSTANT,
    ):
        """
        Args:
            window_size: Size of sliding window (height, width)
            overlap: Overlap ratio between windows (0.0 to < 1.0)
                    0.0 = no overlap, 0.5 = 50% overlap, 0.75 = 75% overlap
                    Higher overlap = more windows = slower but potentially more accurate
            blend_mode: How to combine overlapping predictions
                       'gaussian' - Gaussian weighted averaging (recommended)
                       'linear' - Linear distance-based weighting
                       'constant' - Simple averaging
        """
        self.window_size = window_size
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"Overlap must be in range [0.0, 1.0), got {overlap}")
        self.overlap = overlap
        self.blend_mode = blend_mode

        # Calculate stride based on overlap
        stride_h = int(window_size[0] * (1 - overlap))
        stride_w = int(window_size[1] * (1 - overlap))

        # Ensure minimum stride of 1 pixel
        self.stride = (max(1, stride_h), max(1, stride_w))

        # Pre-compute blend weight map
        self.blend_weights = self._create_blend_weights()

    def _create_blend_weights(self) -> torch.Tensor:
        """Create weight map for blending overlapping windows."""
        h, w = self.window_size

        if self.blend_mode == BlendMode.GAUSSIAN:
            # Gaussian weight map - higher weight at center
            center_h, center_w = h / 2, w / 2
            sigma_h, sigma_w = h / 4, w / 4

            y = torch.arange(h).float() - center_h
            x = torch.arange(w).float() - center_w
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            weights = torch.exp(-(xx**2 / (2 * sigma_w**2) + yy**2 / (2 * sigma_h**2)))

        elif self.blend_mode == BlendMode.LINEAR:
            # Linear weight map - decreases toward edges
            y = torch.linspace(0, 1, h)
            x = torch.linspace(0, 1, w)
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            # Distance from nearest edge
            dist_y = torch.minimum(yy, 1 - yy)
            dist_x = torch.minimum(xx, 1 - xx)
            weights = torch.minimum(dist_y, dist_x) * 2  # Scale to [0, 1]

        elif self.blend_mode == BlendMode.CONSTANT:
            # Equal weighting everywhere
            weights = torch.ones(h, w)

        else:
            raise ValueError(f"Unknown blend mode: {self.blend_mode}")

        return weights

    def _get_window_positions(
        self, image_size: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        """
        Calculate all window positions for sliding window inference.

        Args:
            image_size: (height, width) of input image

        Returns:
            List of (top, left, bottom, right) tuples for each window
        """
        img_h, img_w = image_size
        win_h, win_w = self.window_size
        stride_h, stride_w = self.stride

        positions = []

        # Calculate window positions
        for top in range(0, img_h, stride_h):
            for left in range(0, img_w, stride_w):
                # Adjust bottom-right windows to fit image bounds
                bottom = min(top + win_h, img_h)
                right = min(left + win_w, img_w)

                # Adjust top-left if window extends beyond image
                if bottom == img_h:
                    top = max(0, img_h - win_h)
                if right == img_w:
                    left = max(0, img_w - win_w)

                positions.append((top, left, bottom, right))

        # Remove duplicate positions
        positions = list(dict.fromkeys(positions))

        return positions

    def __call__(
        self, model: nn.Module, image: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor:
        """
        Perform sliding window inference on large image.

        Args:
            model: Segmentation model with forward() method
            image: Input image tensor (B, C, H, W) - typically B=1
            return_logits: If True, return logits; if False, return class predictions

        Returns:
            Segmentation output of same spatial size as input
            - If return_logits=True: (B, num_classes, H, W)
            - If return_logits=False: (B, H, W)
        """
        device = image.device
        batch_size, channels, img_h, img_w = image.shape

        # check if model is in eval mode
        if not model.training:
            model.eval()

        # Get model's number of classes from first forward pass
        with torch.no_grad():
            dummy_window = image[
                :,
                :,
                : min(self.window_size[0], img_h),
                : min(self.window_size[1], img_w),
            ]
            dummy_out = model(dummy_window)
            num_classes = dummy_out.shape[1]

        # Initialize accumulation tensors
        logits_sum = torch.zeros(
            batch_size, num_classes, img_h, img_w, device=device, dtype=torch.float32
        )
        weights_sum = torch.zeros(
            batch_size, 1, img_h, img_w, device=device, dtype=torch.float32
        )

        # Get window positions
        positions = self._get_window_positions((img_h, img_w))
        blend_weights = self.blend_weights.to(device)

        # Process each window
        model.eval()
        with torch.no_grad():
            for top, left, bottom, right in positions:
                # Extract window
                window = image[:, :, top:bottom, left:right]

                # Forward pass
                window_logits = model(window)

                # Get actual window size (might be smaller at edges)
                actual_h, actual_w = window.shape[2:]
                window_blend_weights = blend_weights[:actual_h, :actual_w]

                # Accumulate logits with weights
                logits_sum[:, :, top:bottom, left:right] += (
                    window_logits * window_blend_weights.unsqueeze(0).unsqueeze(0)
                )

                weights_sum[:, :, top:bottom, left:right] += (
                    window_blend_weights.unsqueeze(0).unsqueeze(0)
                )

        # Normalize by accumulated weights
        logits_avg = logits_sum / (weights_sum + 1e-8)

        if return_logits:
            return logits_avg
        else:
            return torch.argmax(logits_avg, dim=1)
