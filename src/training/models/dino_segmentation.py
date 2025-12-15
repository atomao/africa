import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.training.models.dino_head import BN_U_Head_768
from src.training.models.dinov2 import DinoV2Backbone
from src.training.models.dinov3 import (
    DinoV3Backbone,
    REPO_DIR as DINOV3_LOCAL_REPO_PATH, 
     WEIGHTS_PATH as DINOV3_SAT_LARGE_WEIGHTS_PATH
)
from src.sliding_window import SlidingWindowInference, BlendMode


class DinoSegmentor(nn.Module):
    """
    Lightweight inference model for a single head.
    Useful for evaluation, visualization, and production inference.
    """
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        device: str | torch.device,
        out_indices: list[int] = (6, 12, 18, 23),
        align_corners: bool = False
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.out_indices = out_indices
        self.num_classes = num_classes
        self.device = device
        self.align_corners = align_corners

        
        # Initialize backbone
        self.backbone_wrapper = self._load_backbone_wrapper()
        self.feature_dim = self.backbone_wrapper.get_embed_dim()
        
        # Initialize head/decoder
        in_channels = [self.feature_dim] * len(out_indices)
        in_index = list(range(len(out_indices)))
        

        self.head = BN_U_Head_768(
            in_channels=in_channels,
            in_index=in_index,
            num_classes=num_classes,
            align_corners=align_corners
        )
    

    # TODO: Refactor so single_head and multi_head have same baseclass
    def _load_backbone_wrapper(self):
        """Load and return the backbone wrapper."""
        if self.backbone_name == 'dinov2':
            return DinoV2Backbone(
                size="large",
                out_indices=self.out_indices,
                frozen=True,  # Always frozen for inference
                pretrained=True,
                device=self.device
            )
        elif self.backbone_name == 'dinov3':
            return DinoV3Backbone(
                repo_local_dir=DINOV3_LOCAL_REPO_PATH,
                weights_path=DINOV3_SAT_LARGE_WEIGHTS_PATH,
                out_indices=self.out_indices,
                frozen=True, 
                device=self.device
            )
        else:
            raise ValueError("Unsupported backbone. Choose 'dinov2' or 'dinov3'.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and head."""
        with torch.no_grad():
            features = self.backbone_wrapper(x)
        
        seg_logits = self.head(features)
        
        # Resize to input resolution
        original_size = x.shape[2:]
        if seg_logits.shape[2:] != original_size:
            seg_logits = F.interpolate(
                seg_logits,
                size=original_size,
                mode='bilinear',
                align_corners=self.align_corners
            )
        
        return seg_logits


    def predict_sliding_window(
        self,
        x: torch.Tensor,
        window_size: tuple[int, int] = (1024, 1024),
        overlap: float = 0.1,
        blend_mode: str | BlendMode = 'gaussian',
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Perform sliding window inference on large images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            window_size: Size of sliding window (height, width)
            overlap: Overlap ratio (0.0 to < 1.0). Higher values = more windows.
                    Typical range: 0.25-0.5, but can go higher for better quality
            blend_mode: 'gaussian', 'linear', or 'constant'
            return_logits: If True, return logits; if False, return predictions
            
        Returns:
            Segmentation output of same spatial size as input
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D tensor")
        
        if isinstance(blend_mode, str):
            blend_mode = BlendMode(blend_mode)

        inference = SlidingWindowInference(
            window_size=window_size,
            overlap=overlap,
            blend_mode=blend_mode
        )
        
        return inference(self, x, return_logits=return_logits)
 