import torch
import pytorch_lightning as pl

from typing import List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum

from src.training.models.center_padding import CenterPadding

REPO_DIR = "/workspace/dinov3/"
WEIGHTS_PATH= "/workspace/artifacts/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"


class DinoV3Backbone(pl.LightningModule):
    """
    DinoV3 Backbone for PyTorch Lightning with configurable backend.
    
    Args:
        repo_local_dir: Local directory for torch.hub repo (only for torch backend)
        weights_path: Path to weights file (only for torch backend)
        out_indices: List of layer indices to extract features from
        frozen: Whether to freeze backbone weights
        device: Device to load model on ('cuda', 'cpu', or torch.device)
    """
    
    def __init__(
        self,
        weights_path: str = WEIGHTS_PATH,
        repo_local_dir: str = REPO_DIR,
        out_indices: list[int] | tuple[int, ...] = (6, 12, 18, 23),
        frozen: bool = True,
        device: Union[str, torch.device] | None = None,
    ):
        super().__init__()
    
        self.repo_dir = repo_local_dir
        self.weights_path = weights_path
        self.out_indices = list(out_indices) if isinstance(out_indices, tuple) else out_indices
        self.frozen = frozen
        self._target_device = device
        
        
        # Set backbone name
        self.backbone_name = Path(weights_path).stem

        
        # Load backbone
        self._load_torch_hub_backbone()
        
        if self._target_device is not None:
            self.to(self._target_device)
        
        # Patch size handling
        if hasattr(self.backbone, "patch_size"):
            self.center_padding = CenterPadding(self.backbone.patch_size)
        else:
            self.center_padding = None
        
        if self.frozen:
            self._freeze_backbone()
    
    
    def _load_torch_hub_backbone(self):
        """Load backbone from torch.hub"""
        self.backbone = torch.hub.load(
            repo_or_dir=self.repo_dir,
            model='dinov3_vitl16',
            source='local',
            weights=self.weights_path
        )
    
    def get_embed_dim(self) -> int:
        return getattr(self.backbone, 'embed_dim', None)
    
    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda(device)
        self._target_device = f'cuda:{device}' if isinstance(device, int) else 'cuda' if device is None else str(device)
        return self
    
    def cpu(self):
        super().cpu()
        self._target_device = 'cpu'
        return self
    
    def to(self, device: Union[str, torch.device]):
        super().to(device)
        self._target_device = str(device) if not isinstance(device, str) else device
        return self
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.center_padding is not None:
            x = self.center_padding(x)
        
        if self.frozen:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(
                    x, n=self.out_indices, reshape=True
                )
        else:
            features = self.backbone.get_intermediate_layers(
                x, n=self.out_indices, reshape=True
            )
        
        # Convert tuple to list for consistency
        return list(features)
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone weights frozen: {self.frozen}")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.frozen = False
        print("Backbone weights unfrozen")
    
    def get_feature_info(self) -> dict:
        return {
            'backend': self.backend.value,
            'backbone_name': self.backbone_name,
            'out_indices': self.out_indices,
            'frozen': self.frozen,
            'patch_size': getattr(self.backbone, 'patch_size', None),
            'embed_dim': getattr(self.backbone, 'embed_dim', None),
        }