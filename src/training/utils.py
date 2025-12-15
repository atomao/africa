"""
Training utility functions for model management, inference, and evaluation.

This module provides utilities for:
1. Loading and unloading models (checkpoint management)
2. Running inference on images and converting outputs to numpy
3. Computing prediction metrics
4. Model memory management
5. Ensemble predictions
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.training.train_model import SegmentationModel


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_class: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True
) -> SegmentationModel:
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if map_location is None:
        map_location = device
    
    # Load checkpoint
    if model_class is None:
        # Load using Lightning's built-in method
        model = SegmentationModel.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            strict=strict
        )
    else:
        # Load with custom model class
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = model_class(**checkpoint.get('hyper_parameters', {}))
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    
    model.to(device)
    model.eval()
    
    return model


def save_model_checkpoint(
    model: nn.Module,
    save_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def unload_model(model: nn.Module) -> None:
    if model is not None:
        # Move model to CPU
        model.cpu()
        
        # Delete model reference
        del model
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Model unloaded and GPU memory cleared")


def preprocess_image_for_inference(
    image: Union[np.ndarray, str, Path],
    input_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_size = (image.shape[0], image.shape[1])
    
    # Build transformation pipeline
    transforms = []
    
    if input_size is not None:
        transforms.append(A.Resize(height=input_size[0], width=input_size[1]))
    
    if normalize:
        transforms.append(A.Normalize(mean=mean, std=std))
    
    transforms.append(ToTensorV2())
    
    transform = A.Compose(transforms)
    
    # Apply transformations
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_size


@torch.no_grad()
def inference_on_image(
    model: nn.Module,
    image: Union[np.ndarray, str, Path, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    input_size: Optional[Tuple[int, int]] = None,
    return_numpy: bool = True,
    threshold: float = 0.5,
    resize_to_original: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Preprocess image if not already a tensor
    if not isinstance(image, torch.Tensor):
        image_tensor, original_size = preprocess_image_for_inference(
            image, input_size=input_size
        )
    else:
        image_tensor = image
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        original_size = (image_tensor.shape[2], image_tensor.shape[3])
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Run inference
    logits = model(image_tensor)
    
    # Apply sigmoid and threshold
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float()
    
    # Remove batch and channel dimensions
    mask = mask.squeeze(0).squeeze(0)
    
    # Resize to original size if requested
    if resize_to_original and mask.shape != original_size:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        mask = mask.squeeze(0).squeeze(0)
        mask = (mask > threshold).float()
    
    # Convert to numpy if requested
    if return_numpy:
        mask = mask.cpu().numpy()
    
    return mask


@torch.no_grad()
def inference_on_batch(
    model: nn.Module,
    images: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
    threshold: float = 0.5,
    return_numpy: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Move to device
    images = images.to(device)
    
    # Run inference
    logits = model(images)
    
    # Apply sigmoid and threshold
    probs = torch.sigmoid(logits)
    masks = (probs > threshold).float()
    
    # Remove channel dimension
    masks = masks.squeeze(1)
    
    # Convert to numpy if requested
    if return_numpy:
        masks = masks.cpu().numpy()
    
    return masks


def logits_to_mask(
    logits: torch.Tensor,
    threshold: float = 0.5,
    return_numpy: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float()
    
    if return_numpy:
        mask = mask.cpu().numpy()
    
    return mask


def calculate_inference_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6
) -> Dict[str, float]:
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Calculate metrics
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (np.sum(pred_flat) + np.sum(gt_flat) + smooth)
    
    # Precision and Recall
    true_positive = intersection
    false_positive = np.sum(pred_flat) - true_positive
    false_negative = np.sum(gt_flat) - true_positive
    
    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    Get model memory usage statistics.
    
    Parameters
    ----------
    model : nn.Module
        Model to analyze
    
    Returns
    -------
    dict
        Dictionary with memory statistics in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / (1024 ** 2),
        'buffers_mb': buffer_size / (1024 ** 2),
        'total_mb': total_size / (1024 ** 2)
    }