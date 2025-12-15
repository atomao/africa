import numpy as np
import torch

def deaugment_image(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverse Albumentations Normalize + ToTensorV2.
    
    Parameters
    ----------
    tensor_img : torch.Tensor 
        Tensor of shape (C, H, W) and dtype float32, normalized.
    mean : list of float
    std : list of float

    Returns
    -------
    np.ndarray
        uint8 image of shape (H, W, C) in RGB.
    """

    if isinstance(tensor_img, torch.Tensor):
        img = tensor_img.detach().cpu().float().numpy()
    else:
        raise TypeError("Expected torch.Tensor")

    # (C, H, W) -> (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # unnormalize
    img = img * std + mean

    # clip to [0, 1]
    img = np.clip(img, 0, 1)

    # convert to uint8
    img = (img * 255).astype(np.uint8)

    return img
