from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from lightning.pytorch import LightningDataModule


simple_augment = A.Compose([A.ToTensorV2()])
class SegmentationDataset(Dataset):
    """
    Baseline PyTorch Dataset for binary segmentation with Albumentations support.

    Expects a CSV index file (dataset_filepath) with at least two columns:
        x_path,y_path
    where each path is relative to `root_dir` or absolute.

    - x_path: RGB input image path
    - y_path: binary mask path (0/1)

    Parameters
    ----------
    dataset_filepath : str | Path
        Path to CSV file describing dataset pairs (x_path, y_path).
    root_dir : str | Path
        Root directory that contains all images and masks.
    augment : callable, optional
        Albumentations transform (e.g. A.Compose). It should accept
        and return dicts like: {'image': np.ndarray, 'mask': np.ndarray}.
        If None, no augmentations are applied.
    """

    def __init__(
        self,
        dataset_filepath: str | Path ,
        root_dir: str | Path | None = None,
        augment: AlbumentationsTransform = simple_augment,
    ) -> None:
        self.dataset_filepath = Path(dataset_filepath)
        self.root_dir = Path(root_dir)
        self.augment = augment

        self.samples: List[Tuple[Path, Path]] = self._load_index()

    def _load_index(self) -> List[Tuple[Path, Path]]:
        """Read CSV and create list of (x_path, y_path) tuples."""
        samples: List[Tuple[Path, Path]] = []

        with self.dataset_filepath.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "x_path" not in reader.fieldnames or "y_path" not in reader.fieldnames:
                raise ValueError(
                    f"CSV {self.dataset_filepath} must contain 'x_path' and 'y_path' columns."
                )

            for row in reader:
                x_rel = Path(row["x_path"])
                y_rel = Path(row["y_path"])

                x_path = x_rel if x_rel.is_absolute() else self.root_dir / x_rel
                y_path = y_rel if y_rel.is_absolute() else self.root_dir / y_rel

                samples.append((x_path, y_path))

        if not samples:
            raise ValueError(f"No samples found in index file: {self.dataset_filepath}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: Path) -> np.ndarray:
        """
        Read RGB image as np.ndarray of shape (H, W, 3), dtype=uint8.
        """
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        # Convert from BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return img

    def _read_mask(self, path: Path) -> np.ndarray:
        """
        Read binary mask as np.ndarray of shape (H, W), dtype=float32.
        Assumes mask values 0/255 or 0/1 and binarizes them.
        """
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {path}")

        # Binarize: anything > 0 becomes 1
        mask = (mask > 0).astype("float32")
        return mask

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x_path, y_path = self.samples[idx]

        image = self._read_image(x_path)
        mask = self._read_mask(y_path)

        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
