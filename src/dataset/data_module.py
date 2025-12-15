from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset

from src.dataset.dataset import SegmentationDataset, simple_augment


class SegmentationDataModule(LightningDataModule):
    """
    LightningDataModule for SegmentationDataset.

    Parameters
    ----------
    train_indices : Sequence[str | Path]
        List of CSVs with x_path,y_path for training.
    val_indices : Sequence[str | Path], optional
        List of CSVs for validation.
    test_indices : Sequence[str | Path], optional
        List of CSVs for test.
    pred_indices : Sequence[str | Path], optional
        List of CSVs for prediction/inference.
    root_dir : str | Path
        Common root directory for all splits.
    train_augs, val_augs, test_augs, pred_augs : AlbumentationsTransform, optional
        Albumentations pipelines (A.Compose(...)) per split.
    batch_size : int
    num_workers : int
    pin_memory : bool
    persistent_workers : bool | None
    shuffle_train : bool
    drop_last : bool
    """

    def __init__(
        self,
        train_indices: Sequence[str | Path],
        root_dir: str | Path | None = "",
        val_indices: Optional[Sequence[str | Path]] = None,
        test_indices: Optional[Sequence[str | Path]] = None,
        pred_indices: Optional[Sequence[str | Path]] = None,
        train_augs: Optional[AlbumentationsTransform] = simple_augment,
        val_augs: Optional[AlbumentationsTransform] = simple_augment,
        test_augs: Optional[AlbumentationsTransform] = simple_augment,
        pred_augs: Optional[AlbumentationsTransform] = simple_augment,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        shuffle_train: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()

        # store as list[Path]
        self.train_indices = [Path(p) for p in train_indices]
        self.val_indices = (
            [Path(p) for p in val_indices] if val_indices is not None else None
        )
        self.test_indices = (
            [Path(p) for p in test_indices] if test_indices is not None else None
        )
        self.pred_indices = (
            [Path(p) for p in pred_indices] if pred_indices is not None else None
        )

        self.root_dir = Path(root_dir)

        self.train_augs = train_augs
        self.val_augs = val_augs
        self.test_augs = test_augs
        self.pred_augs = pred_augs

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (
            persistent_workers if persistent_workers is not None else (num_workers > 0)
        )
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last

        # Will be set in `setup`
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None
        self.test_dataset: Optional[ConcatDataset] = None
        self.pred_dataset: Optional[ConcatDataset] = None

    def _build_concat_dataset(
        self,
        indices: Sequence[Path],
        augment: Optional[AlbumentationsTransform],
    ) -> ConcatDataset:
        """Create ConcatDataset from multiple CSV index files."""
        datasets = [
            SegmentationDataset(
                dataset_filepath=idx_path,
                root_dir=self.root_dir,
                augment=augment,
            )
            for idx_path in indices
        ]
        if len(datasets) == 1:
            # technically ConcatDataset([ds]) is fine, but this keeps __repr__ nicer
            return ConcatDataset(datasets)
        return ConcatDataset(datasets)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create datasets for different stages.

        stage can be 'fit', 'validate', 'test', 'predict' or None.
        """

        if stage in (None, "fit"):
            self.train_dataset = self._build_concat_dataset(
                indices=self.train_indices,
                augment=self.train_augs,
            )
            if self.val_indices is not None:
                self.val_dataset = self._build_concat_dataset(
                    indices=self.val_indices,
                    augment=self.val_augs,
                )

        if stage in (None, "validate") and self.val_indices is not None:
            if self.val_dataset is None:  # if not already created in "fit"
                self.val_dataset = self._build_concat_dataset(
                    indices=self.val_indices,
                    augment=self.val_augs,
                )

        if stage in (None, "test") and self.test_indices is not None:
            self.test_dataset = self._build_concat_dataset(
                indices=self.test_indices,
                augment=self.test_augs,
            )

        if stage in (None, "predict") and self.pred_indices is not None:
            self.pred_dataset = self._build_concat_dataset(
                indices=self.pred_indices,
                augment=self.pred_augs,
            )

    # ---- Dataloaders ----

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "Train dataset is not initialized. Call `setup('fit')` first."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "Val dataset is not initialized. Provide `val_indices` and call "
                "`setup('fit')` or `setup('validate')`."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "Test dataset is not initialized. Provide `test_indices` and call `setup('test')`."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.pred_dataset is None:
            raise RuntimeError(
                "Predict dataset is not initialized. Provide `pred_indices` and call `setup('predict')`."
            )
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
        )
