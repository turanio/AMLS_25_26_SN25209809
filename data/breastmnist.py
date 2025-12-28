import os
from typing import Dict

import numpy as np
from medmnist import BreastMNIST
from sklearn.model_selection import train_test_split

from utils.logger import CustomLogger
from utils.seed import set_seed


class BreastMNISTLoader:
    """Loader for the BreastMNIST dataset.

    Downloads (if needed), normalizes, and splits the BreastMNIST dataset
    into training, validation, and test sets with stratification.

    Attributes:
        root_dir: Directory to store/load the dataset.
        val_split: Proportion of data to use for validation.
        test_split: Proportion of data to use for testing.
        normalize: Whether to normalize images to [0, 1] range.
        seed: Random seed for reproducibility.
        logger: Logger instance for dataset operations.
    """

    def __init__(
        self,
        root_dir: str,
        val_split: float,
        test_split: float,
        normalize: bool,
        seed: int,
    ) -> None:
        """Initialize the BreastMNIST loader.

        Args:
            root_dir: Directory to store/load the dataset.
            val_split: Proportion of data for validation (e.g., 0.1 for 10%).
            test_split: Proportion of data for testing (e.g., 0.2 for 20%).
            normalize: Whether to normalize images to [0, 1] range.
            seed: Random seed for reproducible splits.
        """
        self.root_dir = root_dir
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.seed = seed

        self.logger = CustomLogger().get_logger(
            name=self.__class__.__name__, log_file="logs/data_loader.log"
        )

        set_seed(seed)

    def load(self) -> Dict[str, tuple]:
        """Load and split the BreastMNIST dataset.

        Downloads the dataset if not already present in root_dir, then loads,
        optionally normalizes, and splits into train/val/test sets.

        Returns:
            Dictionary with keys 'train', 'val', 'test', where each value is
            a tuple (X, y) of images and labels.
        """
        self.logger.info("Loading BreastMNIST dataset")

        os.makedirs(self.root_dir, exist_ok=True)

        train_data = BreastMNIST(split="train", download=False, root=self.root_dir)

        test_data = BreastMNIST(split="test", download=False, root=self.root_dir)

        X = np.concatenate([train_data.imgs, test_data.imgs]).astype("float32")
        y = np.concatenate([train_data.labels, test_data.labels]).squeeze().astype(int)

        self.logger.info(f"Loaded dataset shape: X={X.shape}, y={y.shape}")

        if self.normalize:
            X /= 255.0
            self.logger.info("Applied normalization to [0,1]")

        return self._split(X, y)

    def _split(self, X, y) -> Dict[str, tuple]:
        """Split the dataset into train, validation, and test sets.

        Performs stratified splitting to maintain class balance across splits.

        Args:
            X: Full dataset of images.
            y: Full dataset of labels.

        Returns:
            Dictionary with keys 'train', 'val', 'test', where each value is
            a tuple (X_split, y_split).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_split, stratify=y, random_state=self.seed
        )

        val_ratio = self.val_split / (1.0 - self.test_split)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_ratio,
            stratify=y_train,
            random_state=self.seed,
        )

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
