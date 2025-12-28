import numpy as np
from medmnist import BreastMNIST
from sklearn.model_selection import train_test_split
import os
from typing import Dict

from utils.seed import set_seed
from utils.logger import CustomLogger


class BreastMNISTLoader:
    def __init__(
        self,
        root_dir: str,
        val_split: float,
        test_split: float,
        normalize: bool,
        seed: int,
    ) -> None:
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
        """Loads the BreastMNIST dataset from BreastMNIST if defined directory is empty.
        Otherwise read the file from root_dir.
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
        """Split the datasets into train, val, and test datasets."""
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
