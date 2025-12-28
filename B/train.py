from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from B.config import ModelConfig
from B.model import CNN
from B.transforms import (
    get_train_transforms_with_aug,
    get_train_transforms_without_aug,
)
from utils.base_trainer import BaseTrainer


class NumpyImageDataset(Dataset):
    """PyTorch Dataset wrapper for NumPy image arrays.

    Converts NumPy arrays to PIL images and applies transformations.

    Attributes:
        X: NumPy array of images.
        y: NumPy array of labels.
        transform: Optional torchvision transforms to apply.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            X: NumPy array of images with shape (N, H, W).
            y: NumPy array of labels.
            transform: Optional torchvision transforms to apply.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (transformed_image, label) where the image is a tensor
            and label is a float32 tensor.
        """
        img = self.X[idx]
        label = self.y[idx]

        # NumPy to PIL
        img = Image.fromarray((img * 255).astype("uint8"), mode="L")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


class ModelBTrainer(BaseTrainer):
    """Trainer class for Model B (CNN).

    Manages the complete training workflow including model building,
    data loading, training with PyTorch, and evaluation.

    Attributes:
        config: ModelConfig instance with hyperparameters.
        device: PyTorch device (cpu or cuda).
        model: CNN model instance.
        optimizer: Adam optimizer.
        criterion: BCEWithLogitsLoss for binary classification.
        logger: Logger for training information.
        output_logger: Logger for results output.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the ModelB trainer.

        Args:
            config: ModelConfig instance containing hyperparameters.
        """
        super().__init__(model_name="ModelB", log_dir="logs/modelB", seed=config.seed)

        self.config = config
        self.device = torch.device(config.device)

    def _build_model(self) -> None:
        """Build the CNN model, optimizer, and loss criterion.

        Initializes the CNN architecture and Adam optimizer with the
        configured hyperparameters.
        """
        self.logger.info("Building Model B (CNN)")
        self.model = CNN().to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.criterion = BCEWithLogitsLoss()

    def _train(self, data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Train the CNN model on the provided data.

        Creates DataLoader with optional augmentation and trains the model
        for the configured number of epochs using Adam and BCEWithLogitsLoss.

        Args:
            data: Dictionary containing 'train', 'val', and 'test' splits,
                  where each split is a tuple (X, y).
        """
        X_train, y_train = data["train"]

        transform = (
            get_train_transforms_with_aug()
            if self.config.use_augmentation
            else get_train_transforms_without_aug()
        )

        train_dataset = NumpyImageDataset(X_train, y_train, transform=transform)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Loss={epoch_loss / len(train_loader):.4f}"
            )

    def _evaluate(
        self, data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """Evaluate the trained model on test data.

        Args:
            data: Dictionary containing 'train', 'val', and 'test' splits,
                  where each split is a tuple (X, y).

        Returns:
            Dictionary containing evaluation metrics:
                - accuracy: Classification accuracy
                - precision: Precision score
                - recall: Recall score
                - f1: F1 score
        """
        X_test, y_test = data["test"]

        test_dataset = NumpyImageDataset(
            X_test, y_test, transform=get_train_transforms_without_aug()
        )

        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(labels.numpy())

        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, zero_division=0),
            "recall": recall_score(all_targets, all_preds, zero_division=0),
            "f1": f1_score(all_targets, all_preds, zero_division=0),
        }

        return metrics
