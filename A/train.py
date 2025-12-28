from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from A.config import ModelConfig
from A.features import augment_one_per_image, flatten_features
from A.model import build_model
from utils.base_trainer import BaseTrainer


class ModelATrainer(BaseTrainer):
    """Trainer class for Model A (PCA + Linear SVM).

    Manages the complete training workflow including model building,
    data augmentation, training, and evaluation.

    Attributes:
        config: ModelConfig instance with hyperparameters.
        model: Scikit-learn pipeline containing the trained model.
        logger: Logger for training information.
        output_logger: Logger for results output.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the ModelA trainer.

        Args:
            config: ModelConfig instance containing hyperparameters.
        """
        super().__init__(model_name="ModelA", log_dir="logs/modelA", seed=config.seed)
        self.config = config

    def _build_model(self) -> None:
        """Build the PCA + SVM pipeline.

        Constructs the model using the configuration parameters.
        """
        self.logger.info("Building Model A: PCA + Linear SVM")
        self.logger.info(f"Config: {self.config}")
        self.model = build_model(self.config)

    def _train(self, data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Train the SVM model on the provided data.

        Applies optional augmentation, flattens images, and trains the pipeline.

        Args:
            data: Dictionary containing 'train', 'val', and 'test' splits,
                  where each split is a tuple (X, y).
        """
        X_train, y_train = data["train"]

        if self.config.use_augmentation:
            self.logger.info("Applying augmentation")
            X_train, y_train = augment_one_per_image(
                X_train, y_train, seed=self.config.seed
            )

        X_train = flatten_features(X_train)
        y_train = np.asarray(y_train).astype(int).ravel()

        self.logger.info(f"Train shape after augmentation: {X_train.shape}")
        self.model.fit(X_train, y_train)

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

        X_test = flatten_features(X_test)
        y_test = np.asarray(y_test).astype(int).ravel()

        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        return metrics
