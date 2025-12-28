import numpy as np
from A.config import ModelConfig
from utils.base_trainer import BaseTrainer
from A.model import build_model
from A.features import augment_one_per_image, flatten_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelATrainer(BaseTrainer):
    def __init__(self, config: ModelConfig):
        super().__init__(
            model_name="ModelA",
            log_dir="logs/modelA",
            seed=config.seed
        )
        self.config = config

    def _build_model(self):
        self.logger.info("Building Model A: PCA + Linear SVM")
        self.logger.info(f"Config: {self.config}")
        self.model = build_model(self.config)

    def _train(self, data):
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

    def _evaluate(self, data):
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
