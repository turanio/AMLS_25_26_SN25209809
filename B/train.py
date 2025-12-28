import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

from utils.base_trainer import BaseTrainer
from B.model import CNN
from B.transforms import get_train_transforms_with_aug, get_train_transforms_without_aug


class NumpyImageDataset(Dataset):
    """
    Torch Dataset wrapping NumPy images.
    """

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]

        # NumPy to PIL
        img = Image.fromarray((img * 255).astype("uint8"), mode="L")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


class ModelBTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(model_name="ModelB", log_dir="logs/modelB", seed=config.seed)

        self.config = config
        self.device = torch.device(config.device)

    def _build_model(self):
        self.logger.info("Building Model B (CNN)")
        self.model = CNN().to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.criterion = BCEWithLogitsLoss()

    def _train(self, data):
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

    def _evaluate(self, data):
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
