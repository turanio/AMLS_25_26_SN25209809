from A.config import ModelConfig as SVMModelConfig
from A.train import ModelATrainer
from B.config import ModelConfig as CNNModelConfig
from B.train import ModelBTrainer
from data.breastmnist import BreastMNISTLoader


def main() -> None:
    """Load data and train both models.

    Configures and runs training for:
    1. Model A: PCA + Linear SVM with data augmentation
    2. Model B: Convolutional Neural Network with data augmentation

    The BreastMNIST dataset is loaded with 10% validation split and 20% test split.
    Results are logged to separate directories for each model.
    """
    loader = BreastMNISTLoader(
        root_dir="Datasets",
        val_split=0.1,
        test_split=0.2,
        normalize=True,
        seed=42,
        download=False,
    )
    data = loader.load()

    config = SVMModelConfig(
        seed=42, pca_n_components=64, svm_c=1.0, use_augmentation=True
    )

    svm_trainer = ModelATrainer(config)
    svm_trainer.run(data)

    config = CNNModelConfig(
        seed=42,
        batch_size=128,
        epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_augmentation=True,
        device="cpu",
    )

    cnn_trainer = ModelBTrainer(config)
    cnn_trainer.run(data)


if __name__ == "__main__":
    main()
