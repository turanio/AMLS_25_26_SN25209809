from data.breastmnist import BreastMNISTLoader
from A.config import ModelConfig as SVMModelConfig
from A.train import ModelATrainer
from B.config import ModelBConfig
from B.train import ModelBTrainer


def main():
    loader = BreastMNISTLoader(
        root_dir="datasets",
        val_split=0.1,
        test_split=0.2,
        normalize=True,
        seed=42,
    )
    data = loader.load()

    config = SVMModelConfig(
        seed=42,
        pca_n_components=64,
        svm_c=1.0,
        use_augmentation=True
    )

    trainer = ModelATrainer(config)
    results = trainer.run(data)

    print("Model A results:", results)

    config = ModelBConfig(
        seed=42,
        batch_size=128,
        epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_augmentation=True,
        device="cpu"
    )

    trainer = ModelBTrainer(config)
    results = trainer.run(data)

    print("Model B results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
