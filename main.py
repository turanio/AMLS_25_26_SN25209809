from data.breastmnist import BreastMNISTLoader
from A.config import ModelConfig as SVMModelConfig
from A.train import ModelATrainer


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


if __name__ == "__main__":
    main()
