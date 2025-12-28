from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration parameters for Model A.

    Attributes:
        seed: Random seed for reproducibility.
        use_augmentation: Whether to apply data augmentation during training.
        pca_n_components: Number of principal components to retain in PCA.
        svm_c: Regularization parameter for the SVM classifier.
    """

    seed: int = 42
    use_augmentation: bool = True
    pca_n_components: int = 64
    svm_c: float = 1.0
