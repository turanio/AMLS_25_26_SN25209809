from dataclasses import dataclass


@dataclass
class ModelConfig:
    seed: int = 42

    use_augmentation: bool = True
    pca_n_components: int = 64
    svm_c: float = 1.0
