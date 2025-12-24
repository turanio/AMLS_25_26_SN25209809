from dataclasses import dataclass


@dataclass
class ModelBConfig:
    seed: int = 42

    # Training
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3

    weight_decay: float = 1e-4
    use_augmentation: bool = True
    device: str = "cpu"
