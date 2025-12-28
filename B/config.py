from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration parameters for Model B.

    Attributes:
        seed: Random seed for reproducibility.
        batch_size: Number of samples per training batch.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the Adam optimizer.
        weight_decay: L2 regularization coefficient.
        use_augmentation: Whether to apply data augmentation during training.
        device: Device to use for training ('cpu' or 'cuda').
    """

    seed: int = 42

    # Training
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3

    weight_decay: float = 1e-4
    use_augmentation: bool = True
    device: str = "cpu"
