from torchvision import transforms


def get_train_transforms_with_aug() -> transforms.Compose:
    """Get training transforms with data augmentation.

    Applies the following augmentations:
        - Random horizontal flip
        - Random rotation (±15 degrees)
        - Random affine translation (±10%)
        - Converts to tensor
        - Normalizes with mean=0.5, std=0.5

    Returns:
        Composed torchvision transforms.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def get_train_transforms_without_aug() -> transforms.Compose:
    """Get transforms without data augmentation.

    Only applies:
        - Converts to tensor
        - Normalizes with mean=0.5, std=0.5

    Returns:
        Composed torchvision transforms.
    """
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
