import numpy as np
import random
from scipy.ndimage import rotate
from sklearn.utils import shuffle


def augment_one_per_image(X, y, seed=42):
    """
    Apply exactly ONE random augmentation per image:
    - rotation (±10 degrees)
    - horizontal flip
    - Gaussian noise (o = 0.02)

    This matches the experimental procedure used in testing.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    X = np.asarray(X)
    y = np.asarray(y)

    X_aug = []
    y_aug = []

    for img, label in zip(X, y):
        choice = random.choice(["rotate", "flip", "noise"])
        augmented = img.copy()

        if choice == "rotate":
            angle = rng.uniform(-10, 10)
            augmented = rotate(
                augmented,
                angle,
                reshape=False,
                mode="nearest"
            )

        elif choice == "flip":
            augmented = np.fliplr(augmented)

        elif choice == "noise":
            noise = rng.normal(0, 0.02, img.shape)
            augmented = augmented + noise
            augmented = np.clip(augmented, 0.0, 1.0)

        X_aug.append(augmented)
        y_aug.append(label)

    X_out = np.concatenate([X, np.array(X_aug)])
    y_out = np.concatenate([y, np.array(y_aug)])

    return shuffle(X_out, y_out, random_state=seed)


def flatten_features(X):
    """
    Flatten (N, 28, 28) -> (N, 784)
    """
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1)
