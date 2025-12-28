from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from A.config import ModelConfig


def build_model(config: ModelConfig) -> Pipeline:
    """Build a scikit-learn pipeline for PCA + Linear SVM classification.

    The pipeline consists of:
    1. StandardScaler for feature normalization
    2. PCA for dimensionality reduction
    3. Linear SVM for binary classification

    Args:
        config: ModelConfig instance containing hyperparameters.

    Returns:
        A scikit-learn Pipeline object ready for training.
    """
    steps = []

    steps.append(("scaler", StandardScaler()))
    steps.append(
        ("pca", PCA(n_components=config.pca_n_components, random_state=config.seed))
    )

    steps.append(
        ("svm", SVC(kernel="linear", C=config.svm_c, random_state=config.seed))
    )

    return Pipeline(steps)
