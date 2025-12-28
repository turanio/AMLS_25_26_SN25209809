from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(config):
    """build model function."""
    steps = []

    steps.append(("scaler", StandardScaler()))
    steps.append(
        ("pca", PCA(n_components=config.pca_n_components, random_state=config.seed))
    )

    steps.append(
        ("svm", SVC(kernel="linear", C=config.svm_c, random_state=config.seed))
    )

    return Pipeline(steps)
