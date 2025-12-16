"""
Classification algorithms for SNN output evaluation.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def pca_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: Optional[int] = None,
    variance_ratio: Optional[float] = 0.95,
    whiten: bool = True,
    standardize: bool = True,
    max_iter: int = 1000,
) -> Tuple[Dict[str, float], StandardScaler, PCA, LogisticRegression]:
    """
    Reduce high-dimensional features with PCA, then classify with multinomial Logistic Regression.

    Notes:
    - PCA is fit on train only, then applied to val/test.
    - Use either n_components (int) or variance_ratio (0-1) to set dimensionality.
    """
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
    else:
        scaler = StandardScaler(with_mean=False, with_std=False)
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    if n_components is None and variance_ratio is not None:
        pca = PCA(n_components=variance_ratio, svd_solver="full", whiten=whiten)
    elif n_components is not None:
        pca = PCA(n_components=n_components, svd_solver="full", whiten=whiten)
    else:
        pca = PCA(svd_solver="full", whiten=whiten)

    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=max_iter)
    clf.fit(X_train_p, y_train)

    accs = {
        "train": float(accuracy_score(y_train, clf.predict(X_train_p))),
        "val": float(accuracy_score(y_val, clf.predict(X_val_p))),
        "test": float(accuracy_score(y_test, clf.predict(X_test_p))),
    }

    return accs, scaler, pca, clf


def pca_quadratic_discriminant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: Optional[int] = None,
    variance_ratio: Optional[float] = 0.95,
    whiten: bool = True,
    standardize: bool = True,
    reg_param: float = 0.1,
) -> Tuple[Dict[str, float], StandardScaler, PCA, QuadraticDiscriminantAnalysis]:
    """
    Reduce high-dimensional features with PCA, then classify with QDA (Quadratic Discriminant Analysis).
    - Regularization via reg_param helps when classes are few or covariance is near-singular.
    """
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
    else:
        scaler = StandardScaler(with_mean=False, with_std=False)
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    if n_components is None and variance_ratio is not None:
        pca = PCA(n_components=variance_ratio, svd_solver="full", whiten=whiten)
    elif n_components is not None:
        pca = PCA(n_components=n_components, svd_solver="full", whiten=whiten)
    else:
        pca = PCA(svd_solver="full", whiten=whiten)

    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    clf = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    clf.fit(X_train_p, y_train)

    accs = {
        "train": float(accuracy_score(y_train, clf.predict(X_train_p))),
        "val": float(accuracy_score(y_val, clf.predict(X_val_p))),
        "test": float(accuracy_score(y_test, clf.predict(X_test_p))),
    }

    return accs, scaler, pca, clf
