import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=max_iter
    )
    clf.fit(X_train_p, y_train)

    accs = {
        "train": float(accuracy_score(y_train, clf.predict(X_train_p))),
        "val": float(accuracy_score(y_val, clf.predict(X_val_p))),
        "test": float(accuracy_score(y_test, clf.predict(X_test_p))),
    }

    return accs, scaler, pca, clf


if __name__ == "__main__":
    # Minimal example using existing combined features if desired.
    # Replace with your high-dimensional spiking arrays (flattened time×neurons) as needed.
    try:
        from multiple_log_reg.get_data import get_data

        (
            _img_tr,
            _img_tr_y,
            _img_v,
            _img_v_y,
            _img_te,
            _img_te_y,
            _aud_tr,
            _aud_tr_y,
            _aud_v,
            _aud_v_y,
            _aud_te,
            _aud_te_y,
            comb_tr,
            comb_tr_y,
            comb_v,
            comb_v_y,
            comb_te,
            comb_te_y,
        ) = get_data(imageMNIST=True, audioMNIST=True, combined=True)

        if comb_tr is not None:
            accs, _, _, _ = pca_logistic_regression(
                comb_tr,
                comb_tr_y,
                comb_v,
                comb_v_y,
                comb_te,
                comb_te_y,
                variance_ratio=0.95,
            )
            print("PCA+LogReg on combined ->", accs)
    except Exception as e:
        print("Example skipped:", e)
