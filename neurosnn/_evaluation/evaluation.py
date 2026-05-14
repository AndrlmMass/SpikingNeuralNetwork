class Phi:
    def __init__(self, num_classes: int, eps: float = 1e-10):
        self.num_classes = num_classes
        self.eps = eps
        self.centroids = None

    def fit(self, X, Y):
        import numpy as np

        present = np.unique(Y).astype(int)
        n_components = X.shape[1]

        self.centroids = np.zeros((self.num_classes, n_components))
        for c in present:
            self.centroids[c] = X[Y == c].mean(axis=0)

    def score(self, X, Y):
        import numpy as np

        assert self.centroids is not None, "Call fit() before score()"
        n = X.shape[0]
        present = np.unique(Y).astype(int)
        k = present.size

        wcss = sum(np.sum((X[Y == c] - self.centroids[c]) ** 2) for c in present)

        overall_mean = X.mean(axis=0)
        bcss = sum(
            np.sum(Y == c) * np.sum((self.centroids[c] - overall_mean) ** 2)
            for c in present
        )

        if wcss <= self.eps:
            return 0.0
        return (bcss / max(self.eps, k - 1)) / (wcss / max(self.eps, n - k))


class Evaluator:
    def __init__(
        self,
        xp_var_or_comps,
        whiten: bool = True,
        standardize: bool = True,
        max_iter: int = 1000,
        cv: int = 5,
        n_jobs: int = -1,
        num_classes: int = 10,
        do_phi: bool = True,
        do_LR: bool = True,
        do_pca: bool = True,
    ):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        self.fig = None
        self.ax = None
        self.do_phi = do_phi
        self.do_LR = do_LR
        self.scaler = StandardScaler(with_mean=standardize, with_std=standardize)
        self.pca = (
            PCA(n_components=xp_var_or_comps, svd_solver="full", whiten=whiten)
            if do_pca
            else None
        )
        self.clf = (
            LogisticRegression(
                penalty="l1", solver="saga", max_iter=max_iter, n_jobs=n_jobs
            )
            if do_LR
            else None
        )
        self.phi = Phi(num_classes=num_classes) if do_phi else None

    def fit(self, X, Y):
        import numpy as np

        X = self.scaler.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

        if self.pca:
            max_comps = min(X.shape[0], X.shape[1])
            if self.pca.n_components > max_comps:
                from sklearn.decomposition import PCA
                self.pca = PCA(
                    n_components=max_comps,
                    svd_solver=self.pca.svd_solver,
                    whiten=self.pca.whiten,
                )
            X = self.pca.fit_transform(X)

        if self.phi:
            self.phi.fit(X, Y)

        if self.clf:
            self.clf.fit(X, Y)

    def score(self, X, Y):
        import numpy as np
        from sklearn.metrics import accuracy_score

        X = self.scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0)

        if self.pca:
            X = self.pca.transform(X)
            X = np.nan_to_num(X, nan=0.0)

        if self.clf:
            acc = accuracy_score(Y, self.clf.predict(X))
        else:
            acc = None

        if self.phi:
            phi_score = self.phi.score(X, Y)
        else:
            phi_score = None

        return acc, phi_score
