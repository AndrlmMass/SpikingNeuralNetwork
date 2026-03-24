class Phi:
    def __init__(self, num_classes: int, eps: float = 1e-10):
        self.num_classes = num_classes
        self.eps = eps
        self.centroids = None  # (num_classes, n_components), frozen after fit

    def fit(self, X, Y):
        import numpy as np

        present = np.unique(Y).astype(int)
        n_components = X.shape[1]

        # freeze centroids from training data
        self.centroids = np.zeros((self.num_classes, n_components))
        for c in present:
            self.centroids[c] = X[Y == c].mean(axis=0)

    def score(self, X, Y):
        import numpy as np

        assert self.centroids is not None, "Call fit() before score()"
        n = X.shape[0]
        present = np.unique(Y).astype(int)
        k = present.size

        # WCSS: sum of squared distances from each point to its class centroid
        wcss = sum(np.sum((X[Y == c] - self.centroids[c]) ** 2) for c in present)

        # BCSS: weighted sum of squared distances from each centroid to overall mean
        overall_mean = X.mean(axis=0)
        bcss = sum(
            np.sum(Y == c) * np.sum((self.centroids[c] - overall_mean) ** 2)
            for c in present
        )

        # Calinski-Harabasz: (BCSS / k-1) / (WCSS / n-k)
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
        from sklearn.linear_model import LogisticRegressionCV

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
            LogisticRegressionCV(
                penalty="l1", solver="saga", max_iter=max_iter, cv=cv, n_jobs=n_jobs
            )
            if do_LR
            else None
        )
        self.phi = Phi(num_classes=num_classes) if do_phi else None

    def fit(self, X, Y):
        # scale input
        X = self.scaler.fit_transform(X)

        # reduce dimensionality if do_pca
        if self.pca:
            X = self.pca.fit_transform(X)

        # estimate clustering ability of the network
        if self.phi:
            self.phi.fit(X, Y)

        # fit logistic regression
        if self.clf:
            self.clf.fit(X, Y)

    def score(self, X, Y):
        from sklearn.metrics import accuracy_score

        # scale input
        X = self.scaler.transform(X)

        # reduce dimensionality
        if self.pca:
            X = self.pca.transform(X)

        # compute accuracy
        if self.clf:
            acc = accuracy_score(Y, self.clf.predict(X))
        else:
            acc = None

        # compute phi
        if self.phi:
            phi_score = self.phi.score(X, Y)
        else:
            phi_score = None

        return acc, phi_score
