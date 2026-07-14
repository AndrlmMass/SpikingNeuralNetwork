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
        """Return eta-squared (eta²) = BCSS / (BCSS + WCSS).

        eta² is the fraction of total representational variance explained by
        class structure.  It is bounded in [0, 1] and, crucially, independent
        of sample size — unlike the Calinski-Harabasz F-statistic that was
        used previously, which grows O(n) and caused test-set phi to be ~10×
        larger than val-set phi simply because the test set is 10× bigger.

        Centroids are always recomputed from the data passed to this call so
        that the metric is fully self-contained and does not depend on which
        training batch happened to be used for the last fit().
        """
        import numpy as np

        present = np.unique(Y).astype(int)

        # Recompute centroids from the current data so the metric is
        # self-contained and does not inherit scale artefacts from a
        # differently-sized fitting batch.
        centroids = np.zeros_like(self.centroids)
        for c in present:
            centroids[c] = X[Y == c].mean(axis=0)

        wcss = sum(np.sum((X[Y == c] - centroids[c]) ** 2) for c in present)

        overall_mean = X.mean(axis=0)
        bcss = sum(
            np.sum(Y == c) * np.sum((centroids[c] - overall_mean) ** 2)
            for c in present
        )

        total_ss = bcss + wcss
        if total_ss <= self.eps:
            return 0.0
        return bcss / total_ss


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
        seed: int = 0,
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
                penalty="l1", solver="saga", max_iter=max_iter, n_jobs=n_jobs,
                random_state=seed,
            )
            if do_LR
            else None
        )
        self.phi = Phi(num_classes=num_classes) if do_phi else None

    def fit(self, X, Y):
        import numpy as np

        self._fit_ok = False

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
            try:
                X = self.pca.fit_transform(X)
            except Exception as e:
                print(f"[Evaluator] PCA fit failed ({type(e).__name__}: {e}) — skipping this evaluation step.")
                return

        if self.phi:
            self.phi.fit(X, Y)

        if self.clf:
            self.clf.fit(X, Y)

        self._fit_ok = True

    def score(self, X, Y):
        import numpy as np
        from sklearn.metrics import accuracy_score

        if not getattr(self, "_fit_ok", False):
            return None, None

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
