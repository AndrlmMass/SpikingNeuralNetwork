from neurosnn._evaluation.analysis import eta_squared, group_eta_squared


class Phi:
    """Representational clustering quality = multivariate eta² (BCSS/(BCSS+WCSS)):
    the fraction of total representational variance explained by class structure,
    bounded in [0, 1] and independent of sample size.

    When ``group_assignment`` is provided (grouped WTA architectures), eta² is
    measured on the group-pooled mean rates — the space the group readout uses —
    so it tracks accuracy. On raw sparse WTA rates eta² anti-correlates with
    accuracy: stochastic per-sample winners inflate within-class scatter even
    while the class code stays linearly separable, which is why group pooling is
    the correct measurement space.

    eta² recomputes centroids from the data passed to ``score`` (so it is
    self-contained and sample-size independent); ``fit`` is a no-op kept for the
    Evaluator's fit/score API.
    """

    def __init__(self, num_classes: int, group_assignment=None, eps: float = 1e-10):
        self.num_classes = num_classes
        self.eps = eps
        self.group_assignment = group_assignment
        if group_assignment is not None:
            import numpy as np
            self.n_groups = int(np.asarray(group_assignment).max()) + 1
        else:
            self.n_groups = num_classes

    def fit(self, X, Y):
        return self

    def score(self, X, Y):
        import numpy as np

        X = np.asarray(X)
        if self.group_assignment is not None:
            return group_eta_squared(X, Y, self.group_assignment, self.n_groups)
        return eta_squared(X, Y)


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
        group_assignment=None,
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
        self.phi = (
            Phi(num_classes=num_classes, group_assignment=group_assignment)
            if do_phi else None
        )

    def fit(self, X, Y):
        import numpy as np

        self._fit_ok = False

        # phi is measured on the raw per-neuron rates (group pooling needs the
        # neuron->group map intact); the classifier uses the scaled/PCA features.
        X_raw = np.asarray(X)
        Xt = self.scaler.fit_transform(X_raw)
        Xt = np.nan_to_num(Xt, nan=0.0)

        if self.pca:
            max_comps = min(Xt.shape[0], Xt.shape[1])
            if self.pca.n_components > max_comps:
                from sklearn.decomposition import PCA
                self.pca = PCA(
                    n_components=max_comps,
                    svd_solver=self.pca.svd_solver,
                    whiten=self.pca.whiten,
                )
            try:
                Xt = self.pca.fit_transform(Xt)
            except Exception as e:
                print(f"[Evaluator] PCA fit failed ({type(e).__name__}: {e}) — skipping this evaluation step.")
                return

        if self.phi:
            self.phi.fit(X_raw, Y)

        if self.clf:
            self.clf.fit(Xt, Y)

        self._fit_ok = True

    def score(self, X, Y):
        import numpy as np
        from sklearn.metrics import accuracy_score

        if not getattr(self, "_fit_ok", False):
            return None, None

        X_raw = np.asarray(X)
        Xt = self.scaler.transform(X_raw)
        Xt = np.nan_to_num(Xt, nan=0.0)

        if self.pca:
            Xt = self.pca.transform(Xt)
            Xt = np.nan_to_num(Xt, nan=0.0)

        if self.clf:
            acc = accuracy_score(Y, self.clf.predict(Xt))
        else:
            acc = None

        if self.phi:
            phi_score = self.phi.score(X_raw, Y)
        else:
            phi_score = None

        return acc, phi_score
