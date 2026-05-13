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
        # scale input
        X = self.scaler.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

        # reduce dimensionality if do_pca
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

        # estimate clustering ability of the network
        if self.phi:
            self.phi.fit(X, Y)

        # fit logistic regression
        if self.clf:
            self.clf.fit(X, Y)

    def score(self, X, Y):
        import numpy as np
        from sklearn.metrics import accuracy_score

        # scale input
        X = self.scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0)

        # reduce dimensionality
        if self.pca:
            X = self.pca.transform(X)
            X = np.nan_to_num(X, nan=0.0)

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


# old code for getting elite nodes
def get_elite_nodes_wta(spikes, labels, num_classes, min_total_spikes=10):
    import numpy as np

    mask = (labels >= 0) & (labels < num_classes)
    spikes = spikes[mask]
    labels = labels[mask]

    N = spikes.shape[1]
    # top_k = int(N * narrow_top)

    responses = np.zeros((N, num_classes), dtype=float)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if idx.size > 0:
            responses[:, c] = spikes[idx].sum(axis=0)

    total = responses.sum(axis=1)
    score = np.full_like(responses, -np.inf)
    valid = total >= min_total_spikes
    score[valid] = responses[valid] / total[valid, None]

    pref = np.full(N, -1, dtype=int)
    pref[valid] = np.argmax(score[valid], axis=1)

    return pref, score


#### same here ####
def zscore_vote(
    spikes, pref, baseline_mu, baseline_sigma, num_classes, eps=1e-8, mu_min=1e-3
):
    import numpy as np

    """
    spikes: (B, N) block-reduced spikes
    pref: (N,) neuron->class assignment
    baseline_mu: (N,) mean firing from fit snippet
    baseline_sigma: (N,) std firing from fit snip    # 4) class activations
    acts = np.zeros((B, num_classes), dtype=float)pet
    """
    valid_neurons = baseline_mu >= mu_min

    # Avoid divide-by-zero
    sigma = np.maximum(baseline_sigma, eps)

    z = np.zeros_like(spikes, dtype=float)
    z[:, valid_neurons] = (spikes[:, valid_neurons] - baseline_mu[valid_neurons]) / (
        sigma[valid_neurons] + eps
    )
    z = np.maximum(z, 0.0)

    # 4) class activations
    acts = np.zeros((spikes.shape[0], num_classes), dtype=float)

    for c in range(num_classes):
        idx = np.where((pref == c) & valid_neurons)[0]
        if idx.size:
            acts[:, c] = z[:, idx].mean(axis=1)

    pred = np.argmax(acts, axis=1)
    return pred, acts
