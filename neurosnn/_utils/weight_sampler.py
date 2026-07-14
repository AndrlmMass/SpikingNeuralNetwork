import numpy as np


class WeightSampler:
    """
    Tracks a fixed random subset of w_se and w_ee weights across both the
    awake (per-STDP-update) and regularisation (per-reg-step) phases.

    Usage
    -----
    sampler = WeightSampler(N_x=784, N_exc=1024, n_samples=30)

    # Attach all four callbacks before model.train():
    regularizer.record_fn_se  = sampler.record_se
    regularizer.record_fn_ee  = sampler.record_ee

    # Pass awake callbacks directly to model.train():
    model.train(...,
        record_fn_awake_se=sampler.record_awake_se,
        record_fn_awake_ee=sampler.record_awake_ee,
    )

    Attributes
    ----------
    t_awake : list of int
        Simulation timestep for each awake snapshot.
    awake_se, awake_ee : list of np.ndarray
        Snapshots at every STDP weight update during training.
    t_reg : list of int
        Simulation timestep for each reg snapshot.
    reg_se, reg_ee : list of np.ndarray
        Snapshots at every regularisation step (sleep or normalize).
    """

    def __init__(
        self,
        N_x: int,
        N_exc: int,
        n_samples: int = 30,
        seed: int = 0,
    ) -> None:
        self._n = n_samples // 2          # per weight type (SE / EE)
        self._rng = np.random.default_rng(seed)
        self._idx_se: np.ndarray | None = None   # shape (n, 2)
        self._idx_ee: np.ndarray | None = None   # shape (n, 2)

        # Awake-phase samples (one per STDP update)
        self.t_awake: list[int] = []
        self.awake_se: list[np.ndarray] = []
        self.awake_ee: list[np.ndarray] = []
        # Reg-phase samples (one per sleep step or normalize event)
        self.t_reg: list[int] = []
        self.reg_se: list[np.ndarray] = []
        self.reg_ee: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_idx(self, w_slice: np.ndarray) -> np.ndarray:
        nz = np.argwhere(w_slice > 0)
        k = min(self._n, len(nz))
        return nz[self._rng.choice(len(nz), k, replace=False)]

    def _snap(self, w_slice: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return w_slice[idx[:, 0], idx[:, 1]].copy()

    # ------------------------------------------------------------------
    # Awake-phase callbacks  (→ record_fn_awake_se/ee on model.train)
    # ------------------------------------------------------------------

    def record_awake_se(self, w_slice: np.ndarray, t=None) -> None:
        """Called at every STDP weight update with the w_se slice."""
        if self._idx_se is None:
            self._idx_se = self._pick_idx(w_slice)
        self.t_awake.append(t)
        self.awake_se.append(self._snap(w_slice, self._idx_se))

    def record_awake_ee(self, w_slice: np.ndarray, t=None) -> None:
        """Called at every STDP weight update with the w_ee slice."""
        if self._idx_ee is None:
            self._idx_ee = self._pick_idx(w_slice)
        self.awake_ee.append(self._snap(w_slice, self._idx_ee))

    # ------------------------------------------------------------------
    # Reg-phase callbacks  (→ regularizer.record_fn_se/ee)
    # ------------------------------------------------------------------

    def record_se(self, w_slice: np.ndarray, t=None) -> None:
        """Called at every Sleep.step() / Normalizer.step() with the w_se slice."""
        if self._idx_se is None:
            self._idx_se = self._pick_idx(w_slice)
        self.t_reg.append(t)
        self.reg_se.append(self._snap(w_slice, self._idx_se))

    def record_ee(self, w_slice: np.ndarray, t=None) -> None:
        """Called at every Sleep.step() / Normalizer.step() with the w_ee slice."""
        if self._idx_ee is None:
            self._idx_ee = self._pick_idx(w_slice)
        self.reg_ee.append(self._snap(w_slice, self._idx_ee))
