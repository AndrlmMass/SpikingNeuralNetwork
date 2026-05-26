import matplotlib.pyplot as plt
import numpy as np
import os


def create_3D_weights_plot(weights, title, x_label, y_label, axis_flip, H_, W_):
    total_input = weights.sum(axis=axis_flip)
    Z = total_input.reshape(H_, W_)
    x = np.arange(W_)
    y = np.arange(H_)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()
    return fig, ax


def plot_weights_individual(weights, H_, W_, N, title, dir):
    if dir == "outgoing":
        for j in range(N // 2, N):
            rf = weights[j, :].reshape(H_, W_)
            plt.imshow(rf)
            plt.title(f"{title} {j}")
            plt.colorbar()
            plt.show()
            if input("Press Enter to continue...") == "q":
                break
    elif dir == "incoming":
        for j in range(N // 2, N):
            rf = weights[:, j].reshape(H_, W_)
            plt.imshow(rf)
            plt.title(f"{title} {j}")
            plt.colorbar()
            plt.show()
            if input("Press Enter to continue...") == "q":
                break


def plot_single_neuron_weights(weights, st, ex, H, W, H_e, W_e, id_=None):
    for i in range(st, ex):
        if id_ is not None:
            id = id_ + i - st
        else:
            id = i
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        w_se = weights[:st, id].reshape(H, W)
        w_ee = weights[id, st:ex].reshape(H_e, W_e)
        ih_id = np.argmax(weights[id, ex:]) + ex
        w_ie = weights[st:ex, ih_id].reshape(H_e, W_e)
        w_ei = np.abs(weights[ih_id, st:ex]).reshape(H_e, W_e)

        im1 = ax1.imshow(w_se)
        im2 = ax2.imshow(w_ee)
        im3 = ax4.imshow(w_ei)
        im4 = ax3.imshow(w_ie)

        for ax, im in zip([ax1, ax2, ax3, ax4], [im1, im2, im3, im4]):
            fig.colorbar(im, ax=ax)

        fig.suptitle(f"Incoming and outgoing weights for exc: {id}")
        ax1.set_title("Stimulation to Excitatory")
        ax2.set_title("Excitatory to Excitatory")
        ax3.set_title("Excitatory to Inhibitory")
        ax4.set_title("Inhibitory to Excitatory")
        plt.show()

        id = None
        cont = input("Continue?")
        if cont != "":
            break


def plot_weight_evolution_during_sleep_epoch(weight_tracking_epoch, epoch):
    import matplotlib.pyplot as plt
    import shutil
    from matplotlib.lines import Line2D

    times = np.array(weight_tracking_epoch["times"])
    exc_mean = np.array(weight_tracking_epoch["exc_mean"])
    exc_samples = np.array(weight_tracking_epoch["exc_samples"])
    inh_mean = np.array(weight_tracking_epoch["inh_mean"])
    inh_samples = np.array(weight_tracking_epoch["inh_samples"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(times, exc_mean, color="black", linestyle="-", linewidth=2.0, label="Exc mean")
    if exc_samples.ndim == 2 and exc_samples.shape[0] == times.shape[0]:
        for i in range(min(10, exc_samples.shape[1])):
            ax.plot(times, exc_samples[:, i], color="black", alpha=0.6, linewidth=0.8, linestyle="-")
    ax.set_ylabel("Exc weight")

    ax = axes[1]
    ax.plot(times, inh_mean, color="black", linestyle="--", linewidth=2.0, label="Inh mean")
    if inh_samples.ndim == 2 and inh_samples.shape[0] == times.shape[0]:
        for i in range(min(10, inh_samples.shape[1])):
            ax.plot(times, inh_samples[:, i], color="black", alpha=0.6, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Inh weight")
    ax.set_xlabel("Sleep time (ms)")

    os.makedirs("figures", exist_ok=True)
    pdf_path = os.path.join("plots", f"weight_sleep_epoch_{epoch:03d}.pdf")
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight", dpi=900)
    plt.close(fig)


def plot_weight_evolution(
    weight_evolution: dict, output_path: str = "plots/weights_evolution.pdf"
):
    epochs = np.array(weight_evolution.get("epochs", []), dtype=float)
    if epochs.size == 0:
        raise ValueError("weight_evolution contains no epochs to plot")

    exc_mean = np.array(weight_evolution.get("exc_mean", []), dtype=float)
    exc_min = np.array(weight_evolution.get("exc_min", []), dtype=float)
    exc_max = np.array(weight_evolution.get("exc_max", []), dtype=float)

    inh_mean = np.array(weight_evolution.get("inh_mean", []), dtype=float)
    inh_min = np.array(weight_evolution.get("inh_min", []), dtype=float)
    inh_max = np.array(weight_evolution.get("inh_max", []), dtype=float)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(epochs, exc_mean, label="Exc Mean", linewidth=2, linestyle="-", color="black")
    ax.fill_between(epochs, exc_min, exc_max, facecolor="none", hatch="//",
                    edgecolor="black", label="Exc min/max")
    ax.plot(epochs, inh_mean, label="Inh Mean", linewidth=2, linestyle="--", color="black")
    ax.fill_between(epochs, inh_min, inh_max, facecolor="none", hatch="\\\\",
                    edgecolor="black", label="Inh min/max")

    ax.set_xlabel("Epoch", fontsize=26)
    ax.set_ylabel("Weight", fontsize=26)
    ax.set_xticks(fontsize=16)
    ax.set_yticks(fontsize=16)
    legend = ax.legend(facecolor="white", edgecolor="black", fontsize=22,
                       loc="lower left", framealpha=1.0)
    legend.get_frame().set_alpha(1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=900)
    plt.close(fig)


def plot_weight_evolution_during_sleep(weight_tracking_sleep):
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    if len(weight_tracking_sleep["exc_mean"]) == 0:
        print("Warning: No sleep weight tracking data available")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    sleep_times = np.array(weight_tracking_sleep["times"])
    sleep_exc = np.array(weight_tracking_sleep["exc_mean"])

    try:
        sleep_exc_sm = uniform_filter1d(sleep_exc, size=5)
    except Exception:
        sleep_exc_sm = sleep_exc
    ax.plot(sleep_times, sleep_exc_sm, color="#ff7f68", label="Exc mean (smoothed)")
    if "exc_std" in weight_tracking_sleep:
        exc_std = np.array(weight_tracking_sleep["exc_std"])
        try:
            exc_std_sm = uniform_filter1d(exc_std, size=5)
        except Exception:
            exc_std_sm = exc_std
        ax.fill_between(sleep_times, sleep_exc_sm - exc_std_sm, sleep_exc_sm + exc_std_sm,
                        color="#ffe5e1", alpha=0.6)
    ax.set_ylabel("Exc weight")

    ax = axes[1]
    sleep_inh = np.array(weight_tracking_sleep["inh_mean"])
    try:
        sleep_inh_sm = uniform_filter1d(sleep_inh, size=5)
    except Exception:
        sleep_inh_sm = sleep_inh
    ax.plot(sleep_times, sleep_inh_sm, color="#05af9b", label="Inh mean (smoothed)")
    if "inh_std" in weight_tracking_sleep:
        inh_std = np.array(weight_tracking_sleep["inh_std"])
        try:
            inh_std_sm = uniform_filter1d(inh_std, size=5)
        except Exception:
            inh_std_sm = inh_std
        ax.fill_between(sleep_times, sleep_inh_sm - inh_std_sm, sleep_inh_sm + inh_std_sm,
                        color="#c7fdf7", alpha=0.6)
    ax.set_ylabel("Inh weight")
    ax.set_xlabel("Sleep time (ms)")
    ax.set_xticks(fontsize=16)
    ax.set_yticks(fontsize=16)

    os.makedirs("figures", exist_ok=True)
    save_path = os.path.join("figures", "weight_sleep_all.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_weight_trajectories_with_sleep_epoch(weight_tracking_epoch, max_lines=8):
    import matplotlib.pyplot as plt

    times = np.array(weight_tracking_epoch.get("times", []), dtype=float)
    if times.size == 0:
        return

    exc_samples = weight_tracking_epoch.get("exc_samples", [])
    inh_samples = weight_tracking_epoch.get("inh_samples", [])
    exc_mean = np.array(weight_tracking_epoch.get("exc_mean", []), dtype=float)
    inh_mean = np.array(weight_tracking_epoch.get("inh_mean", []), dtype=float)
    sleep_segments = weight_tracking_epoch.get("sleep_segments", [])

    def _to_array(list_of_lists):
        if not list_of_lists:
            return np.zeros((0, 0), dtype=float)
        max_len = max(len(row) for row in list_of_lists)
        if max_len == 0:
            return np.zeros((len(list_of_lists), 0), dtype=float)
        arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
        for r, row in enumerate(list_of_lists):
            k = min(max_len, len(row))
            if k > 0:
                arr[r, :k] = np.array(row[:k], dtype=float)
        return arr

    exc_arr = _to_array(exc_samples)
    inh_arr = _to_array(inh_samples)

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for s, e in sleep_segments:
        try:
            patch = ax.axvspan(float(s), float(e), facecolor="0.92", edgecolor="0.92", alpha=1.0, zorder=0)
            try:
                patch.set_hatch("//")
            except Exception:
                pass
        except Exception:
            pass

    if exc_arr.size > 0 and exc_arr.shape[0] == times.size:
        lines_to_plot = min(max_lines, exc_arr.shape[1])
        for i in range(lines_to_plot):
            ax.plot(times, exc_arr[:, i], color="black", alpha=0.6, linewidth=0.9, linestyle="-")
    if exc_mean.size == times.size:
        ax.plot(times, exc_mean, color="black", linestyle="-", linewidth=2.0, label="Exc mean")

    if inh_arr.size > 0 and inh_arr.shape[0] == times.size:
        lines_to_plot = min(max_lines, inh_arr.shape[1])
        for i in range(lines_to_plot):
            ax.plot(times, inh_arr[:, i], color="black", alpha=0.6, linewidth=0.9, linestyle="--")
    if inh_mean.size == times.size:
        ax.plot(times, inh_mean, color="black", linestyle="--", linewidth=2.0, label="Inh mean")


def weights_plot(weights_exc, weights_inh):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    mu_weights_exc = np.reshape(weights_exc, (weights_exc.shape[0], -1))
    mu_weights_inh = np.reshape(weights_inh, (weights_inh.shape[0], -1))

    idx_exc = np.where(np.sum(mu_weights_exc, axis=0) == 0)
    mu_weights_exc[:, idx_exc[0]] = None

    idx_inh = np.where(np.sum(mu_weights_inh, axis=0) == 0)
    mu_weights_inh[:, idx_inh[0]] = None

    cmap_exc = plt.get_cmap("autumn")
    cmap_inh = plt.get_cmap("winter")

    for i in range(mu_weights_exc.shape[1]):
        axs[0].plot(mu_weights_exc[:, i], color=cmap_exc(i / mu_weights_exc.shape[1]), alpha=0.7)
    for i in range(mu_weights_inh.shape[1]):
        axs[0].plot(mu_weights_inh[:, i], color=cmap_inh(i / mu_weights_inh.shape[1]), alpha=0.7)

    sum_weights_exc = np.nansum(mu_weights_exc, axis=1)
    sum_weights_inh = np.nansum(mu_weights_inh, axis=1)
    axs[1].plot(sum_weights_exc, color="red", label="excitatory")
    axs[1].plot(sum_weights_inh, color="blue", label="inhibitory")

    axs[0].set_title("Weight Evolution Over Time (Individual Neurons)")
    axs[1].set_title("Total Weight Evolution Over Time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Synaptic Weight")
    plt.show()


def _reg_episodes(t_reg: list, gap_threshold: int = 200):
    """Cluster consecutive reg timesteps into (start, end) episode pairs.

    Two adjacent reg events belong to the same episode when the gap between
    them is ≤ gap_threshold timesteps.  For a single normalise event the
    episode spans just that one timestep.
    """
    if not t_reg:
        return []
    # Filter out None entries (should not happen in normal usage)
    ts = [v for v in t_reg if v is not None]
    if not ts:
        return []
    episodes = []
    start = ts[0]
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > gap_threshold:
            episodes.append((start, ts[i - 1]))
            start = ts[i]
    episodes.append((start, ts[-1]))
    return episodes


def _rf_elongation(patch: np.ndarray) -> float:
    """Eigenvalue ratio λ1/λ2 of the RF's spatial weight covariance.

    Returns 1.0 for a circular/degenerate RF, higher values for elongated ones.
    """
    w = patch.ravel().astype(np.float64)
    total = w.sum()
    if total < 1e-10:
        return 1.0
    w = w / total
    H, W = patch.shape
    py, px = np.mgrid[0:H, 0:W]
    coords = np.stack([px.ravel().astype(np.float64), py.ravel().astype(np.float64)], axis=1)
    mu = (w[:, None] * coords).sum(axis=0)
    diffs = coords - mu
    cov = (w[:, None] * diffs).T @ diffs
    lam = np.linalg.eigvalsh(cov)  # ascending order
    return float(lam[1] / lam[0]) if lam[0] > 1e-10 else 1.0


def plot_oriented_rf_summary(
    W_se: np.ndarray,
    input_size: int,
    n_orientations: int,
    out_path: str,
    n_show: int = 256,
) -> None:
    """Diagnostic figure for oriented elliptical RF weights.

    Panel 1: tiled RF patches sorted by orientation group.
    Panel 2: polar rose diagram of mean elongation (λ1/λ2) per group.

    Parameters
    ----------
    W_se          Weight matrix slice, shape (N_x, N_exc).
    input_size    Square root of N_x (input image side length in pixels).
    n_orientations  Number of orientation bins used during construction.
    out_path      Save path (.pdf or .png).
    n_show        Target number of RFs to tile; rounded to side² ≤ N_exc.
    """
    import matplotlib.gridspec as gridspec

    _, N_exc = W_se.shape

    # Compute side so that side² ≤ min(n_show, N_exc)
    side = int(np.floor(np.sqrt(min(n_show, N_exc))))
    n_show = side * side
    quota = n_show // n_orientations  # neurons per orientation group in tile

    # Select neurons: equal quota per orientation, in construction order
    selected = []
    for g in range(n_orientations):
        indices = [i for i in range(N_exc) if i % n_orientations == g]
        selected.extend(indices[:quota])

    # Compute elongation for every neuron
    all_elongations = np.array([
        _rf_elongation(W_se[:, i].reshape(input_size, input_size))
        for i in range(N_exc)
    ])

    # Mean elongation per orientation group
    group_elongation = []
    for g in range(n_orientations):
        idx = [i for i in range(N_exc) if i % n_orientations == g]
        group_elongation.append(float(np.mean(all_elongations[idx])))

    # ---- Build tile canvas (side × side patches) ----
    canvas = np.zeros((side * input_size, side * input_size), dtype=np.float32)
    for pos, neuron_idx in enumerate(selected):
        r = (pos // side) * input_size
        c = (pos % side) * input_size
        canvas[r:r + input_size, c:c + input_size] = W_se[:, neuron_idx].reshape(
            input_size, input_size
        )

    rows_per_group = max(1, quota // side)
    orientations_deg = [f"{int(round(180 * g / n_orientations))}°" for g in range(n_orientations)]

    # ---- Figure ----
    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig, wspace=0.3)
    ax_tiles = fig.add_subplot(gs[0])
    ax_polar = fig.add_subplot(gs[1], projection="polar")

    # Panel 1: tile grid
    vmax = float(np.percentile(canvas[canvas > 0], 99)) if (canvas > 0).any() else 1.0
    ax_tiles.imshow(canvas, cmap="viridis", vmin=0, vmax=vmax,
                    aspect="auto", interpolation="nearest")
    ax_tiles.set_title("W_se receptive fields (sorted by orientation group)", fontsize=12)
    ax_tiles.set_xlabel(f"{quota} neurons per group  →", fontsize=10)

    # Thin per-neuron grid lines (one box per RF patch)
    for i in range(1, side):
        ax_tiles.axvline(i * input_size - 0.5, color="white", linewidth=0.4, alpha=0.4)
        ax_tiles.axhline(i * input_size - 0.5, color="white", linewidth=0.4, alpha=0.4)

    # Thicker separator lines between orientation groups (on top of per-neuron grid)
    for g in range(1, n_orientations):
        y = g * rows_per_group * input_size - 0.5
        ax_tiles.axhline(y, color="white", linewidth=1.5, alpha=0.9)

    # Y-tick at group centres
    ytick_pos = [(g * rows_per_group + rows_per_group // 2) * input_size
                 for g in range(n_orientations)]
    ax_tiles.set_yticks(ytick_pos)
    ax_tiles.set_yticklabels(orientations_deg, fontsize=9)
    ax_tiles.tick_params(axis="x", bottom=False, labelbottom=False)

    # Panel 2: polar rose diagram
    angles = [np.pi * g / n_orientations for g in range(n_orientations)]
    bar_width = np.pi / n_orientations * 0.85
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_orientations))
    ax_polar.bar(angles, group_elongation, width=bar_width,
                 color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_xticks(angles)
    ax_polar.set_xticklabels(orientations_deg, fontsize=9)
    ax_polar.set_title("Mean RF elongation\n(λ₁/λ₂ per orientation group)",
                       fontsize=11, pad=15)
    ax_polar.set_rlabel_position(90)
    ax_polar.tick_params(axis="y", labelsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"RF summary plot saved -> {out_path}")


def plot_weight_sample_trajectories(
    sampler, config_label: str, output_path: str, x_max: "int | None" = None
) -> None:
    """
    Interleaved timeline plot: true simulation timestep on the x-axis.

    Awake (STDP) and regularisation events are placed at the timesteps where
    they actually occurred.  Sleep / normalise episodes are highlighted with
    thin vertical lines.  Only w_ee (excitatory-to-excitatory) is shown.
    """
    has_awake = bool(sampler.awake_ee)
    has_reg = bool(sampler.reg_ee)
    if not has_awake and not has_reg:
        return

    color_ee = "#d6604d"

    def _to_arr(lst):
        return np.array(lst) if lst else np.empty((0, 0))

    aw_ee = _to_arr(sampler.awake_ee)   # (n_awake, n_ee)
    rg_ee = _to_arr(sampler.reg_ee)     # (n_reg,  n_ee)

    # Build x-axis arrays from recorded timesteps; fall back to index if None
    def _t_axis(t_list, n):
        if not t_list or all(v is None for v in t_list):
            return np.arange(n, dtype=float)
        return np.array([v if v is not None else i
                         for i, v in enumerate(t_list)], dtype=float)

    x_aw = _t_axis(sampler.t_awake, aw_ee.shape[0]) if has_awake else np.empty(0)
    x_rg = _t_axis(sampler.t_reg,   rg_ee.shape[0]) if has_reg   else np.empty(0)

    # Determine gap_threshold for episode clustering.
    # Use half the smallest *positive* inter-event gap so that:
    #   - Sleep sub-steps sharing the same outer t (gap=0) stay merged.
    #   - Consecutive independent episodes (gap=reg_frequency) always split.
    gap_threshold = 5  # safe fallback (handles None-only t_reg)
    if has_reg and len(sampler.t_reg) >= 2:
        valid_ts = [v for v in sampler.t_reg if v is not None]
        if len(valid_ts) >= 2:
            diffs = np.diff(valid_ts)
            positive_diffs = diffs[diffs > 0]
            if positive_diffs.size:
                gap_threshold = int(positive_diffs.min()) // 2

    episodes = _reg_episodes(sampler.t_reg, gap_threshold)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Mark each reg episode with a thin vertical line (works cleanly for
    # both sleep — many sub-steps at one t — and normalize — one event per t).
    for (t_start, t_end) in episodes:
        mid = (t_start + t_end) / 2
        ax.axvline(mid, color="#9999cc", linewidth=0.8, alpha=0.4, zorder=0)

    # Collect all values that will be visible within the x window for y scaling
    _y_vals = []

    def _plot(arr, x, color):
        if arr.size == 0 or x.size == 0:
            return
        mk = "." if len(x) == 1 else None
        # Mask to x window for y-range calculation
        x_lo, x_hi = 0, (x_max if x_max is not None else np.inf)
        mask = (x >= x_lo) & (x <= x_hi)
        if mask.any():
            _y_vals.append(arr[mask])
        for i in range(arr.shape[1]):
            ax.plot(x, arr[:, i], color=color, alpha=0.55, linewidth=0.9,
                    linestyle="-", marker=mk)

    # Awake phase
    _plot(aw_ee, x_aw, color_ee)

    # Reg phase — plotted on the same x-axis at their true positions
    _plot(rg_ee, x_rg, color_ee)

    # Y-axis: scale to the visible data range with a small margin
    if _y_vals:
        all_vals = np.concatenate([v.ravel() for v in _y_vals])
        v_min, v_max = all_vals.min(), all_vals.max()
        margin = (v_max - v_min) * 0.05 or 0.05
        ax.set_ylim(v_min - margin, v_max + margin)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=color_ee, linewidth=1.5, alpha=0.7, label="w_ee"),
    ]
    if episodes:
        legend_handles.append(
            Line2D([0], [0], color="#9999cc", linewidth=1.5, alpha=0.6,
                   label="reg episode")
        )
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)

    ax.set_xlim(left=0, right=x_max if x_max is not None else None)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Synaptic weight")
    ax.set_title(config_label)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
