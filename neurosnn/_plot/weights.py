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
