import os
import numpy as np
from src.utils.platform import configure_matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from datasets.datasets import GEOMFIG_DATASET
import pandas as pd
from typing import Optional
import seaborn as sns
import networkx as nx
import datetime
configure_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_model_accuracy_sleep(
    results_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Create faceted error bar plot comparing model accuracy across sleep rates.
    
    Args:
        results_path: Path to Results_.xlsx
        output_path: Path to save the figure
        show_plot: Whether to display the plot
        
    Returns:
        Path to saved figure
    """
    data = pd.read_excel(results_path)
    data = data.loc[data["Run"] == 1]
    
    # Wide to long format
    ldata = pd.melt(
        frame=data,
        id_vars=["Model", "Sleep_duration", "Lambda"],
        var_name="Dataset",
        value_name="Accuracy"
    )
    
    # Calculate statistics
    stats = ldata.groupby(["Model", "Sleep_duration", "Dataset"])["Accuracy"].agg(
        mean="mean", min="min", max="max"
    ).reset_index()
    
    # Get unique sleep values
    sleep_vals = sorted(data["Sleep_duration"].dropna().unique())
    order = [str(int(v)) for v in sleep_vals]
    stats["Sleep_duration"] = stats["Sleep_duration"].astype(str)
    order_map = {v: i for i, v in enumerate(order)}
    
    # Setup
    dataset_order = ["mnist", "kmnist", "fmnist", "notmnist"]
    hue_order = ["SNN_sleepy", "snntorch"]
    styles = {"SNN_sleepy": "o", "snntorch": "^"}
    cluster_width = 0.8
    base_index = {s: i for i, s in enumerate(order)}
    
    # Create facet grid
    g = sns.FacetGrid(
        data=ldata,
        col="Dataset",
        col_order=dataset_order,
        sharey=True,
        height=4,
        aspect=1.1
    )
    
    def draw_errorbars(data, color=None, **kwargs):
        ax = plt.gca()
        dataset = data["Dataset"].iloc[0]
        subdf = stats[stats["Dataset"] == dataset]
        
        if subdf.empty:
            return
        
        models_here = [m for m in hue_order if m in subdf["Model"].unique()]
        k = len(models_here)
        if k == 0:
            return
        
        offset = {m: (j - (k-1)/2) * (cluster_width / k) for j, m in enumerate(models_here)}
        
        for m in models_here:
            style = styles[m]
            sub_m = subdf[subdf["Model"] == m].copy()
            sub_m["__ord__"] = sub_m["Sleep_duration"].map(order_map)
            sub_m = sub_m.sort_values("__ord__")
            
            if sub_m.empty:
                continue
            
            xs = [base_index[s] + offset[m] for s in sub_m["Sleep_duration"]]
            y_mean = sub_m["mean"].to_numpy()
            y_max = sub_m["max"].to_numpy()
            y_min = sub_m["min"].to_numpy()
            
            yerr_lower = y_mean - y_min
            yerr_upper = y_max - y_mean
            
            plt.errorbar(
                xs, y_mean,
                yerr=[yerr_lower, yerr_upper],
                fmt=style,
                markersize=5,
                linestyle="none",
                capsize=4,
                elinewidth=1.2,
                color="black",
                label=m if dataset == dataset_order[0] else None
            )
            plt.plot(xs, y_mean, linestyle="--", color="black", linewidth=1)
        
        ax.set_xticks(list(base_index.values()))
        ax.set_xticklabels(order)
        ax.set_xlabel("Sleep_duration", fontsize=16)
        ax.set_ylim(top=100)
    
    g.map_dataframe(draw_errorbars)
    
    # Custom titles
    new_titles = {
        "mnist": "MNIST",
        "kmnist": "KMNIST",
        "fmnist": "Fashion-MNIST",
        "notmnist": "NotMNIST"
    }
    
    for ax, ds in zip(g.axes.flatten(), dataset_order):
        ax.set_title(new_titles.get(ds, ds), fontsize=16)
    
    g.set_ylabels("Accuracy", fontsize=18)
    
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if handles:
        g.fig.legend(handles, labels, title="Model", bbox_to_anchor=(0.2, 0.5))
    
    plt.ylim(top=1.0, bottom=min(ldata["Accuracy"]))
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    
    output_path = output_path or "sleep_duration_comparison_models.pdf"
    plt.savefig(output_path, dpi=900)
    print(f"Figure saved to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()
    
    return output_path


def plot_glmm_predictions(
    pred_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Plot GLMM predicted values with confidence intervals.
    
    Args:
        pred_path: Path to pred.xlsx (output from mixed_model2.r)
        output_path: Path to save the figure
        show_plot: Whether to display the plot
        
    Returns:
        Path to saved figure
    """
    data = pd.read_excel(pred_path)
    
    sleep_vals = sorted(data["x"].dropna().unique())
    order = [str(int(v)) for v in sleep_vals]
    data["x"] = data["x"].astype(str)
    order_map = {v: i for i, v in enumerate(order)}
    
    dataset_order = ["mnist", "kmnist", "fmnist", "notmnist"]
    hue_order = ["SNN_sleepy", "snntorch"]
    styles = {"SNN_sleepy": "o", "snntorch": "v"}
    cluster_width = 0.8
    
    g = sns.FacetGrid(
        data=data,
        col="Dataset" if "Dataset" in data.columns else "facet",
        col_order=dataset_order,
        sharey=True,
        height=4,
        aspect=1.1
    )
    
    base_index = {s: i for i, s in enumerate(order)}
    
    def draw_errorbars(data, color=None, **kwargs):
        ax = plt.gca()
        dataset = data["Dataset"].iloc[0] if "Dataset" in data.columns else data["facet"].iloc[0]
        subdf = data.copy()
        
        if subdf.empty:
            return
        
        models_here = [m for m in hue_order if m in subdf["group"].unique()]
        k = len(models_here)
        if k == 0:
            return
        
        offset = {m: (j - (k-1)/2) * (cluster_width / k) for j, m in enumerate(models_here)}
        
        for m in models_here:
            style = styles[m]
            sub_m = subdf[subdf["group"] == m].copy()
            sub_m["__ord__"] = sub_m["x"].map(order_map)
            sub_m = sub_m.sort_values("__ord__")
            
            if sub_m.empty:
                continue
            
            xs = [order_map[s] + offset[m] for s in sub_m["x"]]
            y_mean = sub_m["predicted"].to_numpy()
            y_max = sub_m["conf.high"].to_numpy()
            y_min = sub_m["conf.low"].to_numpy()
            
            yerr_lower = y_mean - y_min
            yerr_upper = y_max - y_mean
            
            plt.errorbar(
                xs, y_mean,
                yerr=[yerr_lower, yerr_upper],
                fmt=style,
                markersize=5,
                linestyle="none",
                capsize=4,
                elinewidth=1.2,
                color="black",
                label=m if dataset == dataset_order[0] else None
            )
            plt.plot(xs, y_mean, linestyle="--", color="black", linewidth=1)
        
        ax.set_xticks(list(order_map.values()))
        ax.set_xticklabels(order, fontsize=18)
        ax.set_xlabel("Sleep_duration", fontsize=16)
        ax.set_ylim(top=1.0, bottom=0.2)
    
    g.map_dataframe(draw_errorbars)
    
    new_titles = {
        "mnist": "MNIST",
        "kmnist": "KMNIST",
        "fmnist": "Fashion-MNIST",
        "notmnist": "NotMNIST"
    }
    
    for ax, ds in zip(g.axes.flatten(), dataset_order):
        ax.set_title(new_titles.get(ds, ds), fontsize=18)
    
    g.set_ylabels("Accuracy (%)", fontsize=16)
    
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if handles:
        g.fig.legend(handles, labels, title="Model", bbox_to_anchor=(0.14, 0.4), framealpha=1.0)
    
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    
    output_path = output_path or "sleep_duration_comparison_glmm.pdf"
    plt.savefig(output_path, dpi=900)
    print(f"Figure saved to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()
    
    return output_path


def plot_glmm_with_raw_accuracy(
    pred_path: str,
    results_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Plot GLMM predictions overlaid with raw observed accuracy.
    
    Args:
        pred_path: Path to pred.xlsx
        results_path: Path to Results_.xlsx
        output_path: Path to save figure
        show_plot: Whether to display
        
    Returns:
        Path to saved figure
    """
    # Load GLMM predictions
    data = pd.read_excel(pred_path)
    data = data.rename(columns={'x': 'Sleep_duration', 'group': 'Model', 'facet': 'Dataset'})
    
    # Load raw results
    data2 = pd.read_excel(results_path)
    data2 = data2[data2["Run"] == 1]
    data2 = data2.drop(["Seed", "Run"], axis=1, errors="ignore")
    
    # Wide to long format
    ldata = pd.melt(
        frame=data2,
        id_vars=["Model", "Sleep_duration", "Lambda"],
        var_name="Dataset",
        value_name="Accuracy"
    )
    
    stats = ldata.groupby(["Model", "Sleep_duration", "Dataset"])["Accuracy"].agg(
        mean="mean", min="min", max="max"
    ).reset_index()
    
    sleep_vals = sorted(data["Sleep_duration"].dropna().unique())
    order = [str(int(v)) for v in sleep_vals]
    stats["Sleep_duration"] = stats["Sleep_duration"].astype(str)
    data["Sleep_duration"] = data["Sleep_duration"].astype(str)
    order_map = {v: i for i, v in enumerate(order)}
    
    dataset_order = ["mnist", "kmnist", "fmnist", "notmnist"]
    hue_order = ["SNN_sleepy", "snntorch"]
    styles = {"SNN_sleepy": "o", "snntorch": "v"}
    cluster_width = 0.8
    
    g = sns.FacetGrid(
        data=data,
        col="Dataset",
        col_order=dataset_order,
        sharey=True,
        height=4,
        aspect=1.1
    )
    
    def draw_errorbars(data, color=None, **kwargs):
        ax = plt.gca()
        dataset = data["Dataset"].iloc[0]
        
        subdf = data.copy()
        subdf2 = stats[stats["Dataset"] == dataset].copy()
        
        if subdf.empty:
            return
        
        models_here = [m for m in hue_order if m in subdf["Model"].unique()]
        k = len(models_here)
        if k == 0:
            return
        
        offset = {m: (j - (k-1)/2) * (cluster_width / k) for j, m in enumerate(models_here)}
        
        for m in models_here:
            style = styles[m]
            sub_m = subdf[subdf["Model"] == m].copy()
            sub_m2 = subdf2[subdf2["Model"] == m].copy()
            
            sub_m["__ord__"] = sub_m["Sleep_duration"].map(order_map)
            sub_m = sub_m.sort_values("__ord__")
            sub_m2["__ord__"] = sub_m2["Sleep_duration"].map(order_map)
            sub_m2 = sub_m2.sort_values("__ord__")
            
            if sub_m.empty:
                continue
            
            xs = [order_map[s] + offset[m] for s in sub_m["Sleep_duration"]]
            y_mean2 = sub_m2["mean"].to_numpy() if not sub_m2.empty else []
            y_mean = sub_m["predicted"].to_numpy()
            y_max = sub_m["conf.high"].to_numpy()
            y_min = sub_m["conf.low"].to_numpy()
            
            yerr_lower = y_mean - y_min
            yerr_upper = y_max - y_mean
            
            # Predicted with CI
            plt.errorbar(
                xs, y_mean,
                yerr=[yerr_lower, yerr_upper],
                fmt=style,
                markersize=5,
                linestyle="none",
                capsize=4,
                elinewidth=1.2,
                color="black",
                label=f"{m} predicted mean" if dataset == dataset_order[0] else None
            )
            
            # Observed means (hollow markers)
            if len(y_mean2) > 0:
                plt.scatter(
                    xs, y_mean2,
                    marker=style,
                    facecolors='none',
                    edgecolors="black",
                    s=50,
                    label=f"{m} observed mean" if dataset == dataset_order[0] else None
                )
            
            plt.plot(xs, y_mean, linestyle="--", color="black", linewidth=1)
        
        ax.set_xticks(list(order_map.values()))
        ax.set_xticklabels(order, fontsize=18)
        ax.set_xlabel("Sleep_duration", fontsize=16)
        ax.set_ylim(top=1.0, bottom=0.0)
    
    g.map_dataframe(draw_errorbars)
    
    new_titles = {
        "mnist": "MNIST",
        "kmnist": "KMNIST",
        "fmnist": "Fashion-MNIST",
        "notmnist": "NotMNIST"
    }
    
    for ax, ds in zip(g.axes.flatten(), dataset_order):
        ax.set_title(new_titles.get(ds, ds), fontsize=18)
    
    g.set_ylabels("Accuracy (%)", fontsize=16)
    
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if handles:
        g.fig.legend(handles, labels, title="Model", bbox_to_anchor=(0.23, 0.55), framealpha=1.0)
    
    sns.despine(offset=10, trim=True)
    
    output_path = output_path or "sleep_duration_comparison_with_accuracy.pdf"
    plt.savefig(output_path, dpi=900)
    print(f"Figure saved to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()
    
    return output_path


def plot_geomfig_comparison(
    sleep_results_path: str,
    no_sleep_results_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Create boxplot comparing accuracy with and without sleep on geomfig dataset.
    """
    import json
    
    with open(no_sleep_results_path, "r") as f:
        no_sleep = json.load(f)
    with open(sleep_results_path, "r") as f:
        sleep = json.load(f)
    
    df = pd.DataFrame(columns=["Run", "Sleep", "Type", "Accuracy"])
    types = ["test_accuracy", "train_accuracy", "val_accuracy"]
    
    # Extract results
    list_dicts1 = no_sleep.get("results_by_dataset", {}).get("geomfig", {}).get("0.1", [])
    list_dicts2 = sleep.get("results_by_dataset", {}).get("geomfig", {}).get("0.1", [])
    
    for item in list_dicts1:
        for typ in types:
            if typ in item:
                df.loc[len(df)] = [item.get("run", 0), "No-sleep", typ, item[typ]]
    
    for item in list_dicts2:
        for typ in types:
            if typ in item:
                df.loc[len(df)] = [item.get("run", 0), "Sleep", typ, item[typ]]
    
    subdf = df[df["Type"] == "test_accuracy"]
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = sns.boxplot(x="Sleep", y="Accuracy", color="black", data=subdf, linewidth=1.5, ax=ax)
    
    for patch in ax.patches:
        patch.set_hatch("////")
        patch.set_facecolor("white")
    
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.xlabel("")
    plt.xticks(fontsize=18)
    
    output_path = output_path or "accuracy_comp_geomfig.pdf"
    plt.savefig(output_path, dpi=900)
    print(f"Figure saved to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()
    
    return output_path

def preview_loaded_data(
    self, num_image_samples: int = 9, save_path: str | None = None
):
    """
    Plot a small grid of images from the loaded dataset once so the user can
    verify that the expected dataset is being used.
    """
    if getattr(self, "_image_preview_done", False):
        return
    # Special handling for geomfig: show N examples per class (0..3)
    if getattr(self, "image_dataset", "").lower() == "geomfig":
        try:
            pixel_size = int(np.sqrt(N_x))
            classes = [0, 1, 2, 3]
            per_class = max(1, int(num_image_samples))
            # Special case: if 1 per class and 4 classes, arrange as 2x2 instead of 4x1
            if per_class == 1 and len(classes) == 4:
                rows, cols = 2, 2
                fig, axes = plt.subplots(rows, cols, figsize=(4.0, 4.0))
                axes = axes.flatten()
                titles = ["Triangle (0)", "Circle (1)", "Square (2)", "X (3)"]
                for idx, cls in enumerate(classes):
                    img = GEOMFIG_DATASET._geomfig_generate_one(
                        cls_id=cls,
                        pixel_size=pixel_size,
                        noise_var=getattr(self, "geom_noise_var", 0.02),
                        jitter=getattr(self, "geom_jitter", False),
                        jitter_amount=getattr(self, "geom_jitter_amount", 0.05),
                    )
                    ax = axes[idx]
                    ax.imshow(img, cmap="gray")
                    ax.set_title(titles[idx], fontsize=10)
                    ax.axis("off")
            else:
                fig, axes = plt.subplots(
                    len(classes),
                    per_class,
                    figsize=(2.0 * per_class, 2.0 * len(classes)),
                )
                if per_class == 1:
                    axes = np.atleast_2d(axes).reshape(len(classes), 1)
                titles = ["Triangle (0)", "Circle (1)", "Square (2)", "X (3)"]
                for r, cls in enumerate(classes):
                    for c in range(per_class):
                        img = GEOMFIG_DATASET._geomfig_generate_one(
                            cls_id=cls,
                            pixel_size=pixel_size,
                            noise_var=getattr(self, "geom_noise_var", 0.02),
                            jitter=getattr(self, "geom_jitter", False),
                            jitter_amount=getattr(self, "geom_jitter_amount", 0.05),
                        )
                        ax = axes[r, c]
                        ax.imshow(img, cmap="gray")
                        if c == 0:
                            ax.set_title(titles[r], fontsize=10)
                        ax.axis("off")
            plt.tight_layout()
            try:
                if save_path is None:
                    os.makedirs("plots", exist_ok=True)
                    save_path = os.path.join("plots", "geomfig_preview.png")
                fig.savefig(save_path)
                print(f"Dataset preview saved to {save_path}")
            except Exception as exc:
                print(f"Failed to save dataset preview ({exc})")
            plt.show()
            plt.close(fig)
        except Exception as exc:
            print(f"Dataset preview skipped ({exc})")
        return
    # Default: use image streamer preview if available
    return

def plot_tsne(tsne_results, segment_labels, train, show_plot, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
    unique_labels = np.unique(segment_labels)

    for i, label in enumerate(unique_labels):
        indices = segment_labels == label
        marker = marker_list[i % len(marker_list)]
        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=f"Class {label}",
            marker=marker,
            color="black",
            s=60,
        )

    plt.xlabel("t-SNE dimension 1", fontsize=26)
    plt.ylabel("t-SNE dimension 2", fontsize=26)
    os.makedirs("plots", exist_ok=True)
    suffix = "train" if train else "test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tsne_path = os.path.join("plots\\tsne", f"tsne_{suffix}_{timestamp}.pdf")
    plt.tight_layout()
    plt.savefig(tsne_path, bbox_inches="tight")
    print(f"t-SNE plot saved to {tsne_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_weight_distribution(N_x, st, ih, weights, N_exc, N_inh, e, weight_evolution, epochs):
    _st, _ex, _ih = N_x, st + 0 - 0, ih + 0 - 0
    _ex = _st + N_exc
    _ih = _ex + N_inh
    W_exc = weights[:_ex, _st:_ih]
    W_inh = weights[_ex:_ih, _st:_ex]

    weight_evolution["epochs"].append(e + 1)

    # Compute stats on non-zero weights only to avoid zero-bias
    W_exc_nz = W_exc[W_exc != 0]
    if W_exc_nz.size > 0:
        weight_evolution["exc_mean"].append(
            float(np.mean(W_exc_nz))
        )
        weight_evolution["exc_std"].append(float(np.std(W_exc_nz)))
        weight_evolution["exc_min"].append(float(np.min(W_exc_nz)))
        weight_evolution["exc_max"].append(float(np.max(W_exc_nz)))
    else:
        weight_evolution["exc_mean"].append(0.0)
        weight_evolution["exc_std"].append(0.0)
        weight_evolution["exc_min"].append(0.0)
        weight_evolution["exc_max"].append(0.0)

    W_inh_nz = W_inh[W_inh != 0]
    if W_inh_nz.size > 0:
        weight_evolution["inh_mean"].append(
            float(np.mean(W_inh_nz))
        )
        weight_evolution["inh_std"].append(float(np.std(W_inh_nz)))
        weight_evolution["inh_min"].append(float(np.min(W_inh_nz)))
        weight_evolution["inh_max"].append(float(np.max(W_inh_nz)))
    else:
        weight_evolution["inh_mean"].append(0.0)
        weight_evolution["inh_std"].append(0.0)
        weight_evolution["inh_min"].append(0.0)
        weight_evolution["inh_max"].append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Plot only non-zero weights for informative histograms
    exc_vals = W_exc.flatten()
    exc_vals = exc_vals[exc_vals != 0]
    if exc_vals.size > 0:
        axes[0].hist(exc_vals, bins=50, color="tomato", alpha=0.8)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No non-zero weights",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    axes[0].set_title("Excitatory weights")
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Count")
    inh_vals = W_inh.flatten()
    inh_vals = inh_vals[inh_vals != 0]
    if inh_vals.size > 0:
        axes[1].hist(
            inh_vals, bins=50, color="steelblue", alpha=0.8
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No non-zero weights",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("Inhibitory weights")
    axes[1].set_xlabel("Weight")
    fig.suptitle(
        f"Epoch {e+1}/{epochs} - Weight Distributions",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"plots/weights/weights_epoch_{e+1:03d}.png", bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"  Saved weights snapshot: plots/weights_epoch_{e+1:03d}.png"
    )


def _plot_weight_matrix(weights,):
    """Visualize the weight matrix."""
    boundaries = [np.min(weights), -0.001, 0.001, np.max(weights)]
    cmap = ListedColormap(["red", "white", "green"])
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    plt.imshow(weights, cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    plt.title("Weights")
    plt.show()

def plot_accuracy_history(save_dir: str = "plots", filename_suffix: str = "", acc_history: dict = None):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    if acc_history.get("train"):
        plt.plot(
            range(1, len(acc_history["train"]) + 1),
            acc_history["train"],
            label="Train",
            color="gray",   
            linestyle="-",
        )
    if acc_history.get("val"):
        plt.plot(
            range(1, len(acc_history["val"]) + 1),
            acc_history["val"],
            label="Val",
            color="black",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    tv_path = os.path.join(save_dir, f"acc_train_val{suffix_str}.png")
    try:
        plt.savefig(tv_path, bbox_inches="tight")
        print(f"Saved train/val accuracy plot to {tv_path}")
    except Exception as exc:
        print(f"Failed to save train/val plot ({exc})")
    plt.close()


def plot_network_graph(weights, N_x, N_exc, N_inh):
    """Visualize the network as a graph."""
    total_nodes = N_x + N_exc + N_inh

    G = nx.from_numpy_array(weights)

    # Partition nodes
    input_nodes = list(range(N_x))
    exc_nodes = list(range(N_x, N_x + N_exc))
    inh_nodes = list(range(N_x + N_exc, total_nodes))

    # Assign positions (vertical columns)
    pos = {}
    for i, node in enumerate(input_nodes):
        y = 1 - (i / (len(input_nodes) - 1)) if len(input_nodes) > 1 else 0.5
        pos[node] = (0, y)

    for i, node in enumerate(exc_nodes):
        y = 1 - (i / (len(exc_nodes) - 1)) if len(exc_nodes) > 1 else 0.5
        pos[node] = (1, y)

    for i, node in enumerate(inh_nodes):
        y = 1 - (i / (len(inh_nodes) - 1)) if len(inh_nodes) > 1 else 0.5
        pos[node] = (2, y)

    # Node colors
    node_colors = {}
    for node in input_nodes:
        node_colors[node] = "skyblue"
    for node in exc_nodes:
        node_colors[node] = "lightgreen"
    for node in inh_nodes:
        node_colors[node] = "salmon"

    colors = [node_colors[node] for node in G.nodes()]

    # Draw
    plt.figure(figsize=(8, 4))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=100)
    edges = G.edges(data=True)
    edge_weights = [data["weight"] for (u, v, data) in edges]
    nx.draw_networkx_edges(G, pos, width=[5 * w for w in edge_weights], alpha=0.1)
    nx.draw_networkx_labels(G, pos, font_size=5, font_color="black")
    plt.title("Partitioned Graph: Input, Excitatory, Inhibitory")
    plt.axis("off")
    plt.show()


def get_elite_nodes(spikes, labels, num_classes, narrow_top):

    # remove unnecessary data periods
    mask_break = (labels != -1) & (labels != -2)
    spikes = spikes[mask_break, :]
    labels = labels[mask_break]

    print(f"Debug get_elite_nodes - spikes shape after filtering: {spikes.shape}")
    print(f"Debug get_elite_nodes - labels shape after filtering: {labels.shape}")
    print(f"Debug get_elite_nodes - unique labels after filtering: {np.unique(labels)}")
    print(f"Debug get_elite_nodes - narrow_top: {narrow_top}")

    # collect responses
    responses = np.zeros(
        (spikes.shape[1], num_classes), dtype=float
    )  # make responses float too

    for cl in range(num_classes):
        indices = np.where(labels == cl)[0]
        summed = np.sum(spikes[indices], axis=0)  # still int at this point
        response = summed.astype(float)  # now convert to float
        response[response == 0] = np.nan  # safe to assign NaN
        responses[:, cl] = response

    # compute discriminatory power
    total_responses = np.sum(spikes, axis=0, dtype=float)
    total_responses[total_responses == 0] = np.nan
    total_responses_reshaped = np.tile(total_responses, (num_classes, 1)).T
    ratio = responses / total_responses_reshaped
    responses *= ratio

    # Now, assign nodes to their preferred class (highest response)
    responses_indices = np.argsort(responses, 0)[::-1, :]
    top_k = int(spikes.shape[1] * narrow_top)

    print(f"Debug get_elite_nodes - total neurons: {spikes.shape[1]}")
    print(f"Debug get_elite_nodes - top_k (neurons per class): {top_k}")
    print(f"Debug get_elite_nodes - total elite neurons: {top_k * num_classes}")

    # Assign top responders
    final_indices = responses_indices[:top_k]

    return final_indices, spikes, labels


def plot_epoch_training(acc, cluster, val_acc=None, val_phi=None):
    fig, ax0 = plt.subplots()

    # Left y-axis
    (line0,) = ax0.plot(cluster, color="tab:blue", label="Cluster")
    ax0.set_ylabel("Cluster", color="tab:blue")
    ax0.tick_params(axis="y", labelcolor="tab:blue")

    # Right y-axis
    ax1 = ax0.twinx()
    (line1,) = ax1.plot(acc, color="tab:red", label="Train Acc")
    if val_acc is not None:
        (line1b,) = ax1.plot(
            val_acc, color="tab:orange", linestyle="--", label="Val Acc"
        )
    ax1.set_ylabel("Accuracy", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Common title and xlabel
    fig.suptitle("Epoch Training")
    fig.supxlabel("Epoch")

    # Common legend
    lines = [line0, line1]
    if val_acc is not None:
        lines.append(line1b)
    if val_phi is not None:
        # Overlay val Phi on left axis for simplicity
        (line2,) = ax0.plot(val_phi, color="tab:green", linestyle=":", label="Val Phi")
        lines.append(line2)
    labels = [line.get_label() for line in lines]
    ax0.legend(lines, labels, loc="upper center", fontsize=14)

    plt.show()


def top_responders_plotted(
    spikes,
    labels,
    num_classes,
    narrow_top,
    smoothening,
    train,
    compute_not_plot,
    n_last_points=None,
):

    # get indicess
    indices, spikes, labels = get_elite_nodes(
        spikes=spikes,
        labels=labels,
        num_classes=num_classes,
        narrow_top=narrow_top,
    )

    if compute_not_plot:
        block_size = smoothening

        # Calculate the number of complete blocks
        num_blocks = spikes.shape[0] // block_size

        # Initialize a list to hold the mean of each block
        means = []
        labs = []

        # Loop through each block, calculate mean along axis=0 (i.e. column-wise)
        for i in range(num_blocks):
            # add spikes
            block = spikes[i * block_size : (i + 1) * block_size]
            block_mean = np.mean(block, axis=0)
            means.append(block_mean)
            # add labels
            block_lab = labels[i * block_size : (i + 1) * block_size]
            block_maj = np.argmax(np.bincount(block_lab))
            labs.append(block_maj)

        # Optionally convert to a NumPy array for further processing
        spikes = np.array(means)
        labels = np.array(labs)

        acts = np.zeros((spikes.shape[0], num_classes))
        for c in range(num_classes):
            acts[:, c] = np.sum(spikes[:, indices[:, c]], axis=1)

        predictions = np.argmax(acts, axis=1)
        precision = np.zeros(spikes.shape[0])
        hit = 0
        for i in range(precision.shape[0]):
            hit += predictions[i] == labels[i]
            precision[i] = hit / (i + 1)

        # Debug: Print some statistics
        print(f"Debug accuracy - Total samples: {len(predictions)}")
        print(f"Debug accuracy - Correct predictions: {hit}")
        print(f"Debug accuracy - Final accuracy: {precision[-1]}")
        print(f"Debug accuracy - Prediction distribution: {np.bincount(predictions)}")
        print(f"Debug accuracy - Label distribution: {np.bincount(labels)}")

        # return the final accuracy measurement
        return precision[-1]
    fig, ax = plt.subplots(2, 1)

    # reduce samples
    cmap = plt.get_cmap("Set3", num_classes)
    colors = cmap.colors

    # Define an intensity factor (values between 0 and 1)
    intensity_factor = 0.5  # 70% of the original brightness

    # Reduce the intensity of each color by scaling its RGB components
    colors_adjusted = [
        tuple(np.clip(np.array(color) * intensity_factor, 0, 1)) for color in colors
    ]

    block_size = smoothening

    # Calculate the number of complete blocks
    num_blocks = spikes.shape[0] // block_size

    # Initialize a list to hold the mean of each block
    means = []
    labs = []

    # Loop through each block, calculate mean along axis=0 (i.e. column-wise)
    for i in range(num_blocks):
        # add spikes
        block = spikes[i * block_size : (i + 1) * block_size]
        block_mean = np.mean(block, axis=0)
        means.append(block_mean)
        # add labels
        block_lab = labels[i * block_size : (i + 1) * block_size]
        block_maj = np.argmax(np.bincount(block_lab))
        labs.append(block_maj)

    # Optionally convert to a NumPy array for further processing
    spikes = np.array(means)
    labels = np.array(labs)

    acts = np.zeros((spikes.shape[0], num_classes))
    for c in range(num_classes):
        activity = np.sum(spikes[:, indices[:, c]], axis=1)
        acts[:, c] = activity

    # Determine the range of points to plot for activity
    if n_last_points is not None and n_last_points < len(acts):
        start_idx = len(acts) - n_last_points
        plot_acts = acts[start_idx:]
        plot_labels = labels[start_idx:]
    else:
        plot_acts = acts
        plot_labels = labels

    # Plot activity for each class
    for c in range(num_classes):
        ax[0].plot(plot_acts[:, c], color=colors[c], label=f"Class {c}")

    # Add the horizontal line below the spikes
    y_offset = 0
    box_height = np.max(plot_acts)

    # We iterate through the time steps to identify contiguous segments
    segment_start = 0
    current_label = plot_labels[0]
    labeled_classes = set()

    # Loop through the labels to draw segments
    for i in range(1, len(plot_labels)):
        if plot_labels[i] != current_label:
            # Draw a rectangle patch for the segment that just ended
            rect = patches.Rectangle(
                (segment_start, y_offset),
                i - segment_start,  # width of the rectangle
                box_height,  # height of the rectangle
                linewidth=2,
                facecolor=colors_adjusted[current_label],
            )
            ax[0].add_patch(rect)

            # Mark this class as having been labeled
            labeled_classes.add(current_label)

            # Update for the new segment
            current_label = plot_labels[i]
            segment_start = i

    # Handle the final segment
    patch_label = (
        f"Class {current_label}" if current_label not in labeled_classes else None
    )
    rect = patches.Rectangle(
        (segment_start, y_offset),
        len(plot_labels) - segment_start,
        box_height,
        linewidth=2,
        edgecolor=colors_adjusted[current_label],
        facecolor=colors_adjusted[current_label],
        label=patch_label,
    )
    ax[0].add_patch(rect)

    if train:
        title = "Top responding nodes by class during training"
    else:
        title = "Top responding nodes by class during testing"
    ax[0].set_ylabel("Spiking rate")

    """
    Plot accuracy in second plot
    """
    predictions = np.argmax(acts, axis=1)
    precision = np.zeros(spikes.shape[0])
    hit = 0
    for i in range(precision.shape[0]):
        hit += predictions[i] == labels[i]
        precision[i] = hit / (i + 1)

    ax[1].plot(precision)
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_xlabel(f"Time (intervals of {smoothening} ms)")
    ax[0].set_title(title)
    ax[0].legend(loc="upper right")
    plt.show()
    return precision[-1]


def spike_plot(data, labels):
    # Validate dimensions
    if len(labels) != data.shape[0]:
        raise ValueError(
            f"Labels length ({len(labels)}) must match the number of time steps ({data.shape[0]})."
        )

    # Debug: Print data information
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data min/max: {data.min()}/{data.max()}")
    print(f"Number of non-zero elements: {np.count_nonzero(data)}")
    print(f"Unique values in data: {np.unique(data)}")

    # Check if there are any spikes at all
    if np.count_nonzero(data) == 0:
        print("WARNING: No spikes found in the data!")
        print("This could be because:")
        print(
            "1. The time window is too small (only last 5% of data is shown by default)"
        )
        print("2. The neurons selected don't have spikes")
        print("3. The spike data format is different than expected")
        print("4. The network hasn't learned to spike yet")
        print("\nSuggestions:")
        print(
            "- Try using a larger time window by setting start_time_spike_plot to an earlier time"
        )
        print("- Check if the network is actually producing spikes during training")
        print("- Verify that the spike data contains non-zero values")
        return

    # Assign colors to unique labels (excluding -1 if desired)
    valid_label_mask = labels != -1
    unique_labels = np.unique(labels[valid_label_mask])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: color for label, color in zip(unique_labels, colors)}

    # Collect spike positions for each neuron
    # Try different spike representations
    if np.any(data > 0):
        # If there are positive values, use those as spikes
        spike_threshold = 0
        print(f"Using positive values as spikes (threshold > {spike_threshold})")
    else:
        # Default to looking for exactly 1
        spike_threshold = 1
        print(f"Using exact value {spike_threshold} as spikes")

    positions = [
        np.where(data[:, n] > spike_threshold)[0] for n in range(data.shape[1])
    ]

    # Debug: Print spike information
    total_spikes = sum(len(pos) for pos in positions)
    print(f"Total spikes found: {total_spikes}")
    print(
        f"Spikes per neuron: {[len(pos) for pos in positions[:10]]}..."
    )  # First 10 neurons

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the spikes
    ax.eventplot(positions, lineoffsets=np.arange(data.shape[1]), colors="black")
    ax.set_ylabel(f"{data.shape[1]} Units")
    ax.set_xlabel("Time (ms)")

    """
    To plot the 
    """

    # We'll collect which labels we've drawn (for legend) so we don't add duplicates
    drawn_labels = set()

    # Add the horizontal line below the spikes
    y_offset = -10  # Position below the spike raster

    # We iterate through the time steps to identify contiguous segments
    segment_start = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        # If the label changes, we close off the old segment (unless it was -1)
        if labels[i] != current_label:
            if current_label != -1:
                if current_label == -2:
                    # For sleep segments, label as "Sleep" only once
                    label_text = "Sleep" if current_label not in drawn_labels else None
                    ax.hlines(
                        y=y_offset,
                        xmin=segment_start,
                        xmax=i,  # up to but not including i
                        color=label_colors.get(current_label, "blue"),
                        linewidth=6,
                        label=label_text,
                    )
                else:
                    label_text = (
                        f"Class {current_label}"
                        if current_label not in drawn_labels
                        else None
                    )
                    ax.hlines(
                        y=y_offset,
                        xmin=segment_start,
                        xmax=i,  # up to but not including i
                        color=label_colors.get(current_label, "black"),
                        linewidth=6,
                        label=label_text,
                    )
                drawn_labels.add(current_label)

            # Update to the new segment
            current_label = labels[i]
            segment_start = i

    # Handle the last segment after exiting the loop
    if current_label != -1:
        if current_label == -2:
            label_text = "Sleep" if current_label not in drawn_labels else None
            ax.hlines(
                y=y_offset,
                xmin=segment_start,
                xmax=len(labels),
                color=label_colors.get(current_label, "blue"),
                linewidth=6,
                label=label_text,
            )
        else:
            label_text = (
                f"Class {current_label}" if current_label not in drawn_labels else None
            )
            ax.hlines(
                y=y_offset,
                xmin=segment_start,
                xmax=len(labels),
                color=label_colors.get(current_label, "black"),
                linewidth=6,
                label=label_text,
            )
        drawn_labels.add(current_label)

    # Create a legend from the existing artists
    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(unique_labels),
    )

    plt.title("Spikes with Class-based Horizontal Lines")
    plt.tight_layout()
    plt.show()


def plot_accuracy(spikes, ih, pp, pn, tp, labels, num_steps, num_classes, test):
    """
    spikes have shape: pp-pn-tp-tn-fp-fn
    """
    pp_ = spikes[:, ih:pp]
    tp_ = spikes[:, pn:tp]

    #### calculate precision (accuracy) ###

    # remove data from all breaks
    mask = labels != -1
    if mask.size != 0:
        labels = labels[mask]
        pp_ = pp_[mask, :]
        tp_ = tp_[mask, :]

    # loop through every num_steps time units and compare activity
    total_images = 0
    current_accuracy = 0
    accuracy = np.zeros((labels.shape[0] // num_steps) + 1)
    total_images2 = np.zeros(num_classes)
    current_accuracy2 = np.zeros(num_classes)
    accuracy2 = np.zeros(((labels.shape[0] // num_steps) + 1, num_classes))
    for t in range(0, labels.shape[0] + 1, num_steps):
        pp_label = np.sum(pp_[t : t + num_steps], axis=0)
        tp_label = np.sum(tp_[t : t + num_steps], axis=0)

        # check if there is no class preference
        if np.sum(tp_label) == 0:
            accuracy[t // num_steps] = accuracy[(t - 1) // num_steps]
        else:
            """
            Look over this logic again. I think argmax might be wrong.
            """
            pp_label_pop = np.argmax(pp_label)
            tp_label_pop = np.argmax(tp_label)
            total_images += 1
            current_accuracy += int(pp_label_pop == tp_label_pop)
            accuracy[t // num_steps] = current_accuracy / total_images

            # update number of data points and accumulated accuracy
            total_images2[tp_label_pop] += 1
            current_accuracy2[tp_label_pop] += int(pp_label_pop == tp_label_pop)
            acc = current_accuracy2[tp_label_pop] / total_images2[tp_label_pop]
            accuracy2[t // num_steps :, tp_label_pop] = acc

    # plot
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        class_accuracy = accuracy2[:, c]
        # Add jitter to the x-values for this class
        jitter = np.random.normal(0, 0.001, size=class_accuracy.shape[0])
        plt.plot(
            class_accuracy + jitter,
            label=f"class:{c}",
            color=colors[c],
            linewidth=0.8,
            linestyle="dashed",
        )

    plt.plot(accuracy, label="All classes", linewidth=3, color="black")
    plt.legend(bbox_to_anchor=(1.1, 0.9), loc="upper right", fontsize=14)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xlabel("Time (t)", fontsize=18)
    if test:
        title = f"Testing accuracy: {accuracy[-1]}"
    else:
        title = f"Training and validation accuracy"
    # plt.title(title, fontsize=20, fontweight="bold")
    plt.show()

    return accuracy[-1]


def get_contiguous_segment(indices):
    """
    Given a sorted 1D array of indices, find contiguous segments
    and return the longest segment.
    """
    if len(indices) == 0:
        return None
    # Find gaps where consecutive indices differ by more than 1
    gaps = np.where(np.diff(indices) != 1)[0]
    segments = np.split(indices, gaps + 1)
    # Return the longest contiguous segment
    return max(segments, key=len)


def plot_floats_and_spikes(images, spikes, spike_labels, img_labels, num_steps):
    """
    Given:
      - images: an array of MNIST images (e.g., shape [num_images, H, W])
      - spikes: a 2D array of spike activity (shape: [time, neurons])
      - spike_labels: an array (length equal to the time dimension of spikes)
                      containing the label of the image that produced that spike train.
      - img_labels: an array of labels for the floating images
    This function plots, for each unique image label, the corresponding MNIST image
    (in the bottom row) and a raster plot of the spike data (in the top row).
    """
    # Determine the unique digit labels from the images.
    unique_labels = np.unique(img_labels)
    n_cols = len(unique_labels)

    # Create subplots: one column per digit, two rows (top for spikes, bottom for image)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))

    # If there's only one column, make sure axs is 2D.
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, label in enumerate(unique_labels):
        # Find the first image with this label
        img_idx = np.where(np.array(img_labels) == label)[0][0]
        # Plot the image in the bottom row
        ax_img = axs[1, i]
        # Ensure the image is 2D (squeeze any singleton dimensions)
        ax_img.imshow(np.squeeze(images[img_idx]), cmap="gray")
        ax_img.set_title(f"Digit {label}")
        ax_img.axis("off")

        # Find all time indices in the spiking data that belong to this label.
        spike_idx_all = np.where(np.array(spike_labels) == label)[0][:num_steps]
        if len(spike_idx_all) == 0:
            print(f"No spiking data found for label {label}.")
            continue

        # Get a contiguous segment from the available indices.
        segment = get_contiguous_segment(spike_idx_all)
        if segment is None or len(segment) == 0:
            print(f"No contiguous segment found for label {label}.")
            continue

        # Extract the spike data for this segment.
        spike_segment = spikes[segment, :]  # shape: [time_segment, neurons]

        # For each neuron, determine the time steps (relative to the segment) where it spiked.
        positions = [
            np.where(spike_segment[:, n] == 1)[0] for n in range(spike_segment.shape[1])
        ]

        # Plot the spike raster on the top row.
        ax_spike = axs[0, i]
        ax_spike.eventplot(positions, colors="black")
        ax_spike.set_title(f"Spikes for {label}")
        ax_spike.set_xlabel("Time steps")
        ax_spike.set_ylabel("Neuron")
        # Optionally, adjust y-limits for clarity:
        ax_spike.set_ylim(-1, spike_segment.shape[1])

    plt.tight_layout()

    plt.savefig("plots/comparison_spike_img.png")
    plt.show()

# Ensure we export what snn.py uses
__all__ = [
    'plot_tsne',
    'plot_weight_distribution',
    'plot_accuracy_history',
    'plot_network_graph',
    'plot_epoch_training',
    'top_responders_plotted',
    'spike_plot',
    'plot_floats_and_spikes',
    'plot_accuracy',
    'get_contiguous_segment',
    'preview_loaded_data',
    'plot_weight_matrix',
    'plot_weight_evolution',
    'plot_weight_evolution_during_sleep',
    'plot_weight_evolution_during_sleep_epoch',
    'plot_weight_trajectories_with_sleep_epoch',
    'save_weight_distribution_gif',
]
