import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns


def plot_glmm_with_raw_accuracy(
    pred_path: str,
    results_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    data = pd.read_excel(pred_path)
    data = data.rename(columns={"x": "Sleep_duration", "group": "Model", "facet": "Dataset"})

    data2 = pd.read_excel(results_path)
    data2 = data2[data2["Run"] == 1]
    data2 = data2.drop(["Seed", "Run"], axis=1, errors="ignore")

    ldata = data2

    stats = (
        ldata.groupby(["Model", "Sleep_duration", "Dataset"])["Accuracy"]
        .agg(mean="mean", min="min", max="max")
        .reset_index()
    )

    sleep_vals = sorted(data["Sleep_duration"].dropna().unique())
    order = [str(int(v)) for v in sleep_vals]
    stats["Sleep_duration"] = stats["Sleep_duration"].astype(str)
    data["Sleep_duration"] = data["Sleep_duration"].astype(str)
    order_map = {v: i for i, v in enumerate(order)}

    dataset_order = ["mnist", "kmnist", "fmnist", "notmnist"]
    hue_order = ["SNN_sleepy", "snntorch"]
    styles = {"SNN_sleepy": "o", "snntorch": "v"}
    cluster_width = 0.8

    g = sns.FacetGrid(data=data, col="Dataset", col_order=dataset_order,
                      sharey=True, height=4, aspect=1.1)

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

        offset = {
            m: (j - (k - 1) / 2) * (cluster_width / k)
            for j, m in enumerate(models_here)
        }

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

            plt.errorbar(xs, y_mean, yerr=[yerr_lower, yerr_upper], fmt=style,
                         markersize=5, linestyle="none", capsize=4, elinewidth=1.2,
                         color="black",
                         label=f"{m} predicted mean" if dataset == dataset_order[0] else None)

            if len(y_mean2) > 0:
                plt.scatter(xs, y_mean2, marker=style, facecolors="none", edgecolors="black",
                            s=50,
                            label=f"{m} observed mean" if dataset == dataset_order[0] else None)

            plt.plot(xs, y_mean, linestyle="--", color="black", linewidth=1)

        ax.set_xticks(list(order_map.values()))
        ax.set_xticklabels(order, fontsize=18)
        ax.set_xlabel("Sleep_duration", fontsize=16)
        ax.set_ylim(top=1.0, bottom=0.0)

    g.map_dataframe(draw_errorbars)

    new_titles = {
        "mnist": "MNIST", "kmnist": "KMNIST",
        "fmnist": "Fashion-MNIST", "notmnist": "NotMNIST",
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


def plot_glmm_predictions(
    pred_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
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
        aspect=1.1,
    )

    base_index = {s: i for i, s in enumerate(order)}

    def draw_errorbars(data, color=None, **kwargs):
        ax = plt.gca()
        dataset = (
            data["Dataset"].iloc[0]
            if "Dataset" in data.columns
            else data["facet"].iloc[0]
        )
        subdf = data.copy()

        if subdf.empty:
            return

        models_here = [m for m in hue_order if m in subdf["group"].unique()]
        k = len(models_here)
        if k == 0:
            return

        offset = {
            m: (j - (k - 1) / 2) * (cluster_width / k)
            for j, m in enumerate(models_here)
        }

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

            plt.errorbar(xs, y_mean, yerr=[yerr_lower, yerr_upper], fmt=style,
                         markersize=5, linestyle="none", capsize=4, elinewidth=1.2,
                         color="black", label=m if dataset == dataset_order[0] else None)
            plt.plot(xs, y_mean, linestyle="--", color="black", linewidth=1)

        ax.set_xticks(list(order_map.values()))
        ax.set_xticklabels(order, fontsize=18)
        ax.set_xlabel("Sleep_duration", fontsize=16)
        ax.set_ylim(top=1.0, bottom=0.2)

    g.map_dataframe(draw_errorbars)

    new_titles = {
        "mnist": "MNIST", "kmnist": "KMNIST",
        "fmnist": "Fashion-MNIST", "notmnist": "NotMNIST",
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
