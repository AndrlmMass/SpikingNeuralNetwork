"""
Paper figure generation for SNN-sleepy research.

This module contains functions to generate the figures used in the paper,
comparing SNN-sleepy with snntorch across MNIST-family datasets.
"""

import os
import argparse
from typing import Optional, Dict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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


def generate_all_paper_figures(
    output_dir: str = "src/analysis/plots",
    analysis_dir: str = "src/analysis",
) -> Dict[str, str]:
    """
    Generate all paper figures from available data.
    
    Args:
        output_dir: Directory to save figures (default: src/analysis/plots)
        analysis_dir: Directory containing Results_.xlsx and pred.xlsx
    
    Returns:
        Dict mapping figure names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = {}
    
    results_path = os.path.join(analysis_dir, "Results_.xlsx")
    pred_path = os.path.join(analysis_dir, "pred.xlsx")
    
    print(f"Looking for results at: {results_path}")
    print(f"Looking for predictions at: {pred_path}")
    
    if os.path.exists(results_path):
        try:
            fig_path = plot_model_accuracy_sleep(
                results_path,
                output_path=os.path.join(output_dir, "sleep_duration_comparison_models.pdf")
            )
            figures["model_accuracy"] = fig_path
        except Exception as e:
            print(f"Warning: Could not generate model accuracy plot: {e}")
    
    if os.path.exists(pred_path):
        try:
            fig_path = plot_glmm_predictions(
                pred_path,
                output_path=os.path.join(output_dir, "sleep_duration_comparison_glmm.pdf")
            )
            figures["glmm_predictions"] = fig_path
        except Exception as e:
            print(f"Warning: Could not generate GLMM predictions plot: {e}")
    
    if os.path.exists(results_path) and os.path.exists(pred_path):
        try:
            fig_path = plot_glmm_with_raw_accuracy(
                pred_path,
                results_path,
                output_path=os.path.join(output_dir, "sleep_duration_comparison_with_accuracy.pdf")
            )
            figures["glmm_with_accuracy"] = fig_path
        except Exception as e:
            print(f"Warning: Could not generate combined GLMM+accuracy plot: {e}")
    
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, help="Path to Results_.xlsx")
    parser.add_argument("--predictions", type=str, help="Path to pred.xlsx")
    parser.add_argument("--output-dir", type=str, default="analysis/plots", help="Output directory")
    parser.add_argument("--model-accuracy", action="store_true", help="Generate model accuracy plot")
    parser.add_argument("--glmm", action="store_true", help="Generate GLMM predictions plot")
    parser.add_argument("--combined", action="store_true", help="Generate combined GLMM+accuracy plot")
    parser.add_argument("--all", action="store_true", help="Generate all available figures")
    parser.add_argument("--show", action="store_true", help="Display plots")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all:
        figures = generate_all_paper_figures(output_dir=args.output_dir)
        print(f"Generated figures: {list(figures.keys())}")
    else:
        if args.model_accuracy and args.results:
            plot_model_accuracy_sleep(
                args.results,
                output_path=os.path.join(args.output_dir, "model_accuracy.pdf"),
                show_plot=args.show
            )
        
        if args.glmm and args.predictions:
            plot_glmm_predictions(
                args.predictions,
                output_path=os.path.join(args.output_dir, "glmm_predictions.pdf"),
                show_plot=args.show
            )
        
        if args.combined and args.predictions and args.results:
            plot_glmm_with_raw_accuracy(
                args.predictions,
                args.results,
                output_path=os.path.join(args.output_dir, "combined.pdf"),
                show_plot=args.show
            )



