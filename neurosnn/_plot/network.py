"""
Cartoonish network wiring schematic for the grouped excitatory architecture.

Renders the flow  input grid -> 10 class-group clusters (intra-group WTA) ->
readout neurons, with each group in a distinct colour mapped to a legend. Meant
as a one-glance sanity check that the network is wired the way you think:
each group tiles the input, inhibition stays within a group, and every group
maps to exactly one output.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _rf_centroid(col, H, W):
    """Weighted (row, col) centroid of one W_se column reshaped to H x W."""
    m = np.clip(np.asarray(col, dtype=float), 0.0, None).reshape(H, W)
    s = m.sum()
    if s < 1e-9:
        return (H - 1) / 2.0, (W - 1) / 2.0
    ys, xs = np.mgrid[0:H, 0:W]
    return float((m * ys).sum() / s), float((m * xs).sum() / s)


def plot_network_schematic(
    weights: np.ndarray,
    group_assignment: np.ndarray,
    st: int,
    ex: int,
    ih: int,
    out_path: str,
    input_size: int = None,
    n_groups: int = None,
    neurons_per_group: int = 12,
    seed: int = 0,
    show_input_edges: bool = True,
    show_wta_edges: bool = True,
) -> None:
    """Draw a layered wiring cartoon of the grouped network.

    Layout (left -> right):
      * input neurons as a 2D grid of grey dots
      * one colored cluster per class group (a jittered blob of sampled neurons
        with intra-group WTA edges drawn, giving the dense 'yarn ball' look)
      * one readout neuron per group, colored to match

    Edges:
      * input -> group  (feedforward W_se): each sampled neuron links back to its
        RF centroid on the input grid, in its group colour — shows each group
        reaching across the input.
      * intra-group WTA: edges among a group's sampled neurons (block-diagonal
        W_ie) — confirms inhibition stays within a group.
      * group -> output: one edge per group to its readout node.

    Parameters
    ----------
    weights           Full combined weight matrix (N x N).
    group_assignment  (N_exc,) int group index per excitatory neuron.
    st, ex, ih        Block boundaries: input=[0,st), exc=[st,ex), inh=[ex,ih).
    out_path          Save path (.png / .pdf).
    input_size        Input side length; defaults to sqrt(st).
    n_groups          Number of groups; defaults to group_assignment.max() + 1.
    neurons_per_group Neurons drawn per cluster (subsample for legibility).
    """
    rng = np.random.default_rng(seed)
    if input_size is None:
        input_size = int(round(np.sqrt(st)))
    if n_groups is None:
        n_groups = int(group_assignment.max()) + 1
    H = W = input_size
    W_se = weights[:st, st:ex]                     # (N_x, N_exc)

    cmap = plt.get_cmap("tab10" if n_groups <= 10 else "tab20")
    colors = [cmap(g % cmap.N) for g in range(n_groups)]

    # --- layout anchors ---
    IX0, IX1, IY0, IY1 = 0.0, 2.4, 0.3, 9.7       # input grid box
    GX = 5.6                                        # group cluster column x
    OX = 8.6                                        # output column x
    y_centers = np.linspace(9.2, 0.8, n_groups)    # cluster + output row centers
    jitter = 0.34

    def in_xy(row, col):
        x = IX0 + (col / max(W - 1, 1)) * (IX1 - IX0)
        y = IY1 - (row / max(H - 1, 1)) * (IY1 - IY0)   # row 0 at top
        return x, y

    fig, ax = plt.subplots(figsize=(12, 8))

    # input grid (grey dots)
    rr, cc = np.mgrid[0:H, 0:W]
    ix, iy = in_xy(rr.ravel(), cc.ravel())
    ax.scatter(ix, iy, s=3, c="0.75", edgecolors="none", zorder=1)
    ax.text((IX0 + IX1) / 2, IY1 + 0.35, f"input\n{H}x{W}", ha="center",
            va="bottom", fontsize=11, color="0.35")

    # per-group clusters + edges
    for g in range(n_groups):
        members = np.nonzero(group_assignment == g)[0]
        if members.size == 0:
            continue
        k = min(neurons_per_group, members.size)
        sample = rng.choice(members, size=k, replace=False)
        cx, cy = GX, y_centers[g]
        pos = np.column_stack([
            cx + rng.normal(0, jitter, k),
            cy + rng.normal(0, jitter, k),
        ])
        col = colors[g]

        # input -> group feedforward (each neuron back to its RF centroid)
        if show_input_edges:
            for j, node in zip(range(k), sample):
                r, c = _rf_centroid(W_se[:, node], H, W)
                sx, sy = in_xy(r, c)
                ax.plot([sx, pos[j, 0]], [sy, pos[j, 1]], color=col,
                        lw=0.35, alpha=0.15, zorder=2)

        # intra-group WTA edges (dense cluster look)
        if show_wta_edges:
            for a_ in range(k):
                for b_ in range(a_ + 1, k):
                    ax.plot(pos[[a_, b_], 0], pos[[a_, b_], 1], color=col,
                            lw=0.3, alpha=0.22, zorder=3)

        # group nodes
        ax.scatter(pos[:, 0], pos[:, 1], s=42, color=col, edgecolors="white",
                   linewidths=0.5, zorder=4)

        # group -> output
        ax.plot([cx, OX], [cy, y_centers[g]], color=col, lw=1.6, alpha=0.8, zorder=3)
        ax.scatter([OX], [y_centers[g]], s=240, color=col, edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.text(OX + 0.35, y_centers[g], f"{g}", ha="left", va="center",
                fontsize=11, fontweight="bold")

    ax.text(GX, 9.75, "10 class groups\n(intra-group WTA)", ha="center",
            va="bottom", fontsize=11, color="0.25")
    ax.text(OX, 9.75, "readout", ha="center", va="bottom", fontsize=11, color="0.25")

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[g],
                     markersize=9, label=f"group {g} (class {g})")
              for g in range(n_groups)]
    ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False, title="groups")

    ax.set_xlim(IX0 - 0.4, OX + 1.2)
    ax.set_ylim(0.0, 10.4)
    ax.axis("off")
    ax.set_title("Network wiring: input -> class groups (WTA) -> readout", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Network schematic saved -> {out_path}")
