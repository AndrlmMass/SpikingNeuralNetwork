"""
Force-directed network graph (ggraph/igraph style) driven by the ACTUAL weight
matrix. Node positions are computed by a spring layout over the real edges, so
the class-group clusters emerge from the genuine connectivity rather than being
drawn by hand.

The honest structure of the grouped architecture:
  * exc-exc coupling comes from W_ie (block-diagonal WTA) and W_ee (recurrence,
    zero in the feedforward config) -> each class group is a densely intra-
    connected clique, so force-direction separates the 10 groups automatically.
  * feedforward drive comes from W_se; here it is shown as edges from pooled
    input-region nodes to the exc neurons they most strongly drive.
  * the readout mapping is group -> output, drawn as one output node per group.

Every edge is read from the weight blocks (or the group/readout assignment); no
synthetic connections are invented.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _pool_se(W_se, input_size, pool):
    """Sum-pool each W_se column (input_size x input_size) into pool x pool
    regions. Returns (pool*pool, N_exc) pooled feedforward weights and the
    (row, col) center of each region in original pixel coords."""
    N_x, N_exc = W_se.shape
    H = W = input_size
    bh = max(1, H // pool)
    bw = max(1, W // pool)
    cols = W_se.reshape(H, W, N_exc)
    pooled = np.zeros((pool, pool, N_exc))
    centers = np.zeros((pool, pool, 2))
    for r in range(pool):
        for c in range(pool):
            r0, r1 = r * bh, (r + 1) * bh if r < pool - 1 else H
            c0, c1 = c * bw, (c + 1) * bw if c < pool - 1 else W
            pooled[r, c] = cols[r0:r1, c0:c1].sum(axis=(0, 1))
            centers[r, c] = [(r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0]
    return pooled.reshape(pool * pool, N_exc), centers.reshape(pool * pool, 2)


def plot_network_graph(
    weights: np.ndarray,
    group_assignment: np.ndarray,
    st: int,
    ex: int,
    ih: int,
    out_path: str,
    input_size: int = None,
    n_groups: int = None,
    neurons_per_group: int = 28,
    include_input: bool = True,
    input_pool: int = 6,
    se_topk: int = 2,
    include_output: bool = True,
    seed: int = 0,
) -> None:
    """Render a force-directed graph of the network from real weights.

    Parameters
    ----------
    weights           Full combined weight matrix (N x N).
    group_assignment  (N_exc,) int group index per excitatory neuron.
    st, ex, ih        Block boundaries: input=[0,st), exc=[st,ex), inh=[ex,ih).
    out_path          Save path (.png / .pdf).
    input_size        Input side length; defaults to sqrt(st).
    n_groups          Group count; defaults to group_assignment.max() + 1.
    neurons_per_group Exc neurons sampled per group (legibility).
    include_input     Add pooled input-region nodes + real W_se edges.
    input_pool        Input pooled into input_pool x input_pool region nodes.
    se_topk           Connect each exc neuron to its top-k input regions by W_se.
    include_output    Add one readout node per group (group->output mapping).
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("plot_network_graph requires networkx (pip install networkx)")

    rng = np.random.default_rng(seed)
    if input_size is None:
        input_size = int(round(np.sqrt(st)))
    if n_groups is None:
        n_groups = int(group_assignment.max()) + 1

    W_se = weights[:st, st:ex]                 # (N_x, N_exc)
    W_ee = weights[st:ex, st:ex]               # (N_exc, N_exc) recurrence
    W_ie = weights[ex:ih, st:ex]               # (N_inh, N_exc) block-diagonal WTA

    cmap = plt.get_cmap("tab10" if n_groups <= 10 else "tab20")
    colors = [cmap(g % cmap.N) for g in range(n_groups)]

    # --- sample neurons per group ---
    sampled = {}
    for g in range(n_groups):
        members = np.nonzero(group_assignment == g)[0]
        k = min(neurons_per_group, members.size)
        sampled[g] = rng.choice(members, size=k, replace=False) if k else np.array([], int)
    all_exc = np.concatenate([sampled[g] for g in range(n_groups)]).astype(int)
    input_drive = np.asarray(W_se[:, all_exc].sum(axis=0)).ravel()  # feedforward strength

    G = nx.Graph()
    for j in all_exc:
        G.add_node(("e", int(j)), kind="exc", group=int(group_assignment[j]))

    # --- real exc-exc edges: W_ie (WTA) + W_ee (recurrence) ---
    intra_edges, recur_edges = [], []
    for g in range(n_groups):
        s = sampled[g]
        for a_ in range(len(s)):
            for b_ in range(a_ + 1, len(s)):
                i, j = int(s[a_]), int(s[b_])
                w = abs(W_ie[i, j]) + abs(W_ie[j, i])
                if w > 0:
                    G.add_edge(("e", i), ("e", j), weight=1.0, kind="wta")
                    intra_edges.append((("e", i), ("e", j)))
    if (W_ee != 0).any():                      # recurrence, if enabled (may cross groups)
        for a_idx, i in enumerate(all_exc):
            row = W_ee[i, all_exc]
            for b_idx, j in enumerate(all_exc):
                if b_idx <= a_idx or row[b_idx] == 0:
                    continue
                G.add_edge(("e", int(i)), ("e", int(j)),
                           weight=0.6 * float(abs(row[b_idx])), kind="recur")
                recur_edges.append((("e", int(i)), ("e", int(j))))

    # --- real feedforward edges: pooled W_se -> exc top-k regions ---
    input_edges = []
    region_centers = None
    if include_input:
        pooled, region_centers = _pool_se(W_se, input_size, input_pool)   # (R, N_exc)
        for r in range(pooled.shape[0]):
            G.add_node(("i", r), kind="input")
        for j in all_exc:
            col = pooled[:, j]
            if col.max() <= 0:
                continue
            top = np.argsort(col)[-se_topk:]
            for r in top:
                if col[r] <= 0:
                    continue
                G.add_edge(("i", int(r)), ("e", int(j)),
                           weight=0.12, kind="se")
                input_edges.append((("i", int(r)), ("e", int(j))))

    # --- readout mapping: group -> output ---
    output_edges = []
    if include_output:
        for g in range(n_groups):
            G.add_node(("o", g), kind="output", group=g)
            for j in sampled[g]:
                G.add_edge(("o", g), ("e", int(j)), weight=0.35, kind="readout")
                output_edges.append((("o", g), ("e", int(j))))

    # --- force-directed layout over the real edges ---
    pos = nx.spring_layout(G, weight="weight", seed=seed,
                           k=1.6 / np.sqrt(max(G.number_of_nodes(), 1)),
                           iterations=200)

    fig, ax = plt.subplots(figsize=(12, 10))

    def seg(edges):
        return [[pos[u], pos[v]] for u, v in edges]

    from matplotlib.collections import LineCollection
    if input_edges:
        ax.add_collection(LineCollection(seg(input_edges), colors="0.75",
                                         linewidths=0.3, alpha=0.35, zorder=1))
    # intra-group WTA edges in group color
    for g in range(n_groups):
        ge = [(u, v) for u, v in intra_edges
              if G.nodes[u].get("group") == g and G.nodes[v].get("group") == g]
        if ge:
            ax.add_collection(LineCollection(seg(ge), colors=[colors[g]],
                                             linewidths=0.35, alpha=0.30, zorder=2))
    if recur_edges:
        ax.add_collection(LineCollection(seg(recur_edges), colors="0.5",
                                         linewidths=0.4, alpha=0.4, zorder=2))
    # readout edges in group color
    for g in range(n_groups):
        oe = [(u, v) for u, v in output_edges if u == ("o", g)]
        if oe:
            ax.add_collection(LineCollection(seg(oe), colors=[colors[g]],
                                             linewidths=0.5, alpha=0.35, zorder=3))

    # nodes
    if include_input:
        ipos = np.array([pos[("i", r)] for r in range(input_pool * input_pool)])
        ax.scatter(ipos[:, 0], ipos[:, 1], s=60, marker="s", c="0.6",
                   edgecolors="white", linewidths=0.5, zorder=4, label="_input")
    exc_pos = np.array([pos[("e", int(j))] for j in all_exc])
    exc_col = [colors[int(group_assignment[j])] for j in all_exc]
    sizes = 30 + 120 * (input_drive - input_drive.min()) / (np.ptp(input_drive) + 1e-9)
    ax.scatter(exc_pos[:, 0], exc_pos[:, 1], s=sizes, c=exc_col,
               edgecolors="white", linewidths=0.4, zorder=5)
    if include_output:
        for g in range(n_groups):
            p = pos[("o", g)]
            ax.scatter([p[0]], [p[1]], s=430, c=[colors[g]], edgecolors="black",
                       linewidths=1.1, zorder=6)
            ax.text(p[0], p[1], str(g), ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white", zorder=7)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[g],
                      markersize=9, label=f"group {g} (class {g})")
               for g in range(n_groups)]
    if include_output:
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
                              markeredgecolor="black", markersize=12, label="readout node"))
    if include_input:
        handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="0.6",
                              markersize=9, label="input region (pooled W_se)"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False, title="legend")

    ax.set_title("Network graph (force-directed, real weights): "
                 "WTA groups + W_se input + readout", fontsize=12)
    ax.axis("off")
    ax.autoscale()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Network graph saved -> {out_path}")
