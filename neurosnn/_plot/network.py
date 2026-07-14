"""
Network wiring graph driven entirely by the ACTUAL weight matrix, for the
grouped feedforward architecture (no recurrence).

Layout, left -> right:
  * input: all N_x neurons as a real HxW pixel grid
  * groups: one blob per class group, each holding its excitatory neurons
    (circles) and their 1:1 inhibitory 'hitmen' (diamonds). Blob shape comes
    from a force-directed layout over the real W_ei (exc->inh) and W_ie
    (inh->exc WTA) edges, so the structure is genuine, not hand-drawn.
  * readout: one output node per group (the group->class mapping)

Every edge is read from a weight block:
  * W_se  input pixel -> exc   (each neuron's strongest pixels, width ~ weight)
  * W_ei  exc -> its own inh   (1:1 identity)
  * W_ie  inh -> exc in group  (block-diagonal winner-take-all)
  * readout  exc -> output     (group assignment)

There is no W_ee term: the architecture is feedforward and none is drawn.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


def _darken(rgba, f=0.55):
    return (rgba[0] * f, rgba[1] * f, rgba[2] * f, 1.0)


def plot_network_graph(
    weights: np.ndarray,
    group_assignment: np.ndarray,
    st: int,
    ex: int,
    ih: int,
    out_path: str,
    input_size: int = None,
    n_groups: int = None,
    neurons_per_group: int = 10,
    se_topk: int = 5,
    seed: int = 0,
) -> None:
    """Draw the feedforward wiring graph from real weights.

    Parameters
    ----------
    weights           Full combined weight matrix (N x N).
    group_assignment  (N_exc,) int group index per excitatory neuron.
    st, ex, ih        Block boundaries: input=[0,st), exc=[st,ex), inh=[ex,ih).
    out_path          Save path (.png / .pdf).
    input_size        Input side length; defaults to sqrt(st).
    n_groups          Group count; defaults to group_assignment.max() + 1.
    neurons_per_group Exc neurons (and their 1:1 inh partners) drawn per group.
    se_topk           W_se edges drawn per exc neuron (its strongest pixels).
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
    H = Wd = input_size
    W_se = weights[:st, st:ex]        # (N_x, N_exc)
    W_ie = weights[ex:ih, st:ex]      # (N_inh, N_exc) inh -> exc (block-diagonal WTA)

    cmap = plt.get_cmap("tab10" if n_groups <= 10 else "tab20")
    colors = [cmap(g % cmap.N) for g in range(n_groups)]

    # --- sample neurons per group (exc idx == its 1:1 inh 'hitman' idx) ---
    sampled = {}
    for g in range(n_groups):
        m = np.nonzero(group_assignment == g)[0]
        k = min(neurons_per_group, m.size)
        sampled[g] = rng.choice(m, size=k, replace=False).astype(int) if k else np.array([], int)

    # --- geometry ---
    IX0, IX1, IY0, IY1 = 0.0, 4.0, 3.0, 7.0        # input pixel grid box (square, centered)
    BX = 8.5                                        # group-blob column x
    OX = 12.5                                       # readout column x
    BY = np.linspace(8.7, 0.6, n_groups)           # per-group row centers
    R = 0.52                                        # blob radius (< half the 0.95 spacing)

    def px(r, c):
        x = IX0 + (np.asarray(c) / max(Wd - 1, 1)) * (IX1 - IX0)
        y = IY1 - (np.asarray(r) / max(H - 1, 1)) * (IY1 - IY0)   # row 0 at top
        return x, y

    # --- force-directed blob per group over real W_ei (1:1) + W_ie (WTA) edges ---
    pos_e, pos_h = {}, {}
    for g in range(n_groups):
        s = sampled[g]
        if len(s) == 0:
            continue
        G = nx.Graph()
        for i in s:
            G.add_edge(("e", int(i)), ("h", int(i)), weight=1.0)      # W_ei exc_i -> inh_i
        for i in s:
            for j in s:
                if i != j and W_ie[int(i), int(j)] != 0:
                    G.add_edge(("h", int(i)), ("e", int(j)), weight=0.5)  # W_ie inh_i -> exc_j
        lp = nx.spring_layout(G, weight="weight", seed=seed, iterations=150)
        P = np.array(list(lp.values()))
        ctr = P.mean(0)
        span = np.abs(P - ctr).max() + 1e-9
        for node, p in lp.items():
            xy = (np.array(p) - ctr) / span * R
            X, Y = BX + xy[0], BY[g] + xy[1]
            (pos_e if node[0] == "e" else pos_h)[node[1]] = (X, Y)

    fig, ax = plt.subplots(figsize=(15, 12))

    # --- W_se edges: strongest pixels -> exc, width/alpha ~ real weight ---
    gmax = max((W_se[:, int(i)].max() for g in range(n_groups) for i in sampled[g]
                if W_se[:, int(i)].max() > 0), default=1.0)
    se_segs, se_rgba = [], []
    for g in range(n_groups):
        base = colors[g]
        for i in sampled[g]:
            ii = int(i)
            col = W_se[:, ii]
            if col.max() <= 0 or ii not in pos_e:
                continue
            for pidx in np.argsort(col)[-se_topk:]:
                w = col[pidx]
                if w <= 0:
                    continue
                r, c = divmod(int(pidx), Wd)
                sx, sy = px(r, c)
                se_segs.append([(float(sx), float(sy)), pos_e[ii]])
                se_rgba.append((base[0], base[1], base[2], float(min(0.35, 0.05 + 0.3 * w / gmax))))
    if se_segs:
        ax.add_collection(LineCollection(se_segs, colors=se_rgba, linewidths=0.3, zorder=1))

    # --- W_ie edges: inh -> exc within group (the WTA cliques) ---
    for g in range(n_groups):
        segs = []
        for i in sampled[g]:
            for j in sampled[g]:
                if i != j and W_ie[int(i), int(j)] != 0 and int(i) in pos_h and int(j) in pos_e:
                    segs.append([pos_h[int(i)], pos_e[int(j)]])
        if segs:
            ax.add_collection(LineCollection(segs, colors=[colors[g]], linewidths=0.3,
                                             alpha=0.15, zorder=2))

    # --- W_ei edges: exc_i -> inh_i (1:1 hitman) ---
    ei_segs = [[pos_e[int(i)], pos_h[int(i)]]
               for g in range(n_groups) for i in sampled[g]
               if int(i) in pos_e and int(i) in pos_h]
    if ei_segs:
        ax.add_collection(LineCollection(ei_segs, colors="0.35", linewidths=0.5,
                                         alpha=0.4, zorder=3))

    # --- readout edges: exc -> output (group mapping) ---
    for g in range(n_groups):
        segs = [[pos_e[int(i)], (OX, BY[g])] for i in sampled[g] if int(i) in pos_e]
        if segs:
            ax.add_collection(LineCollection(segs, colors=[colors[g]], linewidths=0.4,
                                             alpha=0.22, zorder=2))

    # --- nodes ---
    rr, cc = np.mgrid[0:H, 0:Wd]
    ix, iy = px(rr.ravel(), cc.ravel())
    ax.scatter(ix, iy, s=4, marker="s", c="0.8", edgecolors="none", zorder=4)

    for g in range(n_groups):
        ecol = [int(i) for i in sampled[g] if int(i) in pos_e]
        if ecol:
            pts = np.array([pos_e[i] for i in ecol])
            drv = np.array([W_se[:, i].sum() for i in ecol])
            sizes = 28 + 80 * (drv - drv.min()) / (np.ptp(drv) + 1e-9)
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, c=[colors[g]], edgecolors="white",
                       linewidths=0.4, zorder=6)
        hcol = [int(i) for i in sampled[g] if int(i) in pos_h]
        if hcol:
            hpts = np.array([pos_h[i] for i in hcol])
            ax.scatter(hpts[:, 0], hpts[:, 1], s=20, marker="D", c=[_darken(colors[g])],
                       edgecolors="black", linewidths=0.4, zorder=5)
        ax.scatter([OX], [BY[g]], s=430, c=[colors[g]], edgecolors="black",
                   linewidths=1.1, zorder=7)
        ax.text(OX, BY[g], str(g), ha="center", va="center", color="white",
                fontweight="bold", fontsize=10, zorder=8)

    ax.text((IX0 + IX1) / 2, IY1 + 0.3, f"input  {H}x{Wd} = {st} neurons",
            ha="center", va="bottom", fontsize=11, color="0.3")
    ax.text(BX, 9.5, "class groups: exc (●) + inh hitmen (◆), intra-group WTA",
            ha="center", va="bottom", fontsize=11, color="0.25")
    ax.text(OX, 9.5, "readout", ha="center", va="bottom", fontsize=11, color="0.25")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[g],
                      markersize=9, label=f"group {g} (class {g})") for g in range(n_groups)]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markersize=9,
               label="excitatory neuron"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="0.2",
               markeredgecolor="black", markersize=8, label="inhibitory hitman"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.8", markersize=8,
               label="input pixel"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
               markeredgecolor="black", markersize=11, label="readout node"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False, title="legend")

    ax.set_title("Network wiring from real weights: input pixels -> exc/inh groups (WTA) -> readout",
                 fontsize=13)
    ax.set_xlim(IX0 - 0.5, OX + 1.4)
    ax.set_ylim(0.0, 10.2)
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Network graph saved -> {out_path}")
