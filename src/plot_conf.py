import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.set_xlim(-0.5, 14)
ax.set_ylim(-1.8, 3.2)
ax.axis("off")

box_w, box_h = 3.0, 1.5
y0 = 0.0
positions = [1.5, 6.5, 11.5]

labels = [
    "Stimulation\n$N = 784$",
    "Excitatory\n$N_{\\mathrm{exc}} = 1024$",
    "Inhibitory\n$N_{\\mathrm{inh}} = 225$",
]
colors = ["#D6EAF8", "#D5F5E3", "#FADBD8"]
edgecolors = ["#4E79A7", "#F28E2B", "#76B7B2"]

for xc, label, fc, ec in zip(positions, labels, colors, edgecolors):
    rect = mpatches.FancyBboxPatch(
        (xc - box_w / 2, y0),
        box_w,
        box_h,
        boxstyle="round,pad=0.07",
        linewidth=2,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        xc,
        y0 + box_h / 2,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="serif",
    )

ymid = y0 + box_h / 2


def solid_arrow(x1, x2, y, label):
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=2.0, mutation_scale=18),
        zorder=4,
    )
    ax.text((x1 + x2) / 2, y + 0.18, label, ha="center", va="bottom", fontsize=10)


# Input -> Excitatory
solid_arrow(positions[0] + box_w / 2, positions[1] - box_w / 2, ymid, "$P = 5\\%$")

# Excitatory -> Inhibitory
solid_arrow(positions[1] + box_w / 2, positions[2] - box_w / 2, ymid, "$P = 5\\%$")

# Recurrent self-loop (arc above Excitatory)
xc = positions[1]
ytop = y0 + box_h
theta = np.linspace(np.pi, 0, 100)
rx, ry = 0.65, 0.75
lx = xc + rx * np.cos(theta)
ly = ytop + ry * np.sin(theta)
ax.plot(lx, ly, color="#2C3E50", lw=2.0, zorder=4)
ax.annotate(
    "",
    xy=(lx[-1], ly[-1]),
    xytext=(lx[-2], ly[-2]),
    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=2.0, mutation_scale=18),
    zorder=5,
)
ax.text(xc, ytop + ry + 0.15, "$P = 2.5\\%$", ha="center", va="bottom", fontsize=10)

# Inhibitory feedback (dashed arc below)
x_start = positions[2] - 0.5
x_end = positions[1] + 0.5
cx = (x_start + x_end) / 2
rx2 = (x_start - x_end) / 2
ry2 = 0.75
theta2 = np.linspace(0, np.pi, 100)
lx2 = cx + rx2 * np.cos(theta2)
ly2 = y0 - ry2 * np.sin(theta2)
ax.plot(lx2, ly2, color="#2C3E50", lw=2.0, linestyle="dashed", zorder=4)
ax.annotate(
    "",
    xy=(lx2[-1], ly2[-1]),
    xytext=(lx2[-2], ly2[-2]),
    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=2.0, mutation_scale=18),
    zorder=5,
)
ax.text(cx, y0 - ry2 - 0.22, "$P = 5\\%$", ha="center", va="top", fontsize=10)

plt.tight_layout()
plt.savefig("architecture.pdf", bbox_inches="tight", dpi=300)
