import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.use("TkAgg")


T = 1000
eta = 0.997
n_exc = 800
n_inh = 200
N = n_exc + n_inh
w_target_exc = 0.2
w_target_inh = -0.2
# plot weight decay method
weights = np.zeros((T, N))
weights[0] = np.random.uniform(-1, 1, N)
x = np.arange(0, T, 1)

for i in range(1, T):
    weights[i, :-n_inh] = w_target_exc * (weights[i - 1, :-n_inh] / w_target_exc) ** (
        eta
    )
    weights[i, -n_inh:] = w_target_inh * (weights[i - 1, -n_inh:] / w_target_inh) ** (
        eta
    )
# Create color gradients for each group using colorblind-friendly colormaps:
exc_colors = plt.cm.autumn(np.linspace(0, 1, n_exc))
inh_colors = plt.cm.winter(np.linspace(0, 1, n_inh))

# Plot excitatory weights
for i in range(n_exc):
    plt.plot(x, weights[:, i], color=exc_colors[i])
# Plot inhibitory weights
for j in range(n_inh):
    plt.plot(x, weights[:, n_exc + j], color=inh_colors[j])

# Optional: create a custom legend with representative colors.
legend_elements = [
    Line2D([0], [0], color=plt.get_cmap("autumn")(0.5), lw=2, label="Excitatory"),
    Line2D([0], [0], color=plt.get_cmap("winter")(0.5), lw=2, label="Inhibitory"),
    Line2D(
        [0],
        [0],
        color=plt.get_cmap("autumn")(0.5),
        lw=2,
        linestyle="--",
        label="Exc target",
    ),
    Line2D(
        [0],
        [0],
        color=plt.get_cmap("winter")(0.5),
        lw=2,
        linestyle="--",
        label="Inh target",
    ),
]
plt.legend(handles=legend_elements)
plt.axhline(y=w_target_exc, color=plt.get_cmap("autumn")(0.5), linestyle="--", lw=3)
plt.axhline(y=w_target_inh, color=plt.get_cmap("winter")(0.5), linestyle="--", lw=3)
plt.xlabel("Time (ms)", size=16)
plt.ylabel("Δw", size=16)
plt.show()
