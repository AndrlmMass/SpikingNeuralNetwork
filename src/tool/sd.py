# t = np.arange(t_start, t_stop)
# print(t.shape[0])
# MemPot_th_ = np.full(shape=t.shape[0], fill_value=MemPot_th)

# # Get membrane potentials for excitatory and inhibitory neurons
# MemPot_exc = MemPot[t_start:t_stop, :N_excit_neurons]
# MemPot_inh = MemPot[t_start:t_stop, N_excit_neurons:]

# # Compute mean, min, and max membrane potentials over neurons for each time step
# exc_mean = np.mean(MemPot_exc, axis=1)
# exc_min = np.min(MemPot_exc, axis=1)
# exc_max = np.max(MemPot_exc, axis=1)

# inh_mean = np.mean(MemPot_inh, axis=1)
# inh_min = np.min(MemPot_inh, axis=1)
# inh_max = np.max(MemPot_inh, axis=1)

# # Plot mean membrane potentials
# plt.plot(t, exc_mean, color="red", label="Excitatory Mean")
# plt.plot(t, inh_mean, color="blue", label="Inhibitory Mean")
# plt.plot(t, MemPot_th_, color="grey", linestyle="dashed", label="Threshold")

# # Fill between min and max to create shaded area
# plt.fill_between(
#     t, exc_min, exc_max, color="red", alpha=0.2, label="Excitatory Range"
# )
# plt.fill_between(
#     t, inh_min, inh_max, color="blue", alpha=0.2, label="Inhibitory Range"
# )

import numpy as np
import matplotlib.pyplot as plt


d = np.array([[2, 3], [4, 5]])

plt.plot(d)
plt.show()
