import numpy as np

d = np.random.poisson(lam=30, size=1000)
print(np.mean(d))
print(d)
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0  # Total time in seconds
lambda_rate = 20  # Firing rate in Hz
delta_t = 0.001  # Bin size in seconds (1 ms)
num_bins = int(T / delta_t)

# Spike probability per bin
p = lambda_rate * delta_t

# Generate binary sequence
spike_sequence = np.random.rand(num_bins) < p
spike_sequence = spike_sequence.astype(int)

# Visualization
time = np.linspace(0, T, num_bins)
plt.figure(figsize=(12, 2))
plt.eventplot(np.where(spike_sequence)[0] * delta_t, lineoffsets=1, colors="black")
plt.xlabel("Time (s)")
plt.yticks([])
plt.title("Poisson Binary Spike Sequence")
plt.show()
