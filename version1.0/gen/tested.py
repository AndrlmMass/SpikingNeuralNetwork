import numpy as np
from snntorch.spikegen import target_rate_code

total_duration = 1000
rate = 30 / 1000
d = 0.5
p = target_rate_code(num_steps=1000, rate=0.03, firing_pattern="poisson")
print(np.sum(p[0].numpy()))
print(p[0])
