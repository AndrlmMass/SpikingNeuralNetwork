import numpy as np
from snntorch.spikegen import target_rate_code

total_duration = 1000
rate = 30 / 1000
d = 0.5
p = target_rate_code(num_steps=total_duration, rate=rate, firing_pattern="poisson")
print(np.sum(p)(total_duration))
