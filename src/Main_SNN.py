# Main SNN execution file
from main.SNN_functions import SNN

# Check if jax-jit with NVIDIA GPU support is running
import jax

print("Sanity check:", jax.devices())

# Initiate SNN object
snn = SNN()

# Initialize network
snn.build()

# Generate data
snn.gen_data(var_high_freq=0, var_low_freq=0, force_new_data=True)

# Train network
snn.train_()

# Plot training
snn.plot_training(traces=False, overlap=False, tsne=False)
