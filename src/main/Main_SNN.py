# Main SNN execution file
from SNN_functions import SNN

# Initiate SNN object
snn = SNN()

# Initialize network
snn.build()

# Generate data
snn.gen_data()

# Train network
snn.train_(run_njit=False)

# Plot training
snn.plot_training(t_start=0, t_stop=2000)
