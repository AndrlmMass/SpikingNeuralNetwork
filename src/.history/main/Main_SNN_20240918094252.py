# Main SNN execution file
from SNN_functions import SNN

# Initiate SNN object
snn = SNN(num_items=4)

# Initialize network
snn.build(N_excit_neurons=4096, N_inhib_neurons=1024, N_input_neurons=4096)

# Generate data
snn.gen_data()

# Train network
snn.train_(run_njit=False)

# Plot training
snn.plot_training(overlap=False, traces=False, tsne=False)
