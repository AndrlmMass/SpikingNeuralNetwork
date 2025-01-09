# Main SNN execution file
from main.SNN_functions import SNN

# Initiate SNN object
snn = SNN(num_items=20)

# Initialize network
snn.build(load_model_if_available=False)

# Generate data
snn.gen_data(var_high_freq=0, var_low_freq=0, force_new_data=False)

# Train network
snn.train_(force_retrain=True, run_njit=True)

# Plot training
snn.plot_training(traces=False, overlap=False, tsne=False)
