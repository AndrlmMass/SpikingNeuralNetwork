# Main SNN execution file
from SNN_functions import SNN

# Initiate SNN object
snn = SNN(num_items=400)

# Initialize network
snn.build(load_model_if_available=False)

# Generate data
snn.gen_data(var_high_freq=0, var_low_freq=0)

# Train network
snn.train_(force_retrain=True)

# Plot training
snn.plot_training(traces=False, overlap=False, tsne=False)
