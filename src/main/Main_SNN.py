# Main SNN execution file
from SNN_functions import SNN

# Initiate SNN object
snn = SNN(num_items=40)

# Initialize network
snn.build()

# Generate data
snn.gen_data(var_high_freq=0, var_low_freq=0)

# Train network
snn.train_()

# Plot training
snn.plot_training(overlap=False, tsne=False)
