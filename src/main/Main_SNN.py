# Main SNN execution file
from SNN_functions import SNN

# Initiate SNN object
snn = SNN(num_items=60)

# Initialize network
snn.build()

# Generate data
snn.gen_data()

# Train network
snn.train_()

# Plot training
snn.plot_training(overlap=False, traces=False, tsne=False)
