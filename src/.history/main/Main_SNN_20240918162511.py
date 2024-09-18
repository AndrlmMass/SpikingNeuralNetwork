# Main SNN execution file
from SNN_functions import SNN

# Initiate SNN object
snn = SNN(num_items=4)

# Initialize network
snn.build(load_model_if_available=False)

# Generate data
snn.gen_data()

# Train network
snn.train_()

# Plot training
snn.plot_training(overlap=False, tsne=False)
