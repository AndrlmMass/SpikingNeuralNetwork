# Main SNN execution file
from SNN_functions import SNN

# Initiate SNN object
snn = SNN()

# Initialize network
snn.build()

# Generate data
snn.gen_data()

# Train network
snn.train_()

# Plot training
snn.plot_training()
