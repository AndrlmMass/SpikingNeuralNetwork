# Main SNN execution file

# Set cw
import os
import sys

# Set current working directories and add relevant directories to path
if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork"
):
    os.chdir(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
else:
    os.chdir(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
from SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(
    V_th=-50,  # Firing threshold
    V_reset=-70,  # Reset potential
    P=20,  # Ideal weight scaler
    C=1,  # Regulates B which regulates LTP
    R=100,  # Scales up the I_in value
    tau_plus=0.15,  # Time constant in presynaptic learning
    tau_minus=0.15,  # Time constant in postsynaptic learning
    tau_slow=0.09,  # Time constant in slow postsynaptic learning
    tau_m=0.15,  # Time constant in membrane potential
    tau_ht=0.15,  # Time constant in heterosynaptic learning
    tau_hom=0.15,  # Time constant in doublet LTD
    num_items=4,  # Num of training items
    dt=0.001,  # timeunit
    T=0.1,  # total time per item
    V_rest=-60,  # Resting potential
    min_weight=0,  # Minimum weight
    max_weight=5,  # Maximum weight
    num_epochs=1,  # N/A
    init_cals=1,  # N/A
    A=0.0001,  # Regulates hebbian learning -> larger == more hebbian learning - fixed
    B=0.01,  # Regulates hebbian learning -> larger == less hebbian learning
    beta=0.005,  # Regulates heterosynpatic learning -> larger == more heterosynaptic learning - fixed
    delta=0.00002,  # Regulates dopamin_reg - Fixed
    tau_const=0.0001,  # Time constant in ideal_weight update
)

# Initialize & visualize pre-trained network
(
    MemPot,
    spikes,
    W_se,
    W_ee,
    W_ei,
    W_ie,
    W_se_ideal,
    W_ee_ideal,
    W_ei_ideal,
    W_ie_ideal,
) = snn.initialize_network(
    N_input_neurons=484,
    N_excit_neurons=484,
    N_inhib_neurons=121,
    radius_=4,
    W_ee_prob=0.1,
    retur=True,
)
# Generate data
snn.gen_data(
    run=False,
    N_classes=4,
    noise_rand=True,
    noise_rand_ls=[0, 0.01, 0.03, 0.05],
    mean=0,
    blank_variance=0.01,
    input_scaler=20,
    save=True,
    retur=False,
)

# Load data
data, labels = snn.load_data(rand_lvl=0.05, retur=True)

# Visualize network


# Train network
(
    spikes,
    MemPot,
    W_se,
    W_ee,
    W_ei,
    W_ie,
) = snn.train_data(
    retur=True,
    w_p=0.5,
    interactive_tool=False,
    update_frequency=5,
)

# Evaluate performance

# Test network on unseen data and estimate performance

# Visualize training and testing results
snn.plot_training(
    ws_nd_spikes=True,
    idx_start=0,
    idx_stop=100,
    mv=True,
)

snn.plot_I_in()
