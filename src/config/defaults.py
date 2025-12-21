"""
Default parameters for network and training configuration.

These defaults can be overridden when instantiating the network or training.
"""

# Network architecture defaults
DEFAULT_NETWORK_PARAMS = {
    "N_exc": 200,           # Number of excitatory neurons
    "N_inh": 50,            # Number of inhibitory neurons
    "N_x": 225,             # Input neurons (15x15 for images)
    "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Default: 10 MNIST classes
    
    # Weight initialization
    "w_dense_ee": 0.15,     # Excitatory-to-excitatory density
    "w_dense_se": 0.1,      # Stimulus-to-excitatory density
    "w_dense_ei": 0.2,      # Excitatory-to-inhibitory density
    "w_dense_ie": 0.25,     # Inhibitory-to-excitatory density
    "se_weights": 0.15,     # Stimulus-excitatory weight
    "ee_weights": 0.3,      # Excitatory-excitatory weight
    "ei_weights": 0.3,      # Excitatory-inhibitory weight
    "ie_weights": -0.3,     # Inhibitory-excitatory weight
}

# Training defaults
DEFAULT_TRAINING_PARAMS = {
    # Learning rates
    "learning_rate_exc": 0.0005,
    "learning_rate_inh": 0.0005,
    
    # STDP parameters
    "A_plus": 0.5,
    "A_minus": 0.5,
    "tau_LTP": 10,
    "tau_LTD": 10,
    
    # Membrane dynamics
    "tau_m": 30,
    "tau_syn": 30,
    "membrane_resistance": 30,
    "resting_potential": -70,
    "reset_potential": -80,
    "spike_threshold_default": -55,
    
    # Sleep parameters
    "sleep": True,
    "sleep_ratio": 0.2,
    "check_sleep_interval": 35000,
    "sleep_mode": "static",
    
    # Noise parameters
    "noisy_potential": True,
    "noisy_threshold": False,
    "noisy_weights": False,
    "var_noise": 2,
    "mean_noise": 0,
    
    # Spike adaptation
    "spike_adaption": True,
    "tau_adaption": 100,
    "delta_adaption": 3,
    
    # Training control
    "train_weights": True,
    "timing_update": True,
    "trace_update": False,
    "vectorized_trace": False,
    
    # Accuracy estimation
    "accuracy_method": "pca_lr",
    "pca_variance": 0.95,
    "narrow_top": 0.2,
}

# Dataset defaults
DEFAULT_DATA_PARAMS = {
    # Image data
    "all_images_train": 6000,
    "batch_image_train": 400,
    "all_images_test": 1000,
    "batch_image_test": 200,
    "all_images_val": 100,
    "batch_image_val": 100,
    
    # Timing
    "num_steps": 100,      # Timesteps per sample
    "gain": 1.0,            # Rate coding gain
    
    # Preprocessing
    "noisy_data": False,
    "noise_level": 0.0,
}

# Geomfig-specific defaults
GEOMFIG_PARAMS = {
    "classes": [0, 1, 2, 3],  # triangle, circle, square, x
    "pixel_size": 15,
    "noise_var": 0.2,
    "noise_mean": 0.0,
    "jitter": False,
    "jitter_amount": 0.05,
    "gain": 0.5,
}

