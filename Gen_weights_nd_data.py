import numpy as np

def generate_small_world_network_power_law(num_neurons, excit_inhib_ratio, alpha, num_input_neurons):
    n_rows, n_cols = num_neurons, num_neurons

    # Generate power-law distribution for the probability of connections
    connection_probabilities = np.random.power(alpha, size=n_rows * n_cols)

    # Initialize the arrays for weights and signs
    weight_array = np.ones((n_rows, n_cols))
    np.fill_diagonal(weight_array, 0)

    # Add weights to input neurons
    input_indices = np.random.choice(np.arange(num_neurons),num_input_neurons)
    weight_array[input_indices,:] = np.zeros(shape=num_neurons)

    # Assign weights and signs based on connection probability
    for i in range(n_rows):
        for j in range(n_cols):  
            if weight_array[i, j] != 0:
                if connection_probabilities[i * n_cols + j] > np.random.rand():
                    # Assign weights
                    const = 1 if np.random.rand() < excit_inhib_ratio else -1
                    weight_array[i, j] = np.random.rand() * const
                    weight_array[j, i] = 0

                else:
                    weight_array[i, j] = 0

    # Add connections to neurons without post-synaptic connections
    if np.any(np.all(weight_array == 0, axis=0)):
        for j in range(num_neurons):
            if np.all(weight_array[:,j] == 0):
                idx = np.random.choice(num_neurons)
                if idx == j:
                    idx = np.random.choice(num_neurons)
                else:
                    const = 1 if np.random.rand() < excit_inhib_ratio else -1
                    weight_array[idx, j] = np.random.rand() * const
            if np.any(np.all(weight_array == 0, axis=0)) == False:
                break
                    

    # Calculate ratio of excitatory to inhibitory connections
    print(f"This is the current ratio of positive edges to all edges: {np.sum(weight_array > 0)/np.sum(weight_array != 0)}")
    print(np.where(np.all(weight_array == 0, axis=1)))
    print(weight_array)

    return weight_array


def encode_input_poisson(input, num_timesteps, num_neurons, num_items, dt, input_scaler):
    print(input.shape)
    labels = input[:,-1]
    input = input[:,0:num_neurons]
    print(input.shape)
    
    # 2D-array: items x neurons
    poisson_input = np.zeros((num_timesteps, num_neurons, num_items))

    for i in range(num_items):
        for j in range(num_neurons):
            # Calculate the mean spike count for the Poisson distribution
            # Assuming 'input' is the rate (spikes/sec), we multiply by 'dt' to get the average number of spikes per time step
            lambda_poisson = input[i, j]*dt*input_scaler

            # Generate spikes using Poisson distribution
            for t in range(num_timesteps):
                spike_count = np.random.poisson(lambda_poisson)
                poisson_input[t, j, i] = 1 if spike_count > 0 else 0

    return poisson_input, labels

def prep_data(m1, m2, v1, v2,  num_timesteps, num_neurons, 
              num_items, dt, input_scaler):
    
    # Simulate data
    cov1 = np.array([[v1,0],[0,v1]])
    cov2 = np.array([[v2,0],[0,v2]])
    pts1 = np.random.multivariate_normal([m1,m1], cov1, size=200)
    pts2 = np.random.multivariate_normal([m2,m2], cov2, size=200)
    
    # Create labels for the datasets
    labels1 = np.zeros(pts1.shape[0], dtype=int)  # Class 0
    labels2 = np.ones(pts2.shape[0], dtype=int)   # Class 1

    # Combine the datasets
    combined_pts = np.vstack((pts1, pts2))
    combined_labels = np.concatenate((labels1, labels2))

    # Combine the labels with the datasets
    combined_data = np.column_stack((combined_pts, combined_labels))

    # Shuffle the combined data
    np.random.shuffle(combined_data)

    data, classes = encode_input_poisson(combined_data,  num_timesteps, num_neurons, num_items, dt, input_scaler)
    return data, classes