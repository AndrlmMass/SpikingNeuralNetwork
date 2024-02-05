import numpy as np

def generate_multidimensional_data(num_classes, base_mean, mean_increment, variance, num_samples_per_class,
                                   features, num_timesteps, num_items, dt, input_scaler):
    """
    Generate x-dimensional data with specified number of classes from multivariate normal distributions.

    :param num_classes: Number of classes.
    :param base_mean: Base mean for the first distribution.
    :param mean_increment: Increment to be added to the mean for each subsequent class.
    :param variance: Variance for the distributions.
    :param num_samples_per_class: Number of samples to generate per class.
    :param x: Number of dimensions for the data.
    :return: A numpy array of the combined data and labels.
    """
    combined_pts = []
    combined_labels = []

    for class_id in range(num_classes):
        mean = base_mean + mean_increment * class_id
        cov = np.eye(features) * variance
        mean_vector = np.full(features, mean)

        # Generate data points for this class
        pts = np.random.multivariate_normal(mean_vector, cov, num_samples_per_class)
        labels = np.full(num_samples_per_class, class_id, dtype=int)

        combined_pts.append(pts)
        combined_labels.append(labels)

    # Combine and shuffle the datasets
    combined_pts = np.vstack(combined_pts)
    combined_labels = np.concatenate(combined_labels)
    combined_data = np.column_stack((combined_pts, combined_labels))
    np.random.shuffle(combined_data)

    print(combined_data.shape)
    # Convert float-based array to spike-based
    data, classes = encode_input_poisson(combined_data, num_timesteps, features, num_items, dt, input_scaler)
    
    return data, classes