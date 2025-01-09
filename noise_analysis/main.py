from get_data import create_data
from plot import spike_plot, heat_map

data, labels = create_data(
    pixel_size=10,
    num_steps=1000,
    gain=1,
    offset=0,
    first_spike_time=0,
    time_var_input=False,
    download=False,
    num_images=20,
)

# plot spikes
spike_plot(data, min_time=0, max_time=10000)

# plot heatmap of activity
heat_map(data, pixel_size=10)
