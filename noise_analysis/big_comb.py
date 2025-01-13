from get_data import create_data
from plot import spike_plot, heat_map
from create_network import create_weigths, create_arrays


class SNN_noisy:
    def __init__(self, N_exc, N_inh):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N = N_exc + N_inh

    # acquire data
    def prepare_data(
        self,
        plot_spikes=False,
        plot_heat_map=False,
        retur=False,
        pixel_size=10,
        num_steps=1000,
        gain=1,
        offset=0,
        first_spike_time=0,
        time_var_input=False,
        download=False,
        num_images=20,
    ):
        self.T = num_images * num_steps
        self.N_x = pixel_size**2

        self.data, self.labels = create_data(
            pixel_size=pixel_size,
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
            download=download,
            num_images=num_images,
        )

        # plot spikes
        if plot_spikes:
            spike_plot(self.data, min_time=0, max_time=10000)

        # plot heatmap of activity
        if plot_heat_map:
            heat_map(self.data, pixel_size=10)

        # return data and labels if needed
        if retur:
            return self.data, self.labels

    def prepare_training(
        self,
        weight_affinity_hidden=0.1,
        weight_affinity_input=0.05,
        resting_membrane=-70,
        pos_weight=1,
        neg_weight=4,
        plot_weights=False,
        retur=False,
    ):
        # create weights
        self.weights = create_weigths(
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            N_x=self.N_x,
            weight_affinity_hidden=weight_affinity_hidden,
            weight_affinity_input=weight_affinity_input,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            plot_weights=plot_weights,
        )

        # create other arrays
        self.mempot, self.S_trace, self.spikes = create_arrays(
            N=self.N,
            resting_membrane=resting_membrane,
            total_time=self.T,
            data=self.data,
            N_x=self.N_x,
        )

        # return results if retur == True
        if retur:
            return self.weights, self.membrane_potential, self.S_trace

    def train_network(self):
        trained_weights = ...