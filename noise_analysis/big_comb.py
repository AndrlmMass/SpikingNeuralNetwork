from train import train_network
from get_data import create_data
from plot import spike_plot, heat_map, mp_plot
from create_network import create_weigths, create_arrays


class SNN_noisy:
    def __init__(self, N_exc=200, N_inh=50, N_x=100):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_x = N_x
        self.N = N_exc + N_inh + N_x

    # acquire data
    def prepare_data(
        self,
        plot_spikes=False,
        plot_heat_map=False,
        retur=False,
        num_steps=1000,
        gain=1,
        offset=0,
        first_spike_time=0,
        time_var_input=False,
        download=False,
        num_images=20,
        min_time=None,
        max_time=None,
    ):
        self.T = num_images * num_steps

        self.data, self.labels = create_data(
            pixel_size=self.N_x,
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
            if min_time == None:
                min_time = 0
            if max_time == None:
                max_time = self.T
            spike_plot(self.data[min_time:max_time])

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
        self.resting_potential = resting_membrane

        # create other arrays
        self.mp, self.elig_trace, self.spikes = create_arrays(
            N=self.N,
            resting_membrane=self.resting_potential,
            total_time=self.T,
            data=self.data,
            N_x=self.N_x,
        )

        # return results if retur == True
        if retur:
            return self.weights, self.mp, self.elig_trace

    def train_network_(
        self,
        dt=1,
        membrane_resistance=20,
        membrane_conductance=5,
        very_small_value=0.00001,
        spike_threshold=-65,
        reset_potential=-80,
        tau_trace=0.1,
        plot_spikes=False,
        plot_mp=False,
        start_time_spike_plot=None,
        stop_time_spike_plot=None,
        start_index_mp=None,
        stop_index_mp=None,
        time_start_mp=None,
        time_stop_mp=None,
        mean_noise=0,
        var_noise=1,
        retur=False,
        save=True,
    ):
        self.dt = dt
        self.weights, self.spikes, self.elig_trace, self.mp = train_network(
            weights=self.weights,
            mp=self.mp,
            spikes=self.spikes,
            elig_trace=self.elig_trace,
            resting_potential=self.resting_potential,
            membrane_resistance=membrane_resistance,
            membrane_conductance=membrane_conductance,
            very_small_value=very_small_value,
            dt=self.dt,
            spike_threshold=spike_threshold,
            reset_potential=reset_potential,
            tau_trace=tau_trace,
            save=save,
            N_x=self.N_x,
            T=self.T,
            mean_noise=mean_noise,
            var_noise=var_noise,
        )
        if plot_spikes:
            if start_time_spike_plot == None:
                start_time_spike_plot = 0
            if stop_time_spike_plot == None:
                stop_time_spike_plot = self.T

            spike_plot(self.spikes[start_time_spike_plot:stop_time_spike_plot])

        if plot_mp:
            if start_index_mp == None:
                start_index_mp = self.N_x
            if stop_index_mp == None:
                stop_index_mp = -self.N_inh
            if time_start_mp == None:
                time_start_mp = 0
            if time_stop_mp == None:
                time_stop_mp = self.T

            mp_plot(
                mp=self.mp[time_start_mp:time_stop_mp, start_index_mp:stop_index_mp]
            )

        if retur:
            return self.weights, self.spikes, self.elig_trace
