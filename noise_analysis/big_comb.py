import numpy as np
from train import train_network
from get_data import create_data
from plot import spike_plot, heat_map, mp_plot, weights_plot
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
        recreate=False,
    ):
        self.T = num_images * num_steps

        self.data, self.labels = create_data(
            pixel_size=int(np.sqrt(self.N_x)),
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
            download=download,
            num_images=num_images,
            recreate=recreate,
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
        max_time=100,
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
        self.max_time = max_time

        # create other arrays
        self.mp, self.elig_trace, self.spikes, self.spike_times = create_arrays(
            N=self.N,
            resting_membrane=self.resting_potential,
            total_time=self.T,
            data=self.data,
            N_x=self.N_x,
            max_time=self.max_time,
        )

        # return results if retur == True
        if retur:
            return self.weights, self.mp, self.elig_trace, self.spike_times

    def train_network_(
        self,
        dt=1,
        tau_m=10,
        membrane_resistance=100,
        spike_threshold=-65,
        reset_potential=-80,
        plot_spikes=False,
        plot_weights=False,
        plot_mp=False,
        min_weight_exc=0.1,
        max_weight_exc=2,
        min_weight_inh=-0.1,
        max_weight_inh=-8,
        learning_rate_exc=0.0001,
        learning_rate_inh=-0.001,
        tau_pre=0.01,
        tau_post=0.01,
        interval=100,
        start_time_spike_plot=None,
        stop_time_spike_plot=None,
        start_index_mp=None,
        stop_index_mp=None,
        time_start_mp=None,
        time_stop_mp=None,
        mean_noise=0,
        var_noise=1,
        load_weights=False,
        retur=False,
        save=True,
    ):
        self.dt = dt
        self.weights, self.spikes, self.elig_trace, self.mp, self.weights2plot = (
            train_network(
                weights=self.weights,
                mp=self.mp,
                spikes=self.spikes,
                elig_trace=self.elig_trace,
                resting_potential=self.resting_potential,
                membrane_resistance=membrane_resistance,
                min_weight_exc=min_weight_exc,
                max_weight_exc=max_weight_exc,
                min_weight_inh=min_weight_inh,
                max_weight_inh=max_weight_inh,
                N_inh=self.N_inh,
                N_exc=self.N_exc,
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                interval=interval,
                tau_pre=tau_pre,
                tau_post=tau_post,
                tau_m=tau_m,
                dt=self.dt,
                N=self.N,
                spike_threshold=spike_threshold,
                reset_potential=reset_potential,
                spike_times=self.spike_times,
                save=save,
                N_x=self.N_x,
                T=self.T,
                mean_noise=mean_noise,
                var_noise=var_noise,
            )
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

        if plot_weights:
            weights_plot(weights_plot=self.weights2plot, N_x=self.N_x, N_inh=self.N_inh)

        if retur:
            return self.weights, self.spikes, self.elig_trace
