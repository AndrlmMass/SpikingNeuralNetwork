import numpy as np
from train import train_network
from get_data import create_data
from plot import spike_plot, heat_map, mp_plot, weights_plot, spike_threshold_plot
from create_network import create_weights, create_arrays


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
        add_breaks=True,
        break_lengths=[500, 400, 300],
        noisy_data=True,
        noise_level=0.05,
    ):

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
            add_breaks=add_breaks,
            break_lengths=break_lengths,
            noisy_data=noisy_data,
            noise_level=noise_level,
        )
        self.T = self.data.shape[0]

        # plot spikes
        if plot_spikes:
            if min_time == None:
                min_time = 0
            if max_time == None:
                max_time = self.T
            spike_plot(self.data[min_time:max_time], self.labels)

        # plot heatmap of activity
        if plot_heat_map:
            heat_map(self.data, pixel_size=10)

        # return data and labels if needed
        if retur:
            return self.data, self.labels

    def prepare_training(
        self,
        weight_affinity_hidden_exc=0.1,
        weight_affinity_hidden_inh=0.2,
        weight_affinity_input=0.05,
        resting_membrane=-70,
        max_time=100,
        pos_weight=0.5,
        neg_weight=-1,
        plot_weights=False,
        retur=False,
    ):
        # create weights
        self.weights = create_weights(
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            N_x=self.N_x,
            weight_affinity_hidden_exc=weight_affinity_hidden_exc,
            weight_affinity_hidden_inh=weight_affinity_hidden_inh,
            weight_affinity_input=weight_affinity_input,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            plot_weights=plot_weights,
        )
        self.resting_potential = resting_membrane
        self.max_time = max_time

        # create other arrays
        (
            self.mp,
            self.elig_trace,
            self.spikes,
            self.spike_times,
        ) = create_arrays(
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
        tau_m=30,
        membrane_resistance=100,
        spike_threshold_default=int(-65),
        reset_potential=-80,
        noisy_threshold=False,
        spike_slope=-0.1,
        spike_intercept=-4,
        plot_spikes=False,
        plot_weights=False,
        plot_mp=False,
        plot_threshold=False,
        min_weight_exc=0.01,
        max_weight_exc=2,
        min_weight_inh=-4,
        max_weight_inh=-0.01,
        learning_rate_exc=0.5,
        learning_rate_inh=0.5,
        update_weights=True,
        tau_decay_exc=9.5,
        tau_decay_inh=9.5,
        tau_LTP=100,
        tau_LTD=100,
        w_interval=5,
        interval=100,
        spike_adaption=False,
        delta_adaption=0.3,
        tau_adaption=1,
        start_time_spike_plot=None,
        stop_time_spike_plot=None,
        start_index_mp=None,
        stop_index_mp=None,
        time_start_mp=None,
        time_stop_mp=None,
        noisy_potential=False,
        mean_noise=0,
        var_noise=5,
        max_mp=40,
        min_mp=-100,
        weight_decay_rate_exc=1.5,
        weight_decay_rate_inh=1.5,
        noisy_weights=False,
        weight_mean_noise=0.05,
        weight_var_noise=0.005,
        retur=False,
        save=True,
    ):
        self.dt = dt
        (
            self.weights,
            self.spikes,
            self.elig_trace,
            self.mp,
            self.weights2plot,
            self.spike_threshold,
        ) = train_network(
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
            weight_decay_rate_exc=weight_decay_rate_exc,
            weight_decay_rate_inh=weight_decay_rate_inh,
            update_weights_=update_weights,
            learning_rate_exc=learning_rate_exc,
            learning_rate_inh=learning_rate_inh,
            w_interval=w_interval,
            interval=interval,
            tau_decay_exc=tau_decay_exc,
            tau_decay_inh=tau_decay_inh,
            tau_LTP=tau_LTP,
            tau_LTD=tau_LTD,
            tau_m=tau_m,
            max_mp=max_mp,
            min_mp=min_mp,
            dt=self.dt,
            N=self.N,
            spike_adaption=spike_adaption,
            delta_adaption=delta_adaption,
            tau_adaption=tau_adaption,
            spike_threshold_default=spike_threshold_default,
            spike_intercept=spike_intercept,
            spike_slope=spike_slope,
            noisy_threshold=noisy_threshold,
            reset_potential=reset_potential,
            spike_times=self.spike_times,
            noisy_potential=noisy_potential,
            noisy_weights=noisy_weights,
            weight_mean_noise=weight_mean_noise,
            weight_var_noise=weight_var_noise,
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

            spike_plot(
                self.spikes[start_time_spike_plot:stop_time_spike_plot], self.labels
            )

        if plot_threshold:
            spike_threshold_plot(spike_threshold, N_exc)

        if plot_mp:
            if start_index_mp == None:
                start_index_mp = self.N_x
            if stop_index_mp == None:
                stop_index_mp = self.N_exc + self.N_inh
            if time_start_mp == None:
                time_start_mp = 0
            if time_stop_mp == None:
                time_stop_mp = self.T

            mp_plot(
                mp=self.mp[time_start_mp:time_stop_mp],
                N_exc=self.N_exc,
            )

        if plot_weights:
            weights_plot(weights_plot=self.weights2plot, N_x=self.N_x, N_inh=self.N_inh)

        if retur:
            return self.weights, self.spikes, self.elig_trace
