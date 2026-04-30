# performance.py
import threading
import psutil
import os
from plot import heatmap_spike_response


def report_RAM_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory before training: {process.memory_info().rss / 1024**2:.0f} MB")


def _plot_background(kwargs):
    heatmap_spike_response(**kwargs)


def spawn_plot_thread(
    t,
    spikes,
    spike_trace,
    spike_labels,
    weights,
    x_tar_se,
    x_tar_ee,
    num,
    iterations,
    st,
    ex,
    ih,
    dataset,
    run,
    save_plots,
):
    if not save_plots:
        return num, None
    plot_kwargs = dict(
        spikes_exc=spikes[t - iterations - 1 : t - 1, st:ex].copy(),
        spikes_in=spikes[t - iterations - 1 : t - 1, :st].copy(),
        spikes_ih=spikes[t - iterations - 1 : t - 1, ex:].copy(),
        label=spike_labels[t - 1],
        spike_trace=spike_trace.copy(),
        dataset=dataset,
        run=run,
        num=num,
        st=st,
        ex=ex,
        x_target_se=x_tar_se,
        x_target_ex=x_tar_ee,
        weights_st_ex=weights[:st, st:ex].copy(),
        weights_ex_ex=weights[st:ex, st:ex].copy(),
        weights_ex_ih=weights[st:ex, ex:ih].copy(),
        weights_ih_ex=weights[ex:ih, st:ex].copy(),
    )
    thread = threading.Thread(target=_plot_background, args=(plot_kwargs,), daemon=True)
    thread.start()
    return num + 1, thread
