import threading
import queue
import psutil
import os
from neurosnn._plot.spikes import heatmap_spike_response

_plot_queue = queue.Queue()
_plot_worker = None


def _worker_loop():
    while True:
        kwargs = _plot_queue.get()
        if kwargs is None:
            break
        heatmap_spike_response(**kwargs)
        _plot_queue.task_done()


def start_plot_worker():
    global _plot_worker
    _plot_worker = threading.Thread(target=_worker_loop, daemon=True)
    _plot_worker.start()


def stop_plot_worker():
    _plot_queue.put(None)


def report_RAM_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory before training: {process.memory_info().rss / 1024**2:.0f} MB")


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

    _plot_queue.put(
        dict(
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
    )
    return num + 1, None
