import pathlib
import sys
import tempfile
import threading
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from python_spectrometer import daq, Spectrometer
from qutil import const, domains, functools
from qutil.plotting import make_sequential_colormap

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import init, PATH, TEXTWIDTH, MAINSTYLE  # noqa

init(MAINSTYLE, backend='qt')

SEED = 1
rng = np.random.default_rng(SEED)


def spectrum(f, A=1e-4, exp=1, add_colored=True, add_50hz=False, baseline=0, npeaks=None,
             **_):
    f = np.asarray(f)
    S = np.zeros_like(f)

    if add_colored:
        with np.errstate(divide='ignore'):
            S += A / f ** exp

    if add_50hz:
        # sophisticated algorithm!
        harmonics = abs(f % 50) < np.diff(f).mean()
        idx, = harmonics.nonzero()
        p = sc.stats.beta.sf(np.linspace(0, 1, idx.size, endpoint=False), 5, 2)
        idx = rng.choice(idx, size=(min(10, idx.size) if npeaks is None else npeaks),
                         replace=False, p=p/p.sum())
        S[(idx,)] += 5e0 * A / (10 * f[f.nonzero()].min()) ** rng.random(size=idx.size)

    S += baseline
    return S


class MockMFLIDAQ(daq.simulator.DemodulatorQoptColoredNoise):

    @property
    def DAQSettings(self) -> type[daq.DAQSettings]:
        class MFLIDAQSettings(daq.DAQSettings):
            CLOCKBASE = 60e6
            # TODO: always the same for each instrument?
            ALLOWED_FS = domains.ExponentialDiscreteInterval(-23, 0, base=2,
                                                             prefactor=CLOCKBASE / 70)
            DEFAULT_FS = CLOCKBASE / 70 / 2**6

        return MFLIDAQSettings


def stop(timer, fig, path, max_callbacks: int):
    global CALLED_BACK
    if CALLED_BACK < max_callbacks:
        CALLED_BACK += 1
        return

    timer.stop()
    time.sleep(1)  # for good measure
    fig.savefig(path)
    plt.close(fig)


# %%
qopt_daq = MockMFLIDAQ(functools.partial(spectrum, add_50hz=True))
speck = Spectrometer(qopt_daq, savepath=tempfile.mkdtemp(),
                     threaded_acquisition=False, purge_raw_data=False,
                     plot_negative_frequencies=False,
                     procfn=functools.scaled(1e6), processed_unit='μV',
                     figure_kw=dict(layout='constrained'))
settings = dict(f_min=1e1, f_max=1e5, n_avg=10, baseline=2e-16, exp=1, A=1e-13, npeaks=50,
                freq=0, filter_order=3)
# %%
CALLED_BACK = 1

fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH / const.golden * 1.25))
timer = fig.canvas.new_timer(interval=1)
timer.add_callback(stop, timer, fig, PATH / 'pdf/spectrometer/live_view.pdf', max_callbacks=12)

# bordeaux has the largest dynamic range. at index 9 in the cycle
view, = speck.live_view(
    21,
    **settings,
    delay=False,
    in_process=False,
    live_view_kw=dict(
        event_source=timer,
        fig=fig,
        style=['fast', MAINSTYLE, {'axes.prop_cycle': (2 * mpl.rcParams['axes.prop_cycle'])[0:]}],
        img_kw=dict(cmap=make_sequential_colormap('blue').reversed()),
    )
)

# Required for script execution mode
plt.show(block=True)
