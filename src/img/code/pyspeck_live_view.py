import tempfile

import IPython
import matplotlib as mpl
import numpy as np
import scipy as sc

from python_spectrometer import daq, Spectrometer
from qutil import const, functools
from qutil.plotting import make_sequential_colormap

from common import PATH, TEXTWIDTH

ipy = IPython.get_ipython()
ipy.run_line_magic('matplotlib', 'qt')

mpl.rcdefaults()
mpl.style.use('main.mplstyle')

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


def stop(timer, max_callbacks: int):
    global CALLED_BACK
    if CALLED_BACK >= max_callbacks:
        timer.stop()
    else:
        CALLED_BACK += 1


# %%
# TODO: Matplotlib does not automatically relim the y axis based on new x limits.
#       - always index data?
daq = daq.simulator.QoptColoredNoise(functools.partial(spectrum, add_50hz=True))
speck = Spectrometer(daq, savepath=tempfile.mkdtemp(),
                     threaded_acquisition=False, purge_raw_data=False,
                     procfn=functools.scaled(1e6), processed_unit='μV',
                     figure_kw=dict(layout='constrained'))
# speck.plot_negative_frequencies = False
# speck.plot_absolute_frequencies = True

# %%
CALLED_BACK = 1
timer = mpl.backends.backend_qt.TimerQT(1)
timer.add_callback(stop, timer, max_callbacks=12)

# bordeaux has the largest dynamic range. at index 9 in the cycle
view, = speck.live_view(
    21,
    f_min=1e1, f_max=1e5, exp=1, A=1e-13, baseline=1e-16,
    delay=False, in_process=False,
    live_view_kw=dict(
        event_source=timer,
        style=['fast', {'axes.prop_cycle': (2 * mpl.rcParams['axes.prop_cycle'])[0:]}],
        img_kw=dict(cmap=make_sequential_colormap('blue').reversed()),
        fig_kw=dict(figsize=(TEXTWIDTH, TEXTWIDTH / const.golden * 1.25))
    )
)

# %%%
view.stop()
view.fig.savefig(PATH / 'pdf/spectrometer/live_view.pdf')
