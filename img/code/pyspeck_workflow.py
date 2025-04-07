# %%
import sys
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from python_spectrometer import daq, Spectrometer
from qutil import const, functools
from qutil.plotting import RWTH_COLORS

from common import apply_sketch_style, MARGINWIDTH, PATH, TEXTWIDTH

SEED = 1
mpl.use('qtagg')
rng = np.random.default_rng(SEED)


def spectrum(f, A=1e-4, exp=1, add_colored=True, add_50hz=False, baseline=0, **_):
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
        idx = rng.choice(idx, size=min(25, idx.size), replace=False, p=p/p.sum())
        S[(idx,)] += 5e0 * A / (10 * f[f.nonzero()].min()) ** rng.random(size=idx.size)

    S += baseline
    return S


# %%
qopt_daq = daq.simulator.QoptColoredNoise(spectrum)
speck = Spectrometer(qopt_daq, savepath=tempfile.mkdtemp(),
                     threaded_acquisition=False, purge_raw_data=False,
                     procfn=functools.scaled(1e6), processed_unit='μV',
                     plot_style='./main.mplstyle', figure_kw=dict(layout='constrained'))
settings = dict(f_min=1e1, f_max=1e5, n_avg=10, baseline=1e-16, delay=False)

# %%
speck.take('baseline', add_50hz=False, add_colored=False, **settings)
speck.leg.set_loc('lower left')
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 0.75)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_baseline.pdf', backend='pdf')

# %%
speck.take('lifted cable', add_50hz=True, exp=1, A=1e-12, **settings)
speck.take('jumped', add_50hz=True, exp=1, A=1e-12, **settings)
speck.leg.set_loc('lower left')
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_spectra.pdf', backend='pdf')

# %%
speck.plot_timetrace = True
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 1.25)
speck.leg.set_loc('lower left')
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_timetrace.pdf', backend='pdf')

# %%
speck.plot_cumulative = True
speck.plot_cumulative_normalized = False
speck.leg.set_loc('lower left')
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 1.5)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_timetrace_cumulative.pdf', backend='pdf')

# %%
sys.exit(0)
# %%
demod_daq = daq.simulator.DemodulatorQoptColoredNoise(functools.partial(spectrum, add_50hz=True))
speck = Spectrometer(demod_daq, savepath=tempfile.mkdtemp())

# %%
speck.take(n_avg=5, freq=789, fs=13.4e3, nperseg=2 << 11, delay=True, modulate_signal=False)
plt.show()  # necessary if using notebook widget

# %%
speck.plot_negative_frequencies = False
speck.plot_absolute_frequencies = True
speck.ax[0].set_xscale('log')

# %% [markdown]
# For noise hunting, it is often convenient to continuously acquire noise spectra while changing things around the setup. For this, use the `live_view` method.
#
# Note: the plot seems to have some issues in `widget` mode, use `qt` for a more reliable plot.

# %%
views = speck.live_view(11, fs=13.4e3, nperseg=2<<11, delay=True,
                        in_process=False)

# %%
plt.close('all')
