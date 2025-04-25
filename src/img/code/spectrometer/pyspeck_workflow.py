import pathlib
import sys
import tempfile

import IPython
import matplotlib as mpl
import numpy as np
import scipy as sc

from python_spectrometer import daq, Spectrometer
from qutil import const, domains, functools

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import PATH, TEXTWIDTH  # noqa

if (ipy := IPython.get_ipython()) is not None:
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


# %%
qopt_daq = MockMFLIDAQ(spectrum)
speck = Spectrometer(qopt_daq, savepath=tempfile.mkdtemp(),
                     threaded_acquisition=False, purge_raw_data=False,
                     plot_negative_frequencies=False,
                     procfn=functools.scaled(1e6), processed_unit='μV',
                     plot_style='./main.mplstyle',
                     figure_kw=dict(layout='constrained'), legend_kw=dict(loc='lower left'))
settings = dict(f_min=1e1, f_max=1e5, n_avg=10, baseline=1e-16, delay=False,
                freq=0, filter_order=3)

# %%
speck.take('baseline', add_50hz=False, add_colored=False, **settings)
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 0.75)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_baseline.pdf')

# %%
speck.take('connected', add_50hz=True, exp=1, A=1e-13, **settings)
speck.take('lifted cable', add_50hz=True, exp=1, A=1e-13, **settings)
speck.take('jumped', add_50hz=True, exp=1, A=1e-13, **settings)
speck.leg.set_loc('lower left')
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_spectra.pdf')

# %%
speck.plot_timetrace = True
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 1.25)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_timetrace.pdf')

# %%
speck.plot_timetrace = False
speck.plot_cumulative = True
speck.plot_cumulative_normalized = False
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 1.25)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_cumulative.pdf')

# %%
speck.plot_timetrace = False
speck.plot_cumulative = False
speck.plot_amplitude = False
speck.plot_density = False
speck.plot_dB_scale = True
speck.set_reference_spectrum('baseline')
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 0.75)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_db.pdf')

# %%
for _ in range(6):
    speck.take('dud', **settings)
    speck.hide(-1)

# %%
speck.hide('lifted cable')
speck.hide('jumped')
speck.take('fixed', add_50hz=True, exp=1, A=1e-13, npeaks=1, **settings)

speck.plot_amplitude = True
speck.plot_dB_scale = False
speck.plot_density = True
speck.fig.set_size_inches(TEXTWIDTH, TEXTWIDTH / const.golden * 0.75)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_success.pdf')
