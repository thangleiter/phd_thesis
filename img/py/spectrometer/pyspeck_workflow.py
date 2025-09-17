import pathlib
import sys
import tempfile

import numpy as np
import scipy as sp
from python_spectrometer import daq, Spectrometer
from qutil import domains, functools

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import init, PATH, TEXTWIDTH, MAINSTYLE  # noqa

init(MAINSTYLE, backend='qt')

SEED = 42
rng = np.random.default_rng(SEED)
# %% Functions


def spectrum(f, A=1e-4, exp=0.7, add_colored=True, add_50hz=False, baseline=0, npeaks=None,
             seed=None, **_):

    rng = np.random.default_rng(seed)

    f = np.asarray(f)
    S = np.zeros_like(f)

    if add_colored:
        with np.errstate(divide='ignore'):
            S += A / 3 / f ** exp

    if add_50hz:
        # sophisticated algorithm!
        harmonics = abs(f % 50) < np.diff(f).mean()
        idx, = harmonics.nonzero()
        p = sp.stats.beta.sf(np.linspace(0, 1, idx.size, endpoint=False), 5, 2)
        idx = rng.choice(idx, size=(min(15, idx.size) if npeaks is None else npeaks),
                         replace=False, p=p/p.sum())
        S[(idx,)] += 10e0 * A / (10 * f[f.nonzero()].min()) ** rng.random(size=idx.size)

    S += baseline
    return S


class MockMFLIDAQ(daq.simulator.DemodulatorQoptColoredNoise):
    SEED = None

    @property
    def DAQSettings(self) -> type[daq.DAQSettings]:
        class MFLIDAQSettings(daq.DAQSettings):
            CLOCKBASE = 60e6
            ALLOWED_FS = domains.ExponentialDiscreteInterval(-23, 0, base=2,
                                                             prefactor=CLOCKBASE / 70)
            DEFAULT_FS = CLOCKBASE / 70 / 2**6

        return MFLIDAQSettings

    def setup(self, **kwargs):
        # reproducibility
        if self.SEED is None:
            self.SEED = SEED
        else:
            self.SEED += 1
        return super().setup(**(kwargs | {'seed': self.SEED}))


# %% Set up Speck
qopt_daq = MockMFLIDAQ(spectrum)
speck = Spectrometer(qopt_daq, savepath=tempfile.mkdtemp(),
                     threaded_acquisition=False,
                     plot_negative_frequencies=False,
                     procfn=functools.scaled(1e6),
                     processed_unit='μV',
                     plot_style=MAINSTYLE,
                     legend_kw=dict(loc='lower left'),
                     figure_kw=dict(layout='constrained'))
settings = dict(f_min=1e1, f_max=1e5, n_avg=10, baseline=1e-16, delay=False,
                freq=0, filter_order=3)

# %% Take baseline
speck.take('baseline', add_50hz=False, add_colored=False, **settings)
speck.fig.set_size_inches(TEXTWIDTH, 2)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_baseline.pdf')

# %% Take spectra
speck.take('connected', add_50hz=True, exp=1, A=1e-13, **settings)
speck.take('lifted cable', add_50hz=True, exp=1, A=1e-13, **settings)
speck.take('jumped', add_50hz=True, exp=1, A=1e-13, **settings)
speck.leg.set_loc('lower left')
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_spectra.pdf')

# %% Plot timetrace
speck.plot_timetrace = True
speck.fig.set_size_inches(TEXTWIDTH, 3.25)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_timetrace.pdf')

# %% Plot cumulative
speck.plot_timetrace = False
speck.plot_cumulative = True
speck.plot_cumulative_normalized = False
speck.fig.set_size_inches(TEXTWIDTH, 3.25)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_cumulative.pdf')

# %% Plot dB
speck.plot_timetrace = False
speck.plot_cumulative = False
speck.plot_amplitude = False
speck.plot_density = False
speck.plot_dB_scale = True
speck.set_reference_spectrum('connected')
speck.fig.set_size_inches(TEXTWIDTH, 2)
speck.ax[0].legend(handles=[speck._plot_manager.lines[key]['main']['processed']['line']
                            for key in speck._plot_manager.shown],
                   labels=speck._plot_manager.shown,
                   loc='best')
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_db.pdf')

# %% Take some dummy spectra
for _ in range(6):
    speck.take('dud', **(settings | dict(n_avg=1)))

speck.hide(slice(len(speck) - 6, None))

# %% Plot final
speck.hide('lifted cable')
speck.hide('jumped')
speck.take('fixed', add_50hz=True, exp=1, A=1e-13, npeaks=3, **settings)

speck.plot_amplitude = True
speck.plot_dB_scale = False
speck.plot_density = True
speck.fig.set_size_inches(TEXTWIDTH, 2)
speck.fig.savefig(PATH / 'pdf/spectrometer/workflow_success.pdf')
