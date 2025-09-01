# %% Imports
import copy
import os
import pathlib
import sys
import time

import filter_functions as ff
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import qutip as qt
import filter_functions as ff
import lindblad_mc_tools as lmt
from filter_functions import plotting
from filter_functions.util import get_indices_from_identifiers
from cycler import cycler
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_50, make_diverging_colormap
from qutip.control import pulseoptim
from qutip.qip import operations
from qutip.qip.algorithms.qft import qft as qt_qft
from scipy import integrate

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, TEXTWIDTH, TOTALWIDTH, PATH, init, save_pulse_sequence,
                    load_pulse_sequence, prune_nopers)

with np.errstate(divide='ignore', invalid='ignore'):
    DIVERGING_CMAP = make_diverging_colormap(('magenta', 'green'), endpoint='white')

RUN_SIMULATION = os.environ.get('RUN_SIMULATION', "False") == "True"
SEED = 42
THREADS = None
CHUNK_NOISE = 8
SHOW_PROGRESSBAR = False
LINE_COLORS = list(RWTH_COLORS.values())
FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'pgf')
pernanosecond = r' (\unit{\per\nano\second})' if backend == 'pgf' else r' (ns$^{-1}$)'
# %% Load pulses, define parameters
identifiers = [r'$\sigma_y^{(3)}$']
qft_pulse = prune_nopers(load_pulse_sequence(DATA_PATH / 'pulse_sequence_qft_pulse.npz'),
                         identifiers)

qft_pulse_echo = prune_nopers(load_pulse_sequence(DATA_PATH / 'pulse_sequence_qft_pulse_echo.npz'),
                              identifiers)

rng = np.random.default_rng(SEED)
n_MC = 1000
omega = qft_pulse.omega
f = omega/2/np.pi
max_frequency_sample_spacing = f[0]/10

pulses = {'echo': qft_pulse_echo, 'noecho': qft_pulse}
pulses_dc = {key: copy.deepcopy(pulse) for key, pulse in pulses.items()}
pulses_all = {key: copy.deepcopy(pulse) for key, pulse in pulses.items()}
S_0 = {'pink': 1e-9, 'white': 2e-6}
spectra = {'white': lmt.noise.WhiteSpectralFunction(S_0['white']),
           'pink': lmt.noise.PowerLawSpectralFunction(S_0['pink'], -1)}
# %% GKSL
lbs_echo = lmt.LindbladSolver(qft_pulse_echo, spectra['white'], show_progressbar=SHOW_PROGRESSBAR)
lbs_noecho = lmt.LindbladSolver(qft_pulse, spectra['white'], show_progressbar=SHOW_PROGRESSBAR)
# %%% Evaluate
if RUN_SIMULATION:
    lbs_echo.solve()
    np.savez_compressed(DATA_PATH / 'lb_propagator_qft_pulse_echo.npz',
                        complete_propagator=lbs_echo.complete_propagator)
    lbs_noecho.solve()
    np.savez_compressed(DATA_PATH / 'lb_propagator_qft_pulse.npz',
                        complete_propagator=lbs_noecho.complete_propagator)
else:
    with np.load(DATA_PATH / 'lb_propagator_qft_pulse_echo.npz') as arch:
        lbs_echo._complete_propagator = arch['complete_propagator']
    with np.load(DATA_PATH / 'lb_propagator_qft_pulse.npz') as arch:
        lbs_noecho._complete_propagator = arch['complete_propagator']
# %%% Fidelity
lbm_echo = lmt.EntanglementFidelity(lbs_echo)
lbm_noecho = lmt.EntanglementFidelity(lbs_noecho)
lbm_echo.evaluate()
lbm_noecho.evaluate()

print('GKSL infidelities (x 1e3)')
print(f' W/ Echo:\tWhite\t{(1-lbm_echo.value)*1e3:.3g}')
print(f'W/O echo:\tWhite\t{(1-lbm_noecho.value)*1e3:.3g}')
# %% MC
mcs_echo = {
    key: lmt.SchroedingerSolver(
        qft_pulse_echo, spectra[key],
        show_progressbar=SHOW_PROGRESSBAR,
        traces_shape=n_MC,
        f_max=f[-1],
        max_frequency_sample_spacing=max_frequency_sample_spacing,
        chunk_noise=CHUNK_NOISE,
        threads=THREADS,
        seed=SEED
    ) for key in spectra
}
mcs_noecho = {
    key: lmt.SchroedingerSolver(
        qft_pulse, spectra[key],
        show_progressbar=SHOW_PROGRESSBAR,
        traces_shape=n_MC,
        f_max=f[-1],
        max_frequency_sample_spacing=max_frequency_sample_spacing,
        chunk_noise=CHUNK_NOISE,
        threads=THREADS,
        seed=SEED
    ) for key in spectra
}
# %%% Evaluate
if RUN_SIMULATION:
    for key in mcs_echo:
        mcs_echo[key].solve()
        np.savez_compressed(
            DATA_PATH / f'mc_propagator_qft_pulse_echo_{key}.npz',
            complete_propagator_mean=[mcs_echo[key].mean(mcs_echo[key].complete_propagator)],
        )
        mcs_noecho[key].solve()
        np.savez_compressed(
            DATA_PATH / f'mc_propagator_qft_pulse_{key}.npz',
            complete_propagator=[mcs_noecho[key].mean(mcs_noecho[key].complete_propagator)],
        )
else:
    with np.load(DATA_PATH / 'mc_propagator_qft_pulse_echo.npz') as arch:
        mcs_echo._complete_propagator = arch['complete_propagator_mean']
    with np.load(DATA_PATH / 'mc_propagator_qft_pulse.npz') as arch:
        mcs_noecho._complete_propagator = arch['complete_propagator_mean']
# %%% Fidelity
mcm_echo = {key: lmt.EntanglementFidelity(mcs_echo[key]) for key in mcs_echo}
mcm_noecho = {key: lmt.EntanglementFidelity(mcs_noecho[key]) for key in mcs_noecho}

print('MC infidelities (x 1e3)')
for key in mcm_echo:
    print(f' W/ Echo:\t{key}\t{(1-mcm_echo[key].value.mean(0))*1e3:.3g}')
    print(f'W/O echo:\t{key}\t{(1-mcm_noecho[key].value.mean(0))*1e3:.3g}')

# %% FF
ffs_echo = {
    key: lmt.FilterFunctionSolver(
        qft_pulse_echo, spectra[key], show_progressbar=SHOW_PROGRESSBAR, omega=omega,
        second_order=False
    )
    for key in spectra
}
ffs_noecho = {
    key: lmt.FilterFunctionSolver(
        qft_pulse, spectra[key], show_progressbar=SHOW_PROGRESSBAR, omega=omega,
        second_order=False
    )
    for key in spectra
}

# %%% Evaluate
# Nothing to do, the stored pulses have the fidelity filter function cached. Running .solve() on
# the solvers will compute the generalized filter function for the entire error propagator.

# %%% Fidelity
ffm_echo = {key: lmt.metrics.FilterFunctionEntanglementFidelity(ffs_echo[key]) for key in ffs_echo}
ffm_noecho = {key: lmt.metrics.FilterFunctionEntanglementFidelity(ffs_noecho[key])
              for key in ffs_noecho}

print('FF infidelities (x 1e3)')
for key in ffm_echo:
    print(f' W/ Echo:\t{key}\t{(1-ffm_echo[key].value)*1e3:.3g}')
    print(f'W/O echo:\t{key}\t{(1-ffm_noecho[key].value)*1e3:.3g}')

# %% Save fidelities
