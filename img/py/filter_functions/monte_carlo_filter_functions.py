# %% Imports
import pathlib
import sys
import warnings
from typing import Literal

import filter_functions as ff
import lindblad_mc_tools as lmt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe
import scipy as sp
from qutil import misc, ui
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_75, RWTH_COLORS_50, RWTH_COLORS_25

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, MARGINSTYLE, TOTALWIDTH, MARGINWIDTH, PATH, init

try:
    import matlab.engine

    eng = matlab.engine.start_matlab()
    # We require the Matrix Function Toolbox and the Logm Frechet package:
    # https://www.mathworks.com/matlabcentral/fileexchange/20820-the-matrix-function-toolbox
    eng.sqrtm_real(np.eye(2))
    # https://www.mathworks.com/matlabcentral/fileexchange/38894-matrix-logarithm-with-frechet-derivatives-and-condition-number  noqa
    eng.logm_frechet_real(np.eye(2), np.eye(2))
except (ImportError, Exception):
    eng = None

LINE_COLORS = list(RWTH_COLORS.values())
LINE_COLORS_75 = list(RWTH_COLORS_75.values())
LINE_COLORS_50 = list(RWTH_COLORS_50.values())
LINE_COLORS_25 = list(RWTH_COLORS_25.values())
FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'pgf')

pernanosecond = r' (\unit{\per\nano\second})' if backend == 'pgf' else r' (ns$^{-1}$)'
# %%


def monte_carlo_filter_function(pulse, omega, sigma=1.0, traces_shape=400,
                                confidence_method: Literal['frechet', 'bootstrap'] = 'frechet',
                                **solver_kwargs):

    if eng is None and confidence_method == 'frechet':
        warnings.warn('matlab engine / logm_frechet not installed.')
        confidence_method = 'bootstrap'

    expr = oe.contract_expression('...ba,ibc,...cd,jda->...ij',
                                  (traces_shape, pulse.d, pulse.d), pulse.basis,
                                  (traces_shape, pulse.d, pulse.d), pulse.basis,
                                  optimize=[(0, 1), (0, 1), (0, 1)],
                                  constants=[1, 3])

    def to_liouville(U):
        return expr(U.conj(), U).real

    K_mean = np.empty((pulse.d**2, pulse.d**2, len(omega)))
    K_conf = np.empty((2, pulse.d**2, pulse.d**2, len(omega)))
    Us = []

    for i, freq in enumerate(ui.progressbar(omega)):
        solver = lmt.MonochromaticSchroedingerSolver(
            pulse, lmt.noise.MonochromaticSpectralFunction(sigma, freq, frequency_type='angular'),
            sampling_method=solver_kwargs.pop('sampling_method', 'monte carlo'), **solver_kwargs
        )
        U = to_liouville(solver.complete_propagator)
        U_mean = solver.mean(U)
        U_conf = solver.confidence_interval(U, confidence_level=0.6827)

        Uerr_mean = pulse.total_propagator_liouville.T @ U_mean
        Uerr_conf = pulse.total_propagator_liouville.T @ U_conf
        try:
            if confidence_method == 'frechet':
                for j in range(2):
                    X, L = map(np.array, eng.logm_frechet_real(
                        Uerr_mean, Uerr_conf[j] - Uerr_mean, nargout=2
                    ))
                    K_conf[j, ..., i] = X + L
                K_mean[..., i] = X
            else:
                with misc.filter_warnings(action='error', category=np.exceptions.ComplexWarning):
                    K_mean[..., i] = sp.linalg.logm(Uerr_mean)
                    K_conf[..., i] = solver.confidence_interval(
                        sp.linalg.logm(pulse.total_propagator_liouville.T @ U),
                        confidence_level=0.6827
                    )
        except (np.exceptions.ComplexWarning, Exception) as err:
            raise ValueError('U is not positive semidefinite; reduce sigma') from err

        Us.append(U)

    return -K_mean / sigma**2, -K_conf[::-1] / sigma**2, np.array(Us)


def incoherent_filter_function(pulse, omega):
    traces = pulse.basis.four_element_traces
    basis_FF = pulse.get_filter_function(omega, 'generalized', order=1)
    return (
        + oe.contract('...klo,klji->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,kjli->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,kilj->...ijo', basis_FF, traces, backend='sparse').real
        + oe.contract('...klo,kijl->...ijo', basis_FF, traces, backend='sparse').real
    ) / 2


def coherent_filter_function(pulse, omega):
    traces = pulse.basis.four_element_traces
    basis_FF = pulse.get_filter_function(omega, 'generalized', order=2)
    return (
        + oe.contract('...klo,klji->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,lkji->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,klij->...ijo', basis_FF, traces, backend='sparse').real
        + oe.contract('...klo,lkij->...ijo', basis_FF, traces, backend='sparse').real
    ) / 2


def to_symmetric(X, axes=(0, 1)):
    return 0.5*(X + X.conj().swapaxes(*axes))


def to_antisymmetric(X, axes=(0, 1)):
    return 0.5*(X - X.conj().swapaxes(*axes))


# %%
# pulse = testutil.rand_pulse_sequence(2, 10, n_nops=1, commensurable_timesteps=True)
n_dt = 1
X_pi = ff.PulseSequence(
    [[ff.util.paulis[1]/2, [np.pi]*n_dt, 'X']],
    [[ff.util.paulis[3]/2, [1]*n_dt, 'Z']],
    [1 / n_dt]*n_dt
)
Z_pi = ff.PulseSequence(
    [[ff.util.paulis[3]/2, [np.pi]*n_dt, 'Z']],
    [[ff.util.paulis[3]/2, [1]*n_dt, 'Z']],
    [1 / n_dt]*n_dt
)
Idle = ff.PulseSequence(
    [[ff.util.paulis[3]/2, [0]*n_dt, 'Z']],
    [[ff.util.paulis[3]/2, [1]*n_dt, 'Z']],
    [1 / n_dt]*n_dt
)
# pulse.dt *= 1e-2
# pulse.c_coeffs *= 1e2
# %%
pulse = ff.concatenate_periodic(Idle, 10) @ X_pi @ ff.concatenate_periodic(Idle, 10)
omega = ff.util.get_sample_frequencies(pulse, n_samples=300, include_quasistatic=True)

FF_Γ = incoherent_filter_function(pulse, omega)
FF_Δ = coherent_filter_function(pulse, omega)
# %%%
# Too large a sigma results in crass overrotations and bad convergence.
# Too small results in statistical noise
mean, conf, Us = monte_carlo_filter_function(
    pulse, omega, sigma=0.5/pulse.duration,
    confidence_method='frechet',
    oversampling=10, traces_shape=400,
    threads=None
)
# %%
fig, axes = plt.subplots(pulse.d**2 - 1, pulse.d**2 - 1, sharex=True, layout='constrained',
                         figsize=(TOTALWIDTH, 3.5))

axes[0, 1].sharey(axes[0, 2])
axes[0, 2].sharey(axes[1, 2])
axes[1, 2].sharey(axes[1, 0])
axes[1, 0].sharey(axes[2, 0])
axes[2, 0].sharey(axes[2, 1])
axes[0, 0].sharey(axes[1, 1])
axes[1, 1].sharey(axes[2, 2])

for i in range(1, pulse.d**2):
    for j in range(1, pulse.d**2):
        ax1 = axes[i-1, j-1]
        ax2 = ax1
        # ax2 = ax1.twinx()
        # ax1.grid()

        if i == j:
            ax1.set_yscale('log')
        else:
            ax1.set_yscale('asinh', linear_width=0.075)
        ax1.semilogx(omega, to_symmetric(mean)[i, j], color=LINE_COLORS_50[1])
        ax1.fill_between(omega, *to_symmetric(conf, (1, 2))[:, i, j],
                         alpha=0.33, color=LINE_COLORS_25[1])
        ax2.semilogx(omega, to_antisymmetric(mean)[i, j], ls='--', color=LINE_COLORS_50[2])
        ax2.fill_between(omega, *to_antisymmetric(conf, (1, 2))[:, i, j],
                         alpha=0.33, color=LINE_COLORS_25[2], ls='--')

        ax1.semilogx(omega, FF_Γ[0, 0, i, j], '-', color=LINE_COLORS[1])
        ax2.semilogx(omega, FF_Δ[0, 0, i, j], '--', color=LINE_COLORS[2])

axes[0, 1].set_ylim(-10, 10)
